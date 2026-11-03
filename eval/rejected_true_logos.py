#!/usr/bin/env python3
"""T63 follow-up (``spec docs/17``): why the router rejects boxes that match a
gold logo.

Reads ``reports/iou_match_matched.csv`` (written by ``eval/iou_match.py``) and
the run's ``rows.csv``. Writes ``reports/rejected_true_logos.md`` + ``.csv``
and ``rejected_true_logos.png``:

* box size (short side in pixels, ``box_area_ratio``) and the generated
  ``confidence`` for rejected versus kept true logos (kept = ACCEPT, REVIEW,
  UNKNOWN_LOGO; DEDUPED rows are left out of both);
* how often a few words appear in the VLM's ``reasoning`` on the rejected ones,
  and a seeded sample of those reasonings in full;
* a contact sheet of up to ``--sheet`` rejected crops, labelled with
  ``det_id``, the gold class and the generated confidence.

Usage::

    python eval/rejected_true_logos.py --run runs/<video_id>/<run_id> [--sample 25] [--sheet 48] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import md_table, pct, reports_dir, write_csv  # noqa: E402

KEPT = ("ACCEPT", "REVIEW", "UNKNOWN_LOGO")
WORDS = ("text", "blur", "small", "unclear", "background", "banner", "advertis", "pattern", "partial", "sign")
TILE_W, TILE_H, LABEL_H, COLS = 180, 110, 28, 8


def short_side(box) -> float | None:
    if isinstance(box, str):
        box = json.loads(box) if box else None
    if not box or len(box) != 4:
        return None
    return float(min(box[2] - box[0], box[3] - box[1]))


def quartiles(values: list[float]) -> str:
    v = [x for x in values if x is not None and x == x]
    if not v:
        return "—"
    q = np.percentile(np.asarray(v, dtype=float), [25, 50, 75])
    return " / ".join(f"{x:.3g}" for x in q)


def load(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """``(rejected, kept)`` matched rows, each the ``rows.csv`` record plus ``gold_class``."""
    import pandas as pd
    matched = pd.read_csv(Path(run_dir) / "reports" / "iou_match_matched.csv")
    rows = pd.read_csv(Path(run_dir) / "rows.csv").set_index("det_id")
    rejected, kept = [], []
    for m in matched.to_dict("records"):
        if m["det_id"] not in rows.index:
            continue
        r = dict(rows.loc[m["det_id"]])
        r.update(det_id=int(m["det_id"]), gold_class=m["gold_class"], short_side=short_side(r.get("loose_bbox")))
        if m["router_decision"] == "REJECT":
            rejected.append(r)
        elif m["router_decision"] in KEPT:
            kept.append(r)
    return rejected, kept


def contact_sheet(run_dir: Path, rows: list[dict], path: Path) -> int:
    from PIL import Image, ImageDraw
    tiles = []
    for r in rows:
        p = r.get("crop_path")
        if not isinstance(p, str) or not (Path(run_dir) / p).exists():
            continue
        im = Image.open(Path(run_dir) / p).convert("RGB")
        im.thumbnail((TILE_W, TILE_H))
        tiles.append((im, f"{r['det_id']} {r['gold_class']}"[:28], f"conf {r.get('confidence')}"))
    if not tiles:
        return 0
    n_rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE_W, n_rows * (TILE_H + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, top, bottom) in enumerate(tiles):
        x, y = (i % COLS) * TILE_W, (i // COLS) * (TILE_H + LABEL_H)
        sheet.paste(im, (x + (TILE_W - im.width) // 2, y + (TILE_H - im.height) // 2))
        draw.text((x + 3, y + TILE_H + 1), top, fill="black")
        draw.text((x + 3, y + TILE_H + 14), bottom, fill="gray")
    sheet.save(path)
    return len(tiles)


def render(run_dir: Path, rejected: list[dict], kept: list[dict], sample: list[dict], n_sheet: int) -> str:
    def stats(rows):
        return (len(rows), quartiles([r["short_side"] for r in rows]),
                pct(sum(1 for r in rows if r["short_side"] is not None and r["short_side"] < 32), len(rows)),
                quartiles([r.get("box_area_ratio") for r in rows]), quartiles([r.get("confidence") for r in rows]))

    lines = [f"# Rejected true logos — `{run_dir}`", "",
             "Boxes that match a gold logo (IoU ≥ the `iou_match` threshold) and that the router REJECTed, "
             "against the matched boxes it kept (ACCEPT, REVIEW, UNKNOWN_LOGO). Quartiles are p25 / p50 / p75.", ""]
    lines += md_table(["group", "n", "short side px", "short side < 32 px", "box_area_ratio", "generated confidence"],
                      [("rejected", *stats(rejected)), ("kept", *stats(kept))])
    reasons = [str(r.get("reasoning") or "").lower() for r in rejected]
    lines += ["", "## Words in the VLM's reasoning on rejected true logos", ""]
    lines += md_table(["word", "rejected rows mentioning it", "share"],
                      [(w, n, pct(n, len(reasons))) for w in WORDS
                       for n in [sum(1 for t in reasons if w in t)]])
    lines += ["", f"## Sample of {len(sample)} rejected true logos", ""]
    lines += md_table(["det_id", "gold class", "short side px", "confidence", "reasoning"],
                      [(r["det_id"], r["gold_class"], "—" if r["short_side"] is None else int(r["short_side"]),
                        r.get("confidence"), str(r.get("reasoning") or "").replace("|", "/").replace("\n", " "))
                       for r in sample])
    lines += ["", f"Contact sheet: `rejected_true_logos.png` ({n_sheet} crops)." if n_sheet else
              "No crops on disk for a contact sheet."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--sheet", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    rejected, kept = load(run_dir)
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(rejected, min(args.sample, len(rejected))), key=lambda r: r["det_id"])
    out = reports_dir(run_dir, args.out)
    sheet_rows = sorted(rng.sample(rejected, min(args.sheet, len(rejected))), key=lambda r: r["det_id"])
    n_sheet = contact_sheet(run_dir, sheet_rows, out / "rejected_true_logos.png")
    (out / "rejected_true_logos.md").write_text(render(run_dir, rejected, kept, sample, n_sheet), encoding="utf-8")
    write_csv(out / "rejected_true_logos.csv", ["det_id", "gold_class", "group", "short_side_px", "box_area_ratio",
                                                "confidence", "reasoning"],
              [(r["det_id"], r["gold_class"], g, r["short_side"], r.get("box_area_ratio"), r.get("confidence"),
                r.get("reasoning")) for g, rs in (("rejected", rejected), ("kept", kept)) for r in rs])
    print(f"wrote {out / 'rejected_true_logos.md'} ({len(rejected)} rejected, {len(kept)} kept; {n_sheet} crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
