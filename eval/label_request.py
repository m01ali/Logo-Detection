#!/usr/bin/env python3
"""T66 route two (``spec docs/17``): the deterministic label request for the
router-only acceptance gate.

Samples, with ``--seed``: every ACCEPT, every UNKNOWN_LOGO, and ``--per-code``
rows (the 60-80 band) of each remaining reason code that fired at least that
often; codes that fired fewer times contribute all their rows and are listed
separately. Writes ``label_request/labels_to_fill.csv`` (``det_id, frame_idx,
timecode_s, crop_path, enlarged_crop_path, annotated_frame_path,
canonical_brand, router_decision, reason_code, label, true_brand``), copies of
the crops and annotated frames it names, and ``README.md`` stating the counts
against the run's distribution. The client labels in the review UI or the CSV;
``eval/metrics.py --labels`` consumes it.

Usage::

    python eval/label_request.py --run runs/video_1/<run_id> [--per-code 70] [--seed 42] [--out <dir>]
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import joined_rows, md_table, pct  # noqa: E402

BAND = (60, 80)
ALWAYS = ("ACCEPT", "UNKNOWN_LOGO")
COLUMNS = ["det_id", "video_id", "run_id", "frame_idx", "timecode_s", "crop_path", "enlarged_crop_path",
           "annotated_frame_path", "canonical_brand", "router_decision", "reason_code", "label", "true_brand"]


def sample(rows: list[dict], per_code: int, seed: int) -> tuple[list[dict], dict, list[str]]:
    """``(selected rows, {reason_code: n selected}, codes that fired fewer than per_code)``."""
    if not BAND[0] <= per_code <= BAND[1]:
        raise ValueError(f"--per-code must be within {BAND[0]}-{BAND[1]} (17 T66), got {per_code}")
    rng = random.Random(seed)
    chosen: list[dict] = [r for r in rows if r.get("router_decision") in ALWAYS]
    by_code: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("router_decision") not in ALWAYS:
            by_code[str(r.get("reason_code"))].append(r)
    small = []
    for code in sorted(by_code):
        pool = sorted(by_code[code], key=lambda r: r["det_id"])
        if len(pool) < per_code:
            small.append(code)
            chosen += pool
        else:
            chosen += rng.sample(pool, per_code)
    chosen = sorted(chosen, key=lambda r: r["det_id"])
    return chosen, dict(Counter(str(r.get("reason_code")) for r in chosen)), small


def write(run_dir: Path, out: Path, selected: list[dict], counts: dict, small: list[str], rows: list[dict],
          per_code: int, seed: int) -> Path:
    import pandas as pd
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    records = []
    for r in selected:
        rec = {c: r.get(c) for c in COLUMNS}
        rec["label"], rec["true_brand"] = None, None
        for col, sub, name in (("crop_path", "crops", f"det_{r['det_id']:06d}.jpg"),
                               ("enlarged_crop_path", "crops", f"det_{r['det_id']:06d}_enlarged.jpg"),
                               ("annotated_frame_path", "frames", f"frame_{r['frame_idx']:06d}.jpg")):
            src = r.get(col)
            if src and (run_dir / src).exists():
                dst = out / sub / name
                if not dst.exists():
                    shutil.copyfile(run_dir / src, dst)
                rec[col] = f"{sub}/{name}"
        records.append(rec)
    csv_path = out / "labels_to_fill.csv"
    pd.DataFrame(records, columns=COLUMNS).to_csv(csv_path, index=False)

    dist = Counter((r.get("router_decision"), str(r.get("reason_code"))) for r in rows)
    lines = ["# Route-two label request", "",
             f"Run `{run_dir}`: {len(rows)} detections; {len(selected)} rows requested (seed {seed}, "
             f"{per_code} per reason code; every ACCEPT and every UNKNOWN_LOGO).", "",
             "Fill `label` (TP, FP, FN, wrong_brand) and, where the brand is wrong or unknown, `true_brand`, in the "
             "review UI or in `labels_to_fill.csv`; `eval/metrics.py --labels` reads it back.", ""]
    lines += md_table(["decision", "reason code", "rows in run", "share of run", "requested"],
                      [(d, c, n, pct(n, len(rows)), counts.get(c, 0)) for (d, c), n in
                       sorted(dist.items(), key=lambda kv: -kv[1])])
    if small:
        lines += ["", f"Codes that fired fewer than {per_code} times (all their rows are requested): "
                  + ", ".join(f"`{c}`" for c in small) + "."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-code", type=int, default=70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default <run>/label_request")
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    rows = joined_rows(run_dir)
    import pandas as pd
    csv = run_dir / "rows.csv"
    if csv.exists():                                    # crop/frame paths live on rows.csv (output stage)
        paths = pd.read_csv(csv).set_index("det_id")
        for r in rows:
            if r["det_id"] in paths.index:
                for col in ("crop_path", "enlarged_crop_path", "annotated_frame_path"):
                    v = paths.at[r["det_id"], col] if col in paths.columns else None
                    r[col] = v if isinstance(v, str) else r.get(col)
    selected, counts, small = sample(rows, args.per_code, args.seed)
    out = Path(args.out) if args.out else run_dir / "label_request"
    path = write(run_dir, out, selected, counts, small, rows, args.per_code, args.seed)
    print(f"wrote {path} ({len(selected)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
