#!/usr/bin/env python3
"""T63 (``spec docs/17``): IoU matching for the end-to-end evaluation.

Gold boxes come from a Roboflow labels directory (``{video}_frame_{n}_png.rf.*.txt``,
denormalised against each row's ``frame_width``/``frame_height``; frames mode
``frame_idx`` is the ordinal ``n``) or a fresh-frame labels CSV with columns
``frame_idx, class_name, x1, y1, x2, y2`` (absolute pixels). Only frames that
were both sampled and labelled are scored.

Per frame, greedy one-to-one matching by descending IoU (≥ ``--iou``), first on
``loose_bbox`` (the proposal), then on ``selected_bbox`` for the refinement
question. Reports matched, gold never re-detected and detections without a gold
box — overall, per class and without ``evaluation.channel_classes``; router
behaviour on the matched subset only (a matched box is a true logo): the
decision distribution, brand correctness through the alias table, ACCEPT
precision on matched ACCEPTs, "known brand routed unknown"; and recall per
detector source (``--compare-run`` adds the same figure for a second run, e.g.
the merge-off ablation). Detector recall on the Roboflow set is not reported
(out of scope; the set has no false negatives by construction).

Usage::

    python eval/iou_match.py --run runs/14/<run_id> --gold ~/Downloads/logo2.yolov11/train/labels \\
        --names ~/Downloads/logo2.yolov11/data.yaml [--iou 0.5] [--compare-run <dir>]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runio import joined_rows, md_table, pct, read_json, reports_dir, write_csv  # noqa: E402

from logo_pipeline.geometry import iou  # noqa: E402

KEPT = ("ACCEPT", "REVIEW", "UNKNOWN_LOGO")


def match_frame(gold: list[dict], dets: list[dict], iou_t: float, box_key: str) -> list[tuple[int, int, float]]:
    """Greedy one-to-one by descending IoU: ``[(gold_i, det_i, iou)]``."""
    pairs = sorted(((iou(g["box"], d[box_key]), gi, di) for gi, g in enumerate(gold) for di, d in enumerate(dets)),
                   reverse=True)
    used_g, used_d, out = set(), set(), []
    for v, gi, di in pairs:
        if v < iou_t:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        out.append((gi, di, v))
    return out


def gold_from_roboflow(labels_dir: Path, names: list[str], video_id: str, frames: dict[int, tuple[int, int]]) -> dict:
    from faiss_selfmatch import denormalise, frame_stem
    gold: dict[int, list[dict]] = {}
    for lf in sorted(Path(labels_dir).expanduser().glob(f"{video_id}_frame_*.txt")):
        stem = frame_stem(lf)
        if stem is None or stem.split("_frame_")[0] != str(video_id):
            continue
        fi = int(stem.split("_frame_")[1])
        if fi not in frames:
            continue
        w, h = frames[fi]
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                gold.setdefault(fi, []).append({"class_name": names[int(parts[0])],
                                                "box": list(denormalise(*(float(v) for v in parts[1:]), w, h))})
    return gold


def gold_from_csv(path: Path, frames: dict[int, tuple[int, int]]) -> dict:
    import pandas as pd
    gold: dict[int, list[dict]] = {}
    for r in pd.read_csv(path).to_dict("records"):
        fi = int(r["frame_idx"])
        if fi in frames:
            gold.setdefault(fi, []).append({"class_name": str(r["class_name"]),
                                            "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")]})
    return gold


def evaluate(rows: list[dict], gold: dict[int, list[dict]], iou_t: float, box_key: str, brands, channel: set[str]) -> dict:
    from logo_pipeline.brands import normalise
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["frame_idx"] in gold:
            by_frame[r["frame_idx"]].append(r)
    res = {"all": Counter(), "no_channel": Counter(), "per_class": defaultdict(Counter), "decisions": Counter(),
           "brand_correct": 0, "accept": Counter(), "known_routed_unknown": 0, "recall_by_source": Counter(),
           "gold_total": 0}
    for fi, g in gold.items():
        dets = by_frame.get(fi, [])
        pairs = match_frame(g, dets, iou_t, box_key)
        matched_g = {gi for gi, _, _ in pairs}
        matched_d = {di for _, di, _ in pairs}
        for gi, gb in enumerate(g):
            ch = normalise(gb["class_name"]) in channel
            key = "matched" if gi in matched_g else "gold_not_detected"
            for scope in (("all",) if ch else ("all", "no_channel")):
                res[scope][key] += 1
            res["per_class"][gb["class_name"]][key] += 1
            res["gold_total"] += 1
        for di, d in enumerate(dets):
            if di not in matched_d:
                res["all"]["detection_without_gold"] += 1
                res["no_channel"]["detection_without_gold"] += 1
        for gi, di, _ in pairs:
            d, gb = dets[di], g[gi]
            dec = d.get("router_decision")
            res["decisions"][dec] += 1
            gold_c = brands.map_exact(gb["class_name"]) or normalise(gb["class_name"])
            correct = d.get("canonical_brand") == gold_c
            res["brand_correct"] += correct
            if dec == "ACCEPT":
                res["accept"]["correct" if correct else "wrong"] += 1
            if dec == "UNKNOWN_LOGO" and brands.map_exact(gb["class_name"]):
                res["known_routed_unknown"] += 1
            for src in d.get("detection_sources") or []:
                res["recall_by_source"][src] += 1
    return res


def render(run_dir: Path, iou_t: float, loose: dict, sel: dict, compare: dict | None) -> str:
    def counts(res, scope):
        c = res[scope]
        return (c.get("matched", 0), c.get("gold_not_detected", 0), c.get("detection_without_gold", 0))

    lines = [f"# IoU matching (T63) — `{run_dir}`", "", f"Greedy one-to-one matching per frame, IoU ≥ {iou_t}; "
             f"{loose['gold_total']} gold boxes on sampled, labelled frames.", ""]
    lines += md_table(["boxes", "scope", "matched", "gold never re-detected", "detections without gold"],
                      [("loose_bbox", "all classes", *counts(loose, "all")),
                       ("loose_bbox", "without channel classes", *counts(loose, "no_channel")),
                       ("selected_bbox", "all classes", *counts(sel, "all")),
                       ("selected_bbox", "without channel classes", *counts(sel, "no_channel"))])
    m = loose["all"].get("matched", 0)
    lines += ["", "## Router on the matched subset (loose_bbox)", "",
              f"- Matched boxes are true logos. Decisions: "
              + ", ".join(f"{d}: {c}" for d, c in loose["decisions"].most_common()) + ".",
              f"- Kept (ACCEPT/REVIEW/UNKNOWN_LOGO): {sum(loose['decisions'].get(d, 0) for d in KEPT)} of {m} "
              f"({pct(sum(loose['decisions'].get(d, 0) for d in KEPT), m)}).",
              f"- Brand correct (canonical_brand vs gold class through the alias table): {loose['brand_correct']} "
              f"of {m} ({pct(loose['brand_correct'], m)}).",
              f"- ACCEPT precision on matched ACCEPTs: {loose['accept'].get('correct', 0)} of "
              f"{sum(loose['accept'].values())} ({pct(loose['accept'].get('correct', 0), sum(loose['accept'].values()))}).",
              f"- Known brand routed UNKNOWN_LOGO: {loose['known_routed_unknown']}.", "",
              "## Recall on these frames by detector source", ""]
    total = loose["gold_total"]
    rows = [(src, c, pct(c, total)) for src, c in loose["recall_by_source"].most_common()]
    rows.append(("any", m, pct(m, total)))
    if compare is not None:
        rows.append(("any (compare run)", compare["all"].get("matched", 0), pct(compare["all"].get("matched", 0),
                                                                                 compare["gold_total"])))
    lines += md_table(["source", "matched gold", "recall"], rows)
    lines += ["", "## Per class (loose_bbox)", ""]
    lines += md_table(["class", "matched", "gold never re-detected"],
                      [(k, v.get("matched", 0), v.get("gold_not_detected", 0))
                       for k, v in sorted(loose["per_class"].items())])
    return "\n".join(lines) + "\n"


def run_one(run_dir: Path, gold_arg: str, names_arg: str | None, iou_t: float, brands, channel):
    rows = joined_rows(run_dir)
    manifest = read_json(run_dir, "run_manifest.json")
    frames = {r["frame_idx"]: (r["frame_width"], r["frame_height"]) for r in rows}
    for f in read_json(run_dir, "sampled_frames.json", default=[]):
        frames.setdefault(f["frame_idx"], (manifest.get("frame_width"), manifest.get("frame_height")))
    gp = Path(gold_arg).expanduser()
    if gp.is_dir():
        from faiss_selfmatch import load_names
        gold = gold_from_roboflow(gp, load_names(names_arg), str(manifest["video_id"]), frames)
    else:
        gold = gold_from_csv(gp, frames)
    return (evaluate(rows, gold, iou_t, "loose_bbox", brands, channel),
            evaluate(rows, gold, iou_t, "selected_bbox", brands, channel))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", required=True, help="Roboflow labels dir, or a fresh-frame labels CSV")
    ap.add_argument("--names", default="~/Downloads/logo2.yolov11/data.yaml", help="class names (Roboflow gold)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--compare-run", default=None, help="second run on the same frames (e.g. merge off)")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import yaml

    from logo_pipeline.brands import load_brand_list, normalise
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    brands = load_brand_list(cfg["brands"]["list_path"])
    channel = {normalise(c) for c in cfg.get("evaluation", {}).get("channel_classes", [])}
    run_dir = Path(args.run)
    loose, sel = run_one(run_dir, args.gold, args.names, args.iou, brands, channel)
    compare = run_one(Path(args.compare_run), args.gold, args.names, args.iou, brands, channel)[0] \
        if args.compare_run else None
    out = reports_dir(run_dir, args.out)
    (out / "iou_match.md").write_text(render(run_dir, args.iou, loose, sel, compare), encoding="utf-8")
    write_csv(out / "iou_match_per_class.csv", ["class", "matched", "gold_not_detected"],
              [(k, v.get("matched", 0), v.get("gold_not_detected", 0)) for k, v in sorted(loose["per_class"].items())])
    print(f"wrote {out / 'iou_match.md'}: matched {loose['all'].get('matched', 0)} of {loose['gold_total']} gold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
