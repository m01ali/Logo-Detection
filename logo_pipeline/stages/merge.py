"""``stages/merge.py`` (07 MM1 Step 4; T13–T19 in ``spec docs/08`` is MM2).

MM1 ships only the ``merge.enabled: false`` path: every raw box becomes its
own representative, ``merge_count`` 0, ``detection_sources`` a single-source
set, and ``merge_log.jsonl``/``merge_counters.json`` are written empty (T16).
This is deliberate, not a shortcut: the MM1 goal is "no new algorithms", and
T16's own definition of done requires this exact path to keep working
unchanged once MM2 adds the real greedy pooled merge behind
``merge.enabled: true``. Until MM2 lands, this stage runs the same
pass-through regardless of the flag and says so in the stage header.
"""

from __future__ import annotations

from ..geometry import box_area_ratio, box_touches_edge
from .base import (MERGE_COUNTERS_FILE, MERGE_LOG_FILE, MERGED_DETECTIONS_FILE, RAW_DETECTIONS_FILE,
                    RunContext, Stage, StageResult)


class MergeStage(Stage):
    name = "merge"
    device = "cpu"
    inputs = [RAW_DETECTIONS_FILE]
    outputs = [MERGED_DETECTIONS_FILE, MERGE_LOG_FILE, MERGE_COUNTERS_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        raw = ctx.read_jsonl(RAW_DETECTIONS_FILE)
        raw.sort(key=lambda r: (r["frame_idx"], r["raw_det_id"]))  # deterministic det_id order (T13)

        reps: list[dict] = []
        by_frame_source: dict[tuple[int, str], int] = {}
        for det_id, r in enumerate(raw):
            w, h = r["frame_width"], r["frame_height"]
            box = r["box_xyxy"]
            reps.append({
                "det_id": det_id,
                "frame_idx": r["frame_idx"],
                "frame_width": w, "frame_height": h,
                "loose_bbox": box,
                "box_xyxy": box,
                "box_area_ratio": round(box_area_ratio(box, w, h), 6),
                "box_touches_edge": box_touches_edge(box, w, h),
                "det_score": r["score"],
                "detection_sources": sorted({r["source"]}),
                "merge_count": 0,
                "merged_raw_det_ids": [],
                "seed_raw_det_id": r["raw_det_id"],
                "gdino_label": r["label"] if r["source"] == "gdino" else None,
                "yolo_class": r["label"] if r["source"] == "yolo" else None,
                "yolo_class_conf": r["class_conf"] if r["source"] == "yolo" else None,
            })
            by_frame_source[(r["frame_idx"], r["source"])] = by_frame_source.get((r["frame_idx"], r["source"]), 0) + 1

        header = ctx.header("merge", merge_enabled=ctx.cfg.merge.enabled,
                             algorithm="pass_through",
                             note="MM1: pass-through only; the greedy pooled merge is MM2 (spec docs/08, T13-T19)")
        ctx.write_jsonl(MERGED_DETECTIONS_FILE, reps, header=header)
        ctx.write_jsonl(MERGE_LOG_FILE, [], header=header)   # T16: empty log on the false path

        remaining_by_sources = {"gdino": 0, "yolo": 0, "gdino+yolo": 0}
        for rep in reps:
            key = "+".join(rep["detection_sources"])
            remaining_by_sources[key] = remaining_by_sources.get(key, 0) + 1
        boxes_gdino = sum(1 for r in raw if r["source"] == "gdino")
        boxes_yolo = sum(1 for r in raw if r["source"] == "yolo")

        frame_counts: dict[str, dict] = {}
        for f in ctx.sampled_frames():
            fi = f["frame_idx"]
            n = sum(1 for r in reps if r["frame_idx"] == fi)
            frame_counts[str(fi)] = {
                "boxes_total_raw": n, "boxes_remaining": n, "merge_count_sum": 0,
                "pairs_evaluated": 0, "sampled": True,
            }
        counters = {
            "video": {
                "boxes_gdino": boxes_gdino, "boxes_yolo": boxes_yolo, "boxes_total_raw": len(raw),
                "pairs_evaluated": 0, "pairs_merged_intra": 0, "pairs_merged_cross": 0,
                "pairs_skipped_area_floor": 0, "pairs_failed_size_precondition": 0, "pairs_no_overlap": 0,
                "boxes_remaining": len(reps),
                "remaining_by_sources": remaining_by_sources,
                "vlm_calls_saved": len(raw) - len(reps),
            },
            "frames": frame_counts,
        }
        ctx.write_json(MERGE_COUNTERS_FILE, counters)
        return StageResult(counts={"representatives": len(reps), "raw_boxes": len(raw)})
