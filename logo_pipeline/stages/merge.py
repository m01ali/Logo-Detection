"""``stages/merge.py`` (T13-T19, ``spec docs/08``): pooled greedy merge over
the union of GDINO + YOLO raw boxes per frame.

``merge_frame``/``evaluate_pair`` are pure functions of ``geometry.py``
primitives (no config-independent globals, no I/O) so they can be unit
tested directly and reused by ``eval/merge_ablation.py`` (T19). Comparisons
are against the seed only — greedy, NMS-style, no transitive clustering (27
July C0) — and every evaluated pair is logged regardless of outcome, so
``merge_log.jsonl`` carries the full ``iou``/``contain``/``area_ratio``/
``smaller_fraction`` distribution T19 needs.

With ``merge.enabled: false`` (T16) the stage takes the MM1 pass-through
path instead: one representative per raw box, no pairs evaluated, an empty
``merge_log.jsonl``. Both paths share the same counters builder so T18's
counters and the T60 harness read an identical shape either way.
"""

from __future__ import annotations

import math

from ..geometry import box_area, box_area_ratio, box_touches_edge, centre_distance
from .base import (MERGE_COUNTERS_FILE, MERGE_LOG_FILE, MERGED_DETECTIONS_FILE, RAW_DETECTIONS_FILE,
                    RunContext, Stage, StageResult, iter_grouped_by_frame)
from .router.dedup import cluster_boxes, evaluate_box_pair


def _small_box_merge(a: dict, b: dict, cfg) -> bool:
    """**[ASSUMPTION]** optional branch behind ``merge.small_box.enabled``
    (default false, not tuned in M3 — 27 July request, 29 July proposal,
    absent from Revision 3; T13). Only ever consulted for a pair that failed
    the ordinary IoU/containment overlap test: two boxes small enough
    (``min area <= pixel_area_cutoff``) that overlap is a noisy signal, but
    whose centres sit close relative to their own size, merge anyway.
    """
    box_a, box_b = a["box_xyxy"], b["box_xyxy"]
    if min(box_area(box_a), box_area(box_b)) > cfg.small_box.pixel_area_cutoff:
        return False
    diag_a = math.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])
    diag_b = math.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1])
    mean_diag = (diag_a + diag_b) / 2
    if mean_diag <= 0:
        return False
    return (centre_distance(box_a, box_b) / mean_diag) < cfg.small_box.centre_distance_threshold


def evaluate_pair(a: dict, b: dict, w: int, h: int, cfg) -> dict:
    """T13's ``evaluate_pair``: metrics + outcome for one candidate pair.
    ``a``/``b`` are raw-detection-shaped dicts (``box_xyxy``, ``source``,
    ``raw_det_id``); metrics are always computed regardless of outcome so
    T19 can read them off any logged pair, including skipped ones."""
    box_a, box_b = a["box_xyxy"], b["box_xyxy"]
    # Same pair test the router's dedup uses (T45): one implementation, the merge's YAML keys.
    m = evaluate_box_pair(box_a, box_b, cfg.iou_threshold, cfg.containment_threshold, cfg.area_ratio_threshold,
                          cfg.min_merge_area, float(w) * float(h))
    # T13's pseudocode checks the floor first, unconditionally, and small_box
    # is untuned/disabled by default (T13 [ASSUMPTION]) — so note for whoever
    # tunes both later: at the v1 placeholders (min_merge_area 0.0004,
    # small_box.pixel_area_cutoff 400) on a 1080p+ frame, min_merge_area's px²
    # threshold already exceeds pixel_area_cutoff, so small_box pairs are
    # floor-skipped before ever reaching the small_box branch below.
    small_box_triggered = False
    if m["outcome"] == "no_overlap" and cfg.small_box.enabled and _small_box_merge(a, b, cfg):
        m["outcome"] = "merged"
        small_box_triggered = True
    return {
        "a_raw_det_id": a["raw_det_id"], "b_raw_det_id": b["raw_det_id"],
        "pair_type": "intra" if a["source"] == b["source"] else "cross",
        **m, "small_box_triggered": small_box_triggered,
    }


def merge_frame(raw: list[dict], w: int, h: int, cfg) -> tuple[list[dict], list[dict]]:
    """T13's pooled greedy merge for one frame's raw boxes. Returns
    ``(representatives, pair_log)``; representatives are plain dicts (not
    yet assigned a ``det_id`` — that is ``(frame_idx, seed order)`` across
    the whole run, done by the caller)."""
    boxes = sorted(raw, key=lambda d: (-d["score"], d["raw_det_id"]))   # deterministic tie-break
    log: list[dict] = []
    reps: list[dict] = []
    clusters = cluster_boxes(
        [d["box_xyxy"] for d in boxes], cfg.iou_threshold, cfg.containment_threshold, cfg.area_ratio_threshold,
        cfg.min_merge_area, float(w) * float(h),
        evaluate=lambda i, j: evaluate_pair(boxes[i], boxes[j], w, h, cfg),
        on_pair=lambda i, j, m: log.append(m))
    for idx_cluster in clusters:
        seed = boxes[idx_cluster[0]]
        cluster = [boxes[i] for i in idx_cluster]
        rep = min(cluster, key=lambda d: box_area(d["box_xyxy"]))       # locked: keep the tighter box
        reps.append({
            "loose_bbox": rep["box_xyxy"], "det_score": rep["score"],
            "detection_sources": sorted({d["source"] for d in cluster}),
            "merge_count": len(cluster) - 1,
            "merged_raw_det_ids": sorted(d["raw_det_id"] for d in cluster if d is not rep),
            "seed_raw_det_id": seed["raw_det_id"],
            "gdino_label": rep["label"] if rep["source"] == "gdino" else None,
            "yolo_class": rep["label"] if rep["source"] == "yolo" else None,
            "yolo_class_conf": rep["class_conf"] if rep["source"] == "yolo" else None,
        })
    return reps, log


def _pass_through_frame(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """T16: ``merge.enabled: false`` path — one representative per raw box,
    in ``raw_det_id`` order, no pairs evaluated."""
    reps = []
    for r in sorted(raw, key=lambda d: d["raw_det_id"]):
        reps.append({
            "loose_bbox": r["box_xyxy"], "det_score": r["score"], "detection_sources": [r["source"]],
            "merge_count": 0, "merged_raw_det_ids": [], "seed_raw_det_id": r["raw_det_id"],
            "gdino_label": r["label"] if r["source"] == "gdino" else None,
            "yolo_class": r["label"] if r["source"] == "yolo" else None,
            "yolo_class_conf": r["class_conf"] if r["source"] == "yolo" else None,
        })
    return reps, []


def _box_counts(raw: list[dict], reps: list[dict]) -> dict:
    remaining_by_sources = {"gdino": 0, "yolo": 0, "gdino+yolo": 0}
    for rep in reps:
        key = "+".join(rep["detection_sources"])
        remaining_by_sources[key] = remaining_by_sources.get(key, 0) + 1
    return {
        "boxes_gdino": sum(1 for r in raw if r["source"] == "gdino"),
        "boxes_yolo": sum(1 for r in raw if r["source"] == "yolo"),
        "boxes_total_raw": len(raw), "boxes_remaining": len(reps),
        "remaining_by_sources": remaining_by_sources,
        "vlm_calls_saved": len(raw) - len(reps),
    }


def _pair_counts(pairs: list[dict]) -> dict:
    def n(pred):
        return sum(1 for p in pairs if pred(p))
    return {
        "pairs_evaluated": len(pairs),
        "pairs_merged_intra": n(lambda p: p["outcome"] == "merged" and p["pair_type"] == "intra"),
        "pairs_merged_cross": n(lambda p: p["outcome"] == "merged" and p["pair_type"] == "cross"),
        "pairs_skipped_area_floor": n(lambda p: p["outcome"] == "skipped_area_floor"),
        "pairs_failed_size_precondition": n(lambda p: p["outcome"] == "failed_size_precondition"),
        "pairs_no_overlap": n(lambda p: p["outcome"] == "no_overlap"),
    }


class MergeStage(Stage):
    name = "merge"
    device = "cpu"
    inputs = [RAW_DETECTIONS_FILE]
    outputs = [MERGED_DETECTIONS_FILE, MERGE_LOG_FILE, MERGE_COUNTERS_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        mcfg = ctx.cfg.merge
        raw = ctx.read_jsonl(RAW_DETECTIONS_FILE)

        reps: list[dict] = []
        log: list[dict] = []
        by_frame: dict[int, tuple[list[dict], list[dict], list[dict]]] = {}
        det_id = 0
        for frame_idx, frame_raw in iter_grouped_by_frame(raw):
            w, h = frame_raw[0]["frame_width"], frame_raw[0]["frame_height"]
            if mcfg.enabled:
                frame_reps, frame_log = merge_frame(frame_raw, w, h, mcfg)
            else:
                frame_reps, frame_log = _pass_through_frame(frame_raw)
            for m in frame_log:
                m["frame_idx"] = frame_idx
            log.extend(frame_log)

            frame_rows = []
            for rep in frame_reps:
                box = rep["loose_bbox"]
                row = {
                    "det_id": det_id, "frame_idx": frame_idx, "frame_width": w, "frame_height": h,
                    "loose_bbox": box, "box_xyxy": box,
                    "box_area_ratio": round(box_area_ratio(box, w, h), 6),
                    "box_touches_edge": box_touches_edge(box, w, h),
                    "det_score": rep["det_score"], "detection_sources": rep["detection_sources"],
                    "merge_count": rep["merge_count"], "merged_raw_det_ids": rep["merged_raw_det_ids"],
                    "seed_raw_det_id": rep["seed_raw_det_id"], "gdino_label": rep["gdino_label"],
                    "yolo_class": rep["yolo_class"], "yolo_class_conf": rep["yolo_class_conf"],
                }
                reps.append(row)
                frame_rows.append(row)
                det_id += 1
            by_frame[frame_idx] = (frame_raw, frame_rows, frame_log)

        header = ctx.header("merge", merge_enabled=mcfg.enabled,
                             algorithm="pooled_greedy" if mcfg.enabled else "pass_through",
                             iou_threshold=mcfg.iou_threshold, containment_threshold=mcfg.containment_threshold,
                             area_ratio_threshold=mcfg.area_ratio_threshold, min_merge_area=mcfg.min_merge_area)
        ctx.write_jsonl(MERGED_DETECTIONS_FILE, reps, header=header)

        # T16: the false path always writes an empty log; log_failed_pairs trims
        # the true path's persisted log to just the merges (counters still see everything).
        if not mcfg.enabled:
            log_to_write: list[dict] = []
        elif mcfg.log_failed_pairs:
            log_to_write = log
        else:
            log_to_write = [m for m in log if m["outcome"] == "merged"]
        ctx.write_jsonl(MERGE_LOG_FILE, log_to_write, header=header)

        frame_counters: dict[str, dict] = {}
        for f in ctx.sampled_frames():
            fi = f["frame_idx"]
            frame_raw, frame_rows, frame_log = by_frame.get(fi, ([], [], []))
            c = {}
            c.update(_box_counts(frame_raw, frame_rows))
            c.update(_pair_counts(frame_log))
            c["sampled"] = True
            frame_counters[str(fi)] = c

        video_counters = {}
        video_counters.update(_box_counts(raw, reps))
        video_counters.update(_pair_counts(log))
        counters = {"video": video_counters, "frames": frame_counters}
        ctx.write_json(MERGE_COUNTERS_FILE, counters)

        return StageResult(counts={"representatives": len(reps), "raw_boxes": len(raw),
                                    "pairs_evaluated": len(log)})
