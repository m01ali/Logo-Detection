"""``stages/watermark.py`` (T47-T50, ``spec docs/12``): ``watermark_overlap``
as a pre-router step.

Reads the video's region entry from ``configs/watermark_regions.yaml`` and
measures every representative's ``loose_bbox`` against it (T48
[ASSUMPTION]: the box the VLM saw in context; ``selected_bbox`` may have been
tightened onto text inside a larger region and would under-report overlap).
A video without regions writes ``watermark_regions_available: false`` and a
null overlap so Rule 5 takes its "no regions" branch.
"""

from __future__ import annotations

from ..watermark_regions import region_overlap, regions_for_video
from .base import MERGED_DETECTIONS_FILE, RunContext, Stage, StageResult, WATERMARK_FILE


class WatermarkStage(Stage):
    name = "watermark"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE]
    outputs = [WATERMARK_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        wcfg = ctx.cfg.watermark
        entry = regions_for_video(ctx.repo_root, ctx.video, wcfg)
        regions = list(entry.regions) if entry else []
        reps = ctx.read_jsonl(MERGED_DETECTIONS_FILE)

        rows = []
        n_inside = 0
        for r in reps:
            overlap, ratio, available = region_overlap(r["loose_bbox"], regions, r["frame_width"],
                                                       r["frame_height"], wcfg.margin, wcfg.containment_threshold)
            n_inside += bool(overlap)
            rows.append({"det_id": r["det_id"], "watermark_regions_available": available,
                         "watermark_overlap": overlap, "watermark_overlap_ratio": ratio})

        header = ctx.header("watermark", watermark_regions_ref=entry.key if entry else None,
                            regions=[list(r) for r in regions],
                            channel_brands=list(entry.channel_brands) if entry else [],
                            margin=wcfg.margin, containment_threshold=wcfg.containment_threshold)
        ctx.write_jsonl(WATERMARK_FILE, rows, header=header)
        return StageResult(counts={"rows": len(rows), "regions": len(regions), "inside": n_inside})
