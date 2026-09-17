"""``stages/watermark.py`` (07 MM1 Step 5.5): pass-through stub.

Per-video watermark regions and ``watermark_overlap`` (T47–T50) are MM6
(``spec docs/12``); no regions store exists yet, so every row reports
``watermark_regions_available: false`` and a null overlap.
"""

from __future__ import annotations

from .base import MERGED_DETECTIONS_FILE, RunContext, Stage, StageResult, WATERMARK_FILE


class WatermarkStage(Stage):
    name = "watermark"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE]
    outputs = [WATERMARK_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        reps = ctx.read_jsonl(MERGED_DETECTIONS_FILE)
        rows = [{
            "det_id": r["det_id"], "watermark_regions_available": False,
            "watermark_overlap": None, "watermark_overlap_ratio": None,
        } for r in reps]
        header = ctx.header("watermark", note="MM1 stub; regions + overlap are MM6 (spec docs/12, T47-T50)")
        ctx.write_jsonl(WATERMARK_FILE, rows, header=header)
        return StageResult(counts={"rows": len(rows)})
