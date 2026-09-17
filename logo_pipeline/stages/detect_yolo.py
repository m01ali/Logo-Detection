"""``stages/detect_yolo.py`` (07 MM1 Step 3.3): MM1 stub.

Writes nothing; the real ``ultralytics`` class-agnostic pass is T12 (MM2,
``spec docs/08``). This stub exists so the ``raw_detections.jsonl`` file
format (``source: "yolo"``) is fixed now and the stage list is complete
(MM1 definition of done: "All 16 stages exist and run in order").
"""

from __future__ import annotations

from .base import RAW_DETECTIONS_FILE, RunContext, Stage, StageResult


class DetectYoloStage(Stage):
    name = "detect_yolo"
    device = "gpu"
    inputs: list[str] = []
    outputs = [RAW_DETECTIONS_FILE]

    def outputs_exist(self, ctx: RunContext) -> bool:
        return True  # a no-op stage is always "done"

    def run(self, ctx: RunContext) -> StageResult:
        ctx.log.stage_skipped(self.name, "MM2 not yet built (spec docs/08); pass-through stub")
        return StageResult(counts={"boxes_written": 0}, notes=["stub: real YOLO stage is MM2 (T12)"])
