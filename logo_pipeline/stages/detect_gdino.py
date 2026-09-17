"""``stages/detect_gdino.py`` (T11, 07 MM1 Step 3): wraps
``DataExtractionPipeline._run_detection`` — model load, prompt from YAML,
thresholds from YAML, ``min_crop_px`` — and appends ``DetectionRecord``
lines with ``source: "gdino"`` to ``raw_detections.jsonl``.

Idempotent per sampled frame: a frame already carrying ``gdino`` lines is
skipped unless ``ctx.force``. GPU stage; imports ``torch``/``transformers``
only inside ``run`` so the CLI and CPU stages never pay for them.
"""

from __future__ import annotations

import json

from ..geometry import rounded_int_box
from .base import RAW_DETECTIONS_FILE, RunContext, Stage, StageResult, dumps_line


class DetectGdinoStage(Stage):
    name = "detect_gdino"
    device = "gpu"
    inputs: list[str] = []
    outputs = [RAW_DETECTIONS_FILE]

    def outputs_exist(self, ctx: RunContext) -> bool:
        if not ctx.path(RAW_DETECTIONS_FILE).exists():
            return False
        rows = ctx.read_jsonl(RAW_DETECTIONS_FILE)
        sampled = {f["frame_idx"] for f in ctx.sampled_frames()}
        done = {r["frame_idx"] for r in rows if r.get("source") == "gdino"}
        return sampled <= done

    def run(self, ctx: RunContext) -> StageResult:
        gcfg = ctx.cfg.detectors.gdino
        if not gcfg.enabled:
            ctx.log.stage_skipped(self.name, "detectors.gdino.enabled is false")
            return StageResult(counts={"boxes": 0, "enabled": False})

        path = ctx.path(RAW_DETECTIONS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        _, existing_rows = ctx.read_jsonl(RAW_DETECTIONS_FILE, with_header=True)
        done_frames = {r["frame_idx"] for r in existing_rows if r.get("source") == "gdino"} if not ctx.force else set()
        next_raw_id = max([r["raw_det_id"] for r in existing_rows], default=-1) + 1

        sampled = ctx.sampled_frames()
        todo = [f for f in sampled if f["frame_idx"] not in done_frames]
        if not todo:
            ctx.log.stage_skipped(self.name, "all sampled frames already have gdino detections")
            return StageResult(counts={"frames_skipped": len(sampled)})

        from PIL import Image

        from data_extraction_pipeline import DataExtractionPipeline, PipelineConfig

        text_prompt = " . ".join(gcfg.queries) + " ."
        pcfg = PipelineConfig(model_id=gcfg.model_id, text_prompt=text_prompt,
                               box_threshold=gcfg.box_threshold, text_threshold=gcfg.text_threshold,
                               min_crop_px=gcfg.min_crop_px)
        pipeline = DataExtractionPipeline(pcfg)

        write_header = not path.exists() or path.stat().st_size == 0
        n_boxes = 0
        with open(path, "a", encoding="utf-8") as fh:
            if write_header:
                fh.write(dumps_line(ctx.header("detect_gdino", detector_config={
                    "model_id": gcfg.model_id, "box_threshold": gcfg.box_threshold,
                    "text_threshold": gcfg.text_threshold, "min_crop_px": gcfg.min_crop_px,
                    "num_queries": len(gcfg.queries),
                })) + "\n")
                fh.flush()

            for i, frame in enumerate(todo, 1):
                frame_idx = frame["frame_idx"]
                pil = Image.open(ctx.path(frame["path"])).convert("RGB")
                img_w, img_h = pil.size
                boxes, scores, labels = pipeline._run_detection(pil)
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = rounded_int_box(box)
                    if (x2 - x1) < gcfg.min_crop_px or (y2 - y1) < gcfg.min_crop_px:
                        continue
                    row = {
                        "raw_det_id": next_raw_id, "frame_idx": frame_idx, "source": "gdino",
                        "box_xyxy": [round(float(c), 2) for c in box], "score": round(float(score), 4),
                        "label": label, "class_conf": None,
                        "frame_width": img_w, "frame_height": img_h,
                    }
                    fh.write(dumps_line(row) + "\n")
                    next_raw_id += 1
                    n_boxes += 1
                fh.flush()
                ctx.log.progress(i, len(todo), unit="frame")

        return StageResult(counts={"frames_processed": len(todo), "boxes_written": n_boxes})
