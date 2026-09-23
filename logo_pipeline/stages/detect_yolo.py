"""``stages/detect_yolo.py`` (T12, ``spec docs/08``): ``ultralytics.YOLO``
class-agnostic pass over sampled frames, appending ``DetectionRecord`` lines
with ``source: "yolo"`` to ``raw_detections.jsonl`` (the same file
``detect_gdino`` writes into, so the merge stage reads one union).

``agnostic_nms=True`` is what "class-agnostic" means operationally: boxes of
different predicted classes suppress each other in ultralytics' own NMS, so
one physical logo yields one box regardless of which class the model
guessed. The class is logged (``label``/``class_conf``) but drives nothing
downstream (T12.2). Ultralytics letterboxes internally and returns
``boxes.xyxy`` in original-image pixels, so no rescaling is needed (11
August A1). ``min_crop_px`` is the same threshold GDINO uses
(``detectors.gdino.min_crop_px``; T12.3 — "applied identically to both
sources so the counters compare like with like").

Idempotent per sampled frame like ``detect_gdino``: a frame already carrying
``yolo`` lines is skipped unless ``ctx.force``. GPU stage; imports
``ultralytics`` only inside ``run``.
"""

from __future__ import annotations

from ..geometry import rounded_int_box
from .base import RAW_DETECTIONS_FILE, RunContext, Stage, StageResult, dumps_line


class DetectYoloStage(Stage):
    name = "detect_yolo"
    device = "gpu"
    inputs: list[str] = []
    outputs = [RAW_DETECTIONS_FILE]

    def outputs_exist(self, ctx: RunContext) -> bool:
        if not ctx.path(RAW_DETECTIONS_FILE).exists():
            return False
        rows = ctx.read_jsonl(RAW_DETECTIONS_FILE)
        sampled = {f["frame_idx"] for f in ctx.sampled_frames()}
        done = {r["frame_idx"] for r in rows if r.get("source") == "yolo"}
        return sampled <= done

    def run(self, ctx: RunContext) -> StageResult:
        ycfg = ctx.cfg.detectors.yolo
        if not ycfg.enabled:
            ctx.log.stage_skipped(self.name, "detectors.yolo.enabled is false")
            return StageResult(counts={"boxes": 0, "enabled": False})

        min_crop_px = ctx.cfg.detectors.gdino.min_crop_px  # T12.3: shared with GDINO

        # raw_detections.jsonl has one header (base.read_jsonl keeps only the
        # first _header line), and detect_gdino runs before detect_yolo in
        # DEFAULT_STAGES, so it always wins that header. Record this
        # detector's config into run_manifest.json instead, keyed by source,
        # so provenance for both detectors survives regardless of run order.
        detectors_manifest = dict(ctx.manifest.get("detectors", {}))
        detectors_manifest["yolo"] = {
            "weights": str(ycfg.weights), "imgsz": ycfg.imgsz, "conf": ycfg.conf,
            "iou_nms": ycfg.iou_nms, "class_agnostic": ycfg.class_agnostic,
            "log_predicted_class": ycfg.log_predicted_class, "min_crop_px": min_crop_px,
        }
        ctx.update_manifest(detectors=detectors_manifest)

        path = ctx.path(RAW_DETECTIONS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        _, existing_rows = ctx.read_jsonl(RAW_DETECTIONS_FILE, with_header=True)
        done_frames = {r["frame_idx"] for r in existing_rows if r.get("source") == "yolo"} if not ctx.force else set()
        next_raw_id = max([r["raw_det_id"] for r in existing_rows], default=-1) + 1

        sampled = ctx.sampled_frames()
        todo = [f for f in sampled if f["frame_idx"] not in done_frames]
        if not todo:
            ctx.log.stage_skipped(self.name, "all sampled frames already have yolo detections")
            return StageResult(counts={"frames_skipped": len(sampled)})

        from ultralytics import YOLO

        weights_path = ctx.resolve_repo_path(ycfg.weights)
        model = YOLO(str(weights_path))
        names = model.names   # {class_id: class_name}, from the checkpoint itself

        write_header = not path.exists() or path.stat().st_size == 0
        n_boxes = 0
        with open(path, "a", encoding="utf-8") as fh:
            if write_header:
                fh.write(dumps_line(ctx.header("detect_yolo", detector_config={
                    "weights": str(ycfg.weights), "imgsz": ycfg.imgsz, "conf": ycfg.conf,
                    "iou_nms": ycfg.iou_nms, "class_agnostic": ycfg.class_agnostic,
                    "min_crop_px": min_crop_px,
                })) + "\n")
                fh.flush()

            for i, frame in enumerate(todo, 1):
                frame_idx = frame["frame_idx"]
                frame_path = ctx.path(frame["path"])
                results = model.predict(source=str(frame_path), imgsz=ycfg.imgsz, conf=ycfg.conf,
                                         iou=ycfg.iou_nms, agnostic_nms=ycfg.class_agnostic, verbose=False)
                result = results[0]
                img_h, img_w = result.orig_shape
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = rounded_int_box(xyxy)
                    if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
                        continue
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = names[cls_id] if ycfg.log_predicted_class else None
                    class_conf = round(conf, 4) if ycfg.log_predicted_class else None
                    row = {
                        "raw_det_id": next_raw_id, "frame_idx": frame_idx, "source": "yolo",
                        "box_xyxy": [round(float(c), 2) for c in xyxy], "score": round(conf, 4),
                        "label": label, "class_conf": class_conf,
                        "frame_width": img_w, "frame_height": img_h,
                    }
                    fh.write(dumps_line(row) + "\n")
                    next_raw_id += 1
                    n_boxes += 1
                fh.flush()
                ctx.log.progress(i, len(todo), unit="frame")

        return StageResult(counts={"frames_processed": len(todo), "boxes_written": n_boxes})
