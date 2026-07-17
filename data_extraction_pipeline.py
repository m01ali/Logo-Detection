#!/usr/bin/env python3
"""
Automated Data Extraction Pipeline for Logo Detection Dataset Building.

Samples frames from a video at regular intervals or from short segments,
runs Grounding DINO zero-shot detection on the sampled frames, generates
standard and enlarged crops, and saves T-5..T+5 temporal context frames.

All detections are intentionally retained (including false positives) so
the downstream VLM is forced to discriminate between real logos and
background textures.

Output layout:
    extraction_output/{video_stem}/
    ├── detections/
    │   └── frame_{idx:06d}/
    │       ├── frame.jpg               # the sampled frame
    │       └── det_{id:04d}/
    │           ├── crop.jpg            # tight bounding-box crop
    │           ├── crop_enlarged.jpg   # expanded crop with background context
    │           └── metadata.json       # box coords, score, label, timecode
    ├── temporal/
    │   └── frame_{idx:06d}/
    │       ├── T-5_frame{t:06d}.jpg
    │       └── T+5_frame{t:06d}.jpg
    └── manifest.json                   # full summary of all detections + config
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grounding DINO query list
# Each phrase is a separate zero-shot category. They are joined with " . "
# (the separator Grounding DINO expects) at construction time.
# ---------------------------------------------------------------------------

QUERIES_LIST: list[str] = [
    "a television network watermark",
    "a channel logo in the corner of the screen",
    "a digital graphic logo",
    "a sports broadcast logo",
    "a brand logo on a sports shirt",
    "a sponsor logo on athletic shorts",
    "a logo on a tennis racket",
    "a sportswear brand logo",
    "a sponsor logo on a sports uniform",
    "an advertising board sponsor logo",
    "a logo printed on a sports court surface",
    "a sponsor logo on a billiard table border",
    "a tournament logo on a wall",
    "a brand name printed on a banner",
    "a brand name on medicine packaging",
    "a company logo on a food box",
    "a brand logo on a frozen food bag",
    "a toy brand logo",
    "a corporate logo on a product label",
    "a logo on a security camera device",
    "a corporate trademark",
    "a small brand icon",
    "a company symbol",
    "a text-based brand logo",
    "a commercial brand mark",
    "an enterprise emblem",
    "a distinctive brand insignia",
    "a recognizable company badge",
    "a generic logo",
    "an organization's crest",
    "a commercial watermark",
    "a manufacturer's mark",
    "an official logo graphic",
    "a product",
    "a potential logo",
    "a company",
    "a brand",
]

# Grounding DINO expects categories separated by " . "
_DEFAULT_TEXT_PROMPT: str = " . ".join(QUERIES_LIST) + " ."


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    # --- Frame sampling -------------------------------------------------------
    sampling_mode: str = "interval"
    # "interval": sample one frame every `sample_every_n_seconds` seconds
    # "segment" : divide video into segments of `segment_duration_seconds`,
    #             take the middle frame of each segment

    sample_every_n_seconds: float = 2.0
    segment_duration_seconds: float = 4.0

    # --- Grounding DINO -------------------------------------------------------
    model_id: str = "IDEA-Research/grounding-dino-base"
    # Built from QUERIES_LIST by default; override only if you need a custom set.
    text_prompt: str = _DEFAULT_TEXT_PROMPT
    # Low thresholds are intentional: we want false positives in the dataset.
    box_threshold: float = 0.20
    text_threshold: float = 0.20

    # --- Crop settings --------------------------------------------------------
    # Each side of the bounding box is expanded by (enlarge_factor - 1) / 2
    # times the box's own width / height, then clamped to the image boundary.
    # e.g. enlarge_factor=1.5 adds 25 % padding on each side.
    enlarge_factor: float = 1.5
    min_crop_px: int = 16           # skip crops smaller than this in either dim

    # --- Temporal context -----------------------------------------------------
    temporal_window: int = 5        # save frames T-N…T-1, T+1…T+N

    # --- Output ---------------------------------------------------------------
    output_dir: str = "extraction_output"
    save_sampled_frame: bool = True
    jpeg_quality: int = 95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_json_safe(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return _to_json_safe(asdict(obj))
    return obj


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(_to_json_safe(data), indent=2))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DataExtractionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = self._pick_device()
        logger.info("Device: %s", self.device)
        self._load_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        logger.info("Loading Grounding DINO  (%s)", self.config.model_id)
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model_id
        )
        self.model.to(self.device).eval()
        logger.info("Model ready.")

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def _sample_indices(self, cap: cv2.VideoCapture) -> list[int]:
        """Return the sorted list of frame indices to run detection on."""
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if self.config.sampling_mode == "interval":
            step = max(1, round(fps * self.config.sample_every_n_seconds))
            indices = list(range(0, total, step))

        elif self.config.sampling_mode == "segment":
            seg_len = max(1, round(fps * self.config.segment_duration_seconds))
            indices = []
            for seg_start in range(0, total, seg_len):
                seg_end = min(seg_start + seg_len - 1, total - 1)
                indices.append((seg_start + seg_end) // 2)

        else:
            raise ValueError(f"Unknown sampling_mode: {self.config.sampling_mode!r}")

        indices = sorted(set(indices))
        logger.info(
            "Sampling  mode=%s  n=%d  total_frames=%d  fps=%.2f",
            self.config.sampling_mode,
            len(indices),
            total,
            fps,
        )
        return indices

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _run_detection(
        self, pil_image: Image.Image
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """
        Run Grounding DINO on a single PIL image.

        Returns
        -------
        boxes   : list of [x1, y1, x2, y2] in absolute pixel coords
        scores  : list of confidence floats
        labels  : list of text labels
        """
        inputs = self.processor(
            images=pil_image,
            text=self.config.text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # target_sizes expects [(H, W)]
        h, w = pil_image.size[1], pil_image.size[0]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results["labels"]
        return boxes, scores, labels

    # ------------------------------------------------------------------
    # Crop helpers
    # ------------------------------------------------------------------

    def _enlarge_box(
        self, box: list[float], img_w: int, img_h: int
    ) -> list[float]:
        """Expand box symmetrically by `enlarge_factor`, clamped to image."""
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * (self.config.enlarge_factor - 1) / 2
        pad_y = (y2 - y1) * (self.config.enlarge_factor - 1) / 2
        return [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(img_w), x2 + pad_x),
            min(float(img_h), y2 + pad_y),
        ]

    def _crop_image(
        self, pil_image: Image.Image, box: list[float]
    ) -> Optional[Image.Image]:
        """Crop and return None if the result is below min_crop_px."""
        x1, y1, x2, y2 = (int(round(c)) for c in box)
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        if (x2 - x1) < self.config.min_crop_px or (y2 - y1) < self.config.min_crop_px:
            return None
        return pil_image.crop((x1, y1, x2, y2))

    # ------------------------------------------------------------------
    # Temporal frames
    # ------------------------------------------------------------------

    def _save_temporal_frames(
        self,
        cap: cv2.VideoCapture,
        anchor_frame: int,
        out_dir: Path,
        total_frames: int,
        output_root: Path,
    ) -> dict[str, str | None]:
        """
        Write only T-N and T+N frames (no detection, raw images only).

        Returns a dict keyed by offset string (e.g. "-5", "+5") whose values
        are relative paths from output_root, or None if the frame is out of range
        or unreadable.  Store this dict in each detection's metadata so the
        temporal frames can be loaded programmatically later.
        """
        window = self.config.temporal_window
        offsets = [-window, +window]
        paths: dict[str, str | None] = {}

        for offset in offsets:
            key = f"{offset:+d}"
            t = anchor_frame + offset
            if not (0 <= t < total_frames):
                paths[key] = None
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, bgr = cap.read()
            if not ok:
                paths[key] = None
                continue
            fname = out_dir / f"T{offset:+d}_frame{t:06d}.jpg"
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pil.save(fname, quality=self.config.jpeg_quality)
            paths[key] = str(fname.relative_to(output_root))

        return paths

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def run(self, video_path: str | Path) -> Path:
        """
        Process a video end-to-end.

        Parameters
        ----------
        video_path : path to the input video file

        Returns
        -------
        output_root : Path to the output directory
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        output_root = Path(self.config.output_dir) / video_path.stem
        output_root.mkdir(parents=True, exist_ok=True)
        logger.info("Output root: %s", output_root)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample_indices = self._sample_indices(cap)
        all_detections: list[dict] = []
        det_id = 0

        for sample_idx, frame_idx in enumerate(sample_indices):
            logger.info(
                "[%d/%d]  frame=%d  t=%.2fs",
                sample_idx + 1,
                len(sample_indices),
                frame_idx,
                frame_idx / fps,
            )

            # ── Read the sampled frame ──────────────────────────────────────
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                logger.warning("  Could not read frame %d — skipping.", frame_idx)
                continue

            img_h, img_w = bgr.shape[:2]
            pil_frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            timecode_sec = frame_idx / fps

            # ── Create output directories ───────────────────────────────────
            frame_det_dir = (
                output_root / "detections" / f"frame_{frame_idx:06d}"
            )
            frame_det_dir.mkdir(parents=True, exist_ok=True)

            frame_temporal_dir = (
                output_root / "temporal" / f"frame_{frame_idx:06d}"
            )
            frame_temporal_dir.mkdir(parents=True, exist_ok=True)

            # ── Optionally save the sampled frame itself ────────────────────
            if self.config.save_sampled_frame:
                pil_frame.save(
                    frame_det_dir / "frame.jpg",
                    quality=self.config.jpeg_quality,
                )

            # ── Run Grounding DINO ──────────────────────────────────────────
            boxes, scores, labels = self._run_detection(pil_frame)
            logger.info("  %d detection(s)", len(boxes))

            # ── Save T-5 and T+5 temporal frames only (no detection) ─────────
            # Done before crop loop so paths are available in metadata.
            temporal_paths = self._save_temporal_frames(
                cap, frame_idx, frame_temporal_dir, total_frames, output_root
            )

            # ── Generate crops ──────────────────────────────────────────────
            for box, score, label in zip(boxes, scores, labels):
                enlarged_box = self._enlarge_box(box, img_w, img_h)

                crop = self._crop_image(pil_frame, box)
                if crop is None:
                    logger.debug("  det skipped (too small)  score=%.3f", score)
                    continue

                crop_enlarged = self._crop_image(pil_frame, enlarged_box) or crop

                det_dir = frame_det_dir / f"det_{det_id:04d}"
                det_dir.mkdir(exist_ok=True)

                crop_path = det_dir / "crop.jpg"
                crop_enlarged_path = det_dir / "crop_enlarged.jpg"
                crop.save(crop_path, quality=self.config.jpeg_quality)
                crop_enlarged.save(crop_enlarged_path, quality=self.config.jpeg_quality)

                meta = {
                    "det_id": det_id,
                    "sample_idx": sample_idx,
                    "frame_idx": frame_idx,
                    "timecode_seconds": round(timecode_sec, 3),
                    "label": label,
                    "score": round(float(score), 4),
                    "box_xyxy": [round(float(c), 2) for c in box],
                    "box_xyxy_enlarged": [round(float(c), 2) for c in enlarged_box],
                    "frame_width": img_w,
                    "frame_height": img_h,
                    "crop_path": str(crop_path.relative_to(output_root)),
                    "crop_enlarged_path": str(
                        crop_enlarged_path.relative_to(output_root)
                    ),
                    # Relative paths for every temporal neighbour of this frame.
                    # Keys are signed offset strings ("-5"…"-1", "+1"…"+5");
                    # value is None when the frame is out of video range.
                    "temporal_frames": temporal_paths,
                }
                _write_json(det_dir / "metadata.json", meta)
                all_detections.append(meta)
                det_id += 1

        cap.release()

        # ── Write manifest ──────────────────────────────────────────────────
        manifest = {
            "video": str(video_path.resolve()),
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frames": len(sample_indices),
            "total_detections": len(all_detections),
            "config": asdict(self.config),
            "detections": all_detections,
        }
        manifest_path = output_root / "manifest.json"
        _write_json(manifest_path, manifest)

        logger.info(
            "Done.  %d detections across %d sampled frames.  Output: %s",
            len(all_detections),
            len(sample_indices),
            output_root,
        )
        return output_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Data Extraction Pipeline — Grounding DINO + temporal crops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output-dir", default="extraction_output")
    parser.add_argument(
        "--sampling-mode",
        choices=["interval", "segment"],
        default="interval",
        help=(
            "interval: one frame every --sample-every seconds; "
            "segment: middle frame of each --segment-duration-second window"
        ),
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Seconds between sampled frames (interval mode)",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=4.0,
        metavar="SECONDS",
        help="Segment length in seconds (segment mode)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.20,
        help="Grounding DINO box confidence threshold (kept low to retain false positives)",
    )
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--enlarge-factor",
        type=float,
        default=1.5,
        help="Multiply bounding-box dimensions by this factor for the enlarged crop",
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Number of frames before and after each sample to save as temporal context",
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model ID for Grounding DINO",
    )
    parser.add_argument(
        "--text-prompt",
        default=_DEFAULT_TEXT_PROMPT,
        help="Period-separated category list for Grounding DINO (defaults to QUERIES_LIST)",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(
        output_dir=args.output_dir,
        sampling_mode=args.sampling_mode,
        sample_every_n_seconds=args.sample_every,
        segment_duration_seconds=args.segment_duration,
        model_id=args.model_id,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        enlarge_factor=args.enlarge_factor,
        temporal_window=args.temporal_window,
    )

    pipeline = DataExtractionPipeline(cfg)
    out = pipeline.run(args.video)
    print(f"\nOutput directory: {out}")
