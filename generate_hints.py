#!/usr/bin/env python3
"""
Multi-Signal Hints Generation Script.

Precomputes two kinds of hints before VLM inference:

  1. Audio hints  — Whisper transcription → Qwen-text brand extraction
                    → saved as  <extraction_dir>/audio_hints.json
                    (skipped automatically when the file already exists)

  2. FAISS hints  — top-k nearest brands from the CLIP FAISS index
                    → saved as  <det_dir>/hints.json  for every detection

Optionally runs a Qwen VLM with all signals (crop + score, enlarged crop,
full frame, temporal frames, audio brands, FAISS matches) to decide:
  (i)  whether the crop is a logo / partial logo / not a logo
  (ii) if it is a logo, what the brand name is

Output files added to the existing extraction layout:
    <extraction_dir>/
    ├── audio_hints.json                  ← video-level audio brands
    └── detections/frame_*/det_*/
        ├── hints.json                    ← audio brands + FAISS top-k
        └── vlm_prediction.json           ← VLM answer (only with --run-vlm)

Usage:
    python generate_hints.py \\
        --video        path/to/video.mp4 \\
        --extraction-dir extraction_output/video_stem \\
        --faiss-index  FAISS/logo_index.faiss \\
        --faiss-metadata FAISS/metadata.json \\
        [--top-k 5] \\
        [--faiss-threshold 0.0] \\
        [--audio-mode translate|transcribe] \\
        [--temporal-offsets -2 -1 +1 +2] \\
        [--run-vlm] \\
        [--vlm-model Qwen/Qwen2.5-VL-7B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio hint computation
# ---------------------------------------------------------------------------

def compute_audio_hints(
    video_path: Path,
    out_dir: Path,
    mode: str = "translate",
) -> dict[str, Any]:
    """
    Run Whisper + Qwen-text on the video to extract audio brand mentions.

    Returns the hints dict and writes <out_dir>/audio_hints.json.
    Skips the heavy inference if the file already exists.
    """
    hints_path = out_dir / "audio_hints.json"
    if hints_path.exists():
        logger.info("audio_hints.json already exists — skipping Whisper/LLM step.")
        return json.loads(hints_path.read_text())

    logger.info("Computing audio hints for %s …", video_path.name)

    from models.audio_model import WhisperModel, convert_video_to_audio
    from models.description_model import Qwen2_5TextModel
    from utils.prompts import QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT, QWEN_TRANSCRIPT_LOGO_PROMPT

    convert_video_to_audio(str(video_path))

    whisper = WhisperModel()
    if mode == "transcribe":
        transcript, _, lang = whisper.run_example(
            "logo_audio.wav", mode="transcribe", language=None, return_language=True
        )
        lang = lang or "orig"
    else:
        transcript, _, lang = whisper.run_example(
            "logo_audio.wav", mode="translate", language="english", return_language=True
        )
        lang = "english"

    (out_dir / f"transcript.{lang}.txt").write_text(transcript, encoding="utf-8")
    logger.info("Whisper transcript saved (%s, %d chars).", lang, len(transcript))

    qwen_text = Qwen2_5TextModel()
    brand_string = qwen_text.run_example(
        QWEN_TRANSCRIPT_LOGO_PROMPT(transcript),
        QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT,
    )
    brands = sorted({b.lower().strip() for b in brand_string.split("\n") if b.strip()})
    (out_dir / f"brands_audio_{lang}.txt").write_text("\n".join(brands), encoding="utf-8")
    logger.info("Extracted %d audio brands: %s", len(brands), brands)

    hints: dict[str, Any] = {"transcript": transcript, "language": lang, "brands": brands}
    hints_path.write_text(json.dumps(hints, indent=2, ensure_ascii=False))
    return hints


# ---------------------------------------------------------------------------
# FAISS hint computation
# ---------------------------------------------------------------------------

def load_faiss_db(index_path: str, metadata_path: str):
    """Load the CLIP-based LogoDatabaseNew."""
    from models.faiss_db import LogoDatabaseNew
    logger.info("Loading FAISS index from %s …", index_path)
    db = LogoDatabaseNew(index_path=index_path, metadata_path=metadata_path, batch_size=16)
    logger.info("FAISS ready: %d vectors, index loaded.", db.index.ntotal)
    return db


def faiss_top_k(
    db,
    crop_path: Path,
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Query FAISS for the top-k most similar brands to the given crop.

    Deduplicates by brand name (keeps highest similarity per brand).
    """
    crop = Image.open(crop_path).convert("RGB")
    raw = db.search_logo(crop, threshold=threshold, k=top_k * 3)  # over-fetch then dedup

    seen: dict[str, float] = {}
    for r in raw:
        brand = r["brand_name"]
        if brand not in seen or r["similarity"] > seen[brand]:
            seen[brand] = r["similarity"]

    return sorted(
        [{"brand_name": b, "similarity": round(s, 4)} for b, s in seen.items()],
        key=lambda x: x["similarity"],
        reverse=True,
    )[:top_k]


# ---------------------------------------------------------------------------
# VLM prompt assembly
# ---------------------------------------------------------------------------

def _build_vlm_content(
    crop_img: Image.Image,
    enlarged_img: Image.Image,
    frame_img: Image.Image | None,
    temporal_pairs: list[tuple[str, Image.Image]],
    detection: dict[str, Any],
    audio_brands: list[str],
    faiss_hits: list[dict[str, Any]],
) -> list[dict]:
    """
    Assemble the multi-image content list for a single Qwen VLM call.

    Layout (interleaved text + image tokens):
      Signal 1 — tight crop
      Signal 2 — enlarged crop
      Signal 3 — full frame
      Signal 4 — temporal context frames
      Signal 5 — audio brand mentions (text only)
      Signal 6 — FAISS top-k matches (text only)
      Final question
    """
    content: list[dict] = []

    # Signal 1 — tight crop
    content.append({
        "type": "text",
        "text": f"## Signal 1 — Tight crop  (detector confidence: {detection['score']:.3f})\n",
    })
    content.append({"type": "image", "image": crop_img})

    # Signal 2 — enlarged crop
    content.append({"type": "text", "text": "\n## Signal 2 — Enlarged crop with surrounding context\n"})
    content.append({"type": "image", "image": enlarged_img})

    # Signal 3 — full frame
    if frame_img is not None:
        content.append({
            "type": "text",
            "text": f"\n## Signal 3 — Full video frame at t={detection['timecode_seconds']:.1f}s\n",
        })
        content.append({"type": "image", "image": frame_img})

    # Signal 4 — temporal context
    if temporal_pairs:
        content.append({"type": "text", "text": "\n## Signal 4 — Temporal context (nearby frames)\n"})
        for offset, timg in temporal_pairs:
            content.append({"type": "text", "text": f"T{offset}: "})
            content.append({"type": "image", "image": timg})

    # Signal 5 — audio brands (text only)
    if audio_brands:
        brands_str = ", ".join(f'"{b}"' for b in audio_brands)
    else:
        brands_str = "(none detected)"
    content.append({
        "type": "text",
        "text": f"\n## Signal 5 — Brands mentioned in the audio track\n{brands_str}\n",
    })

    # Signal 6 — FAISS visual matches (text only)
    if faiss_hits:
        faiss_lines = "\n".join(
            f"  {i + 1}. {h['brand_name']}  (similarity: {h['similarity']:.3f})"
            for i, h in enumerate(faiss_hits)
        )
    else:
        faiss_lines = "  (no matches above threshold)"
    content.append({
        "type": "text",
        "text": f"\n## Signal 6 — Top visual matches from the brand database (FAISS)\n{faiss_lines}\n",
    })

    # Final question
    content.append({
        "type": "text",
        "text": (
            "\n---\n"
            "You are an expert logo analyst. Using ALL six signals above, answer:\n\n"
            "(i)  Is the detected region a **logo**, a **partial logo**, or **not a logo**?\n"
            "(ii) If it is a logo or partial logo, what is the **brand name**?\n\n"
            "Respond ONLY with a JSON object in exactly this schema (no extra text):\n"
            '{\n'
            '  "is_logo": "logo" | "partial_logo" | "not_logo",\n'
            '  "brand": "<brand name, or null if not_logo>",\n'
            '  "confidence": "high" | "medium" | "low",\n'
            '  "reasoning": "<one or two sentences explaining your decision>"\n'
            '}'
        ),
    })

    return content


# ---------------------------------------------------------------------------
# VLM inference for a single detection
# ---------------------------------------------------------------------------

def run_vlm_on_detection(
    vlm_model,
    processor,
    detection: dict[str, Any],
    extraction_root: Path,
    audio_brands: list[str],
    faiss_hits: list[dict[str, Any]],
    temporal_offsets: list[str],
) -> dict[str, Any]:
    """
    Run Qwen VLM on one detection and return raw text + parsed JSON.
    """
    from qwen_vl_utils import process_vision_info

    # ── Load images ──────────────────────────────────────────────────────
    crop_img = Image.open(extraction_root / detection["crop_path"]).convert("RGB")
    enlarged_img = Image.open(extraction_root / detection["crop_enlarged_path"]).convert("RGB")

    frame_path = (
        extraction_root
        / "detections"
        / f"frame_{detection['frame_idx']:06d}"
        / "frame.jpg"
    )
    frame_img = Image.open(frame_path).convert("RGB") if frame_path.exists() else None

    temporal_pairs: list[tuple[str, Image.Image]] = []
    for offset in temporal_offsets:
        rel = detection["temporal_frames"].get(offset)
        if rel:
            abs_path = extraction_root / rel
            if abs_path.exists():
                temporal_pairs.append((offset, Image.open(abs_path).convert("RGB")))

    # ── Build prompt content ──────────────────────────────────────────────
    content = _build_vlm_content(
        crop_img=crop_img,
        enlarged_img=enlarged_img,
        frame_img=frame_img,
        temporal_pairs=temporal_pairs,
        detection=detection,
        audio_brands=audio_brands,
        faiss_hits=faiss_hits,
    )
    messages = [{"role": "user", "content": content}]

    # ── Tokenise ─────────────────────────────────────────────────────────
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device, vlm_model.dtype)

    # ── Generate ─────────────────────────────────────────────────────────
    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    raw_text: str = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # ── Parse JSON ───────────────────────────────────────────────────────
    parsed: dict | None = None
    try:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
    except json.JSONDecodeError:
        pass

    return {"raw": raw_text, "parsed": parsed}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    video_path: str,
    extraction_dir: str,
    faiss_index: str,
    faiss_metadata: str,
    top_k: int = 5,
    faiss_threshold: float = 0.0,
    audio_mode: str = "translate",
    skip_audio: bool = False,
    run_vlm: bool = False,
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    temporal_offsets: list[str] | None = None,
) -> None:

    if temporal_offsets is None:
        temporal_offsets = ["-2", "-1", "+1", "+2"]

    video_path_obj = Path(video_path)
    extraction_root = Path(extraction_dir)
    manifest_path = extraction_root / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}. "
            "Run data_extraction_pipeline.py first."
        )

    manifest = json.loads(manifest_path.read_text())
    detections: list[dict] = manifest["detections"]
    logger.info("Manifest loaded: %d detections across %d sampled frames.",
                len(detections), manifest.get("sampled_frames", "?"))

    # ── Step 1: audio hints ───────────────────────────────────────────────
    if skip_audio:
        hints_path = extraction_root / "audio_hints.json"
        if hints_path.exists():
            audio_hints = json.loads(hints_path.read_text())
            logger.info("--skip-audio: loaded cached audio_hints.json.")
        else:
            audio_hints = {"brands": [], "transcript": "", "language": "unknown"}
            logger.info("--skip-audio: no cached audio_hints.json found, audio_brands will be empty.")
    else:
        audio_hints = compute_audio_hints(video_path_obj, extraction_root, mode=audio_mode)
    audio_brands: list[str] = audio_hints.get("brands", [])

    # ── Step 2: load FAISS ────────────────────────────────────────────────
    db = load_faiss_db(faiss_index, faiss_metadata)

    # ── Step 3: per-detection FAISS hints ─────────────────────────────────
    logger.info("Computing FAISS top-%d hints for %d detections …", top_k, len(detections))
    for det in tqdm(detections, desc="FAISS hints"):
        det_dir = extraction_root / Path(det["crop_path"]).parent
        hints_path = det_dir / "hints.json"

        if hints_path.exists():
            continue  # already computed

        crop_abs = extraction_root / det["crop_path"]
        faiss_hits = faiss_top_k(db, crop_abs, top_k=top_k, threshold=faiss_threshold)

        hints_out: dict[str, Any] = {
            "det_id": det["det_id"],
            "audio_brands": audio_brands,
            "faiss_top_k": faiss_hits,
        }
        hints_path.write_text(json.dumps(hints_out, indent=2, ensure_ascii=False))

    logger.info("FAISS hints written.")

    # ── Step 4: optional VLM inference ───────────────────────────────────
    if not run_vlm:
        logger.info(
            "Hints precomputed. Pass --run-vlm to also run the VLM on each detection."
        )
        return

    logger.info("Loading VLM: %s …", vlm_model_id)
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_id,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quant_cfg,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        vlm_model_id, use_fast=True, padding_side="left", max_pixels=128 * 28 * 28
    )
    logger.info("VLM ready.")

    for det in tqdm(detections, desc="VLM inference"):
        det_dir = extraction_root / Path(det["crop_path"]).parent
        pred_path = det_dir / "vlm_prediction.json"

        if pred_path.exists():
            continue  # already inferred

        faiss_hits: list[dict] = []
        hints_path = det_dir / "hints.json"
        if hints_path.exists():
            faiss_hits = json.loads(hints_path.read_text()).get("faiss_top_k", [])

        try:
            result = run_vlm_on_detection(
                vlm_model=vlm_model,
                processor=processor,
                detection=det,
                extraction_root=extraction_root,
                audio_brands=audio_brands,
                faiss_hits=faiss_hits,
                temporal_offsets=temporal_offsets,
            )
        except Exception as exc:
            logger.warning("VLM failed on det_id=%d: %s", det["det_id"], exc)
            result = {"raw": None, "parsed": None, "error": str(exc)}

        pred_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    logger.info("VLM inference complete. Results saved as vlm_prediction.json per detection.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Precompute multi-signal hints (Whisper audio + FAISS visual) "
            "and optionally run the Qwen VLM on every detection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--extraction-dir", required=True,
        help="Path to the extraction output directory (must contain manifest.json).",
    )
    parser.add_argument(
        "--faiss-index", default="FAISS/logo_index.faiss",
        help="Path to the FAISS .faiss index file.",
    )
    parser.add_argument(
        "--faiss-metadata", default="FAISS/metadata.json",
        help="Path to the FAISS brand metadata JSON.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top FAISS matches to retrieve per detection.",
    )
    parser.add_argument(
        "--faiss-threshold", type=float, default=0.0,
        help="Minimum FAISS similarity to keep a match (0 = return all top-k).",
    )
    parser.add_argument(
        "--audio-mode", choices=["translate", "transcribe"], default="translate",
        help="Whisper mode: 'translate' outputs English; 'transcribe' keeps original language.",
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Skip Whisper/LLM audio step (uses cached audio_hints.json if present, else empty brands).",
    )
    parser.add_argument(
        "--temporal-offsets", nargs="+", default=["-2", "-1", "+1", "+2"],
        metavar="OFFSET",
        help="Signed frame offsets to include as temporal context (e.g. -2 -1 +1 +2).",
    )
    parser.add_argument(
        "--run-vlm", action="store_true",
        help="After computing hints, run the VLM on each detection.",
    )
    parser.add_argument(
        "--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID for the VLM.",
    )

    args = parser.parse_args()

    run(
        video_path=args.video,
        extraction_dir=args.extraction_dir,
        faiss_index=args.faiss_index,
        faiss_metadata=args.faiss_metadata,
        top_k=args.top_k,
        faiss_threshold=args.faiss_threshold,
        audio_mode=args.audio_mode,
        skip_audio=args.skip_audio,
        run_vlm=args.run_vlm,
        vlm_model_id=args.vlm_model,
        temporal_offsets=args.temporal_offsets,
    )
