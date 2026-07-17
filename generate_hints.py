#!/usr/bin/env python3
"""
Multi-Signal Hints Generation Script.

Precomputes three kinds of hints before VLM inference:

  1. Audio hints  — Whisper transcription → Qwen-text brand extraction
                    → saved as  <extraction_dir>/audio_hints.json
                    (skipped automatically when the file already exists)

  2. FAISS hints  — top-k nearest brands from the CLIP FAISS index
                    → merged into  <det_dir>/hints.json  for every detection

  3. OCR hints    — EasyOCR text read from the cropped region
                    → merged into  <det_dir>/hints.json  for every detection

Optionally runs a Qwen VLM with all enabled signals (tight crop, enlarged
crop, full frame, temporal frames, audio brands, FAISS matches, OCR text)
to decide:
  (i)   whether the crop is a logo / partial logo / not a logo
  (ii)  if it is a logo, what the brand name is (open-set or closed-set)
  (iii) whether the logo is truncated (cut off at the crop edge)

After VLM inference a light text-LLM pass canonicalizes all assigned brand
names — spelling variants of the same brand ("verysure", "verishore",
"verisure") collapse to one canonical spelling stored as `brand_canonical`
next to the original `brand`. Finally it draws every detection on its full
frame with the VLM verdict and brand name. Both post-passes can also run
standalone on cached predictions via --canonicalize / --annotate.

Output files added to the existing extraction layout:
    <extraction_dir>/
    ├── audio_hints.json                        ← video-level audio brands
    ├── detections/frame_*/det_*/
    │   ├── hints.json                          ← audio + FAISS top-k + OCR
    │   └── vlm_prediction[__<exp>].json        ← VLM answer (with --run-vlm)
    ├── brand_canonicalization[__<exp>].json    ← brand variant → canonical map
    ├── annotated_frames[__<exp>]/frame_*.jpg   ← bbox + brand drawn per frame
    ├── frame_annotations[__<exp>].json         ← machine-readable annotations
    └── vlm_predictions[__<exp>].csv            ← flat results table (review UI)

Ablations:
    Use --signals to choose which signals the VLM sees, --brand-mode to
    switch open-set vs closed-set brand naming, and --experiment-name to
    keep outputs of different runs side by side, e.g.:

    python generate_hints.py ... --run-vlm \\
        --signals tight_crop enlarged_crop faiss_matches \\
        --brand-mode closed \\
        --experiment-name no_audio_closed

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
        [--signals tight_crop enlarged_crop full_frame temporal_context audio_brands faiss_matches ocr_text] \\
        [--brand-mode open|closed] \\
        [--channel-watermark "Rai Sport"] \\
        [--experiment-name my_ablation] \\
        [--ocr-langs en it] \\
        [--run-vlm] \\
        [--canonicalize] \\
        [--annotate] \\
        [--export-csv] \\
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
# Canonical signal vocabulary
# ---------------------------------------------------------------------------
# The ONLY names the VLM is allowed to use in `signals_used`, the same names
# used as prompt section headers, and the names outputs are normalized to.

CANONICAL_SIGNALS = [
    "tight_crop",
    "enlarged_crop",
    "full_frame",
    "temporal_context",
    "audio_brands",
    "faiss_matches",
    "ocr_text",
]

# Synonyms observed in VLM outputs → canonical name (used at parse time).
SIGNAL_SYNONYMS = {
    "tight_crop": "tight_crop", "crop": "tight_crop", "crop_visual": "tight_crop",
    "crop_image": "tight_crop", "signal_1": "tight_crop", "signal1": "tight_crop",
    "detected_region": "tight_crop",
    "enlarged_crop": "enlarged_crop", "context_crop": "enlarged_crop",
    "surrounding_context": "enlarged_crop", "signal_2": "enlarged_crop",
    "signal2": "enlarged_crop", "zoomed_out_crop": "enlarged_crop",
    "full_frame": "full_frame", "frame": "full_frame", "video_frame": "full_frame",
    "scene": "full_frame", "signal_3": "full_frame", "signal3": "full_frame",
    "full_video_frame": "full_frame",
    "temporal_context": "temporal_context", "temporal": "temporal_context",
    "temporal_frames": "temporal_context", "nearby_frames": "temporal_context",
    "signal_4": "temporal_context", "signal4": "temporal_context",
    "context_frames": "temporal_context",
    "audio_brands": "audio_brands", "audio": "audio_brands",
    "audio_track": "audio_brands", "audio_mentions": "audio_brands",
    "transcript": "audio_brands", "signal_5": "audio_brands", "signal5": "audio_brands",
    "faiss_matches": "faiss_matches", "faiss": "faiss_matches",
    "database": "faiss_matches", "brand_database": "faiss_matches",
    "visual_matches": "faiss_matches", "db_matches": "faiss_matches",
    "signal_6": "faiss_matches", "signal6": "faiss_matches",
    "ocr_text": "ocr_text", "ocr": "ocr_text", "detected_text": "ocr_text",
    "text": "ocr_text", "crop_text": "ocr_text",
    "signal_7": "ocr_text", "signal7": "ocr_text",
}


def normalize_signals_used(raw: Any) -> list[str]:
    """Map a VLM `signals_used` list onto CANONICAL_SIGNALS (drop the rest)."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")
        canon = SIGNAL_SYNONYMS.get(key)
        if canon is None:  # substring fallback, e.g. "the faiss database matches"
            for c in CANONICAL_SIGNALS:
                if c in key or key in c:
                    canon = c
                    break
        if canon and canon not in out:
            out.append(canon)
    return out


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
# Per-detection hints.json helpers (merge-friendly: FAISS and OCR share it)
# ---------------------------------------------------------------------------

def load_hints(det_dir: Path, det: dict[str, Any]) -> dict[str, Any]:
    hints_path = det_dir / "hints.json"
    if hints_path.exists():
        return json.loads(hints_path.read_text())
    return {"det_id": det["det_id"]}


def save_hints(det_dir: Path, hints: dict[str, Any]) -> None:
    det_dir.mkdir(parents=True, exist_ok=True)
    (det_dir / "hints.json").write_text(json.dumps(hints, indent=2, ensure_ascii=False))


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
# OCR hint computation (EasyOCR)
# ---------------------------------------------------------------------------

def ocr_image(reader, image: Image.Image, min_side: int = 80, upscale: int = 3) -> list[dict]:
    """Run EasyOCR on one image; upscale tiny crops first so text is legible."""
    import numpy as np

    img = image.convert("RGB")
    if min(img.size) < min_side:
        img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    results = reader.readtext(np.array(img))
    return [
        {"text": text.strip(), "confidence": round(float(conf), 3)}
        for _, text, conf in results
        if text.strip()
    ]


def compute_ocr_hints(
    detections: list[dict],
    extraction_root: Path,
    languages: list[str],
) -> None:
    """OCR every detection crop and merge the result into its hints.json."""
    import easyocr

    logger.info("Loading EasyOCR (%s) …", languages)
    reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())

    logger.info("Computing OCR hints for %d detections …", len(detections))
    for det in tqdm(detections, desc="OCR hints"):
        det_dir = extraction_root / Path(det["crop_path"]).parent
        hints = load_hints(det_dir, det)
        if "ocr" in hints:
            continue  # already computed

        # OCR the tight crop first; fall back to the enlarged crop if no text
        lines = ocr_image(reader, Image.open(extraction_root / det["crop_path"]))
        source = "tight_crop"
        if not lines:
            lines = ocr_image(reader, Image.open(extraction_root / det["crop_enlarged_path"]))
            source = "enlarged_crop" if lines else None

        hints["ocr"] = {"source": source, "lines": lines}
        save_hints(det_dir, hints)

    logger.info("OCR hints written.")


# ---------------------------------------------------------------------------
# VLM prompt assembly
# ---------------------------------------------------------------------------

def collect_temporal_pairs(
    detection: dict[str, Any],
    extraction_root: Path,
    offsets: list[str],
) -> list[tuple[str, Image.Image]]:
    """Load the requested temporal-context frames for one detection.

    Falls back to the nearest temporal frames that DO exist when none of the
    requested offsets were saved — older extractions stored only the widest
    offsets (e.g. ±5), which used to leave the temporal_context signal
    silently empty.
    """
    def _load(offset: str) -> Image.Image | None:
        rel = detection["temporal_frames"].get(offset)
        if rel and (extraction_root / rel).exists():
            return Image.open(extraction_root / rel).convert("RGB")
        return None

    pairs = [(o, img) for o in offsets if (img := _load(o)) is not None]
    if pairs:
        return pairs

    available = sorted(
        (k for k, v in detection["temporal_frames"].items()
         if v and (extraction_root / v).exists()),
        key=lambda k: (abs(int(k)), int(k)),
    )[: max(len(offsets), 1)]
    return [(o, img) for o in sorted(available, key=int) if (img := _load(o)) is not None]


def build_closed_set_candidates(
    audio_brands: list[str],
    faiss_hits: list[dict],
    signals: set[str],
) -> list[str]:
    """Candidate brands for closed-set mode: audio brands + FAISS top-k of the
    ENABLED signals (an ablated signal contributes no candidates)."""
    candidates: list[str] = []
    if "audio_brands" in signals:
        candidates += audio_brands
    if "faiss_matches" in signals:
        candidates += [h["brand_name"] for h in faiss_hits]

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        key = c.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def build_final_instruction(
    candidates: list[str],
    brand_mode: str,
    channel_watermark: str,
) -> str:
    """The question + output schema shown to the VLM after all signals."""
    watermark_note = ""
    if channel_watermark:
        watermark_note = (
            f"Known context: the channel watermark in this video is \"{channel_watermark}\". "
            f"If this region is that channel watermark, assign the brand \"{channel_watermark}\". "
            "Attribute all remaining logos to advertisers or sponsors, not to the channel.\n\n"
        )

    if brand_mode == "closed":
        if candidates:
            cand_str = ", ".join(f'"{c}"' for c in candidates)
            brand_rule = (
                "(ii)  If it is a logo or partial logo, assign the brand name STRICTLY from "
                f"this candidate list: [{cand_str}]. "
                'If none of the candidates match what you actually see, output exactly "UNKNOWN". '
                "Never output a brand that is not in the list.\n"
            )
        else:
            brand_rule = (
                "(ii)  No brand candidates are available for this detection. "
                'If it is a logo or partial logo, output exactly "UNKNOWN" as the brand.\n'
            )
    else:
        brand_rule = (
            "(ii)  If it is a logo or partial logo, what is the brand name? "
            "The audio / FAISS / OCR hints are soft evidence — they may be wrong or "
            "incomplete, and you may name a brand that appears in no hint. "
            'If you cannot identify the brand, output exactly "UNKNOWN".\n'
        )

    signal_vocab = ", ".join(f'"{s}"' for s in CANONICAL_SIGNALS)

    return (
        "\n---\n"
        "You are an expert logo analyst. Base your answer ONLY on the signals shown above.\n\n"
        + watermark_note +
        "A LOGO is a distinctive graphic mark, emblem, wordmark or symbol that identifies "
        "a brand, company, product, or sponsoring organisation.\n"
        "Do NOT classify as a logo: plain scene text, captions or headlines, scoreboards "
        "and score bugs, jersey or player numbers, clocks and timers, generic icons or UI "
        "elements, national or regional flags, faces or people, unbranded products, and "
        "random shapes, patterns or textures. These are \"not_logo\".\n\n"
        "Answer:\n"
        "(i)   Is the detected region a **logo**, a **partial logo**, or **not a logo**?\n"
        + brand_rule +
        "(iii) Is the logo **truncated** — visibly cut off at the edge of the crop or frame?\n"
        "(iv)  What is the **probability** (0-100) that this region contains ANY logo at all?\n"
        "(v)   How **confident** (0-100) are you in your answer (see field rules below)?\n\n"
        "Respond ONLY with a JSON object — no extra text, no markdown fences:\n"
        "{\n"
        '  "is_logo": "logo" | "partial_logo" | "not_logo",\n'
        '  "brand": "<brand name>" | "UNKNOWN" | null,\n'
        '  "is_truncated": true | false,\n'
        '  "logo_probability": <integer 0-100>,\n'
        '  "confidence": <integer 0-100>,\n'
        '  "signals_used": ["<signal>", ...],\n'
        '  "reasoning": "<1-3 sentences explaining which signals drove the decision>"\n'
        "}\n\n"
        "Field rules:\n"
        '- brand: null when not_logo; "UNKNOWN" when it is a logo but the brand cannot be determined.\n'
        "- is_truncated: true only if the logo extends past the crop/frame edge and is "
        "visibly cut off; false otherwise (and false when not_logo).\n"
        "- logo_probability: your estimate that this region IS a logo of any brand, "
        "independent of brand identity. 0 = definitely not a logo, 100 = definitely a logo.\n"
        "- confidence: for \"logo\"/\"partial_logo\" — your certainty in the specific brand "
        "name; for \"not_logo\" — your certainty that the region is truly not a logo.\n"
        f"- signals_used: the signals that materially influenced your decision. Use ONLY "
        f"these exact values: [{signal_vocab}]. Never invent other names.\n"
    )


def _build_vlm_content(
    crop_img: Image.Image | None,
    enlarged_img: Image.Image | None,
    frame_img: Image.Image | None,
    temporal_pairs: list[tuple[str, Image.Image]],
    detection: dict[str, Any],
    audio_brands: list[str],
    faiss_hits: list[dict[str, Any]],
    ocr_data: dict[str, Any] | None,
    signals: set[str],
    brand_mode: str,
    channel_watermark: str,
) -> list[dict]:
    """
    Assemble the multi-image content list for a single Qwen VLM call.

    Section headers use the CANONICAL_SIGNALS names so the model can echo
    them back verbatim in `signals_used`. Disabled signals are omitted:
      tight_crop, enlarged_crop, full_frame, temporal_context (images)
      audio_brands, faiss_matches, ocr_text (text)
      Final question (open-set or closed-set)
    """
    content: list[dict] = []

    # Signal — tight crop
    if "tight_crop" in signals and crop_img is not None:
        content.append({
            "type": "text",
            "text": f"## Signal: tight_crop — the detected region  (detector confidence: {detection['score']:.3f})\n",
        })
        content.append({"type": "image", "image": crop_img})

    # Signal — enlarged crop
    if "enlarged_crop" in signals and enlarged_img is not None:
        content.append({"type": "text", "text": "\n## Signal: enlarged_crop — same region with surrounding context\n"})
        content.append({"type": "image", "image": enlarged_img})

    # Signal — full frame
    if "full_frame" in signals and frame_img is not None:
        content.append({
            "type": "text",
            "text": f"\n## Signal: full_frame — full video frame at t={detection['timecode_seconds']:.1f}s\n",
        })
        content.append({"type": "image", "image": frame_img})

    # Signal — temporal context
    if "temporal_context" in signals and temporal_pairs:
        content.append({"type": "text", "text": "\n## Signal: temporal_context — nearby frames\n"})
        for offset, timg in temporal_pairs:
            content.append({"type": "text", "text": f"T{offset}: "})
            content.append({"type": "image", "image": timg})

    # Signal — audio brands (text only)
    if "audio_brands" in signals:
        brands_str = ", ".join(f'"{b}"' for b in audio_brands) if audio_brands else "(none detected)"
        content.append({
            "type": "text",
            "text": f"\n## Signal: audio_brands — brands mentioned in the audio track\n{brands_str}\n",
        })

    # Signal — FAISS visual matches (text only)
    if "faiss_matches" in signals:
        if faiss_hits:
            faiss_lines = "\n".join(
                f"  {i + 1}. {h['brand_name']}  (similarity: {h['similarity']:.3f})"
                for i, h in enumerate(faiss_hits)
            )
        else:
            faiss_lines = "  (no matches above threshold)"
        content.append({
            "type": "text",
            "text": f"\n## Signal: faiss_matches — top visual matches from the brand database\n{faiss_lines}\n",
        })

    # Signal — OCR text (text only)
    if "ocr_text" in signals:
        lines = (ocr_data or {}).get("lines", [])
        source = (ocr_data or {}).get("source") or "cropped region"
        if lines:
            ocr_lines = "\n".join(
                f'  "{l["text"]}"  (OCR confidence: {l["confidence"]:.2f})' for l in lines
            )
            ocr_text = f"\n## Signal: ocr_text — text read by OCR from the {source.replace('_', ' ')}\n{ocr_lines}\n"
        else:
            ocr_text = "\n## Signal: ocr_text — text read by OCR from the cropped region\n  (no text detected)\n"
        content.append({"type": "text", "text": ocr_text})

    # Final question + schema
    candidates = build_closed_set_candidates(audio_brands, faiss_hits, signals)
    content.append({
        "type": "text",
        "text": build_final_instruction(candidates, brand_mode, channel_watermark),
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
    ocr_data: dict[str, Any] | None,
    temporal_offsets: list[str],
    signals: set[str],
    brand_mode: str,
    channel_watermark: str,
) -> dict[str, Any]:
    """
    Run Qwen VLM on one detection and return raw text + parsed JSON.
    """
    from qwen_vl_utils import process_vision_info

    # ── Load images (only for enabled signals) ────────────────────────────
    crop_img = enlarged_img = frame_img = None
    if "tight_crop" in signals:
        crop_img = Image.open(extraction_root / detection["crop_path"]).convert("RGB")
    if "enlarged_crop" in signals:
        enlarged_img = Image.open(extraction_root / detection["crop_enlarged_path"]).convert("RGB")
    if "full_frame" in signals:
        frame_path = (
            extraction_root
            / "detections"
            / f"frame_{detection['frame_idx']:06d}"
            / "frame.jpg"
        )
        frame_img = Image.open(frame_path).convert("RGB") if frame_path.exists() else None

    temporal_pairs: list[tuple[str, Image.Image]] = []
    if "temporal_context" in signals:
        temporal_pairs = collect_temporal_pairs(detection, extraction_root, temporal_offsets)

    # ── Build prompt content ──────────────────────────────────────────────
    content = _build_vlm_content(
        crop_img=crop_img,
        enlarged_img=enlarged_img,
        frame_img=frame_img,
        temporal_pairs=temporal_pairs,
        detection=detection,
        audio_brands=audio_brands,
        faiss_hits=faiss_hits,
        ocr_data=ocr_data,
        signals=signals,
        brand_mode=brand_mode,
        channel_watermark=channel_watermark,
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

    if parsed is not None and "signals_used" in parsed:
        parsed["signals_used_raw"] = parsed.get("signals_used")
        parsed["signals_used"] = normalize_signals_used(parsed.get("signals_used"))

    return {"raw": raw_text, "parsed": parsed}


# ---------------------------------------------------------------------------
# Brand canonicalization pass (light text-LLM, runs once per video)
# ---------------------------------------------------------------------------
# Spelling variants of the same brand assigned across frames ("verysure",
# "verishore", "verisure") collapse to one canonical spelling chosen by the
# model. Adds a "brand_canonical" field to every vlm_prediction file (the
# original "brand" is left untouched) and saves the full mapping for review.

CANONICALIZE_PROMPT = lambda brands: f"""
The following brand-name strings were assigned to logo detections in one video by a vision model.
Different strings may be spelling variants of the SAME real-world brand (OCR errors, transcription
errors, casing or spacing differences) — e.g. "verysure", "verishore" and "verisure" all refer to "Verisure".

Map EVERY input string to a canonical brand spelling:
- Group strings that refer to the same real-world brand and give them ONE identical canonical spelling.
- Prefer the brand's official spelling if you know it; otherwise pick the most plausible variant.
- Do NOT merge genuinely different brands, even if their names look similar.
- If a string is already spelled correctly, map it to itself.

Input strings:
{json.dumps(brands, ensure_ascii=False)}

Respond ONLY with a JSON object mapping every input string (exactly as given) to its canonical form — no extra text:
{{"<input string>": "<canonical brand>", ...}}
"""

CANONICALIZE_SYSTEM_PROMPT = (
    "You are an expert in brand names. You normalize noisy brand-name strings "
    "produced by vision and audio models into canonical spellings."
)


def parse_brand_mapping(raw: str, brands: list[str]) -> dict[str, str]:
    """Parse the mapping JSON; any brand the model missed maps to itself."""
    mapping: dict[str, str] = {}
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            if isinstance(data, dict):
                mapping = {
                    k: v.strip() for k, v in data.items()
                    if isinstance(k, str) and isinstance(v, str) and v.strip()
                }
    except json.JSONDecodeError:
        pass
    return {b: mapping.get(b, b) for b in brands}


def collect_assigned_brands(
    detections: list[dict],
    extraction_root: Path,
    pred_filename: str,
) -> list[str]:
    """Unique brand strings the VLM assigned across all detections (frames)."""
    brands: list[str] = []
    for det in detections:
        pred_path = extraction_root / Path(det["crop_path"]).parent / pred_filename
        if not pred_path.exists():
            continue
        parsed = json.loads(pred_path.read_text()).get("parsed") or {}
        b = parsed.get("brand")
        if isinstance(b, str) and b.strip() and b != "UNKNOWN" and b not in brands:
            brands.append(b)
    return sorted(brands, key=str.lower)


def canonicalize_brands(
    detections: list[dict],
    extraction_root: Path,
    pred_filename: str,
    exp_suffix: str,
    experiment_config: dict[str, Any],
) -> None:
    """Run the canonicalization LLM pass and stamp brand_canonical into every
    prediction file."""
    assigned_brands = collect_assigned_brands(detections, extraction_root, pred_filename)
    logger.info("Distinct brands assigned by the VLM: %d", len(assigned_brands))
    if not assigned_brands:
        logger.info("No brands assigned — skipping canonicalization pass.")
        return

    from models.description_model import Qwen2_5TextModel

    logger.info("Loading text LLM for brand canonicalization …")
    text_model = Qwen2_5TextModel()
    raw_mapping = text_model.run_example(
        CANONICALIZE_PROMPT(assigned_brands),
        CANONICALIZE_SYSTEM_PROMPT,
    )
    del text_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    brand_map = parse_brand_mapping(raw_mapping, assigned_brands)
    changed = {k: v for k, v in brand_map.items() if k != v}
    logger.info("Canonicalization changed %d of %d brand spellings: %s",
                len(changed), len(brand_map), changed)

    map_path = extraction_root / f"brand_canonicalization{exp_suffix}.json"
    map_path.write_text(json.dumps(
        {"config": experiment_config, "input_brands": assigned_brands,
         "mapping": brand_map, "raw_response": raw_mapping},
        indent=2, ensure_ascii=False,
    ))
    logger.info("Mapping saved to %s", map_path)

    for det in detections:
        pred_path = extraction_root / Path(det["crop_path"]).parent / pred_filename
        if not pred_path.exists():
            continue
        data = json.loads(pred_path.read_text())
        parsed = data.get("parsed")
        if not parsed:
            continue
        b = parsed.get("brand")
        if b is None:
            parsed["brand_canonical"] = None
        elif b == "UNKNOWN":
            parsed["brand_canonical"] = "UNKNOWN"
        else:
            parsed["brand_canonical"] = brand_map.get(b, b)
        pred_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("brand_canonical written into all prediction files.")


# ---------------------------------------------------------------------------
# Frame-level bounding-box annotation
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "logo":         (0, 190, 0),
    "partial_logo": (255, 160, 0),
    "not_logo":     (220, 40, 40),
    None:           (128, 128, 128),
}


def _load_font(size: int = 20):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_frames(
    detections: list[dict],
    extraction_root: Path,
    pred_filename: str,
    exp_suffix: str,
    experiment_config: dict[str, Any],
) -> Path:
    """
    Draw every detection on its full frame with the VLM verdict:
      green = logo, orange = partial_logo, red = not_logo, gray = unparsed.
    Writes annotated JPEGs + a machine-readable frame_annotations.json.
    """
    from PIL import ImageDraw

    ann_dir = extraction_root / f"annotated_frames{exp_suffix}"
    ann_dir.mkdir(exist_ok=True)
    font = _load_font(20)

    by_frame: dict[int, list[dict]] = {}
    for det in detections:
        by_frame.setdefault(det["frame_idx"], []).append(det)

    frames_out = []
    for frame_idx, dets in tqdm(sorted(by_frame.items()), desc="Annotating frames"):
        frame_path = extraction_root / "detections" / f"frame_{frame_idx:06d}" / "frame.jpg"
        if not frame_path.exists():
            continue
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        det_entries = []
        for det in dets:
            det_dir = extraction_root / Path(det["crop_path"]).parent
            pred_path = det_dir / pred_filename
            p: dict = {}
            if pred_path.exists():
                p = json.loads(pred_path.read_text()).get("parsed") or {}

            is_logo = p.get("is_logo")
            brand = p.get("brand")
            brand_canonical = p.get("brand_canonical", brand)
            confidence = p.get("confidence")
            color = LABEL_COLORS.get(is_logo, LABEL_COLORS[None])
            box = [float(c) for c in det["box_xyxy"]]

            if is_logo in ("logo", "partial_logo"):
                label = f"{brand_canonical or brand or 'UNKNOWN'}"
                if isinstance(confidence, (int, float)):
                    label += f" {int(confidence)}"
                if is_logo == "partial_logo":
                    label += " (partial)"
                if p.get("is_truncated") is True:
                    label += " [cut off]"
            elif is_logo == "not_logo":
                label = "not a logo"
            else:
                label = "unparsed"

            draw.rectangle(box, outline=color, width=3)
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            x0 = max(0, min(box[0], img.width - tw - 10))
            y0 = max(0, box[1] - th - 8)
            draw.rectangle([x0, y0, x0 + tw + 8, y0 + th + 8], fill=color)
            draw.text((x0 + 4, y0 + 3), label, fill=(255, 255, 255), font=font)

            det_entries.append({
                "det_id":           det["det_id"],
                "box_xyxy":         det["box_xyxy"],
                "detector_score":   round(det["score"], 4),
                "is_logo":          is_logo,
                "brand":            brand,
                "brand_canonical":  brand_canonical,
                "is_truncated":     p.get("is_truncated"),
                "logo_probability": p.get("logo_probability"),
                "confidence":       confidence,
                "signals_used":     p.get("signals_used", []),
            })

        ann_path = ann_dir / f"frame_{frame_idx:06d}.jpg"
        img.save(ann_path, quality=90)

        frames_out.append({
            "frame_idx":        frame_idx,
            "timecode_seconds": dets[0]["timecode_seconds"],
            "frame_path":       str(frame_path.relative_to(extraction_root)),
            "annotated_path":   str(ann_path.relative_to(extraction_root)),
            "detections":       det_entries,
        })

    ann_json = {
        "config": experiment_config,
        "num_frames": len(frames_out),
        "num_detections": sum(len(f["detections"]) for f in frames_out),
        "frames": frames_out,
    }
    ann_json_path = extraction_root / f"frame_annotations{exp_suffix}.json"
    ann_json_path.write_text(json.dumps(ann_json, indent=2, ensure_ascii=False))
    logger.info("Annotated %d frames → %s", len(frames_out), ann_dir)
    logger.info("Frame annotation summary → %s", ann_json_path)
    return ann_dir


# ---------------------------------------------------------------------------
# CSV export (same schema as the Kaggle notebook's results cell)
# ---------------------------------------------------------------------------

def export_predictions_csv(
    detections: list[dict],
    extraction_root: Path,
    pred_filename: str,
    exp_suffix: str,
    audio_brands: list[str],
    experiment_config: dict[str, Any],
) -> Path:
    """Flatten all per-detection predictions + hints into one CSV for the
    review UI (app/review_ui.py). Only adds columns the UI already tolerates."""
    import pandas as pd

    root = extraction_root.resolve()

    def _clamp(val) -> int | None:
        try:
            return max(0, min(100, int(float(val))))
        except (TypeError, ValueError):
            return None

    results = []
    for det in detections:
        det_dir = root / Path(det["crop_path"]).parent
        pred_path = det_dir / pred_filename
        hints_path = det_dir / "hints.json"

        hints: dict = json.loads(hints_path.read_text()) if hints_path.exists() else {}
        faiss_hits: list[dict] = hints.get("faiss_top_k", [])
        ocr_lines: list[dict] = (hints.get("ocr") or {}).get("lines", [])

        faiss_brands = [h["brand_name"] for h in faiss_hits]
        faiss_scores = [h["similarity"] for h in faiss_hits]

        p: dict = {}
        if pred_path.exists():
            p = json.loads(pred_path.read_text()).get("parsed") or {}

        results.append({
            # ── detection metadata ──────────────────────────────────────
            "det_id":           det["det_id"],
            "frame_idx":        det["frame_idx"],
            "timecode_s":       det["timecode_seconds"],
            "det_score":        round(det["score"], 4),
            "box_xyxy":         det["box_xyxy"],
            # ── file paths for the review UI ────────────────────────────
            "crop_path":        str(root / det["crop_path"]),
            "enlarged_crop_path": str(root / det["crop_enlarged_path"]),
            "frame_path":       str(root / "detections" / f"frame_{det['frame_idx']:06d}" / "frame.jpg"),
            "annotated_frame_path": str(root / f"annotated_frames{exp_suffix}" / f"frame_{det['frame_idx']:06d}.jpg"),
            # ── VLM outputs ─────────────────────────────────────────────
            "is_logo":          p.get("is_logo"),
            "brand":            p.get("brand"),
            "brand_canonical":  p.get("brand_canonical", p.get("brand")),
            "is_truncated":     p.get("is_truncated"),
            "logo_probability": _clamp(p.get("logo_probability")),
            "confidence":       _clamp(p.get("confidence")),
            "signals_used":     json.dumps(normalize_signals_used(p.get("signals_used", []))),
            "signals_used_raw": json.dumps(p.get("signals_used_raw", p.get("signals_used", []))),
            "reasoning":        p.get("reasoning"),
            # ── FAISS context ───────────────────────────────────────────
            "faiss_top1":       faiss_brands[0] if faiss_brands else None,
            "faiss_top1_score": faiss_scores[0] if faiss_scores else None,
            "faiss_top5_brands": json.dumps(faiss_brands),
            "faiss_top5_scores": json.dumps(faiss_scores),
            # ── audio context ───────────────────────────────────────────
            "audio_brands":     json.dumps(audio_brands),
            # ── OCR context ─────────────────────────────────────────────
            "ocr_text":         " | ".join(l["text"] for l in ocr_lines),
            "ocr_source":       (hints.get("ocr") or {}).get("source"),
            # ── experiment metadata ─────────────────────────────────────
            "experiment":       experiment_config.get("experiment"),
            "brand_mode":       experiment_config.get("brand_assignment"),
            "enabled_signals":  json.dumps(experiment_config.get("signals", [])),
            # ── review label (filled in by the UI) ──────────────────────
            "label":            None,   # "TP" | "FP" | "FN" | "wrong_brand"
        })

    df = pd.DataFrame(results)
    csv_path = extraction_root / f"vlm_predictions{exp_suffix}.csv"
    df.to_csv(csv_path, index=False)

    logger.info(
        "CSV: %d rows → %s  (logo: %d | partial: %d | not_logo: %d | unparsed: %d)",
        len(df), csv_path,
        (df.is_logo == "logo").sum(), (df.is_logo == "partial_logo").sum(),
        (df.is_logo == "not_logo").sum(), df.is_logo.isna().sum(),
    )
    return csv_path


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
    canonicalize: bool = False,
    annotate: bool = False,
    export_csv: bool = False,
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    temporal_offsets: list[str] | None = None,
    signals: list[str] | None = None,
    brand_mode: str = "open",
    channel_watermark: str = "",
    experiment_name: str = "",
    ocr_langs: list[str] | None = None,
) -> None:

    if temporal_offsets is None:
        temporal_offsets = ["-2", "-1", "+1", "+2"]
    if ocr_langs is None:
        ocr_langs = ["en", "it"]

    enabled_signals: set[str] = set(signals if signals is not None else CANONICAL_SIGNALS)
    unknown = enabled_signals - set(CANONICAL_SIGNALS)
    if unknown:
        raise ValueError(f"Unknown signals {sorted(unknown)}; valid options: {CANONICAL_SIGNALS}")

    exp_suffix = f"__{experiment_name}" if experiment_name else ""
    pred_filename = f"vlm_prediction{exp_suffix}.json"

    experiment_config = {
        "experiment":        experiment_name or "default",
        "brand_assignment":  brand_mode,
        "channel_watermark": channel_watermark,
        "signals":           [s for s in CANONICAL_SIGNALS if s in enabled_signals],
        "vlm_model":         vlm_model_id,
    }
    logger.info("Experiment config: %s", experiment_config)

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

    # ── Step 2+3: per-detection FAISS hints ───────────────────────────────
    if "faiss_matches" in enabled_signals:
        db = load_faiss_db(faiss_index, faiss_metadata)

        logger.info("Computing FAISS top-%d hints for %d detections …", top_k, len(detections))
        for det in tqdm(detections, desc="FAISS hints"):
            det_dir = extraction_root / Path(det["crop_path"]).parent
            hints = load_hints(det_dir, det)
            if "faiss_top_k" in hints:
                continue  # already computed

            crop_abs = extraction_root / det["crop_path"]
            hints["audio_brands"] = audio_brands
            hints["faiss_top_k"] = faiss_top_k(db, crop_abs, top_k=top_k, threshold=faiss_threshold)
            save_hints(det_dir, hints)

        logger.info("FAISS hints written.")
        del db
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        logger.info("faiss_matches signal disabled — skipping FAISS hint computation.")

    # ── Step 3b: per-detection OCR hints ──────────────────────────────────
    if "ocr_text" in enabled_signals:
        compute_ocr_hints(detections, extraction_root, ocr_langs)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        logger.info("ocr_text signal disabled — skipping OCR hint computation.")

    # ── Step 4: optional VLM inference ───────────────────────────────────
    if run_vlm:
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
            pred_path = det_dir / pred_filename

            if pred_path.exists():
                continue  # already inferred

            hints = load_hints(det_dir, det)
            faiss_hits: list[dict] = hints.get("faiss_top_k", [])
            ocr_data = hints.get("ocr")

            try:
                result = run_vlm_on_detection(
                    vlm_model=vlm_model,
                    processor=processor,
                    detection=det,
                    extraction_root=extraction_root,
                    audio_brands=audio_brands,
                    faiss_hits=faiss_hits,
                    ocr_data=ocr_data,
                    temporal_offsets=temporal_offsets,
                    signals=enabled_signals,
                    brand_mode=brand_mode,
                    channel_watermark=channel_watermark,
                )
            except Exception as exc:
                logger.warning("VLM failed on det_id=%d: %s", det["det_id"], exc)
                result = {"raw": None, "parsed": None, "error": str(exc)}

            result["config"] = experiment_config
            pred_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        logger.info("VLM inference complete. Results saved as %s per detection.", pred_filename)

        # Free the VLM before the canonicalization text model loads
        del vlm_model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        logger.info("Hints precomputed. Pass --run-vlm to also run the VLM on each detection.")

    # ── Step 5: brand canonicalization pass ──────────────────────────────
    if run_vlm or canonicalize:
        canonicalize_brands(
            detections=detections,
            extraction_root=extraction_root,
            pred_filename=pred_filename,
            exp_suffix=exp_suffix,
            experiment_config=experiment_config,
        )

    # ── Step 6: frame-level bbox annotations ─────────────────────────────
    if run_vlm or annotate:
        annotate_frames(
            detections=detections,
            extraction_root=extraction_root,
            pred_filename=pred_filename,
            exp_suffix=exp_suffix,
            experiment_config=experiment_config,
        )

    # ── Step 7: flat CSV for the review UI ───────────────────────────────
    if run_vlm or export_csv:
        export_predictions_csv(
            detections=detections,
            extraction_root=extraction_root,
            pred_filename=pred_filename,
            exp_suffix=exp_suffix,
            audio_brands=audio_brands,
            experiment_config=experiment_config,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Precompute multi-signal hints (Whisper audio + FAISS visual + OCR) "
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
        "--signals", nargs="+", default=CANONICAL_SIGNALS, choices=CANONICAL_SIGNALS,
        metavar="SIGNAL",
        help=(
            "Signals to include in the VLM prompt (ablations: omit the ones to "
            f"disable). Options: {' '.join(CANONICAL_SIGNALS)}."
        ),
    )
    parser.add_argument(
        "--brand-mode", choices=["open", "closed"], default="open",
        help=(
            "'open' lets the VLM name any brand; 'closed' restricts it to the "
            "audio + FAISS candidate brands or UNKNOWN."
        ),
    )
    parser.add_argument(
        "--channel-watermark", default="",
        metavar="CHANNEL",
        help=(
            "Pre-declare the channel watermark (e.g. 'Rai Sport'): the prompt then "
            "attributes remaining logos to advertisers, not the channel. Empty disables."
        ),
    )
    parser.add_argument(
        "--experiment-name", default="",
        metavar="NAME",
        help=(
            "Suffix for VLM output files (vlm_prediction__NAME.json, "
            "vlm_predictions__NAME.csv, annotated_frames__NAME/) so ablation runs "
            "don't overwrite each other. Empty keeps the plain filenames."
        ),
    )
    parser.add_argument(
        "--ocr-langs", nargs="+", default=["en", "it"],
        metavar="LANG",
        help="EasyOCR language codes for the crop OCR signal.",
    )
    parser.add_argument(
        "--run-vlm", action="store_true",
        help=(
            "After computing hints, run the VLM on each detection "
            "(also runs brand canonicalization and writes frame annotations)."
        ),
    )
    parser.add_argument(
        "--canonicalize", action="store_true",
        help=(
            "Run the brand canonicalization LLM pass on cached VLM predictions "
            "(adds brand_canonical; runs automatically with --run-vlm)."
        ),
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Write annotated frames + frame_annotations.json from cached VLM predictions (no VLM run needed).",
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help=(
            "Write vlm_predictions[__<exp>].csv for the review UI from cached "
            "predictions (runs automatically with --run-vlm)."
        ),
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
        canonicalize=args.canonicalize,
        annotate=args.annotate,
        export_csv=args.export_csv,
        vlm_model_id=args.vlm_model,
        temporal_offsets=args.temporal_offsets,
        signals=args.signals,
        brand_mode=args.brand_mode,
        channel_watermark=args.channel_watermark,
        experiment_name=args.experiment_name,
        ocr_langs=args.ocr_langs,
    )
