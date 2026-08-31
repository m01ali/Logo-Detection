# =============================================================================
# Multi-Signal Logo Hints — Kaggle Notebook
# =============================================================================
# How to use:
#   1. Create a Kaggle dataset from your local extraction_output folder
#      (the whole folder produced by data_extraction_pipeline.py).
#   2. Create a second Kaggle dataset from your FAISS/ folder
#      (logo_index.faiss + metadata.json).
#   3. Optionally upload the raw video as a third dataset (only needed for
#      audio hints; skip if you already have a transcript).
#   4. Enable GPU accelerator (T4 x2 is fine for 7B models).
#   5. Run the cells top-to-bottom.
#
# Ablations / experiments:
#   - Comment out any line in the SIGNALS dict (Cell 2) to remove that signal
#     from the VLM prompt, set a new EXPERIMENT_NAME, and re-run Cell 6
#     onwards. Hints (audio / FAISS / OCR) are cached on disk, so only the
#     VLM inference re-runs.
#   - Switch BRAND_ASSIGNMENT between "open" and "closed" to compare open-set
#     vs closed-set brand naming.
#
# Output written back to /kaggle/working/:
#   extraction_output/<video_stem>/audio_hints.json
#   extraction_output/<video_stem>/detections/frame_*/det_*/hints.json
#   extraction_output/<video_stem>/detections/frame_*/det_*/vlm_prediction[__<exp>].json
#   extraction_output/<video_stem>/brand_canonicalization[__<exp>].json
#   extraction_output/<video_stem>/vlm_predictions[__<exp>].csv
#   extraction_output/<video_stem>/annotated_frames[__<exp>]/frame_*.jpg
#   extraction_output/<video_stem>/frame_annotations[__<exp>].json
# =============================================================================


# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# %%
# !pip install -q transformers accelerate bitsandbytes faiss-cpu qwen-vl-utils moviepy easyocr
# pip install huggingface_hub[hf_xet] for faster downloads (optional)


# ── Cell 2 ── Imports & config ───────────────────────────────────────────────
# %%
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# ── Paths ── adjust dataset names to match what you uploaded on Kaggle ───────
EXTRACTION_DIR   = Path("/kaggle/input/<your-extraction-dataset>/extraction_output/<video-stem>")
FAISS_INDEX      = "/kaggle/input/<your-faiss-dataset>/FAISS/logo_index.faiss"
FAISS_METADATA   = "/kaggle/input/<your-faiss-dataset>/FAISS/metadata.json"
VIDEO_PATH       = "/kaggle/input/<your-video-dataset>/video_3-trimmed.mp4"  # only for audio

# Working output directory (writable on Kaggle)
OUT_DIR = Path("/kaggle/working/extraction_output") / EXTRACTION_DIR.name
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Copy the read-only input data to working dir so we can write hints.json files
if not (OUT_DIR / "manifest.json").exists():
    shutil.copytree(str(EXTRACTION_DIR), str(OUT_DIR), dirs_exist_ok=True)
    print(f"Copied extraction data to {OUT_DIR}")

# ── Model IDs ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_ID   = "openai/whisper-large-v3"
TEXT_MODEL_ID      = "Qwen/Qwen3-8B"           # brand extraction from transcript
VLM_MODEL_ID       = "Qwen/Qwen3-VL-8B-Instruct"   # multi-signal logo inference

TOP_K              = 5    # FAISS nearest neighbours
TEMPORAL_OFFSETS   = ["-2", "-1", "+1", "+2"]  # context frames included in VLM prompt
SKIP_AUDIO         = False   # set True if you already have audio_hints.json
OCR_LANGUAGES      = ["en", "it"]   # EasyOCR languages (Italian TV content)

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG — the knobs to touch for ablations
# ═════════════════════════════════════════════════════════════════════════════
# SIGNALS: comment out (or set to False) any line to remove that signal from
# the VLM prompt. A missing key counts as disabled. Hints stay cached on disk,
# so flipping these and re-running Cell 6 onwards is cheap.
SIGNALS = {
    "tight_crop":       True,   # the detected region itself — keep this on
    "enlarged_crop":    True,   # region + surrounding context
    "full_frame":       True,   # entire video frame
    "temporal_context": True,   # nearby frames (TEMPORAL_OFFSETS)
    "audio_brands":     True,   # brands extracted from the audio transcript
    "faiss_matches":    True,   # top-k visual matches from the brand database
    "ocr_text":         True,   # OCR of the cropped region (EasyOCR)
}

# Suffix for all VLM output files so ablation runs don't overwrite each other.
# "" keeps the plain names (vlm_prediction.json / vlm_predictions.csv).
EXPERIMENT_NAME = ""

# "open"   → VLM may name any brand it recognises (hints are soft evidence).
# "closed" → VLM must pick ONLY from the candidate list built from the audio
#            brands + FAISS top-k of the enabled signals, or output "UNKNOWN".
BRAND_ASSIGNMENT = "open"

# Channel watermark pre-declaration. When set, the prompt tells the VLM that
# this channel's watermark appears in the video and remaining logos belong to
# advertisers/sponsors. Set to "" to disable the line.
CHANNEL_WATERMARK = "Rai Sport"

# ── Canonical signal vocabulary ───────────────────────────────────────────────
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


def signal_enabled(name: str) -> bool:
    """Commented-out / missing keys in SIGNALS count as disabled."""
    return bool(SIGNALS.get(name, False))


EXP_SUFFIX     = f"__{EXPERIMENT_NAME}" if EXPERIMENT_NAME else ""
PRED_FILENAME  = f"vlm_prediction{EXP_SUFFIX}.json"

print("Experiment config")
print(f"  name:              {EXPERIMENT_NAME or '(default)'}")
print(f"  brand assignment:  {BRAND_ASSIGNMENT}")
print(f"  channel watermark: {CHANNEL_WATERMARK or '(none)'}")
print(f"  enabled signals:   {[s for s in CANONICAL_SIGNALS if signal_enabled(s)]}")


# ── Cell 3 ── Prompts ────────────────────────────────────────────────────────
# %%
TRANSCRIPT_SYSTEM_PROMPT = (
    "You are an expert in analyzing advertisements. "
    "You will be given a transcript from an advertisement video. "
    "Give me a thorough and complete list of ALL brand names mentioned in this "
    "transcript using your knowledge and the contextual information present."
)

TRANSCRIPT_USER_PROMPT = lambda transcript: f"""
You are an expert assistant tasked with extracting brand and product names from advertisement transcripts.

Your goal is to extract a newline-separated list of all proper brand or product names mentioned below.

Include: pharmaceutical brands, digital platforms, food/fashion brands, TV channels, any named brand entity.
Exclude: personal names, countries, cities, sports teams (unless branded), general nouns or slogans.
Do not include duplicates. Some brand names may have minor spelling errors.

Transcript:
\"\"\"{transcript.strip()}\"\"\"

List the brand names below, one per line:
"""


# ── Cell 4 ── Audio hints (Whisper + Qwen3-text) ─────────────────────────────
# %%
def compute_audio_hints(video_path: str, out_dir: Path, skip: bool = False) -> dict:
    """
    Transcribe video audio with Whisper, then extract brand names with Qwen3.
    Writes out_dir/audio_hints.json.  Safe to re-run (cached).
    """
    hints_path = out_dir / "audio_hints.json"
    if hints_path.exists():
        print("audio_hints.json already exists — skipping.")
        return json.loads(hints_path.read_text())

    if skip:
        print("SKIP_AUDIO=True and no cached file — audio brands will be empty.")
        hints = {"transcript": "", "language": "unknown", "brands": []}
        hints_path.write_text(json.dumps(hints, indent=2))
        return hints

    # ── Extract audio ────────────────────────────────────────────────────────
    from moviepy import VideoFileClip
    audio_wav = str(out_dir / "audio.wav")
    print("Extracting audio …")
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_wav, logger=None)

    # ── Whisper transcription ────────────────────────────────────────────────
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading Whisper on {device} …")

    w_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    w_proc = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=w_model,
        tokenizer=w_proc.tokenizer,
        feature_extractor=w_proc.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=4,
        device=device,
        torch_dtype=dtype,
        return_timestamps=True,
    )
    print("Transcribing …")
    result = asr(audio_wav, generate_kwargs={"task": "translate", "language": "english"})
    transcript: str = result.get("text", "")
    (out_dir / "transcript.en.txt").write_text(transcript, encoding="utf-8")
    print(f"Transcript ({len(transcript)} chars): {transcript[:200]} …")

    # Free Whisper from GPU memory before loading the text LLM
    del w_model, w_proc, asr
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Qwen3 brand extraction ───────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {TEXT_MODEL_ID} for brand extraction …")
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    t_model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quant_cfg,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, use_fast=True)

    messages = [
        {"role": "system", "content": TRANSCRIPT_SYSTEM_PROMPT},
        {"role": "user",   "content": TRANSCRIPT_USER_PROMPT(transcript)},
    ]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        # Qwen3 supports a thinking toggle — disable for this extraction task
        enable_thinking=False,
    )
    inputs = tokenizer([text_input], return_tensors="pt").to(t_model.device)
    with torch.no_grad():
        output_ids = t_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    output_ids_trimmed = output_ids[0][len(inputs.input_ids[0]):]
    brand_string = tokenizer.decode(output_ids_trimmed, skip_special_tokens=True)

    brands = sorted({b.lower().strip() for b in brand_string.split("\n") if b.strip()})
    (out_dir / "brands_audio.txt").write_text("\n".join(brands), encoding="utf-8")
    print(f"Extracted {len(brands)} brands: {brands}")

    del t_model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    hints = {"transcript": transcript, "language": "english", "brands": brands}
    hints_path.write_text(json.dumps(hints, indent=2, ensure_ascii=False))
    return hints


audio_hints = compute_audio_hints(VIDEO_PATH, OUT_DIR, skip=SKIP_AUDIO)
audio_brands = audio_hints["brands"]
print(f"\nAudio brands ({len(audio_brands)}): {audio_brands}")


# ── Cell 5 ── FAISS visual hints ─────────────────────────────────────────────
# %%
def load_faiss_db(index_path: str, metadata_path: str):
    import faiss
    from transformers import CLIPProcessor, CLIPModel

    print("Loading FAISS index …")
    index = faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"  {index.ntotal} vectors, {len(set(metadata))} unique brands")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device} …")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    return index, metadata, clip_proc, clip_model, device


def embed_image(image: Image.Image, clip_proc, clip_model, device: str) -> np.ndarray:
    inputs = clip_proc(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
    arr = emb.cpu().numpy()
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr.astype("float32")


def faiss_top_k(
    image: Image.Image,
    index, metadata, clip_proc, clip_model, device,
    top_k: int = 5,
) -> list[dict]:
    emb = embed_image(image, clip_proc, clip_model, device)
    distances, indices = index.search(emb, top_k * 3)

    seen: dict[str, float] = {}
    for idx, dist in zip(indices[0], distances[0]):
        similarity = round(float(1 - dist / 4), 4)
        brand = metadata[idx] if idx < len(metadata) else "unknown"
        if brand not in seen or similarity > seen[brand]:
            seen[brand] = similarity

    return sorted(
        [{"brand_name": b, "similarity": s} for b, s in seen.items()],
        key=lambda x: x["similarity"],
        reverse=True,
    )[:top_k]


def load_hints(det_dir: Path, det: dict) -> dict:
    """Load an existing hints.json or start a fresh one (merge-friendly)."""
    hints_path = det_dir / "hints.json"
    if hints_path.exists():
        return json.loads(hints_path.read_text())
    return {"det_id": det["det_id"]}


def save_hints(det_dir: Path, hints: dict) -> None:
    det_dir.mkdir(parents=True, exist_ok=True)
    (det_dir / "hints.json").write_text(json.dumps(hints, indent=2, ensure_ascii=False))


# Load manifest
manifest = json.loads((OUT_DIR / "manifest.json").read_text())
detections = manifest["detections"]
print(f"Manifest: {len(detections)} detections")

if signal_enabled("faiss_matches"):
    # Load FAISS + CLIP
    faiss_index, faiss_meta, clip_proc, clip_model, clip_device = load_faiss_db(
        FAISS_INDEX, FAISS_METADATA
    )

    # Compute per-detection FAISS hints (cached in hints.json)
    print(f"\nComputing FAISS top-{TOP_K} hints …")
    for det in tqdm(detections, desc="FAISS hints"):
        det_dir = OUT_DIR / Path(det["crop_path"]).parent
        hints = load_hints(det_dir, det)
        if "faiss_top_k" in hints:
            continue

        crop = Image.open(OUT_DIR / det["crop_path"]).convert("RGB")
        hints["audio_brands"] = audio_brands
        hints["faiss_top_k"] = faiss_top_k(
            crop, faiss_index, faiss_meta, clip_proc, clip_model, clip_device, top_k=TOP_K
        )
        save_hints(det_dir, hints)

    print("FAISS hints written.")

    # Free CLIP from GPU before loading the OCR / VLM models
    del clip_model, clip_proc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
else:
    print("faiss_matches signal disabled — skipping FAISS hint computation.")


# ── Cell 5b ── OCR hints (EasyOCR on the cropped region) ─────────────────────
# %%
def ocr_image(reader, image: Image.Image, min_side: int = 80, upscale: int = 3) -> list[dict]:
    """Run EasyOCR on one image; upscale tiny crops first so text is legible."""
    img = image.convert("RGB")
    if min(img.size) < min_side:
        img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    results = reader.readtext(np.array(img))
    return [
        {"text": text.strip(), "confidence": round(float(conf), 3)}
        for _, text, conf in results
        if text.strip()
    ]


if signal_enabled("ocr_text"):
    import easyocr

    print(f"Loading EasyOCR ({OCR_LANGUAGES}) …")
    ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=torch.cuda.is_available())

    print("Computing OCR hints …")
    for det in tqdm(detections, desc="OCR hints"):
        det_dir = OUT_DIR / Path(det["crop_path"]).parent
        hints = load_hints(det_dir, det)
        if "ocr" in hints:
            continue

        # OCR the tight crop first; fall back to the enlarged crop if no text
        lines = ocr_image(ocr_reader, Image.open(OUT_DIR / det["crop_path"]))
        source = "tight_crop"
        if not lines:
            lines = ocr_image(ocr_reader, Image.open(OUT_DIR / det["crop_enlarged_path"]))
            source = "enlarged_crop" if lines else None

        hints["ocr"] = {"source": source, "lines": lines}
        save_hints(det_dir, hints)

    del ocr_reader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("OCR hints written.")
else:
    print("ocr_text signal disabled — skipping OCR hint computation.")


# ── Cell 6 ── Multi-signal VLM inference (Qwen3-VL) ──────────────────────────
# %%
def build_closed_set_candidates(audio_brands: list[str], faiss_hits: list[dict]) -> list[str]:
    """Candidate brands for closed-set mode: the channel watermark (if
    declared) + audio brands + FAISS top-k of the ENABLED signals (an
    ablated signal contributes no candidates).

    The watermark is always added when declared, regardless of which
    signals are enabled/ablated, since it comes from CHANNEL_WATERMARK, not
    from a hint signal — otherwise the prompt's watermark note ("assign the
    brand X if this is the watermark") contradicts the closed-set rule
    ("assign STRICTLY from this list"), and the model can never legally
    name the watermark brand.
    """
    candidates: list[str] = []
    if CHANNEL_WATERMARK:
        candidates.append(CHANNEL_WATERMARK)
    if signal_enabled("audio_brands"):
        candidates += audio_brands
    if signal_enabled("faiss_matches"):
        candidates += [h["brand_name"] for h in faiss_hits]

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        key = c.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def build_final_instruction(candidates: list[str]) -> str:
    """The question + output schema shown to the VLM after all signals."""
    watermark_note = ""
    if CHANNEL_WATERMARK:
        watermark_note = (
            f"Known context: the channel watermark in this video is \"{CHANNEL_WATERMARK}\". "
            f"If this region is that channel watermark, assign the brand \"{CHANNEL_WATERMARK}\". "
            "Attribute all remaining logos to advertisers or sponsors, not to the channel.\n\n"
        )

    if BRAND_ASSIGNMENT == "closed":
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


def collect_temporal_pairs(det: dict, root: Path, offsets: list[str]) -> list[tuple[str, Image.Image]]:
    """Load the requested temporal-context frames for one detection.

    Falls back to the nearest temporal frames that DO exist when none of the
    requested offsets were saved — older extractions stored only the widest
    offsets (e.g. ±5), which used to leave the temporal_context signal
    silently empty.
    """
    def _load(offset: str) -> Image.Image | None:
        rel = det["temporal_frames"].get(offset)
        if rel and (root / rel).exists():
            return Image.open(root / rel).convert("RGB")
        return None

    pairs = [(o, img) for o in offsets if (img := _load(o)) is not None]
    if pairs:
        return pairs

    available = sorted(
        (k for k, v in det["temporal_frames"].items() if v and (root / v).exists()),
        key=lambda k: (abs(int(k)), int(k)),
    )[: max(len(offsets), 1)]
    return [(o, img) for o in sorted(available, key=int) if (img := _load(o)) is not None]


def build_vlm_content(
    crop_img: Image.Image | None,
    enlarged_img: Image.Image | None,
    frame_img: Image.Image | None,
    temporal_pairs: list[tuple[str, Image.Image]],
    det: dict,
    audio_brands: list[str],
    faiss_hits: list[dict],
    ocr_data: dict | None,
) -> list[dict]:
    """Build the interleaved text+image content list for the VLM prompt.

    Section headers use the CANONICAL_SIGNALS names so the model can echo them
    back verbatim in `signals_used`. Disabled signals are omitted entirely.
    """
    content = []

    # Signal — tight crop
    if signal_enabled("tight_crop") and crop_img is not None:
        content += [
            {"type": "text", "text": f"## Signal: tight_crop — the detected region  (detector confidence: {det['score']:.3f})\n"},
            {"type": "image", "image": crop_img},
        ]

    # Signal — enlarged crop
    if signal_enabled("enlarged_crop") and enlarged_img is not None:
        content += [
            {"type": "text", "text": "\n## Signal: enlarged_crop — same region with surrounding context\n"},
            {"type": "image", "image": enlarged_img},
        ]

    # Signal — full frame
    if signal_enabled("full_frame") and frame_img is not None:
        content += [
            {"type": "text", "text": f"\n## Signal: full_frame — full video frame at t={det['timecode_seconds']:.1f}s\n"},
            {"type": "image", "image": frame_img},
        ]

    # Signal — temporal context
    if signal_enabled("temporal_context") and temporal_pairs:
        content.append({"type": "text", "text": "\n## Signal: temporal_context — nearby frames\n"})
        for offset, timg in temporal_pairs:
            content += [
                {"type": "text", "text": f"T{offset}: "},
                {"type": "image", "image": timg},
            ]

    # Signal — audio brands (text only)
    if signal_enabled("audio_brands"):
        brands_str = ", ".join(f'"{b}"' for b in audio_brands) if audio_brands else "(none detected)"
        content.append({
            "type": "text",
            "text": f"\n## Signal: audio_brands — brands mentioned in the audio track\n{brands_str}\n",
        })

    # Signal — FAISS hits (text only)
    if signal_enabled("faiss_matches"):
        if faiss_hits:
            faiss_lines = "\n".join(
                f"  {i+1}. {h['brand_name']}  (similarity: {h['similarity']:.3f})"
                for i, h in enumerate(faiss_hits)
            )
        else:
            faiss_lines = "  (no matches)"
        content.append({
            "type": "text",
            "text": f"\n## Signal: faiss_matches — top visual matches from the brand database\n{faiss_lines}\n",
        })

    # Signal — OCR text (text only)
    if signal_enabled("ocr_text"):
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
    candidates = build_closed_set_candidates(audio_brands, faiss_hits)
    content.append({"type": "text", "text": build_final_instruction(candidates)})
    return content


def run_vlm(vlm_model, processor, messages: list) -> tuple[str, dict | None]:
    """Run a single multi-image VLM call and return (raw_text, parsed_json_or_None)."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device, vlm_model.dtype)

    with torch.no_grad():
        gen_ids = vlm_model.generate(**inputs, max_new_tokens=256, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    parsed = None
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
    except json.JSONDecodeError:
        pass

    if parsed is not None and "signals_used" in parsed:
        parsed["signals_used_raw"] = parsed.get("signals_used")
        parsed["signals_used"] = normalize_signals_used(parsed.get("signals_used"))

    return raw, parsed


# ── Load VLM ──────────────────────────────────────────────────────────
print(f"Loading VLM: {VLM_MODEL_ID} …")
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quant_cfg,
).eval()

vlm_processor = AutoProcessor.from_pretrained(
    VLM_MODEL_ID,
    use_fast=True,
    padding_side="left",
    max_pixels=128 * 28 * 28
)
print("VLM ready.")

# Snapshot of the experiment config, stored inside every prediction file
EXPERIMENT_CONFIG = {
    "experiment":        EXPERIMENT_NAME or "default",
    "brand_assignment":  BRAND_ASSIGNMENT,
    "channel_watermark": CHANNEL_WATERMARK,
    "signals":           [s for s in CANONICAL_SIGNALS if signal_enabled(s)],
    "vlm_model":         VLM_MODEL_ID,
}

# Run inference on every detection
print(f"\nRunning VLM on {len(detections)} detections …  (experiment: {EXPERIMENT_CONFIG['experiment']})")
for det in tqdm(detections, desc="VLM inference"):
    det_dir = OUT_DIR / Path(det["crop_path"]).parent
    pred_path = det_dir / PRED_FILENAME
    if pred_path.exists():
        continue

    # ── Load images (only for enabled signals) ────────────────────────────
    crop_img = enlarged_img = frame_img = None
    if signal_enabled("tight_crop"):
        crop_img = Image.open(OUT_DIR / det["crop_path"]).convert("RGB")
    if signal_enabled("enlarged_crop"):
        enlarged_img = Image.open(OUT_DIR / det["crop_enlarged_path"]).convert("RGB")
    if signal_enabled("full_frame"):
        frame_path = OUT_DIR / "detections" / f"frame_{det['frame_idx']:06d}" / "frame.jpg"
        frame_img = Image.open(frame_path).convert("RGB") if frame_path.exists() else None

    temporal_pairs = []
    if signal_enabled("temporal_context"):
        temporal_pairs = collect_temporal_pairs(det, OUT_DIR, TEMPORAL_OFFSETS)

    # ── Load pre-computed hints (FAISS + OCR) ─────────────────────────────
    hints_path = det_dir / "hints.json"
    hints = json.loads(hints_path.read_text()) if hints_path.exists() else {}
    faiss_hits = hints.get("faiss_top_k", [])
    ocr_data = hints.get("ocr")

    # ── Build prompt and run ──────────────────────────────────────────────
    content = build_vlm_content(
        crop_img, enlarged_img, frame_img, temporal_pairs,
        det, audio_brands, faiss_hits, ocr_data,
    )
    messages = [{"role": "user", "content": content}]

    try:
        raw, parsed = run_vlm(vlm_model, vlm_processor, messages)
    except Exception as e:
        raw, parsed = None, None
        print(f"  ✗ det_id={det['det_id']}: {e}")

    pred_path.write_text(json.dumps(
        {"raw": raw, "parsed": parsed, "config": EXPERIMENT_CONFIG},
        indent=2, ensure_ascii=False,
    ))

print("\nVLM inference complete.")


# ── Cell 6b ── Brand canonicalization pass ───────────────────────────────────
# %%
# A second, text-only LLM pass over ALL brand names assigned in this video.
# Spelling variants of the same brand ("verysure", "verishore", "verisure")
# collapse to one canonical spelling chosen by the model. Adds a
# "brand_canonical" field to every vlm_prediction file (the original "brand"
# is left untouched) and saves the full mapping for review.

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


def run_vlm_text_only(vlm_model, processor, prompt: str, max_new_tokens: int = 1024) -> str:
    """Text-only call to the already-loaded VLM (no images)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(vlm_model.device)
    with torch.no_grad():
        gen_ids = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


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


def collect_assigned_brands(detections: list[dict], out_dir: Path) -> list[str]:
    """Unique brand strings the VLM assigned across all detections (frames)."""
    brands: list[str] = []
    for det in detections:
        pred_path = out_dir / Path(det["crop_path"]).parent / PRED_FILENAME
        if not pred_path.exists():
            continue
        parsed = json.loads(pred_path.read_text()).get("parsed") or {}
        b = parsed.get("brand")
        if isinstance(b, str) and b.strip() and b != "UNKNOWN" and b not in brands:
            brands.append(b)
    return sorted(brands, key=str.lower)


assigned_brands = collect_assigned_brands(detections, OUT_DIR)
print(f"Distinct brands assigned by the VLM: {len(assigned_brands)}")

if assigned_brands:
    raw_mapping = run_vlm_text_only(vlm_model, vlm_processor, CANONICALIZE_PROMPT(assigned_brands))
    brand_map = parse_brand_mapping(raw_mapping, assigned_brands)

    changed = {k: v for k, v in brand_map.items() if k != v}
    print(f"Canonicalization changed {len(changed)} of {len(brand_map)} brand spellings:")
    for k, v in changed.items():
        print(f"  {k!r} → {v!r}")

    map_path = OUT_DIR / f"brand_canonicalization{EXP_SUFFIX}.json"
    map_path.write_text(json.dumps(
        {"config": EXPERIMENT_CONFIG, "input_brands": assigned_brands,
         "mapping": brand_map, "raw_response": raw_mapping},
        indent=2, ensure_ascii=False,
    ))
    print(f"Mapping saved to {map_path}")

    # Stamp brand_canonical into every prediction file
    for det in detections:
        pred_path = OUT_DIR / Path(det["crop_path"]).parent / PRED_FILENAME
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
    print("brand_canonical written into all prediction files.")
else:
    print("No brands assigned — skipping canonicalization pass.")


# ── Cell 7 ── Results summary + CSV ─────────────────────────────────────────
# %%
import pandas as pd

results = []
for det in detections:
    det_dir   = OUT_DIR / Path(det["crop_path"]).parent
    pred_path = det_dir / PRED_FILENAME
    hints_path = det_dir / "hints.json"

    # Load hints (FAISS + OCR) for this detection
    hints: dict = json.loads(hints_path.read_text()) if hints_path.exists() else {}
    faiss_hits: list[dict] = hints.get("faiss_top_k", [])
    ocr_lines: list[dict] = (hints.get("ocr") or {}).get("lines", [])

    faiss_brands = [h["brand_name"] for h in faiss_hits]          # ordered list
    faiss_scores = [h["similarity"]  for h in faiss_hits]

    # Absolute paths for the review UI (relative to OUT_DIR so they work on any machine)
    crop_abs  = str(OUT_DIR / det["crop_path"])
    frame_abs = str(OUT_DIR / "detections" / f"frame_{det['frame_idx']:06d}" / "frame.jpg")
    annotated_abs = str(OUT_DIR / f"annotated_frames{EXP_SUFFIX}" / f"frame_{det['frame_idx']:06d}.jpg")

    p: dict = {}
    if pred_path.exists():
        p = json.loads(pred_path.read_text()).get("parsed") or {}

    def _clamp(val) -> int | None:
        try:
            return max(0, min(100, int(float(val))))
        except (TypeError, ValueError):
            return None

    results.append({
        # ── detection metadata ──────────────────────────────────────────
        "det_id":           det["det_id"],
        "frame_idx":        det["frame_idx"],
        "timecode_s":       det["timecode_seconds"],
        "det_score":        round(det["score"], 4),
        "box_xyxy":         det["box_xyxy"],
        # ── file paths for the review UI ────────────────────────────────
        "crop_path":        crop_abs,
        "enlarged_crop_path": str(OUT_DIR / det["crop_enlarged_path"]),
        "frame_path":       frame_abs,
        "annotated_frame_path": annotated_abs,
        # ── VLM outputs ─────────────────────────────────────────────────
        "is_logo":          p.get("is_logo"),
        "brand":            p.get("brand"),
        "brand_canonical":  p.get("brand_canonical", p.get("brand")),  # canonical spelling (Cell 6b)
        "is_truncated":     p.get("is_truncated"),                # logo cut off at crop edge
        "logo_probability": _clamp(p.get("logo_probability")),   # P(crop is any logo) 0-100
        "confidence":       _clamp(p.get("confidence")),         # certainty of assigned label 0-100
        "signals_used":     json.dumps(normalize_signals_used(p.get("signals_used", []))),
        "signals_used_raw": json.dumps(p.get("signals_used_raw", p.get("signals_used", []))),
        "reasoning":        p.get("reasoning"),
        # ── FAISS context ────────────────────────────────────────────────
        "faiss_top1":       faiss_brands[0] if faiss_brands else None,
        "faiss_top1_score": faiss_scores[0] if faiss_scores else None,
        "faiss_top5_brands": json.dumps(faiss_brands),
        "faiss_top5_scores": json.dumps(faiss_scores),
        # ── audio context ────────────────────────────────────────────────
        "audio_brands":     json.dumps(audio_brands),
        # ── OCR context ──────────────────────────────────────────────────
        "ocr_text":         " | ".join(l["text"] for l in ocr_lines),
        "ocr_source":       (hints.get("ocr") or {}).get("source"),
        # ── experiment metadata ──────────────────────────────────────────
        "experiment":       EXPERIMENT_NAME or "default",
        "brand_mode":       BRAND_ASSIGNMENT,
        "enabled_signals":  json.dumps([s for s in CANONICAL_SIGNALS if signal_enabled(s)]),
        # ── review label (filled in by the UI) ──────────────────────────
        "label":            None,   # "TP" | "FP" | "FN" | "wrong_brand"
    })

df = pd.DataFrame(results)

print(f"\nTotal predictions: {len(df)}   (experiment: {EXPERIMENT_NAME or 'default'}, brand mode: {BRAND_ASSIGNMENT})")
print(f"  logo:         {(df.is_logo == 'logo').sum()}")
print(f"  partial_logo: {(df.is_logo == 'partial_logo').sum()}")
print(f"  not_logo:     {(df.is_logo == 'not_logo').sum()}")
print(f"  unparsed:     {df.is_logo.isna().sum()}")
print(f"  UNKNOWN brand: {(df.brand == 'UNKNOWN').sum()}")
print(f"  truncated:    {(df.is_truncated == True).sum()}")
print()

logo_df = df[df.is_logo.isin(["logo", "partial_logo"])]
print(logo_df[["det_id", "timecode_s", "brand", "brand_canonical", "logo_probability", "confidence", "is_truncated", "faiss_top1", "signals_used"]].to_string(index=False))

csv_path = OUT_DIR / f"vlm_predictions{EXP_SUFFIX}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved {len(df)} rows to {csv_path}")


# ── Cell 8 ── Frame-level bounding-box annotations ───────────────────────────
# %%
# Draws every detection on its full frame with the VLM verdict:
#   green = logo, orange = partial_logo, red = not_logo, gray = unparsed.
# Writes annotated JPEGs + a machine-readable frame_annotations.json.
from PIL import ImageDraw, ImageFont

LABEL_COLORS = {
    "logo":         (0, 190, 0),
    "partial_logo": (255, 160, 0),
    "not_logo":     (220, 40, 40),
    None:           (128, 128, 128),
}


def _load_font(size: int = 20):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_frames(detections: list[dict], out_dir: Path) -> Path:
    ann_dir = out_dir / f"annotated_frames{EXP_SUFFIX}"
    ann_dir.mkdir(exist_ok=True)
    font = _load_font(20)

    by_frame: dict[int, list[dict]] = {}
    for det in detections:
        by_frame.setdefault(det["frame_idx"], []).append(det)

    frames_out = []
    for frame_idx, dets in tqdm(sorted(by_frame.items()), desc="Annotating frames"):
        frame_path = out_dir / "detections" / f"frame_{frame_idx:06d}" / "frame.jpg"
        if not frame_path.exists():
            continue
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        det_entries = []
        for det in dets:
            det_dir = out_dir / Path(det["crop_path"]).parent
            pred_path = det_dir / PRED_FILENAME
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
            "frame_path":       str(frame_path.relative_to(out_dir)),
            "annotated_path":   str(ann_path.relative_to(out_dir)),
            "detections":       det_entries,
        })

    ann_json = {
        "config": EXPERIMENT_CONFIG,
        "num_frames": len(frames_out),
        "num_detections": sum(len(f["detections"]) for f in frames_out),
        "frames": frames_out,
    }
    ann_json_path = out_dir / f"frame_annotations{EXP_SUFFIX}.json"
    ann_json_path.write_text(json.dumps(ann_json, indent=2, ensure_ascii=False))
    print(f"Annotated {len(frames_out)} frames → {ann_dir}")
    print(f"Frame annotation summary → {ann_json_path}")
    return ann_dir


annotate_frames(detections, OUT_DIR)
