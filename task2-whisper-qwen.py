# =============================================================================
# Multi-Signal Logo Hints — Local Script (RTX 3090)
# Converted from Kaggle Notebook: task-2-whisper-qwen.ipynb
# =============================================================================
#
# Setup:
#   pip install transformers accelerate bitsandbytes faiss-cpu qwen-vl-utils
#   pip install "moviepy<2.0" "huggingface_hub[hf_xet]" pandas tqdm pillow
#
# Usage:
#   Set the paths in the CONFIG section below to match your local directory
#   structure, then run:
#       python task2_whisper_qwen.py
#
# Output written to OUT_DIR/:
#   audio_hints.json
#   brands_audio.txt
#   transcript.<lang>.txt           (lang = "it" or "en" depending on TRANSCRIPT_LANGUAGE)
#   scene_descriptions.csv
#   detections/frame_*/scene_description.txt
#   detections/frame_*/det_*/hints.json
#   detections/frame_*/det_*/vlm_prediction.json
#   vlm_predictions.csv
# =============================================================================

import gc
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

# ── Set your HuggingFace token (or set HF_TOKEN env variable before running) ─
os.environ["HF_TOKEN"] = "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"   # ← paste here, or export HF_TOKEN=... in shell

# =============================================================================
# CONFIG — edit these paths to match your local setup
# =============================================================================
EXTRACTION_DIR = Path(r"C:\Users\Admin\Downloads\Logo-Detection\extraction_output\video_1")         # your extracted video folder
FAISS_INDEX    = r"C:\Users\Admin\Downloads\Logo-Detection\FAISS\logo_index.faiss"                  # FAISS index file
FAISS_METADATA = r"C:\Users\Admin\Downloads\Logo-Detection\FAISS\metadata.json"                     # FAISS metadata JSON
VIDEO_PATH     = r"C:\Users\Admin\Downloads\Logo-Detection\video_1.mp4"                      # source video (for audio)

# Working output directory (writable)
OUT_DIR = Path("working/extraction_output") / EXTRACTION_DIR.name
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Predicted brands output directory
PREDICTED_BRANDS_DIR = Path("predicted_brands")

# Model IDs
WHISPER_MODEL_ID = "openai/whisper-large-v3"
TEXT_MODEL_ID    = "Qwen/Qwen3-8B"               # brand extraction from transcript
VLM_MODEL_ID     = "Qwen/Qwen3-VL-8B-Instruct"  # multi-signal logo inference

TOP_K            = 5                              # FAISS nearest neighbours
TEMPORAL_OFFSETS = ["-2", "-1", "+1", "+2"]       # context frames for VLM prompt
SKIP_AUDIO       = False                          # set True to skip Whisper step

# ── Whisper transcription language ──────────────────────────────────────────
# TRANSCRIPT_TASK     : "transcribe" keeps the original language, "translate" forces English.
# TRANSCRIPT_LANGUAGE : source-audio language. For Italian ads use "italian".
TRANSCRIPT_TASK     = "transcribe"
TRANSCRIPT_LANGUAGE = "italian"
_LANG_CODE          = {"italian": "it", "english": "en"}
TRANSCRIPT_LANG_CODE = _LANG_CODE.get(TRANSCRIPT_LANGUAGE, TRANSCRIPT_LANGUAGE[:2])

# ── Scene description (Signal 7) ────────────────────────────────────────────
# When True, the VLM is invoked once per unique sampled frame to produce a
# short scene description, which is then fed into the per-detection prompt as
# Signal 7. Descriptions are cached on disk and aggregated into
# scene_descriptions.csv next to vlm_predictions.csv.
GENERATE_SCENE_DESCRIPTIONS = True
SCENE_DESC_MAX_NEW_TOKENS   = 96
# =============================================================================

# Copy extraction data to working dir if needed (mirrors Kaggle read-only → working copy)
if not (OUT_DIR / "manifest.json").exists():
    shutil.copytree(str(EXTRACTION_DIR), str(OUT_DIR), dirs_exist_ok=True)
    print(f"Copied extraction data to {OUT_DIR}")


# =============================================================================
# Prompts
# =============================================================================
TRANSCRIPT_SYSTEM_PROMPT = (
    "You are an expert in analyzing advertisements. "
    "You will be given a transcript from an advertisement video. "
    "Give me a thorough and complete list of ALL brand names mentioned in this "
    "transcript using your knowledge and the contextual information present."
)

def TRANSCRIPT_USER_PROMPT(transcript: str, language: str = "english") -> str:
    lang_hint = (
        f"The transcript below is in {language.capitalize()}. Whisper may have "
        "split or mis-spelled brand names phonetically — normalise them to their "
        "correct canonical spelling (e.g. 'very sure' → 'Verisure', "
        "'neo borosilina' → 'Neo Borocillina', 'monifarma' → 'Moneyfarm'). "
        "Ignore obvious transcription artefacts such as 'amara.org' (a known "
        "Whisper subtitle-watermark hallucination) and generic Italian nouns "
        "that are not brands (e.g. 'merluzzo' = cod)."
    )
    return f"""
You are an expert assistant tasked with extracting brand and product names from advertisement transcripts.

{lang_hint}

Your goal is to extract a newline-separated list of all proper brand or product names mentioned below.

Include: pharmaceutical brands, digital platforms, food/fashion brands, TV channels, any named brand entity.
Exclude: personal names, countries, cities, sports teams (unless branded), general nouns or slogans.
Do not include duplicates. Some brand names may have minor spelling errors.

Transcript:
\"\"\"{transcript.strip()}\"\"\"

List the brand names below, one per line:
"""


# =============================================================================
# Cell 4 — Audio hints (Whisper + Qwen3-text)
# =============================================================================
def _save_predicted_brands(brands: list, output_path: Path) -> None:
    """Write the predicted brand list to a standalone JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "audio_transcript",
        "model": TEXT_MODEL_ID,
        "count": len(brands),
        "brands": brands,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Predicted brands JSON saved → {output_path}")


def compute_audio_hints(
    video_path: str,
    out_dir: Path,
    skip: bool = False,
    brands_output_dir: Path = PREDICTED_BRANDS_DIR,
) -> dict:
    """
    Transcribe video audio with Whisper, then extract brand names with Qwen3.
    Writes out_dir/audio_hints.json and brands_output_dir/predicted_brands.json.
    Cache is invalidated when TRANSCRIPT_LANGUAGE / TRANSCRIPT_TASK change so that
    switching from English-translation back to Italian-transcribe (or vice-versa)
    forces a re-run instead of silently reusing the stale transcript.
    """
    hints_path       = out_dir / "audio_hints.json"
    brands_json_path = brands_output_dir / "predicted_brands.json"

    if hints_path.exists():
        cached = json.loads(hints_path.read_text())
        cached_lang = cached.get("language", "")
        cached_task = cached.get("task", "")
        if cached_lang == TRANSCRIPT_LANGUAGE and cached_task == TRANSCRIPT_TASK:
            print(f"audio_hints.json already exists for ({TRANSCRIPT_TASK}, {TRANSCRIPT_LANGUAGE}) — skipping.")
            _save_predicted_brands(cached["brands"], brands_json_path)
            return cached
        print(
            f"Cached audio_hints.json is for ({cached_task or '?'}, {cached_lang or '?'}); "
            f"current config is ({TRANSCRIPT_TASK}, {TRANSCRIPT_LANGUAGE}) — regenerating."
        )

    if skip:
        print("SKIP_AUDIO=True and no usable cache — audio brands will be empty.")
        hints = {"transcript": "", "language": "unknown", "task": "skip", "brands": []}
        hints_path.write_text(json.dumps(hints, indent=2))
        _save_predicted_brands([], brands_json_path)
        return hints

    # ── Extract audio ────────────────────────────────────────────────────────
    from moviepy.editor import VideoFileClip

    audio_wav = str(out_dir / "audio.wav")
    print("Extracting audio …")
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_wav, logger=None)

    # ── Whisper transcription ────────────────────────────────────────────────
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers import pipeline as hf_pipeline

    # 3090 always has CUDA; keep float16 for memory efficiency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading Whisper on {device} …")

    w_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    w_proc = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=w_model,
        tokenizer=w_proc.tokenizer,
        feature_extractor=w_proc.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=4,           # safe for 24 GB VRAM; lower to 2 if OOM
        device=device,
        torch_dtype=dtype,
        return_timestamps=True,
    )
    print(f"Transcribing (task={TRANSCRIPT_TASK}, language={TRANSCRIPT_LANGUAGE}) …")
    result     = asr(
        audio_wav,
        generate_kwargs={"task": TRANSCRIPT_TASK, "language": TRANSCRIPT_LANGUAGE},
    )
    transcript: str = result.get("text", "")
    transcript_path = out_dir / f"transcript.{TRANSCRIPT_LANG_CODE}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    print(f"Transcript saved → {transcript_path} ({len(transcript)} chars): {transcript[:200]} …")

    # Free Whisper before loading Qwen
    del w_model, w_proc, asr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        {"role": "user",   "content": TRANSCRIPT_USER_PROMPT(transcript, TRANSCRIPT_LANGUAGE)},
    ]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hints = {
        "transcript": transcript,
        "language":   TRANSCRIPT_LANGUAGE,
        "task":       TRANSCRIPT_TASK,
        "brands":     brands,
    }
    hints_path.write_text(json.dumps(hints, indent=2, ensure_ascii=False))
    _save_predicted_brands(brands, brands_json_path)

    return hints


# =============================================================================
# Cell 5 — FAISS visual hints
# =============================================================================
def load_faiss_db(index_path: str, metadata_path: str):
    import faiss
    from transformers import CLIPModel, CLIPProcessor

    print("Loading FAISS index …")
    index = faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"  {index.ntotal} vectors, {len(set(metadata))} unique brands")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device} …")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    return index, metadata, clip_proc, clip_model, device


def embed_image(image: Image.Image, clip_proc, clip_model, device: str) -> np.ndarray:
    inputs = clip_proc(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"])

        # Unwrap HF dataclass if needed
        if not isinstance(emb, torch.Tensor):
            if hasattr(emb, "image_embeds") and emb.image_embeds is not None:
                emb = emb.image_embeds
            elif hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            else:
                emb = emb[0]

        # Apply visual projection only if not yet projected
        if hasattr(clip_model, "visual_projection"):
            in_dim = clip_model.visual_projection.in_features
            if emb.shape[-1] == in_dim:
                emb = clip_model.visual_projection(emb)

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


# =============================================================================
# Cell 5b — Scene descriptions (Signal 7)
# =============================================================================
SCENE_DESCRIPTION_PROMPT = (
    "Briefly describe this video frame in 1-2 sentences. Focus on: "
    "the setting (sport/event/indoor/outdoor), people and their actions, "
    "visible objects, any on-screen text or graphics, and any branding or "
    "logos you can spot (sponsor boards, jerseys, packaging, screen overlays). "
    "Be concise and factual — no speculation."
)


def describe_frame(vlm_model, processor, frame_img: Image.Image) -> str:
    """Single-image VLM call that returns a short natural-language scene description."""
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": frame_img},
            {"type": "text",  "text": SCENE_DESCRIPTION_PROMPT},
        ],
    }]
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
        gen_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=SCENE_DESC_MAX_NEW_TOKENS,
            do_sample=False,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    desc = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return desc


def generate_scene_descriptions(
    detections: list,
    out_dir: Path,
    vlm_model,
    vlm_processor,
) -> dict[int, str]:
    """
    For each unique frame_idx in detections, generate a short scene description
    using the VLM. Caches per-frame to detections/frame_NNNNNN/scene_description.txt
    and aggregates into out_dir/scene_descriptions.csv (one row per unique frame).
    Returns {frame_idx: description}.
    """
    import pandas as pd

    # Map frame_idx → frame.jpg path (deduplicate by frame_idx)
    frame_jobs: dict[int, dict] = {}
    for det in detections:
        fi = det["frame_idx"]
        if fi in frame_jobs:
            continue
        frame_path = out_dir / "detections" / f"frame_{fi:06d}" / "frame.jpg"
        frame_jobs[fi] = {
            "frame_idx":     fi,
            "timecode_s":    det["timecode_seconds"],
            "frame_path":    frame_path,
            "frame_rel":     f"detections/frame_{fi:06d}/frame.jpg",
        }

    descriptions: dict[int, str] = {}
    rows = []
    print(f"\nGenerating scene descriptions for {len(frame_jobs)} unique frames …")
    for fi in tqdm(sorted(frame_jobs), desc="Scene descriptions"):
        job        = frame_jobs[fi]
        cache_path = job["frame_path"].parent / "scene_description.txt"

        if cache_path.exists():
            desc = cache_path.read_text(encoding="utf-8").strip()
        else:
            if not job["frame_path"].exists():
                desc = ""
            else:
                try:
                    img  = Image.open(job["frame_path"]).convert("RGB")
                    desc = describe_frame(vlm_model, vlm_processor, img)
                except Exception as e:
                    print(f"  ✗ frame_idx={fi}: {e}")
                    desc = ""
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(desc, encoding="utf-8")

        descriptions[fi] = desc
        rows.append({
            "frame_idx":   fi,
            "timecode_s":  job["timecode_s"],
            "frame_path":  job["frame_rel"],   # CSV-relative
            "description": desc,
        })

    csv_path = out_dir / "scene_descriptions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {len(rows)} scene descriptions → {csv_path}")
    return descriptions


# =============================================================================
# Cell 6 — Qwen3 VLM inference
# =============================================================================
def build_vlm_content(
    crop_img: Image.Image,
    enlarged_img: Image.Image,
    frame_img: Image.Image | None,
    temporal_pairs: list[tuple[str, Image.Image]],
    det: dict,
    audio_brands: list[str],
    faiss_hits: list[dict],
    frame_description: str | None = None,
    temporal_descriptions: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Build the interleaved text+image content list for the VLM prompt."""
    content = []

    # Signal 1 — tight crop
    content += [
        {"type": "text",  "text": f"## Signal 1 — Tight crop  (detector confidence: {det['score']:.3f})\n"},
        {"type": "image", "image": crop_img},
    ]

    # Signal 2 — enlarged crop
    content += [
        {"type": "text",  "text": "\n## Signal 2 — Enlarged crop with surrounding context\n"},
        {"type": "image", "image": enlarged_img},
    ]

    # Signal 3 — full frame
    if frame_img is not None:
        content += [
            {"type": "text",  "text": f"\n## Signal 3 — Full frame at t={det['timecode_seconds']:.1f}s\n"},
            {"type": "image", "image": frame_img},
        ]

    # Signal 4 — temporal context
    if temporal_pairs:
        content.append({"type": "text", "text": "\n## Signal 4 — Temporal context frames\n"})
        for offset, timg in temporal_pairs:
            content += [
                {"type": "text",  "text": f"T{offset}: "},
                {"type": "image", "image": timg},
            ]

    # Signal 5 — audio brands (text only)
    brands_str = ", ".join(f'"{b}"' for b in audio_brands) if audio_brands else "(none detected)"
    content.append({
        "type": "text",
        "text": f"\n## Signal 5 — Brands mentioned in the audio track\n{brands_str}\n",
    })

    # Signal 6 — FAISS hits (text only)
    if faiss_hits:
        faiss_lines = "\n".join(
            f"  {i+1}. {h['brand_name']}  (similarity: {h['similarity']:.3f})"
            for i, h in enumerate(faiss_hits)
        )
    else:
        faiss_lines = "  (no matches)"
    content.append({
        "type": "text",
        "text": f"\n## Signal 6 — Top visual matches from brand database (FAISS)\n{faiss_lines}\n",
    })

    # Signal 7 — scene descriptions (text only)
    desc_lines = []
    if frame_description:
        desc_lines.append(
            f"- Main frame (t={det['timecode_seconds']:.1f}s, the frame the cropped region "
            f"was taken from):\n    {frame_description}"
        )
    if temporal_descriptions:
        for offset, desc in temporal_descriptions:
            desc_lines.append(f"- T{offset} (temporal context frame):\n    {desc}")
    if not desc_lines:
        desc_lines.append("- (no scene descriptions available)")

    content.append({
        "type": "text",
        "text": (
            "\n## Signal 7 — Scene descriptions of the surrounding video frames\n"
            "Below are natural-language descriptions of the video frame the cropped region "
            "was taken from, and (when available) of the temporal-context frames. Use them "
            "as additional context about the setting, on-screen objects, and any text or "
            "branding cues that may help identify the logo. Treat them as hints — they may "
            "be incomplete or noisy, so weigh them against the visual signals.\n"
            + "\n".join(desc_lines) + "\n"
        ),
    })

    # Final question
    content.append({
        "type": "text",
        "text": (
            "\n---\n"
            "You are an expert logo analyst. Using ALL seven signals above, answer:\n\n"
            "(i)   Is the detected region a **logo**, a **partial logo**, or **not a logo**?\n"
            "(ii)  If partial, *which kind* of partial — see partial_type below.\n"
            "(iii) If it is a logo or partial logo, what is the **brand name**?\n"
            "(iv)  What is the **probability** (0-100) that this crop contains ANY logo at all?\n"
            "(v)   How **confident** are you in the specific brand name (0-100)?\n\n"
            "Respond ONLY with a JSON object — no extra text, no markdown fences:\n"
            '{\n'
            '  "is_logo": "logo" | "partial_logo" | "not_logo",\n'
            '  "partial_type": "cropped_by_box" | "truncated_by_frame" | null,\n'
            '  "brand": "<brand name, or null if not_logo>",\n'
            '  "logo_probability": <integer 0-100>,\n'
            '  "confidence": <integer 0-100>,\n'
            '  "signals_used": ["<signal1>", "<signal2>", ...],\n'
            '  "reasoning": "<1-3 sentences explaining which signals drove the decision>"\n'
            '}\n\n'
            "partial_type — only set when is_logo == \"partial_logo\"; otherwise null:\n"
            "  * \"cropped_by_box\"       — the detector box covers only PART of a logo "
            "that is itself FULLY VISIBLE in the surrounding frame (use Signals 2 and 3 to check).\n"
            "  * \"truncated_by_frame\"   — the logo itself runs OFF the edge of the video "
            "frame, so even the full frame does not contain the whole logo.\n"
            "logo_probability: your estimate that this crop IS a logo (any brand), "
            "independent of brand identification. 0 = definitely not a logo, 100 = definitely a logo.\n"
            "confidence: your certainty about the specific brand name. "
            "Only meaningful when is_logo is logo or partial_logo; set to 0 for not_logo.\n"
            "For signals_used: list the signal names that most influenced your answer. "
            "Use ONLY these exact strings, do NOT invent variants: "
            '"crop_visual", "enlarged_crop", "full_frame", "temporal_context", '
            '"audio_brands", "faiss_matches", "scene_description".'
        ),
    })
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
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    parsed = None
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
    except json.JSONDecodeError:
        pass

    return raw, parsed


# =============================================================================
# Cell 7 — Aggregate results into a DataFrame / CSV
# =============================================================================
def _posix_rel(rel: str) -> str:
    """Normalise an OUT_DIR-relative path to forward-slashes for cross-platform safety."""
    return str(Path(rel).as_posix())


def _box_touches_edge(box, w, h, tol: int = 2) -> bool:
    """True iff the detector box reaches any frame edge within `tol` px."""
    if not box or w is None or h is None:
        return False
    x1, y1, x2, y2 = box
    return (x1 <= tol) or (y1 <= tol) or (x2 >= w - tol) or (y2 >= h - tol)


def build_results_df(detections: list, out_dir: Path, audio_brands: list) -> "pd.DataFrame":
    import pandas as pd

    results = []
    for det in detections:
        det_dir    = out_dir / Path(det["crop_path"]).parent
        pred_path  = det_dir / "vlm_prediction.json"
        hints_path = det_dir / "hints.json"

        faiss_hits: list[dict] = []
        frame_description: str | None = None
        if hints_path.exists():
            _hints            = json.loads(hints_path.read_text())
            faiss_hits        = _hints.get("faiss_top_k", [])
            frame_description = _hints.get("frame_description")

        faiss_brands = [h["brand_name"] for h in faiss_hits]
        faiss_scores = [h["similarity"]  for h in faiss_hits]

        # Q10: paths stored relative to the CSV's directory (== out_dir), using
        # forward slashes so the same CSV resolves on Windows/macOS/Linux.
        crop_rel     = _posix_rel(det["crop_path"])
        enlarged_rel = _posix_rel(det["crop_enlarged_path"])
        frame_rel    = _posix_rel(f"detections/frame_{det['frame_idx']:06d}/frame.jpg")

        p: dict = {}
        if pred_path.exists():
            p = json.loads(pred_path.read_text()).get("parsed") or {}

        def _clamp(val) -> int | None:
            try:
                return max(0, min(100, int(float(val))))
            except (TypeError, ValueError):
                return None

        # Q9: deterministic edge-touching flag — distinguishes "partial because
        # the logo is cut off by the frame edge" (true here) from "partial
        # because the detector box is too tight on a fully-visible logo" (false).
        box_edge = _box_touches_edge(
            det.get("box_xyxy"),
            det.get("frame_width"),
            det.get("frame_height"),
        )

        results.append({
            "det_id":               det["det_id"],
            "frame_idx":            det["frame_idx"],
            "timecode_s":           det["timecode_seconds"],
            "det_score":            round(det["score"], 4),
            "box_xyxy":             det["box_xyxy"],
            "box_touches_edge":     box_edge,
            "crop_path":            crop_rel,
            "enlarged_crop_path":   enlarged_rel,
            "frame_path":           frame_rel,
            "is_logo":              p.get("is_logo"),
            "partial_type":         p.get("partial_type"),
            "brand":                p.get("brand"),
            "logo_probability":     _clamp(p.get("logo_probability")),
            "confidence":           _clamp(p.get("confidence")),
            "signals_used":         json.dumps(p.get("signals_used", [])),
            "reasoning":            p.get("reasoning"),
            "faiss_top1":           faiss_brands[0] if faiss_brands else None,
            "faiss_top1_score":     faiss_scores[0] if faiss_scores else None,
            "faiss_top5_brands":    json.dumps(faiss_brands),
            "faiss_top5_scores":    json.dumps(faiss_scores),
            "audio_brands":         json.dumps(audio_brands),
            "frame_description":    frame_description,
            "label":                None,   # "TP" | "FP" | "FN" | "wrong_brand"
        })

    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================
def main():
    # ── Step 1: Audio hints ──────────────────────────────────────────────────
    audio_hints  = compute_audio_hints(VIDEO_PATH, OUT_DIR, skip=SKIP_AUDIO)
    audio_brands = audio_hints["brands"]
    print(f"\nAudio brands ({len(audio_brands)}): {audio_brands}")

    # ── Step 2: Load manifest ────────────────────────────────────────────────
    manifest   = json.loads((OUT_DIR / "manifest.json").read_text())
    detections = manifest["detections"]
    print(f"Manifest: {len(detections)} detections")

    # ── Step 3: FAISS visual hints ───────────────────────────────────────────
    faiss_index, faiss_meta, clip_proc, clip_model, clip_device = load_faiss_db(
        FAISS_INDEX, FAISS_METADATA
    )

    print(f"\nComputing FAISS top-{TOP_K} hints …")
    for det in tqdm(detections, desc="FAISS hints"):
        det_dir    = OUT_DIR / Path(det["crop_path"]).parent
        hints_path = det_dir / "hints.json"
        if hints_path.exists():
            continue

        crop = Image.open(OUT_DIR / det["crop_path"]).convert("RGB")
        hits = faiss_top_k(crop, faiss_index, faiss_meta, clip_proc, clip_model, clip_device, top_k=TOP_K)

        hints_out = {
            "det_id":       det["det_id"],
            "audio_brands": audio_brands,
            "faiss_top_k":  hits,
        }
        det_dir.mkdir(parents=True, exist_ok=True)
        hints_path.write_text(json.dumps(hints_out, indent=2, ensure_ascii=False))

    print("FAISS hints written.")

    # Free CLIP from GPU before loading VLM
    del clip_model, clip_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU free after CLIP: {torch.cuda.mem_get_info()[0]/1e9:.2f} GiB")

    # ── Step 4: VLM inference ────────────────────────────────────────────────
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    print(f"\nLoading VLM: {VLM_MODEL_ID} …")
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
        max_pixels=128 * 28 * 28,
    )
    print("VLM ready.")

    # ── Step 4b: Scene descriptions (Signal 7) ───────────────────────────────
    if GENERATE_SCENE_DESCRIPTIONS:
        scene_descriptions = generate_scene_descriptions(
            detections, OUT_DIR, vlm_model, vlm_processor,
        )
    else:
        scene_descriptions = {}

    print(f"\nRunning VLM on {len(detections)} detections …")
    for det in tqdm(detections, desc="VLM inference"):
        det_dir   = OUT_DIR / Path(det["crop_path"]).parent
        pred_path = det_dir / "vlm_prediction.json"
        if pred_path.exists():
            continue

        crop_img     = Image.open(OUT_DIR / det["crop_path"]).convert("RGB")
        enlarged_img = Image.open(OUT_DIR / det["crop_enlarged_path"]).convert("RGB")

        frame_path = OUT_DIR / "detections" / f"frame_{det['frame_idx']:06d}" / "frame.jpg"
        frame_img  = Image.open(frame_path).convert("RGB") if frame_path.exists() else None

        temporal_pairs = []
        for offset in TEMPORAL_OFFSETS:
            rel = det["temporal_frames"].get(offset)
            if rel and (OUT_DIR / rel).exists():
                temporal_pairs.append((offset, Image.open(OUT_DIR / rel).convert("RGB")))

        hints_path = det_dir / "hints.json"
        faiss_hits = json.loads(hints_path.read_text()).get("faiss_top_k", []) if hints_path.exists() else []

        frame_description = scene_descriptions.get(det["frame_idx"]) or None

        # Persist the description into hints.json so the per-detection cache is self-contained
        if frame_description and hints_path.exists():
            try:
                _cached = json.loads(hints_path.read_text())
                if _cached.get("frame_description") != frame_description:
                    _cached["frame_description"] = frame_description
                    hints_path.write_text(json.dumps(_cached, indent=2, ensure_ascii=False))
            except Exception:
                pass

        content  = build_vlm_content(
            crop_img, enlarged_img, frame_img, temporal_pairs,
            det, audio_brands, faiss_hits,
            frame_description=frame_description,
        )
        messages = [{"role": "user", "content": content}]

        try:
            raw, parsed = run_vlm(vlm_model, vlm_processor, messages)
        except Exception as e:
            raw, parsed = None, None
            print(f"  ✗ det_id={det['det_id']}: {e}")

        pred_path.write_text(json.dumps({"raw": raw, "parsed": parsed}, indent=2, ensure_ascii=False))

    print("\nVLM inference complete.")

    # Free VLM
    del vlm_model, vlm_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU free after VLM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GiB")

    # ── Step 5: Aggregate results ────────────────────────────────────────────
    df = build_results_df(detections, OUT_DIR, audio_brands)

    print(f"\nTotal predictions: {len(df)}")
    print(f"  logo:         {(df.is_logo == 'logo').sum()}")
    print(f"  partial_logo: {(df.is_logo == 'partial_logo').sum()}")
    print(f"  not_logo:     {(df.is_logo == 'not_logo').sum()}")
    print(f"  unparsed:     {df.is_logo.isna().sum()}")
    print()

    logo_df = df[df.is_logo.isin(["logo", "partial_logo"])]
    print(logo_df[[
        "det_id", "timecode_s", "brand",
        "logo_probability", "confidence", "faiss_top1", "signals_used",
    ]].to_string(index=False))

    csv_path = OUT_DIR / "vlm_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()