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
# Output written back to /kaggle/working/:
#   extraction_output/<video_stem>/audio_hints.json
#   extraction_output/<video_stem>/detections/frame_*/det_*/hints.json
#   extraction_output/<video_stem>/detections/frame_*/det_*/vlm_prediction.json
# =============================================================================


# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# %%
# !pip install -q transformers accelerate bitsandbytes faiss-cpu qwen-vl-utils moviepy
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
                                                # swap to Qwen/Qwen3-VL-7B-Instruct
                                                # once available on HuggingFace

TOP_K              = 5    # FAISS nearest neighbours
TEMPORAL_OFFSETS   = ["-2", "-1", "+1", "+2"]  # context frames included in VLM prompt
SKIP_AUDIO         = False   # set True if you already have audio_hints.json


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


# Load manifest
manifest = json.loads((OUT_DIR / "manifest.json").read_text())
detections = manifest["detections"]
print(f"Manifest: {len(detections)} detections")

# Load FAISS + CLIP
faiss_index, faiss_meta, clip_proc, clip_model, clip_device = load_faiss_db(
    FAISS_INDEX, FAISS_METADATA
)

# Compute per-detection FAISS hints
print(f"\nComputing FAISS top-{TOP_K} hints …")
for det in tqdm(detections, desc="FAISS hints"):
    det_dir = OUT_DIR / Path(det["crop_path"]).parent
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
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ── Cell 6 ── Multi-signal VLM inference (Qwen3/Qwen2.5-VL) ─────────────────
# %%
def build_vlm_content(
    crop_img: Image.Image,
    enlarged_img: Image.Image,
    frame_img: Image.Image | None,
    temporal_pairs: list[tuple[str, Image.Image]],
    det: dict,
    audio_brands: list[str],
    faiss_hits: list[dict],
) -> list[dict]:
    """Build the interleaved text+image content list for the VLM prompt."""
    content = []

    # Signal 1 — tight crop
    content += [
        {"type": "text", "text": f"## Signal 1 — Tight crop  (detector confidence: {det['score']:.3f})\n"},
        {"type": "image", "image": crop_img},
    ]

    # Signal 2 — enlarged crop
    content += [
        {"type": "text", "text": "\n## Signal 2 — Enlarged crop with surrounding context\n"},
        {"type": "image", "image": enlarged_img},
    ]

    # Signal 3 — full frame
    if frame_img is not None:
        content += [
            {"type": "text", "text": f"\n## Signal 3 — Full frame at t={det['timecode_seconds']:.1f}s\n"},
            {"type": "image", "image": frame_img},
        ]

    # Signal 4 — temporal context
    if temporal_pairs:
        content.append({"type": "text", "text": "\n## Signal 4 — Temporal context frames\n"})
        for offset, timg in temporal_pairs:
            content += [
                {"type": "text", "text": f"T{offset}: "},
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

    # Final question
    content.append({
        "type": "text",
        "text": (
            "\n---\n"
            "You are an expert logo analyst. Using ALL six signals above, answer:\n\n"
            "(i)   Is the detected region a **logo**, a **partial logo**, or **not a logo**?\n"
            "(ii)  If it is a logo or partial logo, what is the **brand name**?\n"
            "(iii) What is the **probability** (0-100) that this crop contains ANY logo at all?\n"
            "(iv)  How **confident** are you in the specific brand name (0-100)?\n\n"
            "Respond ONLY with a JSON object — no extra text, no markdown fences:\n"
            '{\n'
            '  "is_logo": "logo" | "partial_logo" | "not_logo",\n'
            '  "brand": "<brand name, or null if not_logo>",\n'
            '  "logo_probability": <integer 0-100>,\n'
            '  "confidence": <integer 0-100>,\n'
            '  "signals_used": ["<signal1>", "<signal2>", ...],\n'
            '  "reasoning": "<1-3 sentences explaining which signals drove the decision>"\n'
            '}\n\n'
            "logo_probability: your estimate that this crop IS a logo (any brand), "
            "independent of brand identification. 0 = definitely not a logo, 100 = definitely a logo.\n"
            "confidence: your certainty about the specific brand name. "
            "Only meaningful when is_logo is logo or partial_logo; set to 0 for not_logo.\n"
            "For signals_used: list the signal names that most influenced your answer "
            '(e.g. "crop_visual", "enlarged_crop", "full_frame", "temporal_context", '
            '"audio_brands", "faiss_matches").'
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
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    parsed = None
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
    except json.JSONDecodeError:
        pass

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


# Run inference on every detection
print(f"\nRunning VLM on {len(detections)} detections …")
for det in tqdm(detections, desc="VLM inference"):
    det_dir = OUT_DIR / Path(det["crop_path"]).parent
    pred_path = det_dir / "vlm_prediction.json"
    if pred_path.exists():
        continue

    # ── Load images ──────────────────────────────────────────────────────
    crop_img     = Image.open(OUT_DIR / det["crop_path"]).convert("RGB")
    enlarged_img = Image.open(OUT_DIR / det["crop_enlarged_path"]).convert("RGB")

    frame_path = OUT_DIR / "detections" / f"frame_{det['frame_idx']:06d}" / "frame.jpg"
    frame_img  = Image.open(frame_path).convert("RGB") if frame_path.exists() else None

    temporal_pairs = []
    for offset in TEMPORAL_OFFSETS:
        rel = det["temporal_frames"].get(offset)
        if rel and (OUT_DIR / rel).exists():
            temporal_pairs.append((offset, Image.open(OUT_DIR / rel).convert("RGB")))

    # ── Load pre-computed FAISS hints ─────────────────────────────────────
    hints_path = det_dir / "hints.json"
    faiss_hits = json.loads(hints_path.read_text()).get("faiss_top_k", []) if hints_path.exists() else []

    # ── Build prompt and run ──────────────────────────────────────────────
    content  = build_vlm_content(crop_img, enlarged_img, frame_img, temporal_pairs, det, audio_brands, faiss_hits)
    messages = [{"role": "user", "content": content}]

    try:
        raw, parsed = run_vlm(vlm_model, vlm_processor, messages)
    except Exception as e:
        raw, parsed = None, None
        print(f"  ✗ det_id={det['det_id']}: {e}")

    pred_path.write_text(json.dumps({"raw": raw, "parsed": parsed}, indent=2, ensure_ascii=False))

print("\nVLM inference complete.")


# ── Cell 7 ── Results summary + CSV ─────────────────────────────────────────
# %%
import pandas as pd

results = []
for det in detections:
    det_dir   = OUT_DIR / Path(det["crop_path"]).parent
    pred_path = det_dir / "vlm_prediction.json"
    hints_path = det_dir / "hints.json"

    # Load FAISS hits for this detection
    faiss_hits: list[dict] = []
    if hints_path.exists():
        faiss_hits = json.loads(hints_path.read_text()).get("faiss_top_k", [])

    faiss_brands = [h["brand_name"] for h in faiss_hits]          # ordered list
    faiss_scores = [h["similarity"]  for h in faiss_hits]

    # Absolute paths for the review UI (relative to OUT_DIR so they work on any machine)
    crop_abs  = str(OUT_DIR / det["crop_path"])
    frame_abs = str(OUT_DIR / "detections" / f"frame_{det['frame_idx']:06d}" / "frame.jpg")

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
        # ── VLM outputs ─────────────────────────────────────────────────
        "is_logo":          p.get("is_logo"),
        "brand":            p.get("brand"),
        "logo_probability": _clamp(p.get("logo_probability")),   # P(crop is any logo) 0-100
        "confidence":       _clamp(p.get("confidence")),         # certainty of brand name 0-100
        "signals_used":     json.dumps(p.get("signals_used", [])),
        "reasoning":        p.get("reasoning"),
        # ── FAISS context ────────────────────────────────────────────────
        "faiss_top1":       faiss_brands[0] if faiss_brands else None,
        "faiss_top1_score": faiss_scores[0] if faiss_scores else None,
        "faiss_top5_brands": json.dumps(faiss_brands),
        "faiss_top5_scores": json.dumps(faiss_scores),
        # ── audio context ────────────────────────────────────────────────
        "audio_brands":     json.dumps(audio_brands),
        # ── review label (filled in by the UI) ──────────────────────────
        "label":            None,   # "TP" | "FP" | "FN" | "wrong_brand"
    })

df = pd.DataFrame(results)

print(f"\nTotal predictions: {len(df)}")
print(f"  logo:         {(df.is_logo == 'logo').sum()}")
print(f"  partial_logo: {(df.is_logo == 'partial_logo').sum()}")
print(f"  not_logo:     {(df.is_logo == 'not_logo').sum()}")
print(f"  unparsed:     {df.is_logo.isna().sum()}")
print()

logo_df = df[df.is_logo.isin(["logo", "partial_logo"])]
print(logo_df[["det_id", "timecode_s", "brand", "logo_probability", "confidence", "faiss_top1", "signals_used"]].to_string(index=False))

csv_path = OUT_DIR / "vlm_predictions.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved {len(df)} rows to {csv_path}")
