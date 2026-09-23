# 01 — Research Findings: Codebase Audit

Audit date: 16 September 2026. Branch `m2-revision-Jul08` at `129aaf0`. Everything below was read from the files; nothing is inferred from the correspondence alone.

## 1. Repository map and era

| Path | Era | Lines | What it is | M3 role |
|------|-----|-------|------------|---------|
| `data_extraction_pipeline.py` | M2 | 576 | Grounding DINO (`IDEA-Research/grounding-dino-base`) over sampled frames; writes `extraction_output/<stem>/` with `manifest.json`, per-detection `metadata.json`, tight + 1.5× enlarged crops, T-5..T+5 temporal frames. Sampling `interval` (2 s) or `segment`. Thresholds box 0.20 / text 0.20. `min_crop_px` 16. Prompt = 37 phrases joined with " . " | **Wrap** into `stages/ingest`, `sampling`, `detect_gdino`. The 37-phrase prompt moves to YAML. |
| `generate_hints.py` | M2 | 1,336 | Local multi-signal pipeline: audio hints (Whisper large-v3 + Qwen2.5-7B text), FAISS top-k (CLIP), EasyOCR hints, Qwen2.5-VL-7B 4-bit call with seven signals, open/closed brand mode, channel-watermark line, LLM canonicalisation, frame annotation, CSV export | **Wrap** the hint builders, the VLM call, annotation and CSV export. Canonicalisation is replaced by T37 (LLM merge kept as optional off-by-default step). |
| `kaggle_multi_signal_hints.py` | M2 | 1,113 | Cell-structured copy of the above for Kaggle (Qwen3-VL-8B, Qwen3-8B text) | Reference only. The pipeline must be runnable as a script on the GPU machine; the notebook form is not carried forward. |
| `task2-whisper-qwen.py` | M2 | 868 | RTX 3090 variant with Italian transcription, brand-name normalisation prompt, per-frame scene descriptions (Signal 7), `box_touches_edge` flag, relative CSV paths | Reuse the improved transcript prompt and `box_touches_edge`. **Contains a hard-coded HF token on line 40 — revoke and remove.** Scene descriptions are not in M3 scope. |
| `train_yolov11_m2.py`, `predict_yolov11_m2.py`, `Notebooks/Yolov11-M2.ipynb` | M2 | 243 / 67 | YOLO11-L training on the Roboflow export (70/30 split, seed 42, 50 epochs, imgsz 640) and a video predict script | `predict` shows the ultralytics call shape; `stages/detect_yolo` is new. |
| `best.pt` (repo root, git-ignored) | M2 | 51.7 MB | The trained 321-class YOLO11-L weights | Required input for T12. Must be copied to the GPU machine. Record in T76. |
| `docs/yolov11-m2-validation/` | M2 | — | mAP50 0.850, mAP50-95 0.677 on the random val split (1,749 images, 267 classes present) | Acceptance-evidence caveat: trained on a random split of the same labelled set. |
| `app/review_ui.py` | M2 | 502 | Streamlit review screen over the per-run CSV. Labels TP/FP/FN/wrong_brand written back to the same CSV. Filters: label, `is_logo`, `confidence`, `logo_probability`. Path resolution: absolute, then CSV-dir-relative, then cwd | **Extend** for T73 (router_decision / reason_code filters, Rule 5 evidence panel) and T75 (Accept into FAISS). |
| `app/streamlit_faiss_app.py` | Phase 1/M2 | 1,780 | The FAISS manager. Eight tabs. All enrolment goes through `LogoDatabaseNew.add_logos()` at eight call sites (see §4) | **Instrument** via T55; unchanged otherwise. |
| `app/streamlit_faiss_app_{old,18_10_25,19_10_25}.py` | Phase 1 | — | Older copies | Ignore. |
| `models/faiss_db.py` | Phase 1 | 630 | `LogoDatabase`, `LogoDatabaseNew` (CLIP ViT-L/14, 768-d, `IndexFlatL2`, similarity = 1 − d/4), `LogoDatabaseSigLIP2`. Metadata is a flat JSON list of brand strings. `add_logos` appends; no dedup; optional albumentations augmentation | **Extend** (T55): records instead of strings for new entries, legacy read path. Query path reused for hints, T51, T54. |
| `models/audio_model.py` | Phase 1 | 316 | Whisper wrapper (translate/transcribe, chunked); `convert_video_to_audio` writes `logo_audio.wav` in cwd | Wrap; fix the cwd write to go into the run directory. |
| `models/description_model.py` | Phase 1 | 387 | Qwen2.5-VL, Qwen2.5 text (`temperature=1.2, do_sample=True`), Florence, Gemma, LLaMA wrappers | Only `Qwen2_5TextModel` is used by M2 audio extraction. **Its sampling makes audio brand extraction non-deterministic**; the pipeline sets greedy decoding (see `04`). |
| `models/detector_model.py` | Phase 1 | 273 | OWLv2 detector | Not used in M3 (Grounding DINO replaced it in M2). |
| `utils/shot_detection.py` | Phase 1 | 74 | PySceneDetect `AdaptiveDetector`, keyframe = midpoint between peaks of `adaptive_ratio (w=2)` at quantile 0.98. Returns frames, frame numbers, timecodes. **No shot index** | **Adapt** for T57: emit a scene list with start/end frames and ordinal `shot_id`; `per_shot` sampling uses the same list. |
| `utils/fuzzy_matching.py` | Phase 1 | 178 | rapidfuzz `WRatio` matchers, short-string threshold bumps (`+15` under 4 chars), `smart_score` (Jaro-Winkler for ≤6 chars, WRatio for multi-word, partial_ratio otherwise). **Instantiates an EasyOCR reader at import time** (English only) | **Adapt** the scoring functions for T27; do not import the module as-is (import-time model load). |
| `utils/prompts.py` | Phase 1 | 33 | Transcript → brand list prompts | Reuse (the `task2` variant with the Italian normalisation hint is better). |
| `utils/video_annotations_to_crops.py` | Phase 1 | 98 | Label Studio JSON → crops for the FAISS manager | Untouched (one of the eight enrolment paths). |
| `pipelines/pipeline.py` | Phase 1 | 1,097 | Gradio-era pipeline: OWLv2 → heuristic/CLIP filters → brand chain (CLIP, OCR-fuzzy, FAISS) → `annotate_video` (cv2 `mp4v`, writes only frames with detections ± window, loads the whole video into memory) | Only `annotate_video` is a pattern for T59, and it must be rewritten to stream frames and write **all sampled frames**. |
| `pipelines/brand_recognition_chain.py` | Phase 1 | 953 | Chain-of-techniques brand assignment | Not used in M3. |
| `filtering/heuristic_filtering.py` | Phase 1 | 118 | `area_aspect_filter` with `min_area=500`, `max_area_ratio=0.5` — the legacy thresholds T17 forbids in the new path | Do not wire in. |
| `filtering/clip_filtering.py`, `llm_filtering.py`, `post_filtering.py` | Phase 1 | — | CLIP/LLM false-positive filters, Qwen "Correct/Other/Incorrect" post-filter | Not used in M3. |
| `benchmarks/` | Phase 1 | — | Leave-one-out and crop retrieval benchmarks for CLIP/SigLIP2/DINOv2/hybrid; `metrics.py` has `recall_at_k`, `mean_reciprocal_rank`, `threshold_sensitivity`; k = [1, 3, 5]; production threshold 0.85. Config paths point at machines that no longer exist | Reuse `metrics.py` and the k-values for T54. |
| `build_siglip2_index.py` | Phase 1 | 150 | Batch-build a SigLIP2 index from a CSV + folder | Not needed; the CLIP index is the production index. |
| `scripts/compare_open_closed.py` | M2 | 69 | Compares two experiment CSVs (open vs closed) | Pattern for T64's agreement report. |
| `gradio_app.py`, `app/logo_app.py`, `app/app_image.py`, `test.py` | Phase 1 | — | Gradio demos, OWLv2 video test | Ignore. |
| `runs/detect/yolo results/video_3-trimmed/` | M2 | 338 label files + `.avi` | YOLO predict output over every frame of video_3-trimmed (conf 0.25) | Should be git-ignored; it is tracked. Shows YOLO label format `cls cx cy w h conf`. |
| `predicted_brands/predicted_brands.json` | M2 | — | Audio brands for video_1 from Qwen3-8B: 14 items including `rai sport`, `rai 2`, `orogel`, `very sure`, `frost`, `barbie`, `neo borosilina`, `monifarma`, `amara.org` (Whisper artefact) | Test fixture for T37 (audio-brand aliasing: `very sure` → Verisure) |

## 2. What M2 actually produces today (the row the review UI reads)

`export_predictions_csv` in `generate_hints.py` writes these columns, and `app/review_ui.py` reads the subset marked ★:

`det_id`★, `frame_idx`★, `timecode_s`★, `det_score`★, `box_xyxy`★, `crop_path`★, `enlarged_crop_path`★, `frame_path`★, `annotated_frame_path`, `is_logo`★, `brand`★, `brand_canonical`, `is_truncated`, `logo_probability`★, `confidence`★, `signals_used`★, `signals_used_raw`, `reasoning`★, `faiss_top1`, `faiss_top1_score`, `faiss_top5_brands`★, `faiss_top5_scores`★, `audio_brands`★, `ocr_text`, `ocr_source`, `experiment`, `brand_mode`, `enabled_signals`, `label`★.

T73 requires the M3 CSV to keep every ★ column with the same name and semantics. The frozen schema in `05` does that and adds the M3 fields.

## 3. The M2 VLM call, as shipped

- Model: `Qwen/Qwen2.5-VL-7B-Instruct` 4-bit (local) or `Qwen/Qwen3-VL-8B-Instruct` (Kaggle). `max_pixels = 128·28·28`, `max_new_tokens = 256`, greedy.
- Signals: `tight_crop`, `enlarged_crop`, `full_frame`, `temporal_context` (offsets −2, −1, +1, +2 with fallback), `audio_brands`, `faiss_matches` (top-5 brand-deduplicated), `ocr_text`. Each is a `## Signal: <name>` section; disabled signals are omitted from the prompt **and not computed** (the defect T10 fixes).
- Output JSON: `is_logo` (logo/partial_logo/not_logo), `brand` (name/UNKNOWN/null), `is_truncated`, `logo_probability` 0–100, `confidence` 0–100, `signals_used`, `reasoning`.
- Open mode says hints are soft evidence and allows any brand; closed mode restricts to the candidate list (watermark + audio + FAISS) or UNKNOWN. Revision 3 runs **open** for both the hinted and blind pass (T33).
- Measured cost: 13.8 s per detection on the 4-bit 7B model (M2 report, video_3-trimmed, 212 detections).
- The generated `logo_probability` took only the values 0/95/100 across the 1,739 rows (client feedback 24 July). That is why T30–T32 exist.

## 4. The eight enrolment call sites (T55 / T75 evidence)

All eight call `LogoDatabaseNew.add_logos(images, brand_names, augmentations, num_augments)` in `app/streamlit_faiss_app.py`:

| # | Tab | Line | Action |
|---|-----|------|--------|
| 1 | Add Logo | 239 | One uploaded image + typed brand |
| 2 | Video Object Cropping with Annotations | 342 | Bulk crops from Label Studio JSON |
| 3 | Review Unknown Images | 723 | Per-image "Add to FAISS" with typed label |
| 4 | Review Unknown Images | 771/773 | Bulk label + add (with or without augmentation) |
| 5 | Review Known Logos | 1305 | "Update" after brand edit |
| 6 | Review Known Logos | 1315 | "FALSE_LOGO" |
| 7 | Review Known Logos | 1396 | Bulk correction (typo / wrong / false positive) |
| 8 | Review Known Logos | 1434 | Bulk "Confirm" |

None records where the image came from. `add_logos` appends brand strings to a flat list; `save()` rewrites `metadata.json` wholesale. T55 changes the record type for new entries only.

## 5. Reuse map for the T7 stage list

| T7 stage | Existing code | Disposition |
|----------|---------------|-------------|
| Ingest | `data_extraction_pipeline.py` frame reading; none for frame folders | Wrap + **new frame-folder mode** (needed for `fixed-images`) |
| Shot detection | `utils/shot_detection.py` | Adapt to emit scenes + `shot_id` (T57 new build) |
| Frame sampling | `_sample_indices` (interval / segment) | Wrap; add `per_shot` mode; `segment` retained as a third mode |
| Detection | Grounding DINO in extraction pipeline; YOLO in `predict_yolov11_m2.py` | Wrap GDINO; **new** YOLO stage; common `DetectionRecord` (T12) |
| Merge | nothing | **New** (T13–T19) |
| Crop refinement | nothing | **New** (T20–T24) |
| OCR | `ocr_image` (lines only, boxes discarded) in `generate_hints.py`; EasyOCR | Wrap reader; **new** cheap boxes call and verification pass (T21, T25–T29) |
| FAISS query | `faiss_top_k` in `generate_hints.py` | Wrap |
| Audio | `compute_audio_hints` | Wrap; greedy decoding; run-dir output |
| VLM | `run_vlm_on_detection`, `_build_vlm_content` | Wrap; **new** blind pass, hint-composition log, UNKNOWN wording, scoring hook (T67, T30–T34) |
| Canonicalisation | `canonicalize_brands` (LLM, non-deterministic) | **New** deterministic pass (T37); LLM merge kept as optional off-by-default step |
| Router | nothing | **New** (T38–T46, T72) |
| Watermark | `--channel-watermark` prompt line only | **New** region config + overlap (T47–T50); the prompt line is kept as-is |
| Output | `annotate_frames`, `export_predictions_csv`, `annotate_video` (Phase 1) | Wrap annotation/CSV; **rewrite** video writer to stream all sampled frames (T58, T59) |
| Evaluation | `benchmarks/evaluation/metrics.py`, `scripts/compare_open_closed.py` | Partial reuse; scripts are **new** (T61–T66, T71) |

## 6. Discrepancies between the correspondence and the code

These matter because the client will quote the correspondence back.

1. **ByteTrack / `track_id`.** The 29 July response states "ByteTrack is already in place on our side, so track_id is not a placeholder". There is no tracker anywhere in the repository. Revision 3's T56 field list does not include `track_id`, so the schema in `05` omits it. Do not re-promise it.
2. **Canonicalisation.** T69 was answered correctly: the M2 pass is an LLM spelling-merge with no reference list. T37 is new build.
3. **OCR text boxes.** T68/A9 answered correctly: `ocr_image` keeps text and confidence only. The refinement-side call (T21) must call `reader.readtext` and keep the box quadrilaterals.
4. **`annotate_video` in Phase 1** writes only frames carrying detections and reads the whole video into memory. T58/T59 require all sampled frames; the writer is rewritten.
5. **Audio brand extraction is non-deterministic** in `generate_hints.py` (`Qwen2_5TextModel` samples at temperature 1.2). The Kaggle and `task2` variants decode greedily. The pipeline uses greedy decoding so a re-run reproduces the same audio brand list; this also matters for the deterministic T37 map.
6. **`convert_video_to_audio` writes `logo_audio.wav` into the current directory** (a 2 MB copy sits in the repo root). The wrapped stage writes into the run directory.
7. **The `utils/fuzzy_matching.py` module loads EasyOCR at import** with English only. The pipeline must not import it at module level; the scoring helpers are copied into `stages/ocr.py`.
8. **Review UI paths.** M2 CSVs carry absolute paths (`/Users/alii/...` or `/kaggle/working/...`). The review UI resolves CSV-relative paths first if they exist. The M3 CSV writes paths **relative to the CSV's own directory** so it loads on any machine (T73).
9. **The `.gitignore` change adding `fixed-images/` is uncommitted.** It should be committed in MM0 together with `runs/`.
10. **A live Hugging Face token** is committed in `task2-whisper-qwen.py`. Revoke it on huggingface.co, replace with `os.environ.get("HF_TOKEN")`, and commit the fix in MM1. Earlier commits already had to remove exposed tokens once (`d97c621`).
11. **The Revision 2 statement that Roboflow images are "screen captures at roughly 1884 by 1000 pixels rather than native video frames"** is half right. `fixed-images` shows three size families (see `02`), and every Roboflow image is one of these frames unchanged. No resolution mapping is needed for route three when the frames themselves are used.

## 7. Environment facts

| Item | Value |
|------|-------|
| Mac venv | Python 3.11; has faiss-cpu, easyocr, scenedetect, rapidfuzz, PyYAML, streamlit, pandas, opencv, pillow; **no torch, transformers, ultralytics**; no CUDA |
| GPU machine | RTX 3090 (24 GB) per `task2-whisper-qwen.py`; Kaggle T4×2 and Colab used for VLM and YOLO |
| Pinned requirements | `requirements-macos.txt` (torch 2.7.0, transformers 4.52.3, easyocr 1.7.2, faiss-cpu 1.12.0, scenedetect 0.6.6, RapidFuzz 3.13.0, streamlit 1.45.1); `requirements-win.txt` |
| Missing from both requirement files | `ultralytics`, `bitsandbytes`, `qwen-vl-utils` is present on mac list; `textdistance` (imported by `utils/fuzzy_matching.py`) is absent |

Consequence for the build: stages are tagged **CPU** or **GPU** in `03`; the CPU-only path (merge → refine → OCR → router → outputs → evaluation) must be runnable on the Mac from persisted raw detections, which is exactly what T11 and T61 require anyway.

## 8. Key numbers to quote

| Fact | Value | Source |
|------|-------|--------|
| video_1 | 960×560, 30 fps, 202.2 s, 6,067 frames, AAC 48 kHz stereo | ffprobe |
| video_1 M2 extraction | 102 sampled frames (interval 2 s), 1,739 detections, GDINO base, box 0.20 / text 0.20 | `extraction_output/video_1/manifest.json` |
| video_3-trimmed | 960×560, 30 fps, 11.3 s, 340 frames | ffprobe |
| video_3-trimmed M2 extraction | 4 sampled frames (segment 3 s), 212 detections, box 0.15 | manifest |
| Local hints/VLM files for video_1 | none (0 `hints.json`, 0 `vlm_prediction*.json`) | find |
| FAISS index | 158,701 vectors, 3,016 distinct brand strings, 487.5 MB, CLIP 768-d L2 | `FAISS/` |
| Unexplained vectors | `rai sport` 17, `verisure` 10 | `metadata.json` |
| YOLO weights | `best.pt`, 51.7 MB, 321 classes | repo root |
| M2 VLM cost | 13.8 s / detection (4-bit 7B) | M2 report |
