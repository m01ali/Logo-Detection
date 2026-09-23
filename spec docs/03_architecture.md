# 03 — Target Architecture (T7, T8, T11)

## 1. Goals the architecture must satisfy

From the correspondence, restated as constraints:

1. A single YAML drives a complete run; no numeric threshold in code (T8). The sampling interval is a plain value the client edits (Revision 3 §2).
2. Discrete stages with persisted intermediate outputs so a failed run resumes without reprocessing (24 July definition of done; T11 for detections).
3. The merge and the router are pure, replayable functions over persisted inputs (T11, T16, T61).
4. Stage-level logging with progress: current stage, frames processed / total, per-stage timing.
5. Outputs load into the existing review screen without conversion (T73).
6. Every run output records the config hash, the brand-list version and hash, the code revision, and the signal switches (T5, T9, T10).
7. CPU-only stages run on the Mac; GPU stages run on the GPU machine; the boundary is a directory that can be copied.

## 2. Package layout

```
logo_pipeline/
  __init__.py
  cli.py                  # entry points: run, stage, router-only, eval-*, brands-*
  config.py               # YAML loader, schema validation, defaults, config hash
  schema.py               # frozen row schema (05), CSV column order, enums
  brands.py               # canonical list loader: version, hash, aliases, normalise()
  geometry.py             # iou, containment, area_ratio, centre_distance, clamp, normalise coords
  logging_utils.py        # stage progress logger, timings, stage_status.json
  frame_source.py         # FrameSource ABC: VideoFrameSource, FolderFrameSource
  stages/
    ingest.py             # opens the source, writes run_manifest.json, extracts audio (video only)
    shots.py              # PySceneDetect scene list -> shots.json, shot_id per frame (T57)
    sampling.py           # interval | per_shot | segment -> sampled_frames.json
    detect_gdino.py       # GPU. Grounding DINO -> raw_detections.jsonl (source=gdino)
    detect_yolo.py        # GPU. YOLO11-L class-agnostic -> raw_detections.jsonl (source=yolo)
    merge.py              # CPU. pure fn over raw_detections -> merged_detections.jsonl (+ logs)
    refine.py             # CPU (+EasyOCR). heuristics 1,2,5 -> refinement.jsonl
    ocr.py                # CPU/GPU. reader singleton; cheap boxes call; verification pass -> ocr.jsonl
    faiss_query.py        # GPU pref. CLIP embed + top-k -> faiss_hints.jsonl
    audio.py              # GPU. Whisper + text LLM -> audio_hints.json
    vlm.py                # GPU. hinted + blind calls -> vlm/hinted.jsonl, vlm/blind.jsonl, vlm/hint_composition.jsonl
    vlm_scoring.py        # GPU. T30/T31 closed-set decision scoring (spike-gated)
    canonicalise.py       # CPU. deterministic map -> canonical.jsonl (T37)
    watermark.py          # CPU. watermark_overlap -> watermark.jsonl (T48)
    router_input.py       # CPU. joins all of the above into router_input.jsonl (the frozen row minus router fields)
    router/
      engine.py           # six-block evaluation, first-match-wins, firing log (T38, T46)
      rules.py            # Rule 1..18 as functions returning the five-value tuple (T44)
      reason_codes.py     # the enum (T44)
      dedup.py            # shared dedup function, two call sites (T45)
      inputs.py           # RouterInput dataclass; missing field -> raise (no .get defaults)
    output.py             # CPU. rows.csv, annotated_frames/, annotated.mp4 (T56, T58, T59)
  eval/
    router_only.py        # T61
    metrics.py            # T62
    iou_match.py          # T63
    blind_vs_hinted.py    # T64
    ordering_checks.py    # T65
    merge_ablation.py     # T18/T19 reporting, merge_enabled false/true comparison
    faiss_selfmatch.py    # T51/T53
    faiss_rebenchmark.py  # T54
    watermark_report.py   # T71
    refinement_report.py  # T24
    sampling_rate_check.py# T66 route three, perceptual hash vs fixed-images
  faiss_store/
    records.py            # enrolment record type, legacy adapter (T55)
    enrol.py              # the one instrumented method every enrolment path calls (T55, T75)
configs/
  pipeline.yaml           # the single YAML (04)
  brands/roboflow_2026-04-28.yaml   # canonical list snapshot (06)
  videos/                 # one file per video_id: source path, channel, watermark regions (T47, T50)
  reason_codes.md         # generated from reason_codes.py (T44 deliverable)
app/
  review_ui.py            # extended (T73, T75)
  streamlit_faiss_app.py  # routed through faiss_store.enrol (T55)
tests/                    # see 21_test_plan.md
```

Existing M2 modules are imported by the wrappers, not copied, except where §6 of `01` says a copy is necessary (fuzzy helpers).

## 3. Stage contract

Every stage implements:

```python
class Stage(Protocol):
    name: str                      # e.g. "merge"
    device: Literal["cpu", "gpu"]
    inputs: list[str]              # run-dir relative files it reads
    outputs: list[str]             # run-dir relative files it writes
    def run(self, ctx: RunContext) -> StageResult: ...
```

Rules:

- A stage reads only from the run directory and the config; it never reaches back into a previous stage's Python objects. This is what makes the CPU/GPU split a directory copy.
- A stage is idempotent. If its outputs exist and `stage_status.json` marks it `done` with the same config hash, it is skipped unless `--force`.
- Outputs are JSON Lines keyed by `det_id` (per-detection stages) or `frame_idx` (per-frame stages), plus one summary JSON per stage. JSON Lines are chosen over per-detection folders because 1,739-row runs already produce ~5,000 small files in M2 and the review UI needs one CSV anyway.
- Crops are cut on demand from `frames/` using stored coordinates (29 July: "nothing is pre-baked as a crop"). The tight and enlarged crop JPEGs are still written under `crops/` because the review UI displays them by path.
- Every stage logs `stage=<name> frame=i/N det=j/M elapsed=…` at a configurable interval and writes `timings` into `stage_status.json`.

## 4. Run directory layout

```
runs/<video_id>/<run_id>/            run_id = YYYYMMDD-HHMMSS-<config_hash[:8]>
  run_manifest.json                  source, video_id, channel, fps (or assumed_fps), frame size,
                                     config snapshot + hash, brand list version + hash, git sha,
                                     signal switches, stage list, created_at
  stage_status.json                  per stage: status, started, finished, config_hash, counts
  frames/frame_XXXXXX.jpg            every sampled frame (T58 needs all of them)
  temporal/frame_XXXXXX/T±k.jpg      context frames (video) or neighbouring ordinals (folder)
  audio/audio.wav, transcript.<lang>.txt, audio_hints.json
  shots.json                         [{shot_id, start_frame, end_frame, start_s, end_s}]
  sampled_frames.json                [{frame_idx, timecode_s, shot_id, path}]
  raw_detections.jsonl               T11 — one line per raw box from each detector
  merged_detections.jsonl            T13 — representatives, detection_sources, merge_count, merged_det_ids
  merge_log.jsonl                    T19 — every evaluated pair with iou/containment/area_ratio/outcome
  merge_counters.json                T18 — per video and per frame
  crops/det_XXXXXX/{crop.jpg,crop_enlarged.jpg,selected.jpg}
  refinement.jsonl                   T20–T23
  ocr.jsonl                          T25–T29 (verification pass) + refinement-side boxes
  faiss_hints.jsonl
  vlm/hinted.jsonl, vlm/blind.jsonl  T33
  vlm/hint_composition.jsonl         T34
  vlm/scores.jsonl                   T31 (when enabled)
  canonical.jsonl                    T37
  watermark.jsonl                    T48
  router_input.jsonl                 the input rows T61 replays
  router_output.jsonl                decision, reason_code, confidence, risk, trust per row
  firing_log.jsonl                   T46
  rows.csv                           the per-run CSV (05), review-UI compatible
  annotated_frames/frame_XXXXXX.jpg  T58
  annotated.mp4                      T59
  reports/                           evaluation outputs when run against this run
```

`router_input.jsonl` is the artefact that makes T61 deterministic: the router-only runner takes this file and a YAML and emits `router_output.jsonl` without touching anything upstream.

## 5. Ingest modes

### 5.1 Video (default)

`VideoFrameSource(path)` via OpenCV; fps and frame count from the container; audio extracted with moviepy into `audio/`. `video_id` is taken from `configs/videos/<video_id>.yaml` or `--video-id`, never from the filename alone (A7 asks for the client's identifier to be carried on the rows).

### 5.2 Frame folder (new, for `fixed-images`)

`FolderFrameSource(dir, pattern)` reads `{video_id}_frame_{ordinal}.png` in ordinal order. `frame_idx = ordinal`, `timecode_s = (ordinal − 1) / assumed_fps`, `fps = assumed_fps` recorded with `fps_is_assumed: true` in the manifest. Shot detection runs PySceneDetect's `ContentDetector` over the frame sequence (adaptive detection needs a continuous stream; on sparse samples the content detector with a higher threshold is the honest choice, and the manifest records the detector used). Sampling mode `interval` maps to every k-th ordinal with `k = round(interval_s × assumed_fps)`; with the defaults (2 s, 1 fps) every second frame is processed, and `interval_s: 1.0` processes every frame. Temporal context is the ±1, ±2 neighbouring ordinals. Audio is `signal_unavailable`.

Route three (T66) uses this mode with `interval_s` set so that every labelled frame is processed (the YAML per-video file lists `process_all_frames: true` for evaluation runs).

## 6. CLI

```
python -m logo_pipeline run      --config configs/pipeline.yaml --video-config configs/videos/video_1.yaml [--from-stage merge] [--to-stage router] [--force]
python -m logo_pipeline run      --config … --frames-dir fixed-images --video-id 14 [--stages …]
python -m logo_pipeline router-only --config … --run runs/video_1/<run_id> [--out reports/router_v2]
python -m logo_pipeline eval <name> --config … --run … [--labels labels.csv]
python -m logo_pipeline brands validate|hash|add --list configs/brands/roboflow_2026-04-28.yaml
python -m logo_pipeline faiss selfmatch|rebenchmark|enrol …
```

`--from-stage`/`--to-stage` are how the GPU machine runs `ingest…detect_yolo` and `faiss_query…vlm`, and the Mac runs everything else on a copied run directory.

## 7. Device split

| Stage | Device | Why |
|-------|--------|-----|
| ingest, shots, sampling | CPU | OpenCV / PySceneDetect |
| detect_gdino, detect_yolo | GPU | transformers / ultralytics |
| merge, refine (geometry), canonicalise, watermark, router_input, router, output | CPU | pure Python |
| ocr (both calls) | CPU acceptable, GPU faster | EasyOCR; 1,739 crops ≈ 10–20 min on CPU |
| faiss_query | GPU preferred | CLIP embedding of every crop |
| audio | GPU | Whisper large-v3 + 7B/8B text LLM 4-bit |
| vlm, vlm_scoring | GPU | Qwen2.5-VL-7B 4-bit, ~13.8 s per call |
| eval/* | CPU | pandas |
| review UI, FAISS manager | CPU | Streamlit; the FAISS manager loads CLIP for enrolment (CPU works, slowly) |

## 8. Determinism and provenance

- Config hash = SHA-256 of the canonicalised YAML (sorted keys, resolved defaults). Written into every JSONL line's run header and into `rows.csv` as `config_hash`.
- Brand list hash = SHA-256 of the canonicalised list content (06). Written as `brand_list_version` and `brand_list_hash` on every row (T5, T26).
- VLM decoding is greedy; audio LLM decoding is greedy; EasyOCR is deterministic for a given model version; FAISS flat search is exact. Merge, refinement, canonicalisation and the router are pure functions. The only residual non-determinism is the VLM under different quantisation kernels, which is recorded via `vlm_model_id` and `vlm_quantisation` in the manifest.
- Git SHA and `pip freeze` snapshot are written into `run_manifest.json`.

## 9. Two-switch signal config (T10)

Each of `faiss`, `ocr`, `audio` has `compute_and_log` and `include_in_vlm_prompt`. The hinted call includes a section only if both are true; the blind call includes none; computation happens whenever `compute_and_log` is true regardless of the prompt. Image signals (`tight_crop`, `enlarged_crop`, `full_frame`, `temporal_context`) keep a single `include_in_vlm_prompt` switch because they have nothing to log separately. The M2 behaviour (disabling a signal also skipped its computation) is retired.
