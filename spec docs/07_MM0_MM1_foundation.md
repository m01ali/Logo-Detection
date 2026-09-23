# 07 — MM0 Preparation (T5, T76) and MM1 Foundation (T7–T11)

## MM0 — Preparation and housekeeping

**Effort:** 1 day (Revision 3 Group 0). **Device:** CPU. **Gates:** nothing; MM1 starts in parallel on day 1.

### T5 — Canonical brand list snapshot

1. Write `scripts/make_brand_snapshot.py` (one-off): reads `~/Downloads/logo2.yolov11/data.yaml`, writes `configs/brands/roboflow_2026-04-28.yaml` in the `06` format with empty alias lists, `channel: true` for the broadcaster names (`boing`, `discovery plus`, `infinity plus`, `m6`, `mediaset`, `nove`, `paramount plus`, `rai`, `rai fiction`, `rete 4`, `sky`, `sky wifi`, `sport italia`, `tim`, `x factor` — **[ASSUMPTION]**, confirm with the client; `tim` and `sky wifi` are advertisers as often as channels).
2. Add the alias seeds from `06` §2.
3. Implement `logo_pipeline/brands.py` (`06` §5) and `python -m logo_pipeline brands validate|hash`.
4. Commit the snapshot. Record version `2026-04-28` and the hash in `20_received_artefacts_record.md`.

**Done when:** `brands validate` passes; `brands hash` is stable across two runs; unit tests in `21` §MM0 pass (alias collision rejected, short-name exact rule, accent stripping).

### T76 — Received-artefacts record

Create `docs/received_artefacts.md` from `20_received_artefacts_record.md` (this spec set holds the seed). Update it whenever an artefact arrives. It is a deliverable at close.

### Housekeeping (not a task ID, but blocks a clean start)

- Commit the pending `.gitignore` change (`fixed-images/`) and add `runs/`, `logo_audio.wav`, `*.avi`. Remove `runs/detect/` from tracking (`git rm -r --cached`).
- **Revoke the Hugging Face token in `task2-whisper-qwen.py:40`** on huggingface.co, replace the line with `os.environ.setdefault("HF_TOKEN", "")`, commit. Check `git log -p` history is acceptable to the user; a history rewrite is the user's decision (`19`).
- Add `ultralytics`, `bitsandbytes`, `qwen-vl-utils`, `textdistance` to a new `requirements-gpu.txt`; keep `requirements-macos.txt` for the CPU path. Pin versions from the GPU machine's environment when first run there.
- Copy `best.pt`, `FAISS/`, `video_1.mp4`, `video_3-trimmed.mp4` to the GPU machine; record locations in T76.

## MM1 — Foundation: the configured pipeline

**Effort:** 3 days (Revision 3 Group 1). **Device:** CPU for the skeleton; one GPU run to prove the wrapped M2 path. **Gates:** everything after it.

**Goal.** A `python -m logo_pipeline run` that reproduces the M2 output for `video_3-trimmed` under the new layout, with the YAML, the stage contract, persisted raw detections and the frozen schema in place. No new algorithms in MM1; the merge, refinement, router and blind pass are stubs that pass rows through unchanged.

### Step 1 — Package skeleton and config (T8) — 0.5 day

1. Create `logo_pipeline/` per `03` §2 with empty stage modules.
2. `config.py`: load `pipeline.yaml`, overlay `configs/videos/<id>.yaml` `overrides`, validate against a dataclass schema (every key in `04`), reject unknown keys, compute `config_hash`.
3. `logging_utils.py`: `StageLogger` writing `stage=… i/N` lines and `stage_status.json`.
4. `cli.py` with `run`, `--from-stage`, `--to-stage`, `--force`, `--frames-dir`, `--video-id`.

**Check:** `python -m logo_pipeline run --config configs/pipeline.yaml --video-config configs/videos/video_3-trimmed.yaml --to-stage ingest` creates the run directory and manifest.

### Step 2 — Frame sources and ingest — 0.5 day

1. `frame_source.py`: `VideoFrameSource` (OpenCV; fps, frame count, seek) and `FolderFrameSource` (sorted ordinals, `assumed_fps`).
2. `stages/ingest.py`: writes `run_manifest.json` (`03` §4), extracts audio to `audio/audio.wav` in video mode (moviepy, as `models/audio_model.convert_video_to_audio` does but into the run dir).
3. `stages/shots.py`: PySceneDetect scene list → `shots.json`; `shot_id` = ordinal; frames mode uses `ContentDetector` (`04` `shots.detector` forced). This is the T57 producer; MM7 only copies the id onto rows.
4. `stages/sampling.py`: `interval` (as M2: `step = round(fps × interval_s)`), `per_shot` (middle or first frame of each scene), `segment` (M2 legacy). Writes `sampled_frames.json` with `shot_id` per sampled frame and saves `frames/` and `temporal/`.

**Check:** on `video_1.mp4` with `interval_s: 2.0` the sampled frame indices equal the 102 M2 indices (0, 60, 120, …); `shots.json` is non-empty; frames mode on `fixed-images` video 14 lists 46 frames with `fps_is_assumed: true`.

### Step 3 — Detection records and persistence (T11, groundwork for T12) — 0.5 day

1. Define `DetectionRecord` in `schema.py`: `raw_det_id, frame_idx, source ∈ {gdino, yolo}, box_xyxy, score, label, class_conf, frame_width, frame_height`.
2. `stages/detect_gdino.py`: wrap `DataExtractionPipeline._run_detection` (model load, prompt from YAML, thresholds from YAML, `min_crop_px`). Writes `raw_detections.jsonl` lines with `source: gdino`.
3. `stages/detect_yolo.py`: stub in MM1 that writes nothing (real implementation in MM2), so the file format is fixed now.
4. Both detectors append to the same `raw_detections.jsonl`; a stage header line records the detector config.

**Check:** `raw_detections.jsonl` for `video_3-trimmed` contains 212 GDINO lines with the same boxes as the M2 manifest (box threshold 0.15 for this video, per its M2 config; use a per-video override to reproduce).

### Step 4 — Pass-through merge and the frozen schema (T9) — 0.5 day

1. `stages/merge.py` in MM1: `merge.enabled: false` path only — every raw box becomes a representative, `merge_count 0`, `detection_sources {source}`, `det_id` assigned sequentially. This is the T16 false path and it must keep working after MM2.
2. `schema.py`: the `Row` dataclass with every field in `05`, `CSV_COLUMNS` in that order, enums for `router_decision`, `refinement_status`, `brand_evidence`, etc.
3. `stages/router_input.py`: joins whatever per-stage JSONL files exist into `router_input.jsonl`; missing stages leave their fields null.
4. `scripts/export_schema_doc.py`: renders `05` from `schema.py` (field, type, stage) so the document sent to the client is generated from code and cannot drift.

**Deliverable:** the schema document to the client (Revision 3: due with T7). Send `05` after this step; the client confirms within one round.

### Step 5 — Two-switch signal config (T10) and the wrapped M2 hint/VLM path — 1 day (GPU)

1. `stages/audio.py`: wrap `compute_audio_hints` from `generate_hints.py`; force greedy decoding on the text model; prompt variant from YAML (`task2_italian` default); output `audio/audio_hints.json`; `audio_signal_status`.
2. `stages/faiss_query.py`: wrap `load_faiss_db` + `faiss_top_k`; compute `faiss_score_gap`; crops cut from `selected_bbox` (which equals `loose_bbox` until MM3).
3. `stages/ocr.py`: reader singleton; in MM1 only the verification-pass *hint* form (lines + confidences, M2 behaviour) so the VLM prompt is unchanged; matching arrives in MM4.
4. `stages/vlm.py`: wrap `_build_vlm_content` + `run_vlm_on_detection`; sections included only when `compute_and_log && include_in_vlm_prompt`; computation governed by `compute_and_log` alone. Hinted pass only in MM1. Add the T67 UNKNOWN wording now (it is one line and MM9 would otherwise re-touch the prompt): "(ii) … If you cannot identify the brand, or it is a logo of a brand you do not recognise, output exactly \"UNKNOWN\"." — keep the M2 field rules.
5. `stages/canonicalise.py`, `stages/watermark.py`, `stages/router/`: pass-through stubs writing null fields; `router_decision` null in MM1.
6. `stages/output.py`: wrap `export_predictions_csv` (relative paths) and `annotate_frames`; in MM1 the annotated frames are written for **all sampled frames**, drawing brand only (decision/reason arrive in MM7).

**Check (the MM1 acceptance run):** on the GPU machine, `run` over `video_3-trimmed` with all signals on produces `rows.csv` whose M2 columns match the M2 `vlm_predictions.csv` in row count (212) and column set; `app/review_ui.py --csv runs/video_3-trimmed/<run>/rows.csv` loads and shows crops. Copy the run directory to the Mac; `run --from-stage merge --to-stage output` re-generates `rows.csv` identically (idempotency and the CPU/GPU split).

### Step 6 — Documentation

`README.md` gains a "Running M3" section: environment files, the YAML, the per-video file, the two ingest modes, `--from-stage/--to-stage`, where outputs land. Required by the 24 July definition of done.

### MM1 definition of done

- [ ] All 16 stages exist and run in order from one YAML; unknown keys fail validation.
- [ ] `raw_detections.jsonl` persisted; `merge.enabled: false` path executes.
- [ ] `router_input.jsonl` and `rows.csv` carry every field of `05` (nulls allowed).
- [ ] Two-switch config verified: with `faiss.include_in_vlm_prompt: false` the prompt has no FAISS section and `faiss_hints.jsonl` is still written.
- [ ] Schema document exported and sent (T9).
- [ ] Review UI loads `rows.csv` unchanged.
- [ ] Token removed, `.gitignore` committed, `requirements-gpu.txt` present.
