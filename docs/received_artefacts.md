# Received-Artefacts Record (T76)

Copied from `spec docs/20_received_artefacts_record.md` (the seed) and kept
current from here. One row per artefact; add a row when anything is
re-supplied rather than editing the old one. Locations are relative to the
repository root unless stated.

| # | Artefact | Received | From | Where stored (this machine) | Also on | Hash / size | Notes |
|---|----------|----------|------|------------------------------|---------|-------------|-------|
| 1 | `fixed-images/` — 20,222 PNG frames, 100 videos | folder mtime 17 Mar 2026; identified as the M3 client folder 16 Sep 2026 | client | `repo/fixed-images/` (git-ignored) | to copy to GPU machine | 12 GB | Frame pool of the Roboflow labels. Rate believed 1 fps, unconfirmed |
| 2 | Roboflow YOLOv11 export `logo2.yolov11` (5,830 images, 12,774 boxes, 321 classes) | export 28 Apr 2026; on disk 2 May 2026 | client (Roboflow project antos-workspace-cttem) | `~/Downloads/logo2.yolov11/` | training machine (Colab, Drive) | 5,830 + 5,830 files | Used to train `best.pt`. Filenames map to artefact 1. Consumed by `scripts/make_brand_snapshot.py` for artefact 9 |
| 3 | YOLO11-L weights `best.pt` | 4 Sep 2026 (copied into repo root) | own training run `logo2_raw` | `repo/best.pt` (git-ignored) | Google Drive `MyDrive/LogoDet/weights/<run>/best.pt` | 51.7 MB | 321-class; run class-agnostic in M3 (T12). **Not yet copied to a GPU machine** — no GPU machine chosen yet (open question, see `spec docs/19` §B3) |
| 4 | FAISS index `logo_index.faiss` + `metadata.json` | 29 Mar 2026 (file dates) | own build (LogoDet-3K-style public set) + client enrichment via the FAISS manager | `repo/FAISS/` (git-ignored) | — | 487.5 MB + 2.2 MB; 158,701 vectors, 3,016 brands | No provenance per vector; 27 vectors `rai sport`/`verisure`. **Not yet copied to a GPU machine** |
| 5 | `video_1.mp4` (Rai Sport advertising block, 202 s) | 14 Mar 2026 | client | `repo/video_1.mp4` (git-ignored) | GPU machine (per `task2` script paths) | 49.3 MB | The corpus video. **Not yet copied to a GPU machine** |
| 6 | `video_3-trimmed.mp4` (11 s) | 19 Mar 2026 | client / trimmed locally | `repo/video_3-trimmed.mp4` | — | 3.3 MB | Smoke-test video. **Not yet copied to a GPU machine** |
| 7 | M2 extraction outputs (`extraction_output/video_1`, `video_3-trimmed`) | 3 May 2026 / 19 Apr 2026 | own runs | `repo/extraction_output/` (git-ignored) | — | 82 MB | 1,739 and 212 detections; no hints/VLM files for video_1 locally |
| 8 | M2 VLM CSVs for video_3-trimmed | 22–26 Apr 2026 | own Kaggle runs | `~/Downloads/vlm_predictions*.csv`, `repo/extraction_output/video_3-trimmed/vlm_predictions.csv` | Kaggle | — | The July video_1 VLM output is lost (Revision 2 A1) |
| 9 | Canonical brand list snapshot `roboflow_2026-04-28.yaml` | created 17 Sep 2026 (MM0, T5) | derived from artefact 2 via `scripts/make_brand_snapshot.py` | `repo/configs/brands/roboflow_2026-04-28.yaml` (committed) | — | `sha256:f1f139865d2559fd3a7463730054e7eb820d561e6d95e25e4489c5d22bacd857` — 321 brands, 15 channel brands, 37 short names (≤4 chars), 11 with alias seeds | The `channel: true` set is **[ASSUMPTION]**, confirm with the client (`spec docs/07` MM0 T5, `spec docs/19` §A5) |
| 10 | Fresh videos (3–5, MP4/H.264/AAC, one Drive sub-folder each) | **pending** | client | — | — | — | Notice date per `spec docs/18` |
| 11 | Watermark coordinates per fresh video | **pending** (trails 10) | client | `configs/watermark_regions.yaml` when received | — | — | T47 (MM6, not yet built) |
| 12 | Route-two labels | **pending** (after the video_1 run) | client | `runs/video_1/<run>/label_request/labels_filled.csv` | — | — | T66 (MM11, not yet built) |
| 13 | M3 correspondence (18 Jul → 6 Sep, 17 documents) | various | both | `~/Downloads/M3_*.docx`, repo root (4 files) | — | — | Markdown conversions were session-scratch only |
| 14 | The `logo_pipeline/` M3 package itself (MM0 + MM1: T5, T7–T11) | 17 Sep 2026 | own build, this session | `repo/logo_pipeline/`, `repo/configs/`, `repo/scripts/`, `repo/tests/` (committed) | — | — | Implements `spec docs/07`. `config_hash`/`brand_list_hash`/`git_sha` are written into every run's `run_manifest.json` (03 §8) |

## MM0 housekeeping performed (this session)

- Committed the pending `.gitignore` change (`fixed-images/`); added `runs/`, `logo_audio.wav`, `*.avi`.
- Untracked `runs/detect/` (339 YOLO-predict label files + 1 `.avi`, artefact of the M2 `predict_yolov11_m2.py` run) and `logo_audio.wav` from git (`git rm --cached`); both remain on disk.
- **The Hugging Face token committed in `task2-whisper-qwen.py:40` was replaced with `os.environ.setdefault("HF_TOKEN", "")`.** The token itself **must still be revoked on huggingface.co by the account owner** — that action cannot be performed from this session. It has been in git history since commit `0da8613`; whether to rewrite history (`git filter-repo`) or leave the tip fix only is the user's decision (`spec docs/19` §B2), not made here.
- Added `requirements-gpu.txt` (`ultralytics`, `bitsandbytes`, `qwen-vl-utils`, `textdistance`), unpinned pending a real run on the chosen GPU machine.
