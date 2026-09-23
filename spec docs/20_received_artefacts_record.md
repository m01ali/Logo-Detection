# 20 — Received-Artefacts Record (T76 seed)

Copy to `docs/received_artefacts.md` in MM0 and keep it current. One row per artefact; add a row when anything is re-supplied rather than editing the old one.

| # | Artefact | Received | From | Where stored (this machine) | Also on | Hash / size | Notes |
|---|----------|----------|------|------------------------------|---------|-------------|-------|
| 1 | `fixed-images/` — 20,222 PNG frames, 100 videos | folder mtime 17 Mar 2026; identified as the M3 client folder 16 Sep 2026 | client | `repo/fixed-images/` (git-ignored) | to copy to GPU machine | 12 GB | Frame pool of the Roboflow labels (`02` §3). Rate believed 1 fps, unconfirmed |
| 2 | Roboflow YOLOv11 export `logo2.yolov11` (5,830 images, 12,774 boxes, 321 classes) | export 28 Apr 2026; on disk 2 May 2026 | client (Roboflow project antos-workspace-cttem) | `~/Downloads/logo2.yolov11/` | training machine (Colab, Drive) | 5,830 + 5,830 files | Used to train `best.pt`. Filenames map to artefact 1 |
| 3 | YOLO11-L weights `best.pt` | 4 Sep 2026 (copied into repo root) | own training run `logo2_raw` | `repo/best.pt` (git-ignored) | Google Drive `MyDrive/LogoDet/weights/<run>/best.pt` | 51.7 MB | 321-class; run class-agnostic in M3 |
| 4 | FAISS index `logo_index.faiss` + `metadata.json` | 29 Mar 2026 (file dates) | own build (LogoDet-3K-style public set) + client enrichment via the FAISS manager | `repo/FAISS/` (git-ignored) | — | 487.5 MB + 2.2 MB; 158,701 vectors, 3,016 brands | No provenance per vector; 27 vectors `rai sport`/`verisure` |
| 5 | `video_1.mp4` (Rai Sport advertising block, 202 s) | 14 Mar 2026 | client | `repo/video_1.mp4` (git-ignored) | GPU machine (per `task2` script paths) | 49.3 MB | The corpus video |
| 6 | `video_3-trimmed.mp4` (11 s) | 19 Mar 2026 | client / trimmed locally | `repo/video_3-trimmed.mp4` | — | 3.3 MB | Smoke-test video |
| 7 | M2 extraction outputs (`extraction_output/video_1`, `video_3-trimmed`) | 3 May 2026 / 19 Apr 2026 | own runs | `repo/extraction_output/` (git-ignored) | — | 82 MB | 1,739 and 212 detections; no hints/VLM files for video_1 locally |
| 8 | M2 VLM CSVs for video_3-trimmed | 22–26 Apr 2026 | own Kaggle runs | `~/Downloads/vlm_predictions*.csv`, `repo/extraction_output/video_3-trimmed/vlm_predictions.csv` | Kaggle | — | The July video_1 VLM output is lost (Revision 2 A1) |
| 9 | Canonical brand list snapshot `roboflow_2026-04-28.yaml` | to be created in MM0 | derived from artefact 2 | `repo/configs/brands/` (committed) | — | hash recorded on creation | T5 |
| 10 | Fresh videos (3–5, MP4/H.264/AAC, one Drive sub-folder each) | **pending** | client | — | — | — | Notice date per `18` |
| 11 | Watermark coordinates per fresh video | **pending** (trails 10) | client | `configs/watermark_regions.yaml` when received | — | — | T47 |
| 12 | Route-two labels | **pending** (after the video_1 run) | client | `runs/video_1/<run>/label_request/labels_filled.csv` | — | — | T66 |
| 13 | M3 correspondence (18 Jul → 6 Sep, 17 documents) | various | both | `~/Downloads/M3_*.docx`, repo root (4 files) | — | — | Markdown conversions were session-scratch only |
