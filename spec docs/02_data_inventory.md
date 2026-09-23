# 02 — Data Inventory and the `fixed-images` Finding

Everything in this document was measured on 16 September 2026 with scripts run against the files on this machine. The scripts are reproducible from the descriptions and should be re-run as part of T76 when any artefact is re-supplied.

## 1. `fixed-images/` — the client folder for M3

| Property | Value |
|----------|-------|
| Location | `/Users/alii/Documents/Code/Logo-Detection/fixed-images/` (git-ignored by the uncommitted `.gitignore` change) |
| Files | 20,222 PNG, no subfolders, 12 GB |
| Naming | `{video_id}_frame_{ordinal:04d}.png`, ordinal contiguous from 0001 per video, no gaps, no other patterns |
| Videos | 100 distinct `video_id` values, 1 to 100 |
| Frames per video | min 36 (video 25), max 608 (video 82), typical 100–300 |
| Colour | RGB PNG; Roboflow re-encoded them as PNG too |
| Folder mtime | 17 March 2026 (predates the 28 April Roboflow export) |

### 1.1 Resolution families

Frame size is constant within a video and falls into three families:

| Family | Typical size | Videos |
|--------|--------------|--------|
| A | ~1,844–1,898 × 954–1,028 (some 1,878–1,898 × 1,162–1,180) | 1–37, 77, 83, 84, 95, 97, 98, 100 |
| B | ~1,236–1,304 × 662–724 | 39–76, 78–82, 85–94, 96, 99 |
| C | 842 × 472 | 38 |

Sizes are not exact multiples of 16:9 and vary by a few pixels per video, which is consistent with cropped screen recordings. Box coordinates in the Roboflow labels are normalised, so this only matters for `min_merge_area` (a fraction of frame area, T14) and for the annotated outputs.

### 1.2 Content (visual spot-check of 24 frames)

Italian broadcast television: Rai News 24 studio and sport, Rai 1/2 idents, Mediaset (Rete 4, Italia 1, Canale 5, Iris, Mediaset Extra), Nove, Sport Italia, Boing; advertising blocks (Vicks, Lidl, Mavosten, Whirlpool, Zucchi, Ferrari Formaggi, ACE, Chef, Kinder Bueno, Arborea, Acqua di Nepi, San Benedetto, Gazzetta / Corriere / Tuttosport front pages); football and tennis broadcasts with perimeter boards and kit sponsors. Channel bugs sit top-left (Rai) or bottom-right (Mediaset). This is the population the router's Rule 5 and the canonical list were designed around.

## 2. Roboflow export `~/Downloads/logo2.yolov11/` (exported 28 April 2026)

| Property | Value |
|----------|-------|
| Images | 5,830 (`train/images`), no valid/test split in the export |
| Labels | 5,830 files, **12,774 boxes**, YOLO format `cls cx cy w h` (normalised) |
| Classes | 321 (`data.yaml` `names`), all 321 have at least one box |
| Background images | 46 (zero boxes) |
| Max boxes per image | 77 |
| Filenames | `{video}_frame_{n}_png.rf.{hash}.png`; the stem before `_png` is the `fixed-images` name |
| Source videos | 40 distinct ids: 1, 2, 3, 9–39, 95–100 |
| Pre-processing | "No pre-processing or augmentation was applied" (README) |

### 2.1 Class distribution

`rai` alone has 4,384 boxes (34.3 %). Next: `mediaset` 332, `boing` 320, `rete 4` 281, `sport italia` 232, `wind tre` 202, `nove` 198, `italo` 197, `corriere della sera` 196, `eni` 158. Channel-bug classes therefore dominate the gold set; the T54 present/absent split and the T63 matched-subset precision must be reported with and without channel classes or they will be swamped by `rai`.

Twelve classes have exactly one box (`dkv`, `kinder colazione piu`, `kinder brioss`, `muller`, `cygames`, `diadora`, `discovery plus`, `zalando`, `mbe`, `msc`, …). `wolkswagen` is a misspelling of Volkswagen in the client's own list; it stays the canonical spelling because the list is the client's (T5) and gets `volkswagen` as an alias.

### 2.2 Box size distribution (fraction of frame area)

| Statistic | Value |
|-----------|-------|
| min | 0.00001 |
| p5 | 0.0012 |
| median | 0.0032 |
| p95 | 0.033 |
| max | 0.575 |
| share below 0.0004 (the `min_merge_area` placeholder) | 0.8 % |
| share below 0.001 | 3.8 % |

The 0.0004 placeholder sits below essentially every labelled logo, so "erring high" (client steer) has headroom: raising it to 0.001 would still leave 96 % of labelled boxes above the floor.

## 3. Verified mapping: Roboflow ↔ `fixed-images`

Method: strip `_png.rf.<hash>.png`, look up `fixed-images/<stem>.png`, compare sizes, then compare decoded pixels on 40 random pairs.

| Check | Result |
|-------|--------|
| Roboflow stems found in `fixed-images` | **5,830 of 5,830** |
| Duplicate stems in the export | 0 |
| Size mismatches | 0 of 40 |
| Pixel-identical | 0 of 40 |
| Mean absolute pixel difference | ≈ 0.5 (on a 0–255 scale) — Roboflow's re-encode, not a resize or crop |

Labelled fraction per video (fixed-images frames → Roboflow images): 100 % for most of videos 2, 3, 9–38, 96, 98, 99; partial for video 1 (37 of 216, 17 %), 13 (88 %), 14 (28 of 46, 61 %), 17 (89 %), 25 (92 %), 29 (92 %), 39 (15 of 109, 14 %), 95 (94 of 282, 33 %), 97 (69 %), 100 (128 of 140, 91 %). Videos 4–8 and 40–94 (60 videos, 14,392 frames) carry no labels.

### 3.1 Consequences for the task list

1. **Route three (T66) no longer needs the sampling rate or a resolution mapping to match boxes.** The gold boxes are defined on frames we hold. Running the pipeline in frame-folder mode over `fixed-images/{v}_frame_*.png` produces router-input rows whose frame identity is the Roboflow image, and T63's IoU matching runs on identical pixel grids.
2. **The sampling-rate check still has a purpose**, but a smaller one: it converts ordinals to timecodes (exposure duration, the commercial output). Until a source video arrives, frame-folder runs carry `timecode_s = (ordinal − 1) / assumed_fps` with `assumed_fps` recorded in the run manifest as an assumption (client believes 1 fps).
3. **No audio in frame-folder mode.** `audio_brands` is empty and logged as `signal_unavailable`; Rule 11 cannot fire; T26 matches OCR against the canonical list alone. Temporal context comes from neighbouring ordinals (±1, ±2 frames of the folder) rather than ±1..2 video frames, and is labelled as such in the hint-composition log (T34).
4. **The five nominated route-three videos are all here**: 14 (46 frames, 28 labelled, 584 boxes of which ~540 non-channel), 1 (216 / 37, 499 boxes / 463 non-channel), 100 (140 / 128, 750 / 644), 98 (185 / 185, 568 / ~353), 27 (153 / 150, 363 / 230). The doc's counts are reproduced to within the definition of "channel" used.
5. **A fallback for the fresh-frame recall sample exists.** If the MP4 fresh videos are late, 30–50 frames can be drawn from the 60 unlabelled `fixed-images` videos, on which neither detector has trained. This is offered in `19` as an option, not assumed.
6. **T76 record.** `fixed-images` is an artefact received from the client and must be listed with date, source and location.

## 4. FAISS index `FAISS/`

| Property | Value |
|----------|-------|
| `logo_index.faiss` | 487.5 MB, `IndexFlatL2`, 768-d CLIP ViT-L/14 image embeddings, L2-normalised |
| `metadata.json` | flat JSON list, 158,701 strings, 3,016 distinct |
| Similarity convention | `1 − L2²/4` (cosine on unit vectors) |
| Largest brands | `76` 302, `mclaren` 252, `zendium` 249, `lexus-1` 223, … (LogoDet-3K style) |
| Client-domain brands present | `rai sport` 17 vectors, `verisure` 10 — the 27 unexplained vectors |
| Overlap with the 321-name list | 26 by exact match (Revision 2 finding) |
| `FALSE_LOGO` entries | 0 |

The metadata format (strings only) is why T55 introduces records `{brand, source_id, enrolled_at, …}` for new entries and reads existing entries as legacy.

## 5. M2 extraction outputs `extraction_output/`

| Video | Frames sampled | Detections | Files | Hints / VLM |
|-------|----------------|------------|-------|-------------|
| `video_1` | 102 (every 60 frames = 2 s) | 1,739 | `manifest.json` 1.26 MB, per-det `metadata.json`, `crop.jpg`, `crop_enlarged.jpg`, `frame.jpg`, temporal T±5 only (older extraction) | none locally |
| `video_3-trimmed` | 4 (segment 3 s) | 212 | + `vlm_predictions.csv` (open_baseline, 33 logo / 179 not_logo), `transcript.english.txt` | CSV only |

Detection record fields available today: `det_id, sample_idx, frame_idx, timecode_seconds, label (GDINO phrase), score, box_xyxy, box_xyxy_enlarged, frame_width, frame_height, crop_path, crop_enlarged_path, temporal_frames{offset: path}`.

The 1,739 rows are historical (Revision 3, T61). They remain useful as a **CPU-only development fixture** for merge, refinement, OCR and router work before the GPU machine is available, and `19` proposes using them that way.

## 6. Other artefacts

| Artefact | Location | Note |
|----------|----------|------|
| `video_1.mp4`, `video_3-trimmed.mp4` | repo root (git-ignored) | The only source videos held |
| `~/Downloads/vlm_predictions*.csv` | Downloads | Three video_3-trimmed exports (an early one without `logo_probability`) |
| `~/Downloads/cropped_logos*`, `owlv2_logos*.zip`, `grounding_dino_logos.zip`, `yoloe_logos.zip` | Downloads | Phase 1 detector comparison crops |
| `~/Downloads/logo_for_database/` | Downloads | Reference logos used for the Phase 1 leave-one-out benchmark (108 queries) |
| `benchmarks/results/loo_summary.csv` | repo | CLIP recall@1 0.98 on clean reference logos — the number the client contrasts with "observed FAISS quality is poor" on video crops |
| `M3_*.docx` | repo root and `~/Downloads` | Full correspondence chain; markdown conversions used for this audit are in the session scratchpad only |
