# 04 — The Single YAML (T8): Key-by-Key Specification

The file is `configs/pipeline.yaml`. Every numeric threshold in the specs appears here with its v1 value and owning task. Per-video facts (path, channel, watermark regions) live in `configs/videos/<video_id>.yaml` and are merged at load time; the merged, resolved config is what gets hashed.

Types: `int`, `float`, `bool`, `str`, `enum{…}`, `list`, `map`. Values marked **placeholder** are expected to change after the first full run (Revision 3 §3).

```yaml
version: 1                                   # config schema version, bump on breaking change

run:
  output_root: runs                           # str
  run_id: null                                # str|null; null => auto YYYYMMDD-HHMMSS-<hash8>
  log_every_n: 10                             # int; progress log cadence (frames / detections)
  device: auto                                # enum{auto,cpu,cuda}
  seed: 42                                    # int; used only by augmentation in enrolment
  stages:                                     # list; the ordered stage plan (T7)
    [ingest, shots, sampling, detect_gdino, detect_yolo, merge, refine, ocr,
     faiss_query, audio, vlm, canonicalise, watermark, router_input, router, output]

ingest:
  mode: video                                 # enum{video, frames}; frames = fixed-images style folder
  frames:
    dir: null                                 # str; e.g. fixed-images
    pattern: "{video_id}_frame_*.png"         # str
    assumed_fps: 1.0                          # float; client belief; recorded as fps_is_assumed (02 §3.1)
  save_frame_jpeg_quality: 95                 # int (from data_extraction_pipeline)

shots:                                        # T57
  enabled: true                               # bool
  detector: adaptive                          # enum{adaptive, content}; frames-mode forces content
  adaptive_threshold: 3.0                     # float (PySceneDetect default)
  content_threshold: 27.0                     # float (PySceneDetect default)
  min_scene_len_frames: 15                    # int

sampling:                                     # Revision 3 §2 "Frame sampling"
  mode: interval                              # enum{interval, per_shot, segment}
  interval_s: 2.0                             # float; the plain value the client edits
  per_shot:
    position: middle                          # enum{middle, first}
  segment:
    duration_s: 4.0                           # float (legacy M2 mode, kept)
  temporal_window: 5                          # int; ±k frames saved
  temporal_offsets_for_vlm: [-2, -1, 1, 2]    # list[int]

detectors:                                    # T12
  gdino:
    enabled: true
    model_id: IDEA-Research/grounding-dino-base
    box_threshold: 0.20                       # float (M2 value)
    text_threshold: 0.20                      # float
    queries: [ "a television network watermark", …, "a brand" ]   # the 37 M2 phrases, verbatim
    min_crop_px: 16                           # int; boxes smaller than this in either dim are dropped at detection
  yolo:
    enabled: true
    weights: weights/best.pt                  # str
    imgsz: 640                                # int
    conf: 0.25                                # float (predict_yolov11_m2 default)
    iou_nms: 0.7                              # float (ultralytics default; internal NMS, intra-YOLO)
    class_agnostic: true                      # bool; must stay true (decision locked)
    log_predicted_class: true                 # bool; yolo_class / yolo_class_conf logged, drive nothing

merge:                                        # T13–T19
  enabled: true                               # bool; T16: the false path must run
  iou_threshold: 0.70                         # float
  containment_threshold: 0.90                 # float
  area_ratio_threshold: 0.80                  # float; min(A,B)/max(A,B)
  min_merge_area: 0.0004                      # float placeholder; fraction of frame area, tested on the smaller box (T14)
  representative: tighter                     # enum{tighter, higher_score}; locked: tighter
  small_box:                                  # [ASSUMPTION] centre-distance branch asked for on 27 Jul, proposed 29 Jul, absent from Rev 3 → shipped disabled
    enabled: false
    pixel_area_cutoff: 400                    # int
    centre_distance_threshold: 0.5            # float; fraction of the mean box diagonal
  log_failed_pairs: true                      # bool; T19 distributions
  legacy_filters_enabled: false               # bool; must remain false (T17); no key for 500px/0.5 exists

refinement:                                   # T20–T24
  enabled: true
  heuristic_1_provenance: true                # always on when enabled
  heuristic_2_ocr_tightening:
    enabled: true
    ocr_min_confidence: 0.40                  # float; text box must have EasyOCR conf ≥ this
    brand_match_min_score: 80                 # int; fuzzy score of the box text vs canonical+audio list to tighten on it
    padding_ratio: 0.10                       # float; pad the text box by this fraction of its size
    min_area_ratio_vs_loose: 0.05             # float; refined box must keep ≥ 5 % of the loose area
    max_area_ratio_vs_loose: 0.95             # float; must reduce area by ≥ 5 %
    min_side_px: 12                           # int
    max_aspect: 8.0                           # float
  heuristic_5_oversized_guard:
    enabled: true
    area_ratio: 0.45                          # float; box_area / frame_area; the Rule 4 input
  heuristics_3_4_enabled: false               # bool; out of scope, no other keys

ocr:                                          # T25–T29, A9
  languages: [en, it]                         # list[str]
  gpu: auto                                   # enum{auto,true,false}
  upscale_min_side: 80                        # int; crops with min side below this are upscaled
  upscale_factor: 3                           # int
  fallback_to_enlarged_crop: true             # bool (M2 behaviour, kept for the verification pass)
  normalise:
    lowercase: true
    strip_non_alphanumeric: true              # keep [a-z0-9 ] after unicode NFKD + accent strip
    collapse_whitespace: true
  matching:                                   # T27
    short_name_cutoff: 4                      # int; strings of length ≤ cutoff use exact matching
    exact_match_short: true                   # bool; T28: failing exact match => ocr_brand_match None
    general_fuzzy_threshold: 90               # int; the "general threshold" (T72)
    scorer: WRatio                            # enum{WRatio, token_set_ratio, smart_score}
    match_lists: [canonical, audio]           # list; T26: never faiss
  cheap_call_keep_boxes: true                 # bool; refinement-side call keeps quadrilaterals

faiss:
  index_path: FAISS/logo_index.faiss
  metadata_path: FAISS/metadata.json
  embedder: clip                              # enum{clip, siglip2}
  top_k: 5                                    # int
  overfetch_factor: 3                         # int (M2: k*3 then dedup by brand)
  similarity_threshold: 0.0                   # float; 0 = return all top-k
  rule7_top1_threshold: 0.97                  # float; Rule 7 (disabled)
  rule7_score_gap: 0.08                       # float; Rule 7 (disabled)
  rule12_top1_threshold: 0.97                 # float; Rule 12 second variant
  rule14_top1_threshold: 0.90                 # float; Rule 14
  abstain_threshold: null                     # float|null; set only after T54 evidence

audio:
  enabled: true
  whisper_model_id: openai/whisper-large-v3
  task: transcribe                            # enum{transcribe, translate}
  language: italian                           # str|null
  text_model_id: Qwen/Qwen2.5-7B-Instruct     # str; Qwen/Qwen3-8B on the GPU machine if available
  text_decoding: greedy                       # enum{greedy}; sampling is not allowed (determinism)
  brand_prompt: task2_italian                 # enum{m2, task2_italian}; the prompt variant

signals:                                      # T10 two-switch config
  tight_crop:       {include_in_vlm_prompt: true}
  enlarged_crop:    {include_in_vlm_prompt: true}
  full_frame:       {include_in_vlm_prompt: true}
  temporal_context: {include_in_vlm_prompt: true}
  audio:            {compute_and_log: true, include_in_vlm_prompt: true}
  faiss:            {compute_and_log: true, include_in_vlm_prompt: true}   # T35 (removal) is M4
  ocr:              {compute_and_log: true, include_in_vlm_prompt: true}

vlm:
  model_id: Qwen/Qwen2.5-VL-7B-Instruct
  quantisation: 4bit                          # enum{4bit, 8bit, none}
  max_pixels: 100352                          # int = 128*28*28
  max_new_tokens: 256
  decoding: greedy
  brand_mode: open                            # enum{open, closed}; locked: open for both passes
  unknown_explicit_option: true               # T67
  channel_watermark_line: true                # bool; uses video config channel name
  hinted_pass: true
  blind_pass: true                            # T33
  blind_pass_signals: [tight_crop, enlarged_crop, full_frame, temporal_context]  # image signals only
  log_hint_composition: true                  # T34
  scoring:                                    # T30–T32
    enabled: false                            # stays false until the distribution check passes
    labels: [logo, partial_logo, not_logo, uncertain]
    method: sequence_logprob                  # enum{sequence_logprob, first_token_softmax}; see 15 T30
    distribution_check_required: true

canonicalisation:                             # T37
  alias_first: true
  exact_then_fuzzy: true
  fuzzy_threshold: 90                         # int; onto canonical + audio brands
  short_name_cutoff: 4                        # int; same rule as OCR
  audio_brands_as_targets: true
  llm_merge:
    enabled: false                            # the M2 LLM pass, optional, logged
    model_id: Qwen/Qwen2.5-7B-Instruct

watermark:                                    # T47–T50
  margin: 0.02                                # float; fraction of frame size added around each region
  containment_threshold: 0.50                 # float; region∩box / min(region, box)
  regions_ref_required: false                 # bool; a video without regions runs with an empty set

router:                                       # T38–T46, T72
  enabled: true
  rules:                                      # enable flags; a disabled rule is skipped, never defaulted
    rule_1: true
    rule_1_fallback_logprob: false            # needs T31 score
    rule_2: true
    rule_3: true
    rule_4: true
    rule_5_review: true
    rule_5_reject: false                      # locked off for all of M3
    rule_6: true
    rule_7: false                             # pending T54
    rule_8: false                             # locked off for all of M3
    rule_9: true
    rule_10: true
    rule_11_review: true
    rule_11_reject: true                      # both variants adopted (24 Jul §5.4)
    rule_12: true
    rule_13: true
    rule_14: true
    rule_15: true
    rule_16: true
    rule_17: true
    rule_18: true
  thresholds:
    rule1_not_logo_confidence: 0.80
    rule1_logo_probability_max: 20
    rule4_area_ratio: 0.45                    # same value as refinement.heuristic_5; router reads its own key
    rule5_ocr_fuzzy: 90
    rule5_reject_ocr_fuzzy: 70
    rule6_ocr_fuzzy_threshold: 95             # T72; only Rule 6 reads this
    rule8_logo_probability: 95
    rule9_strong_ocr: 95
    rule11_ocr_fuzzy: 80
    rule12_ocr_fuzzy: 80
    rule12_ocr_fuzzy_second: 90
    rule13_ocr_fuzzy: 85
    rule16_ocr_fuzzy: 80
  dedup:                                      # T45
    dedup_iou_threshold: 0.50
    dedup_containment_threshold: 0.80
    dedup_area_ratio_threshold: 0.80
    post_vlm_min_merge_area_enabled: false    # locked: disabled at the post-VLM site
  scoring_layer:                              # 18 July §8; computed and logged, never decides
    compute_trust_risk: true
  brand_evidence:                             # T43
    compute: true
  firing_log: true                            # T46

output:                                       # T56, T58, T59, T73
  csv_name: rows.csv
  csv_paths: relative                         # enum{relative, absolute}; relative to the CSV directory
  annotated_frames: true                      # all sampled frames
  annotated_video: true
  video_codec: mp4v                           # str; re-encode to H.264 with ffmpeg if available
  draw: {brand: true, decision: true, reason_code: true, det_id: true}

brands:                                       # T5
  list_path: configs/brands/roboflow_2026-04-28.yaml
  require_hash_match: false                   # bool; if true, refuse to run when the stored hash ≠ recomputed

evaluation:                                   # T62–T66
  labels_path: null                           # str|null; the client label file
  iou_match_threshold: 0.50                   # float; T63
  roboflow_labels_dir: null                   # str; ~/Downloads/logo2.yolov11/train/labels for route three
  channel_classes: [rai, rai fiction, mediaset, rete 4, boing, nove, sport italia, sky, sky wifi,
                    discovery plus, infinity plus, paramount plus, m6, x factor, tim]   # [ASSUMPTION] reported with/without
```

## Per-video file `configs/videos/<video_id>.yaml`

```yaml
video_id: video_1                             # the identifier carried on every row (A7)
source: video_1.mp4                           # or frames_dir + pattern when ingest.mode = frames
channel: rai_sport                            # str|null; key into watermark.regions and known_channel_brands
channel_brand: rai sport                      # str|null; the canonical brand name of the bug (prompt line + Rule 5)
watermark_regions_ref: rai_sport_default      # str|null; key into the regions store (T50)
overrides: {}                                 # any pipeline.yaml key may be overridden per video
```

## Regions store `configs/watermark_regions.yaml` (T50, shape only)

```yaml
rai_sport_default:
  channel_brands: [rai sport, rai]            # known_channel_brands for Rule 5
  regions:                                    # normalised x1 y1 x2 y2, several allowed
    - [0.02, 0.03, 0.14, 0.11]
```

## Loader behaviour

- Unknown keys fail validation; every key above has a default except those marked `str` paths in `detectors.yolo.weights`, `faiss.*`, `brands.list_path`.
- `router.rules.*` false ⇒ the rule function is not called. A rule whose required input field is `None` on the row raises `MissingRouterInput` (27 July: no `.get()` defaults).
- The resolved config is written verbatim into `run_manifest.json` and hashed.
