# 05 — Frozen Row Schema (T9, T56)

This is the document that goes to the client for one round of confirmation. One row per detection that survives the merge; `rows.csv` and `router_input.jsonl` carry the same fields (the JSONL keeps lists as lists; the CSV serialises lists as JSON strings, as M2 did).

Column order in the CSV is exactly the order below. The columns marked **M2** exist today with the same name and meaning and are what `app/review_ui.py` reads (T73). Nothing below changes the meaning of an M2 column.

## A. Identity and geometry

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `video_id` | str | ingest | | From the per-video config (A7). |
| `run_id` | str | ingest | | |
| `det_id` | int | merge | M2 | Sequential over the run, assigned to representatives after the merge. Raw detections keep `raw_det_id`. |
| `frame_idx` | int | sampling | M2 | Video frame index, or the ordinal in frames mode. |
| `timecode_s` | float | sampling | M2 | `frame_idx / fps`; in frames mode `(ordinal−1)/assumed_fps`. |
| `shot_id` | int | shots | | Ordinal of the scene containing the frame (T57). |
| `frame_width`, `frame_height` | int | ingest | | |
| `det_score` | float | detect | M2 | Score of the representative box (its own detector's score). |
| `box_xyxy` | list[float] | merge | M2 | `loose_bbox` of the representative, absolute pixels. Kept under the M2 name. |
| `loose_bbox` | list[float] | merge | | Identical to `box_xyxy`; the name the correspondence uses. |
| `box_area_ratio` | float | merge | | `box_area / frame_area`. Rule 4 input. |
| `box_touches_edge` | bool | merge | | Any side within 2 px of the frame edge (from `task2`). |
| `detection_sources` | set[str] | merge | | JSON list in CSV; values from `{gdino, yolo}` (T15). |
| `merge_count` | int | merge | | Boxes absorbed into this representative (T15). Merged ⇔ `> 0`. |
| `merged_raw_det_ids` | list[int] | merge | | Provenance; `[]` when `merge_count = 0`. **[ASSUMPTION]** replaces the `union_bbox` idea; the union is recoverable from `raw_detections.jsonl`. |
| `gdino_label` | str|null | detect | | The GDINO phrase that produced the box (M2 `label`). |
| `yolo_class`, `yolo_class_conf` | str|null, float|null | detect | | Logged, drive nothing (T12). |

## B. Crop refinement (T20–T23)

| Field | Type | Stage | Notes |
|-------|------|-------|-------|
| `refined_bbox` | list[float]|null | refine | Candidate tight box or null. |
| `refinement_method` | enum | refine | `none`, `ocr_guided`, `oversized_guard` |
| `refinement_status` | enum | refine | `applied`, `fallback_to_loose`, `rejected_safety_check`, `oversized_region`, `no_candidate` |
| `selected_bbox` | list[float] | refine | `refined_bbox` when status is `applied`, else `loose_bbox`. What OCR, FAISS and the VLM crop see. |
| `has_valid_tight_bbox` | bool | refine | `refinement_status == applied` (T23). Rule 4 input. |
| `refinement_area_ratio` | float|null | refine | `area(refined)/area(loose)`. |

## C. Crop paths (review UI)

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `crop_path` | str | output | M2 | `crops/det_XXXXXX/crop.jpg`, cut from `selected_bbox`. |
| `enlarged_crop_path` | str | output | M2 | 1.5× of `selected_bbox`, clamped. |
| `loose_crop_path` | str | output | | Cut from `loose_bbox` when it differs from `selected_bbox`, else same file. |
| `frame_path` | str | output | M2 | `frames/frame_XXXXXX.jpg` |
| `annotated_frame_path` | str | output | M2 | `annotated_frames/frame_XXXXXX.jpg` |

All paths are relative to the CSV's directory (T73).

## D. OCR verification pass (T25–T29)

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `ocr_text` | str | ocr | M2 | Normalised, lines joined with " | " as in M2. Empty string when nothing read. |
| `ocr_text_raw` | str | ocr | | Un-normalised lines. |
| `ocr_source` | enum|null | ocr | M2 | `tight_crop`, `enlarged_crop`, null |
| `ocr_max_confidence` | float|null | ocr | | Best EasyOCR line confidence. |
| `ocr_brand_match` | str|null | ocr | | Canonical spelling of the best match, or null (T28). |
| `ocr_fuzzy_score` | int | ocr | | 0–100; 0 when no text. Rule inputs. |
| `ocr_match_list` | enum|null | ocr | | `canonical`, `audio`, null (T29). |
| `ocr_match_mode` | enum|null | ocr | | `exact_short`, `fuzzy` (T27). |

## E. FAISS

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `faiss_top1` | str|null | faiss_query | M2 | Brand string as stored in the index. |
| `faiss_top1_score` | float|null | faiss_query | M2 | |
| `faiss_top5_brands` | list[str] | faiss_query | M2 | |
| `faiss_top5_scores` | list[float] | faiss_query | M2 | |
| `faiss_top1_brand_canonical` | str|null | canonicalise | | Mapped onto the canonical list where possible. |
| `faiss_score_gap` | float|null | faiss_query | | top1 − top2 (brand-deduplicated). Rule 7 input. |
| `faiss_agrees_with_vlm` | bool|null | canonicalise | | `canonical(faiss_top1) == canonical_brand`. |
| `brand_in_database` | bool|null | canonicalise | | `canonical_brand` (or an alias) appears among index brand strings. |

## F. Audio

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `audio_brands` | list[str] | audio | M2 | Video-level list, repeated on every row (M2 convention). |
| `audio_brands_canonical` | list[str] | canonicalise | | Mapped where possible. |
| `brand_from_audio` | bool | canonicalise | | `canonical_brand` ∈ `audio_brands_canonical`. Rules 5, 11. |
| `audio_signal_status` | enum | audio | | `ok`, `empty`, `signal_unavailable` (frames mode), `disabled` |

## G. VLM, hinted pass

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `is_logo` | enum|null | vlm | M2 | `logo`, `partial_logo`, `not_logo`, null (unparsed). Kept under the M2 name; `vlm_decision` is its alias in the router. |
| `vlm_decision` | enum|null | vlm | | Same value as `is_logo` (the correspondence name). |
| `brand` | str|null | vlm | M2 | Raw VLM brand string, `UNKNOWN`, or null. |
| `vlm_brand` | str|null | vlm | | Same as `brand`. |
| `is_truncated` | bool|null | vlm | M2 | |
| `logo_probability` | int|null | vlm | M2 | The generated 0–100 field (M2). Kept for the UI. |
| `confidence` | int|null | vlm | M2 | Generated 0–100. |
| `signals_used`, `signals_used_raw` | list[str] | vlm | M2 | |
| `reasoning` | str|null | vlm | M2 | |
| `vlm_parse_ok` | bool | vlm | | False when the JSON did not parse. |
| `logo_probability_scored` | float|null | vlm_scoring | | T31 continuous score, null until enabled. |
| `not_logo_confidence` | float|null | vlm_scoring | | T31, null until enabled. Rule 1 primary variant input. |
| `decision_scores` | map|null | vlm_scoring | | `{logo, partial_logo, not_logo, uncertain}` renormalised. |

`brand_type`, `is_brand_identifier` and `evidence_text` from the 18 July "enhanced fields" are **not** in the M2 prompt and are not added in M3 (the prompt is not redesigned). Rules 2 and 3, which read them, are implemented but cannot fire; see `11`.

## H. VLM, blind pass (T33)

| Field | Type | Stage | Notes |
|-------|------|-------|-------|
| `vlm_decision_blind` | enum|null | vlm | |
| `vlm_brand_blind` | str|null | vlm | |
| `vlm_brand_blind_canonical` | str|null | canonicalise | |
| `blind_hinted_brand_agree` | bool|null | canonicalise | For T64. |
| `hint_composition` | list[map] | vlm | T34: `[{item, source}]`, source ∈ `{audio, faiss, ocr, watermark}`; JSON string in CSV. |
| `hinted_brand_in_faiss_hints` | bool|null | canonicalise | For T64 FAISS-origin attribution. |

## I. Canonicalisation (T37)

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `canonical_brand` | str|null | canonicalise | | Deterministic map of `vlm_brand`; `UNKNOWN` stays `UNKNOWN`; null for `not_logo`. |
| `brand_canonical` | str|null | canonicalise | M2 | Same value under the M2 column name. |
| `canonical_match_mode` | enum|null | canonicalise | | `alias`, `exact`, `fuzzy`, `audio`, `unmatched`, `llm_merge` |
| `canonical_match_score` | int|null | canonicalise | | |
| `brand_list_version`, `brand_list_hash` | str | canonicalise | | From the snapshot (T5). |

## J. Watermark (T47–T48)

| Field | Type | Stage | Notes |
|-------|------|-------|-------|
| `channel` | str|null | ingest | Per-video config. |
| `channel_brand_canonical` | str|null | ingest | |
| `watermark_regions_available` | bool | watermark | False when the video has no regions. |
| `watermark_overlap` | bool|null | watermark | Null when no regions (rule reads null → Rule 5 uses the 29 July no-region form). |
| `watermark_overlap_ratio` | float|null | watermark | The containment value. |

## K. Router (T38–T46)

| Field | Type | Stage | Notes |
|-------|------|-------|-------|
| `brand_evidence` | enum | router_input | `ocr_confirmed`, `faiss_confirmed`, `both`, `none` (T43). |
| `has_conflicting_ocr` | bool | router_input | Derived per 18 July. |
| `strong_visual_match` | bool | router_input | |
| `audio_only_prediction`, `faiss_only_prediction` | bool | router_input | |
| `router_decision` | enum | router | `ACCEPT`, `REVIEW`, `REJECT`, `UNKNOWN_LOGO`, `DEDUPED` |
| `router_reason` | str | router | A member of the reason-code enum (T44). |
| `reason_code` | str | router | Same as `router_reason` (the name T73 uses for the filter). |
| `router_rule` | int|null | router | 1–18, or null for the default. |
| `router_confidence`, `risk_score`, `trust_score` | float, int, int | router | The five-value tuple's remaining members (T44); scoring layer logged only. |
| `review_priority` | enum|null | router | `low`, `normal`, `high` |
| `dedup_cluster_id` | int|null | router | Rules 17/18. |
| `dedup_representative` | bool|null | router | |
| `rules_matched_preempted` | list[int] | router | Rules whose condition held but were pre-empted by ordering (T46). |

## L. Run provenance and review

| Field | Type | Stage | M2 | Notes |
|-------|------|-------|----|-------|
| `config_hash` | str | ingest | | |
| `experiment` | str | ingest | M2 | Run label; defaults to `run_id`. |
| `brand_mode` | str | vlm | M2 | `open` |
| `enabled_signals` | list[str] | vlm | M2 | Signals in the hinted prompt. |
| `label` | str|null | review UI | M2 | `TP`, `FP`, `FN`, `wrong_brand`, null. Written by the review screen. |
| `enrolled_to_faiss` | bool | review UI | | Set by T75. |

## M. Rows for frames without detections

The CSV has no row for a frame with no detections (the review UI is per detection). Frames are listed in `sampled_frames.json` and every one gets an annotated image (T58). `merge_counters.json` carries per-frame counts including zero.

## N. Brand-list procedure (T5, repeated here because the client asked for it in the schema document)

See `06`. Summary: edit `configs/brands/<version>.yaml`, add the entry with `canonical` and `aliases`, set `version` to the date of the change, run `python -m logo_pipeline brands hash` to recompute the content hash, commit. Every run output carries `brand_list_version` and `brand_list_hash`. No code change.
