# Frozen Row Schema (T9, T56)

Generated from `logo_pipeline/schema.py` by `scripts/export_schema_doc.py`; do not hand-edit — edit the dataclass and regenerate.

One row per detection that survives the merge. Column order in `rows.csv` is exactly the order below. Columns marked ★ exist in M2 today with the same name and meaning (`app/review_ui.py` reads them).

## A. Identity and geometry

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `video_id` | str |  | From the per-video config (A7). |
| `run_id` | str |  |  |
| `det_id` | int | ★ | Sequential over the run, assigned to representatives after the merge. |
| `frame_idx` | int | ★ | Video frame index, or the ordinal in frames mode. |
| `timecode_s` | float | ★ | frame_idx / fps; in frames mode (ordinal−1)/assumed_fps. |
| `shot_id` | Optional[int] |  | Ordinal of the scene containing the frame (T57). |
| `frame_width` | int |  |  |
| `frame_height` | int |  |  |
| `det_score` | float | ★ | Score of the representative box (its own detector's score). |
| `box_xyxy` | list[float] | ★ | loose_bbox of the representative, absolute pixels. |
| `loose_bbox` | list[float] |  | Identical to box_xyxy; the name the correspondence uses. |
| `box_area_ratio` | float |  | box_area / frame_area. Rule 4 input. |
| `box_touches_edge` | bool |  | Any side within 2 px of the frame edge (from task2). |
| `detection_sources` | list[str] |  | Values from {gdino, yolo} (T15). |
| `merge_count` | int |  | Boxes absorbed into this representative (T15). Merged ⇔ > 0. |
| `merged_raw_det_ids` | list[int] |  | Provenance; [] when merge_count = 0. |
| `gdino_label` | Optional[str] |  | The GDINO phrase that produced the box (M2 label). |
| `yolo_class` | Optional[str] |  | Logged, drives nothing (T12). |
| `yolo_class_conf` | Optional[float] |  |  |

## B. Crop refinement (T20–T23)

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `refined_bbox` | Optional[list[float]] |  | Candidate tight box or null. |
| `refinement_method` | str |  | none | ocr_guided | oversized_guard |
| `refinement_status` | str |  | applied | fallback_to_loose | rejected_safety_check | oversized_region | no_candidate |
| `selected_bbox` | list[float] |  | refined_bbox when applied, else loose_bbox. |
| `has_valid_tight_bbox` | bool |  | refinement_status == applied (T23). Rule 4 input. |
| `refinement_area_ratio` | Optional[float] |  | area(refined)/area(loose). |

## C. Crop paths (review UI)

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `crop_path` | str | ★ | crops/det_XXXXXX/crop.jpg, cut from selected_bbox. |
| `enlarged_crop_path` | str | ★ | 1.5x of selected_bbox, clamped. |
| `loose_crop_path` | str |  | Cut from loose_bbox when it differs from selected_bbox, else same file. |
| `frame_path` | str | ★ | frames/frame_XXXXXX.jpg |
| `annotated_frame_path` | Optional[str] | ★ | annotated_frames/frame_XXXXXX.jpg |

## D. OCR verification pass (T25–T29)

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `ocr_text` | str | ★ | Normalised, lines joined with ' | ' as in M2. |
| `ocr_text_raw` | str |  | Un-normalised lines. |
| `ocr_source` | Optional[str] | ★ | tight_crop, enlarged_crop, null |
| `ocr_max_confidence` | Optional[float] |  | Best EasyOCR line confidence. |
| `ocr_brand_match` | Optional[str] |  | Canonical spelling of the best match, or null (T28). |
| `ocr_fuzzy_score` | int |  | 0-100; 0 when no text. Rule inputs. |
| `ocr_match_list` | Optional[str] |  | canonical, audio, null (T29). |
| `ocr_match_mode` | Optional[str] |  | exact_short, fuzzy (T27). |

## E. FAISS

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `faiss_top1` | Optional[str] | ★ | Brand string as stored in the index. |
| `faiss_top1_score` | Optional[float] | ★ |  |
| `faiss_top5_brands` | list[str] | ★ |  |
| `faiss_top5_scores` | list[float] | ★ |  |
| `faiss_top1_brand_canonical` | Optional[str] |  | Mapped onto the canonical list where possible. |
| `faiss_score_gap` | Optional[float] |  | top1 − top2 (brand-deduplicated). Rule 7 input. |
| `faiss_agrees_with_vlm` | Optional[bool] |  | canonical(faiss_top1) == canonical_brand. |
| `brand_in_database` | Optional[bool] |  | canonical_brand (or an alias) appears among index brand strings. |

## F. Audio

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `audio_brands` | list[str] | ★ | Video-level list, repeated on every row (M2 convention). |
| `audio_brands_canonical` | list[str] |  | Mapped where possible. |
| `brand_from_audio` | bool |  | canonical_brand in audio_brands_canonical. Rules 5, 11. |
| `audio_signal_status` | str |  | ok, empty, signal_unavailable (frames mode), disabled |

## G. VLM, hinted pass

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `is_logo` | Optional[str] | ★ | logo, partial_logo, not_logo, null (unparsed). |
| `vlm_decision` | Optional[str] |  | Same value as is_logo (the correspondence name). |
| `brand` | Optional[str] | ★ | Raw VLM brand string, UNKNOWN, or null. |
| `vlm_brand` | Optional[str] |  | Same as brand. |
| `is_truncated` | Optional[bool] | ★ |  |
| `logo_probability` | Optional[int] | ★ | The generated 0-100 field (M2). Kept for the UI. |
| `confidence` | Optional[int] | ★ | Generated 0-100. |
| `signals_used` | list[str] | ★ |  |
| `signals_used_raw` | list[str] | ★ |  |
| `reasoning` | Optional[str] | ★ |  |
| `vlm_parse_ok` | bool |  | False when the JSON did not parse. |
| `logo_probability_scored` | Optional[float] |  | T31 continuous score, null until enabled. |
| `not_logo_confidence` | Optional[float] |  | T31, null until enabled. Rule 1 primary variant input. |
| `decision_scores` | Optional[dict] |  | {logo, partial_logo, not_logo, uncertain} renormalised. |

## H. VLM, blind pass (T33)

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `vlm_decision_blind` | Optional[str] |  |  |
| `vlm_brand_blind` | Optional[str] |  |  |
| `vlm_brand_blind_canonical` | Optional[str] |  |  |
| `blind_hinted_brand_agree` | Optional[bool] |  | For T64. |
| `hint_composition` | list[dict] |  | [{item, source}], source in {audio, faiss, ocr, watermark}. |
| `hinted_brand_in_faiss_hints` | Optional[bool] |  | For T64 FAISS-origin attribution. |

## I. Canonicalisation (T37)

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `canonical_brand` | Optional[str] |  | Deterministic map of vlm_brand; UNKNOWN stays UNKNOWN; null for not_logo. |
| `brand_canonical` | Optional[str] | ★ | Same value under the M2 column name. |
| `canonical_match_mode` | Optional[str] |  | alias, exact, fuzzy, audio, unmatched, llm_merge |
| `canonical_match_score` | Optional[int] |  |  |
| `brand_list_version` | str |  | From the snapshot (T5). |
| `brand_list_hash` | str |  |  |

## J. Watermark (T47–T48)

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `channel` | Optional[str] |  | Per-video config. |
| `channel_brand_canonical` | Optional[str] |  |  |
| `watermark_regions_available` | bool |  | False when the video has no regions. |
| `watermark_overlap` | Optional[bool] |  | Null when no regions (rule reads null). |
| `watermark_overlap_ratio` | Optional[float] |  | The containment value. |

## K. Router (T38–T46)

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `brand_evidence` | str |  | ocr_confirmed, faiss_confirmed, both, none (T43). |
| `has_conflicting_ocr` | bool |  | Derived per 18 July. |
| `strong_visual_match` | bool |  |  |
| `audio_only_prediction` | bool |  |  |
| `faiss_only_prediction` | bool |  |  |
| `router_decision` | Optional[str] |  | ACCEPT, REVIEW, REJECT, UNKNOWN_LOGO, DEDUPED |
| `router_reason` | Optional[str] |  | A member of the reason-code enum (T44). |
| `reason_code` | Optional[str] |  | Same as router_reason (the name T73 uses for the filter). |
| `router_rule` | Optional[int] |  | 1-18, or null for the default. |
| `router_confidence` | Optional[float] |  |  |
| `risk_score` | Optional[int] |  |  |
| `trust_score` | Optional[int] |  |  |
| `review_priority` | Optional[str] |  | low, normal, high |
| `dedup_cluster_id` | Optional[int] |  | Rules 17/18. |
| `dedup_representative` | Optional[bool] |  |  |
| `rules_matched_preempted` | list[int] |  | Rules whose condition held but were pre-empted (T46). |

## L. Run provenance and review

| Field | Type | M2 | Notes |
|-------|------|----|-------|
| `config_hash` | str |  |  |
| `experiment` | str | ★ | Run label; defaults to run_id. |
| `brand_mode` | str | ★ | open |
| `enabled_signals` | list[str] | ★ | Signals in the hinted prompt. |
| `label` | Optional[str] | ★ | TP, FP, FN, wrong_brand, null. Written by the review screen. |
| `enrolled_to_faiss` | bool |  | Set by T75. |

**Total fields:** 103

## CSV column order

```
video_id
run_id
det_id
frame_idx
timecode_s
shot_id
frame_width
frame_height
det_score
box_xyxy
loose_bbox
box_area_ratio
box_touches_edge
detection_sources
merge_count
merged_raw_det_ids
gdino_label
yolo_class
yolo_class_conf
refined_bbox
refinement_method
refinement_status
selected_bbox
has_valid_tight_bbox
refinement_area_ratio
crop_path
enlarged_crop_path
loose_crop_path
frame_path
annotated_frame_path
ocr_text
ocr_text_raw
ocr_source
ocr_max_confidence
ocr_brand_match
ocr_fuzzy_score
ocr_match_list
ocr_match_mode
faiss_top1
faiss_top1_score
faiss_top5_brands
faiss_top5_scores
faiss_top1_brand_canonical
faiss_score_gap
faiss_agrees_with_vlm
brand_in_database
audio_brands
audio_brands_canonical
brand_from_audio
audio_signal_status
is_logo
vlm_decision
brand
vlm_brand
is_truncated
logo_probability
confidence
signals_used
signals_used_raw
reasoning
vlm_parse_ok
logo_probability_scored
not_logo_confidence
decision_scores
vlm_decision_blind
vlm_brand_blind
vlm_brand_blind_canonical
blind_hinted_brand_agree
hint_composition
hinted_brand_in_faiss_hints
canonical_brand
brand_canonical
canonical_match_mode
canonical_match_score
brand_list_version
brand_list_hash
channel
channel_brand_canonical
watermark_regions_available
watermark_overlap
watermark_overlap_ratio
brand_evidence
has_conflicting_ocr
strong_visual_match
audio_only_prediction
faiss_only_prediction
router_decision
router_reason
reason_code
router_rule
router_confidence
risk_score
trust_score
review_priority
dedup_cluster_id
dedup_representative
rules_matched_preempted
config_hash
experiment
brand_mode
enabled_signals
label
enrolled_to_faiss
```
