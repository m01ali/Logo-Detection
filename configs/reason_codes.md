# Router reason codes (T44)

Generated from `logo_pipeline/stages/router/reason_codes.py` — do not edit by hand (`python scripts/export_reason_codes.py`).

Evaluation: blocks 1-6 in order; within a block rules in numeric order; the first rule whose condition holds decides. Disabled rules are skipped. Conditions below are the **v1** forms: with `router.enhanced_vlm_fields_available: false` Rules 2 and 3 cannot fire and the `brand_type == brand` / `is_brand_identifier is True` clauses of Rules 6, 7, 8, 15 and 16 are omitted.

## Block 1 — Rejects

### `high_confidence_not_logo`

- **Rule:** 1
- **Decision:** REJECT
- **Condition (v1):** vlm_decision == not_logo and not_logo_confidence >= rule1_not_logo_confidence (0.80)
- **Fields read:** `vlm_decision`, `not_logo_confidence`
- **Reviewer should check:** Open the crop: is it really not a logo? A wrong REJECT is invisible in the default queue, so sample these.
- **Note:** not_logo_confidence comes from the generated `confidence/100` until router.not_logo_confidence_source is 'scored'.

### `low_logo_probability_not_logo`

- **Rule:** 1 (fallback)
- **Decision:** REJECT
- **Condition (v1):** vlm_decision == not_logo and logo_probability_scored <= rule1_logo_probability_max (20)
- **Fields read:** `vlm_decision`, `logo_probability_scored`
- **Reviewer should check:** Same as Rule 1: confirm the crop is not a logo.
- **Note:** Disabled until T31/T32 pass (rule_1_fallback_logprob).

### `non_brand_content_generic_word`

- **Rule:** 2
- **Decision:** REJECT
- **Condition (v1):** brand_type == generic_word
- **Fields read:** `brand_type`
- **Reviewer should check:** Confirm the text is a common word rather than a brand.
- **Note:** Unreachable in v1: brand_type is not produced.

### `non_brand_content_ingredient`

- **Rule:** 2
- **Decision:** REJECT
- **Condition (v1):** brand_type == ingredient
- **Fields read:** `brand_type`
- **Reviewer should check:** Confirm the text is an ingredient name.
- **Note:** Unreachable in v1: brand_type is not produced.

### `non_brand_content_slogan`

- **Rule:** 2
- **Decision:** REJECT
- **Condition (v1):** brand_type == slogan
- **Fields read:** `brand_type`
- **Reviewer should check:** Confirm the text is a slogan.
- **Note:** Unreachable in v1: brand_type is not produced.

### `non_brand_content_ui_icon`

- **Rule:** 2
- **Decision:** REJECT
- **Condition (v1):** brand_type == ui_icon
- **Fields read:** `brand_type`
- **Reviewer should check:** Confirm the region is a UI icon.
- **Note:** Unreachable in v1: brand_type is not produced.

### `non_brand_content_sign`

- **Rule:** 2
- **Decision:** REJECT
- **Condition (v1):** brand_type == sign
- **Fields read:** `brand_type`
- **Reviewer should check:** Confirm the region is a generic sign.
- **Note:** Unreachable in v1: brand_type is not produced.

### `not_a_brand_identifier`

- **Rule:** 3
- **Decision:** REJECT
- **Condition (v1):** is_brand_identifier is False and vlm_decision in {not_logo, partial_logo, uncertain}
- **Fields read:** `is_brand_identifier`, `vlm_decision`
- **Reviewer should check:** Confirm the text does not identify a brand.
- **Note:** Unreachable in v1: is_brand_identifier is not produced.

### `oversized_scene_region_without_tight_logo`

- **Rule:** 4
- **Decision:** REJECT
- **Condition (v1):** box_area_ratio >= rule4_area_ratio (0.45) and not has_valid_tight_bbox
- **Fields read:** `box_area_ratio`, `has_valid_tight_bbox`
- **Reviewer should check:** Check whether a real logo was inside the scene-sized box that refinement could not tighten.
- **Note:** With a valid tight box the row continues (the crops already use it).

### `likely_channel_audio_contamination`

- **Rule:** 5 (reject)
- **Decision:** REJECT
- **Condition (v1):** canonical_brand is a channel brand and watermark_overlap is False and brand_from_audio and ocr_fuzzy_score < rule5_reject_ocr_fuzzy (70)
- **Fields read:** `canonical_brand`, `watermark_overlap`, `brand_from_audio`, `ocr_fuzzy_score`
- **Reviewer should check:** Confirm the channel name came from the commentary, not the picture.
- **Note:** Disabled for all of M3 (rule_5_reject).

## Block 2 — Accepts

### `clear_logo_with_strong_ocr_match`

- **Rule:** 6
- **Decision:** ACCEPT
- **Condition (v1):** vlm_decision == logo and is_truncated is False and ocr_fuzzy_score >= rule6_ocr_fuzzy_threshold (95) and ocr_brand_c == canonical_brand
- **Fields read:** `vlm_decision`, `is_truncated`, `ocr_fuzzy_score`, `ocr_brand_match`, `canonical_brand`
- **Reviewer should check:** Spot-check that OCR read the same text the VLM named; this is the only auto-accept in v1.
- **Note:** brand_type == brand / is_brand_identifier clauses omitted in v1.

### `clear_logo_with_strong_visual_database_match`

- **Rule:** 7
- **Decision:** ACCEPT
- **Condition (v1):** vlm_decision == logo and is_truncated is False and strong_visual_match and not has_conflicting_ocr
- **Fields read:** `vlm_decision`, `is_truncated`, `faiss_agrees_with_vlm`, `faiss_top1_score`, `faiss_score_gap`, `has_conflicting_ocr`
- **Reviewer should check:** Compare the crop with the FAISS reference for the brand.
- **Note:** Disabled pending T54 (rule_7).

### `high_confidence_visual_logo_no_conflicts`

- **Rule:** 8
- **Decision:** ACCEPT
- **Condition (v1):** vlm_decision == logo and logo_probability_scored >= rule8_logo_probability (95) and is_truncated is False and not has_conflicting_ocr and not audio_only_prediction and not faiss_only_prediction and box_area_ratio < 0.45
- **Fields read:** `vlm_decision`, `logo_probability_scored`, `is_truncated`, `has_conflicting_ocr`, `audio_only_prediction`, `faiss_only_prediction`, `box_area_ratio`
- **Reviewer should check:** Confirm the VLM's visual judgement without any supporting signal.
- **Note:** Locked off for all of M3 (rule_8).

## Block 3 — Review and conflict

### `channel_brand_outside_watermark_region`

- **Rule:** 5
- **Decision:** REVIEW
- **Condition (v1):** canonical_brand is a channel brand and watermark_overlap is False and not (ocr_fuzzy_score >= 90 and ocr_brand_c == canonical_brand)
- **Fields read:** `canonical_brand`, `watermark_overlap`, `ocr_fuzzy_score`, `ocr_brand_match`, `ocr_text`, `faiss_top1`, `faiss_top1_score`, `brand_from_audio`
- **Reviewer should check:** Is this the channel bug in a place the region misses, or the channel name appearing elsewhere in the scene?
- **Note:** The row already carries ocr_text, ocr_fuzzy_score, faiss_top1, faiss_top1_score and brand_from_audio (T49).

### `channel_brand_needs_watermark_check`

- **Rule:** 5
- **Decision:** REVIEW
- **Condition (v1):** canonical_brand is a channel brand and watermark_overlap is None and ocr_fuzzy_score < rule5_ocr_fuzzy (90)
- **Fields read:** `canonical_brand`, `watermark_overlap`, `ocr_fuzzy_score`
- **Reviewer should check:** The video has no watermark regions: decide by eye whether this is the broadcaster's bug.
- **Note:** watermark_overlap is null when the video has no region set (never treated as 'outside').

### `partial_logo_with_strong_ocr_support`

- **Rule:** 9a
- **Decision:** REVIEW
- **Condition (v1):** vlm_decision == partial_logo and ocr_fuzzy_score >= rule9_strong_ocr (95) and ocr_brand_c == canonical_brand and is_truncated is False
- **Fields read:** `vlm_decision`, `ocr_fuzzy_score`, `ocr_brand_match`, `canonical_brand`, `is_truncated`
- **Reviewer should check:** Quick confirm: OCR supports the brand, the crop is a partial view.
- **Note:** review_priority = low.

### `partial_logo_default_review`

- **Rule:** 9b
- **Decision:** REVIEW
- **Condition (v1):** vlm_decision == partial_logo
- **Fields read:** `vlm_decision`
- **Reviewer should check:** partial_logo is the least reliable VLM class: judge whether the visible part identifies the brand.

### `truncated_logo_requires_review`

- **Rule:** 10
- **Decision:** REVIEW
- **Condition (v1):** is_truncated is True
- **Fields read:** `is_truncated`
- **Reviewer should check:** The visible fragment may mislead: check the full frame for the complete logo.

### `weak_visual_crop_rescued_by_audio_hint`

- **Rule:** 11 (reject)
- **Decision:** REJECT
- **Condition (v1):** brand_from_audio and vlm_decision in {partial_logo, uncertain} and ocr_text == '' and not strong_visual_match
- **Fields read:** `brand_from_audio`, `vlm_decision`, `ocr_text`, `strong_visual_match`
- **Reviewer should check:** Confirm the brand was named only because the audio suggested it, over a crop with no readable content.
- **Note:** Evaluated before the REVIEW variant. Rule 9 pre-empts partial_logo rows, so in practice this sees `uncertain`.

### `audio_only_brand_prediction`

- **Rule:** 11 (review)
- **Decision:** REVIEW
- **Condition (v1):** brand_from_audio and ocr_fuzzy_score < rule11_ocr_fuzzy (80) and not strong_visual_match
- **Fields read:** `brand_from_audio`, `ocr_fuzzy_score`, `strong_visual_match`
- **Reviewer should check:** Is the brand actually visible, or only mentioned in the commentary?

### `faiss_only_or_faiss_conflicting_prediction`

- **Rule:** 12a
- **Decision:** REVIEW
- **Condition (v1):** brand_from_faiss and ocr_fuzzy_score < rule12_ocr_fuzzy (80) and not faiss_agrees_with_vlm
- **Fields read:** `brand_from_faiss`, `ocr_fuzzy_score`, `faiss_agrees_with_vlm`
- **Reviewer should check:** The VLM's brand came from a FAISS hint that FAISS itself ranks lower: which one is right?

### `faiss_score_not_discriminative_enough`

- **Rule:** 12b
- **Decision:** REVIEW
- **Condition (v1):** brand_from_faiss and faiss_top1_score < rule12_top1_threshold (0.97) and ocr_fuzzy_score < rule12_ocr_fuzzy_second (90)
- **Fields read:** `brand_from_faiss`, `faiss_top1_score`, `ocr_fuzzy_score`
- **Reviewer should check:** FAISS supports the brand but not decisively, and OCR does not back it up.

### `ocr_vlm_brand_conflict`

- **Rule:** 13
- **Decision:** REVIEW
- **Condition (v1):** has_conflicting_ocr (ocr_brand_c != canonical_brand, both real, ocr_fuzzy_score >= rule13_ocr_fuzzy (85))
- **Fields read:** `ocr_brand_match`, `canonical_brand`, `ocr_fuzzy_score`
- **Reviewer should check:** OCR and the VLM name different brands: read the text in the crop.

### `faiss_vlm_brand_conflict`

- **Rule:** 14
- **Decision:** REVIEW
- **Condition (v1):** faiss_top1_brand_canonical != canonical_brand (both real) and faiss_top1_score >= rule14_top1_threshold (0.90)
- **Fields read:** `faiss_top1_brand_canonical`, `canonical_brand`, `faiss_top1_score`
- **Reviewer should check:** FAISS confidently names a different brand: compare with the reference logos.

## Block 4 — UNKNOWN_LOGO

### `logo_like_but_brand_unknown`

- **Rule:** 15
- **Decision:** UNKNOWN_LOGO
- **Condition (v1):** vlm_decision == logo and canonical_brand in (None, '', UNKNOWN)
- **Fields read:** `vlm_decision`, `canonical_brand`
- **Reviewer should check:** A logo the VLM could not name: candidate for enrolment or the add-a-brand procedure.
- **Note:** brand_type / is_brand_identifier clauses omitted in v1.

### `readable_brand_like_text_not_in_database`

- **Rule:** 16
- **Decision:** UNKNOWN_LOGO
- **Condition (v1):** vlm_decision == logo and ocr_text != '' and ocr_fuzzy_score < rule16_ocr_fuzzy (80)
- **Fields read:** `vlm_decision`, `ocr_text`, `ocr_fuzzy_score`
- **Reviewer should check:** Readable text that matches no known brand: a local sponsor, or a brand missing from the list.
- **Note:** brand_type clause omitted in v1.

## Block 5 — Dedup

### `duplicate_same_brand_box`

- **Rule:** 17
- **Decision:** DEDUPED
- **Condition (v1):** same canonical_brand, iou >= 0.50 or containment >= 0.80 (and area_ratio >= 0.80)
- **Fields read:** `canonical_brand`, `selected_bbox`, `router_decision`
- **Reviewer should check:** Confirm the kept representative covers the logo.
- **Note:** The cluster representative keeps its own decision.

### `subset_box_same_brand`

- **Rule:** 18
- **Decision:** DEDUPED
- **Condition (v1):** same canonical_brand, box mostly contained (>= 0.80) in a non-oversized box
- **Fields read:** `canonical_brand`, `selected_bbox`, `box_area_ratio`
- **Reviewer should check:** Confirm the containing box is the intended one.

### `oversized_container_replaced_by_tight_box`

- **Rule:** 18
- **Decision:** DEDUPED
- **Condition (v1):** same canonical_brand, oversized box (>= 0.45 of the frame) containing a box with a valid tight bbox
- **Fields read:** `canonical_brand`, `selected_bbox`, `box_area_ratio`, `has_valid_tight_bbox`
- **Reviewer should check:** Confirm the tight box (kept) is the logo and the oversized one was scene context.

## Block 6 — Default

### `default_uncertain_or_insufficient_evidence`

- **Rule:** D
- **Decision:** REVIEW
- **Condition (v1):** no rule above fired
- **Fields read:** —
- **Reviewer should check:** Default queue. A not_logo with mid confidence lands here by design (24 July).
- **Note:** review_priority = normal.

### `vlm_output_unparsed`

- **Rule:** X
- **Decision:** REVIEW
- **Condition (v1):** vlm_parse_ok is False and no rule above fired
- **Fields read:** `vlm_parse_ok`
- **Reviewer should check:** The VLM's JSON did not parse: inspect `raw` in vlm/hinted.jsonl.
- **Note:** [ASSUMPTION] block-6 special case.
