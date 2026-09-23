"""The reason-code enum (T44, ``spec docs/11`` Part D) and the reference document
generated from it (``configs/reason_codes.md``, a deliverable).

Each code carries the rule it belongs to, its block, the decision it produces,
the condition text (v1, i.e. with the enhanced-VLM-field clauses omitted), the
fields the rule reads and what a reviewer should check. The generated document
is rendered from this table so the two cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

ACCEPT, REVIEW, REJECT, UNKNOWN_LOGO, DEDUPED = "ACCEPT", "REVIEW", "REJECT", "UNKNOWN_LOGO", "DEDUPED"


class ReasonCode(str, Enum):
    HIGH_CONFIDENCE_NOT_LOGO = "high_confidence_not_logo"
    LOW_LOGO_PROBABILITY_NOT_LOGO = "low_logo_probability_not_logo"
    NON_BRAND_CONTENT_GENERIC_WORD = "non_brand_content_generic_word"
    NON_BRAND_CONTENT_INGREDIENT = "non_brand_content_ingredient"
    NON_BRAND_CONTENT_SLOGAN = "non_brand_content_slogan"
    NON_BRAND_CONTENT_UI_ICON = "non_brand_content_ui_icon"
    NON_BRAND_CONTENT_SIGN = "non_brand_content_sign"
    NOT_A_BRAND_IDENTIFIER = "not_a_brand_identifier"
    OVERSIZED_SCENE_REGION_WITHOUT_TIGHT_LOGO = "oversized_scene_region_without_tight_logo"
    CHANNEL_BRAND_OUTSIDE_WATERMARK_REGION = "channel_brand_outside_watermark_region"
    CHANNEL_BRAND_NEEDS_WATERMARK_CHECK = "channel_brand_needs_watermark_check"
    LIKELY_CHANNEL_AUDIO_CONTAMINATION = "likely_channel_audio_contamination"
    CLEAR_LOGO_WITH_STRONG_OCR_MATCH = "clear_logo_with_strong_ocr_match"
    CLEAR_LOGO_WITH_STRONG_VISUAL_DATABASE_MATCH = "clear_logo_with_strong_visual_database_match"
    HIGH_CONFIDENCE_VISUAL_LOGO_NO_CONFLICTS = "high_confidence_visual_logo_no_conflicts"
    PARTIAL_LOGO_WITH_STRONG_OCR_SUPPORT = "partial_logo_with_strong_ocr_support"
    PARTIAL_LOGO_DEFAULT_REVIEW = "partial_logo_default_review"
    TRUNCATED_LOGO_REQUIRES_REVIEW = "truncated_logo_requires_review"
    WEAK_VISUAL_CROP_RESCUED_BY_AUDIO_HINT = "weak_visual_crop_rescued_by_audio_hint"
    AUDIO_ONLY_BRAND_PREDICTION = "audio_only_brand_prediction"
    FAISS_ONLY_OR_FAISS_CONFLICTING_PREDICTION = "faiss_only_or_faiss_conflicting_prediction"
    FAISS_SCORE_NOT_DISCRIMINATIVE_ENOUGH = "faiss_score_not_discriminative_enough"
    OCR_VLM_BRAND_CONFLICT = "ocr_vlm_brand_conflict"
    FAISS_VLM_BRAND_CONFLICT = "faiss_vlm_brand_conflict"
    LOGO_LIKE_BUT_BRAND_UNKNOWN = "logo_like_but_brand_unknown"
    READABLE_BRAND_LIKE_TEXT_NOT_IN_DATABASE = "readable_brand_like_text_not_in_database"
    DUPLICATE_SAME_BRAND_BOX = "duplicate_same_brand_box"
    SUBSET_BOX_SAME_BRAND = "subset_box_same_brand"
    OVERSIZED_CONTAINER_REPLACED_BY_TIGHT_BOX = "oversized_container_replaced_by_tight_box"
    DEFAULT_UNCERTAIN_OR_INSUFFICIENT_EVIDENCE = "default_uncertain_or_insufficient_evidence"
    VLM_OUTPUT_UNPARSED = "vlm_output_unparsed"


@dataclass(frozen=True)
class CodeDoc:
    rule: str            # "1", "5", "D"...
    block: int
    decision: str
    condition: str
    fields: tuple[str, ...]
    check: str
    note: str = ""


RC = ReasonCode
_D = CodeDoc

CODE_DOCS: dict[ReasonCode, CodeDoc] = {
    RC.HIGH_CONFIDENCE_NOT_LOGO: _D(
        "1", 1, REJECT, "vlm_decision == not_logo and not_logo_confidence >= rule1_not_logo_confidence (0.80)",
        ("vlm_decision", "not_logo_confidence"),
        "Open the crop: is it really not a logo? A wrong REJECT is invisible in the default queue, so sample these.",
        "not_logo_confidence comes from the generated `confidence/100` until router.not_logo_confidence_source is 'scored'."),
    RC.LOW_LOGO_PROBABILITY_NOT_LOGO: _D(
        "1 (fallback)", 1, REJECT, "vlm_decision == not_logo and logo_probability_scored <= rule1_logo_probability_max (20)",
        ("vlm_decision", "logo_probability_scored"),
        "Same as Rule 1: confirm the crop is not a logo.", "Disabled until T31/T32 pass (rule_1_fallback_logprob)."),
    RC.NON_BRAND_CONTENT_GENERIC_WORD: _D(
        "2", 1, REJECT, "brand_type == generic_word", ("brand_type",),
        "Confirm the text is a common word rather than a brand.", "Unreachable in v1: brand_type is not produced."),
    RC.NON_BRAND_CONTENT_INGREDIENT: _D(
        "2", 1, REJECT, "brand_type == ingredient", ("brand_type",),
        "Confirm the text is an ingredient name.", "Unreachable in v1: brand_type is not produced."),
    RC.NON_BRAND_CONTENT_SLOGAN: _D(
        "2", 1, REJECT, "brand_type == slogan", ("brand_type",),
        "Confirm the text is a slogan.", "Unreachable in v1: brand_type is not produced."),
    RC.NON_BRAND_CONTENT_UI_ICON: _D(
        "2", 1, REJECT, "brand_type == ui_icon", ("brand_type",),
        "Confirm the region is a UI icon.", "Unreachable in v1: brand_type is not produced."),
    RC.NON_BRAND_CONTENT_SIGN: _D(
        "2", 1, REJECT, "brand_type == sign", ("brand_type",),
        "Confirm the region is a generic sign.", "Unreachable in v1: brand_type is not produced."),
    RC.NOT_A_BRAND_IDENTIFIER: _D(
        "3", 1, REJECT, "is_brand_identifier is False and vlm_decision in {not_logo, partial_logo, uncertain}",
        ("is_brand_identifier", "vlm_decision"), "Confirm the text does not identify a brand.",
        "Unreachable in v1: is_brand_identifier is not produced."),
    RC.OVERSIZED_SCENE_REGION_WITHOUT_TIGHT_LOGO: _D(
        "4", 1, REJECT, "box_area_ratio >= rule4_area_ratio (0.45) and not has_valid_tight_bbox",
        ("box_area_ratio", "has_valid_tight_bbox"),
        "Check whether a real logo was inside the scene-sized box that refinement could not tighten.",
        "With a valid tight box the row continues (the crops already use it)."),
    RC.CHANNEL_BRAND_OUTSIDE_WATERMARK_REGION: _D(
        "5", 3, REVIEW, "canonical_brand is a channel brand and watermark_overlap is False and not (ocr_fuzzy_score >= 90 "
        "and ocr_brand_c == canonical_brand)",
        ("canonical_brand", "watermark_overlap", "ocr_fuzzy_score", "ocr_brand_match", "ocr_text", "faiss_top1",
         "faiss_top1_score", "brand_from_audio"),
        "Is this the channel bug in a place the region misses, or the channel name appearing elsewhere in the scene?",
        "The row already carries ocr_text, ocr_fuzzy_score, faiss_top1, faiss_top1_score and brand_from_audio (T49)."),
    RC.CHANNEL_BRAND_NEEDS_WATERMARK_CHECK: _D(
        "5", 3, REVIEW, "canonical_brand is a channel brand and watermark_overlap is None and ocr_fuzzy_score < rule5_ocr_fuzzy (90)",
        ("canonical_brand", "watermark_overlap", "ocr_fuzzy_score"),
        "The video has no watermark regions: decide by eye whether this is the broadcaster's bug.",
        "watermark_overlap is null when the video has no region set (never treated as 'outside')."),
    RC.LIKELY_CHANNEL_AUDIO_CONTAMINATION: _D(
        "5 (reject)", 1, REJECT, "canonical_brand is a channel brand and watermark_overlap is False and brand_from_audio "
        "and ocr_fuzzy_score < rule5_reject_ocr_fuzzy (70)",
        ("canonical_brand", "watermark_overlap", "brand_from_audio", "ocr_fuzzy_score"),
        "Confirm the channel name came from the commentary, not the picture.", "Disabled for all of M3 (rule_5_reject)."),
    RC.CLEAR_LOGO_WITH_STRONG_OCR_MATCH: _D(
        "6", 2, ACCEPT, "vlm_decision == logo and is_truncated is False and ocr_fuzzy_score >= rule6_ocr_fuzzy_threshold (95) "
        "and ocr_brand_c == canonical_brand",
        ("vlm_decision", "is_truncated", "ocr_fuzzy_score", "ocr_brand_match", "canonical_brand"),
        "Spot-check that OCR read the same text the VLM named; this is the only auto-accept in v1.",
        "brand_type == brand / is_brand_identifier clauses omitted in v1."),
    RC.CLEAR_LOGO_WITH_STRONG_VISUAL_DATABASE_MATCH: _D(
        "7", 2, ACCEPT, "vlm_decision == logo and is_truncated is False and strong_visual_match and not has_conflicting_ocr",
        ("vlm_decision", "is_truncated", "faiss_agrees_with_vlm", "faiss_top1_score", "faiss_score_gap", "has_conflicting_ocr"),
        "Compare the crop with the FAISS reference for the brand.", "Disabled pending T54 (rule_7)."),
    RC.HIGH_CONFIDENCE_VISUAL_LOGO_NO_CONFLICTS: _D(
        "8", 2, ACCEPT, "vlm_decision == logo and logo_probability_scored >= rule8_logo_probability (95) and is_truncated is False "
        "and not has_conflicting_ocr and not audio_only_prediction and not faiss_only_prediction and box_area_ratio < 0.45",
        ("vlm_decision", "logo_probability_scored", "is_truncated", "has_conflicting_ocr", "audio_only_prediction",
         "faiss_only_prediction", "box_area_ratio"),
        "Confirm the VLM's visual judgement without any supporting signal.", "Locked off for all of M3 (rule_8)."),
    RC.PARTIAL_LOGO_WITH_STRONG_OCR_SUPPORT: _D(
        "9a", 3, REVIEW, "vlm_decision == partial_logo and ocr_fuzzy_score >= rule9_strong_ocr (95) and ocr_brand_c == canonical_brand "
        "and is_truncated is False",
        ("vlm_decision", "ocr_fuzzy_score", "ocr_brand_match", "canonical_brand", "is_truncated"),
        "Quick confirm: OCR supports the brand, the crop is a partial view.", "review_priority = low."),
    RC.PARTIAL_LOGO_DEFAULT_REVIEW: _D(
        "9b", 3, REVIEW, "vlm_decision == partial_logo", ("vlm_decision",),
        "partial_logo is the least reliable VLM class: judge whether the visible part identifies the brand."),
    RC.TRUNCATED_LOGO_REQUIRES_REVIEW: _D(
        "10", 3, REVIEW, "is_truncated is True", ("is_truncated",),
        "The visible fragment may mislead: check the full frame for the complete logo."),
    RC.WEAK_VISUAL_CROP_RESCUED_BY_AUDIO_HINT: _D(
        "11 (reject)", 3, REJECT, "brand_from_audio and vlm_decision in {partial_logo, uncertain} and ocr_text == '' "
        "and not strong_visual_match",
        ("brand_from_audio", "vlm_decision", "ocr_text", "strong_visual_match"),
        "Confirm the brand was named only because the audio suggested it, over a crop with no readable content.",
        "Evaluated before the REVIEW variant. Rule 9 pre-empts partial_logo rows, so in practice this sees `uncertain`."),
    RC.AUDIO_ONLY_BRAND_PREDICTION: _D(
        "11 (review)", 3, REVIEW, "brand_from_audio and ocr_fuzzy_score < rule11_ocr_fuzzy (80) and not strong_visual_match",
        ("brand_from_audio", "ocr_fuzzy_score", "strong_visual_match"),
        "Is the brand actually visible, or only mentioned in the commentary?"),
    RC.FAISS_ONLY_OR_FAISS_CONFLICTING_PREDICTION: _D(
        "12a", 3, REVIEW, "brand_from_faiss and ocr_fuzzy_score < rule12_ocr_fuzzy (80) and not faiss_agrees_with_vlm",
        ("brand_from_faiss", "ocr_fuzzy_score", "faiss_agrees_with_vlm"),
        "The VLM's brand came from a FAISS hint that FAISS itself ranks lower: which one is right?"),
    RC.FAISS_SCORE_NOT_DISCRIMINATIVE_ENOUGH: _D(
        "12b", 3, REVIEW, "brand_from_faiss and faiss_top1_score < rule12_top1_threshold (0.97) and ocr_fuzzy_score < "
        "rule12_ocr_fuzzy_second (90)",
        ("brand_from_faiss", "faiss_top1_score", "ocr_fuzzy_score"),
        "FAISS supports the brand but not decisively, and OCR does not back it up."),
    RC.OCR_VLM_BRAND_CONFLICT: _D(
        "13", 3, REVIEW, "has_conflicting_ocr (ocr_brand_c != canonical_brand, both real, ocr_fuzzy_score >= rule13_ocr_fuzzy (85))",
        ("ocr_brand_match", "canonical_brand", "ocr_fuzzy_score"),
        "OCR and the VLM name different brands: read the text in the crop."),
    RC.FAISS_VLM_BRAND_CONFLICT: _D(
        "14", 3, REVIEW, "faiss_top1_brand_canonical != canonical_brand (both real) and faiss_top1_score >= rule14_top1_threshold (0.90)",
        ("faiss_top1_brand_canonical", "canonical_brand", "faiss_top1_score"),
        "FAISS confidently names a different brand: compare with the reference logos."),
    RC.LOGO_LIKE_BUT_BRAND_UNKNOWN: _D(
        "15", 4, UNKNOWN_LOGO, "vlm_decision == logo and canonical_brand in (None, '', UNKNOWN)",
        ("vlm_decision", "canonical_brand"),
        "A logo the VLM could not name: candidate for enrolment or the add-a-brand procedure.",
        "brand_type / is_brand_identifier clauses omitted in v1."),
    RC.READABLE_BRAND_LIKE_TEXT_NOT_IN_DATABASE: _D(
        "16", 4, UNKNOWN_LOGO, "vlm_decision == logo and ocr_text != '' and ocr_fuzzy_score < rule16_ocr_fuzzy (80)",
        ("vlm_decision", "ocr_text", "ocr_fuzzy_score"),
        "Readable text that matches no known brand: a local sponsor, or a brand missing from the list.",
        "brand_type clause omitted in v1."),
    RC.DUPLICATE_SAME_BRAND_BOX: _D(
        "17", 5, DEDUPED, "same canonical_brand, iou >= 0.50 or containment >= 0.80 (and area_ratio >= 0.80)",
        ("canonical_brand", "selected_bbox", "router_decision"),
        "Confirm the kept representative covers the logo.", "The cluster representative keeps its own decision."),
    RC.SUBSET_BOX_SAME_BRAND: _D(
        "18", 5, DEDUPED, "same canonical_brand, box mostly contained (>= 0.80) in a non-oversized box",
        ("canonical_brand", "selected_bbox", "box_area_ratio"),
        "Confirm the containing box is the intended one."),
    RC.OVERSIZED_CONTAINER_REPLACED_BY_TIGHT_BOX: _D(
        "18", 5, DEDUPED, "same canonical_brand, oversized box (>= 0.45 of the frame) containing a box with a valid tight bbox",
        ("canonical_brand", "selected_bbox", "box_area_ratio", "has_valid_tight_bbox"),
        "Confirm the tight box (kept) is the logo and the oversized one was scene context."),
    RC.DEFAULT_UNCERTAIN_OR_INSUFFICIENT_EVIDENCE: _D(
        "D", 6, REVIEW, "no rule above fired", (),
        "Default queue. A not_logo with mid confidence lands here by design (24 July).",
        "review_priority = normal."),
    RC.VLM_OUTPUT_UNPARSED: _D(
        "X", 6, REVIEW, "vlm_parse_ok is False and no rule above fired", ("vlm_parse_ok",),
        "The VLM's JSON did not parse: inspect `raw` in vlm/hinted.jsonl.",
        "[ASSUMPTION] block-6 special case."),
}

assert set(CODE_DOCS) == set(ReasonCode), "every reason code needs a CodeDoc"

BLOCK_TITLES = {1: "Rejects", 2: "Accepts", 3: "Review and conflict", 4: "UNKNOWN_LOGO", 5: "Dedup", 6: "Default"}


def render_markdown() -> str:
    """The content of ``configs/reason_codes.md``."""
    out = [
        "# Router reason codes (T44)", "",
        "Generated from `logo_pipeline/stages/router/reason_codes.py` — do not edit by hand "
        "(`python scripts/export_reason_codes.py`).", "",
        "Evaluation: blocks 1-6 in order; within a block rules in numeric order; the first rule whose condition "
        "holds decides. Disabled rules are skipped. Conditions below are the **v1** forms: with "
        "`router.enhanced_vlm_fields_available: false` Rules 2 and 3 cannot fire and the `brand_type == brand` / "
        "`is_brand_identifier is True` clauses of Rules 6, 7, 8, 15 and 16 are omitted.", "",
    ]
    for block, title in BLOCK_TITLES.items():
        out += [f"## Block {block} — {title}", ""]
        for code, doc in CODE_DOCS.items():
            if doc.block != block:
                continue
            out += [f"### `{code.value}`", "",
                    f"- **Rule:** {doc.rule}", f"- **Decision:** {doc.decision}",
                    f"- **Condition (v1):** {doc.condition}",
                    f"- **Fields read:** {', '.join(f'`{f}`' for f in doc.fields) or '—'}",
                    f"- **Reviewer should check:** {doc.check}"]
            if doc.note:
                out.append(f"- **Note:** {doc.note}")
            out.append("")
    return "\n".join(out)
