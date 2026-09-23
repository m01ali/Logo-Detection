"""Rules 1-16 (T38-T42, T72), v1 forms (``spec docs/11`` Part C).

Every rule is a pure, side-effect-free ``rule_n(inp, cfg) -> Verdict | None``
so the engine can evaluate the ones that lost to an earlier rule for the
firing log (T46). ``cfg`` is the ``PipelineCfg``. Fields a rule needs are read
through ``inp.req(...)`` at the point of use, which raises
``MissingRouterInput`` on ``None`` (no defaults); each rule's ``requires`` is
what the engine checks against the config before the first row.

Rules 17-18 (dedup) are ``dedup.py``. Enhanced VLM fields (``brand_type``,
``is_brand_identifier``) are not produced in M3: Rules 2 and 3 cannot fire and
their clauses in 6, 7, 8, 15 and 16 are omitted unless
``router.enhanced_vlm_fields_available``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .inputs import RouterInput, real_brand
from .reason_codes import ACCEPT, REJECT, REVIEW, UNKNOWN_LOGO, ReasonCode as RC

NON_BRAND_TYPES = ("generic_word", "ingredient", "slogan", "ui_icon", "sign")


@dataclass(frozen=True)
class Verdict:
    decision: str
    reason_code: RC
    rule: int | None
    review_priority: str | None = None
    payload: dict | None = None      # extra context for the firing log (Rule 5)


def _review(code: RC, rule: int, priority: str = "normal", payload: dict | None = None) -> Verdict:
    return Verdict(REVIEW, code, rule, priority, payload)


def _enhanced_ok(i: RouterInput, cfg, *, identifier: bool, brand_types: tuple[str, ...]) -> bool:
    """The ``brand_type`` / ``is_brand_identifier`` clauses; vacuously true when the fields are not available."""
    if not cfg.router.enhanced_vlm_fields_available:
        return True
    return (i.is_brand_identifier is True or not identifier) and i.brand_type in brand_types


def _same_real_brand(a: str | None, b: str | None) -> bool:
    return real_brand(a) and real_brand(b) and a == b


# -- Block 1: rejects ----------------------------------------------------------

def rule_1(i: RouterInput, cfg) -> Verdict | None:
    if i.vlm_decision == "not_logo" and i.req("not_logo_confidence") >= cfg.router.thresholds.rule1_not_logo_confidence:
        return Verdict(REJECT, RC.HIGH_CONFIDENCE_NOT_LOGO, 1)
    return None


def rule_1_fallback(i: RouterInput, cfg) -> Verdict | None:
    if i.vlm_decision == "not_logo" and i.req("logo_probability_scored") <= cfg.router.thresholds.rule1_logo_probability_max:
        return Verdict(REJECT, RC.LOW_LOGO_PROBABILITY_NOT_LOGO, 1)
    return None


def rule_2(i: RouterInput, cfg) -> Verdict | None:
    if i.brand_type in NON_BRAND_TYPES:
        return Verdict(REJECT, RC(f"non_brand_content_{i.brand_type}"), 2)
    return None


def rule_3(i: RouterInput, cfg) -> Verdict | None:
    if i.is_brand_identifier is False and i.vlm_decision in ("not_logo", "partial_logo", "uncertain"):
        return Verdict(REJECT, RC.NOT_A_BRAND_IDENTIFIER, 3)
    return None


def rule_4(i: RouterInput, cfg) -> Verdict | None:
    if i.box_area_ratio >= cfg.router.thresholds.rule4_area_ratio and not i.has_valid_tight_bbox:
        return Verdict(REJECT, RC.OVERSIZED_SCENE_REGION_WITHOUT_TIGHT_LOGO, 4)
    return None


def rule_5_reject(i: RouterInput, cfg) -> Verdict | None:
    if (i.is_channel_brand and i.watermark_overlap is False and i.brand_from_audio
            and i.ocr_fuzzy_score < cfg.router.thresholds.rule5_reject_ocr_fuzzy):
        return Verdict(REJECT, RC.LIKELY_CHANNEL_AUDIO_CONTAMINATION, 5)
    return None


# -- Block 2: accepts ----------------------------------------------------------

def rule_6(i: RouterInput, cfg) -> Verdict | None:
    if (i.vlm_decision == "logo" and i.is_truncated is False
            and i.ocr_fuzzy_score >= cfg.router.thresholds.rule6_ocr_fuzzy_threshold
            and _same_real_brand(i.ocr_brand_c, i.vlm_brand_c)
            and _enhanced_ok(i, cfg, identifier=True, brand_types=("brand",))):
        return Verdict(ACCEPT, RC.CLEAR_LOGO_WITH_STRONG_OCR_MATCH, 6)
    return None


def rule_7(i: RouterInput, cfg) -> Verdict | None:
    if (i.vlm_decision == "logo" and i.is_truncated is False and i.strong_visual_match
            and not i.has_conflicting_ocr and _enhanced_ok(i, cfg, identifier=True, brand_types=("brand",))):
        return Verdict(ACCEPT, RC.CLEAR_LOGO_WITH_STRONG_VISUAL_DATABASE_MATCH, 7)
    return None


def rule_8(i: RouterInput, cfg) -> Verdict | None:
    if (i.vlm_decision == "logo" and i.req("logo_probability_scored") >= cfg.router.thresholds.rule8_logo_probability
            and i.is_truncated is False and not i.has_conflicting_ocr and not i.audio_only_prediction
            and not i.faiss_only_prediction and i.box_area_ratio < cfg.router.thresholds.rule4_area_ratio
            and _enhanced_ok(i, cfg, identifier=True, brand_types=("brand",))):
        return Verdict(ACCEPT, RC.HIGH_CONFIDENCE_VISUAL_LOGO_NO_CONFLICTS, 8)
    return None


# -- Block 3: review and conflict ---------------------------------------------

def rule_9(i: RouterInput, cfg) -> Verdict | None:
    if i.vlm_decision != "partial_logo":
        return None
    if (i.ocr_fuzzy_score >= cfg.router.thresholds.rule9_strong_ocr and _same_real_brand(i.ocr_brand_c, i.vlm_brand_c)
            and i.is_truncated is False):
        return _review(RC.PARTIAL_LOGO_WITH_STRONG_OCR_SUPPORT, 9, "low")
    return _review(RC.PARTIAL_LOGO_DEFAULT_REVIEW, 9)


def rule_10(i: RouterInput, cfg) -> Verdict | None:
    return _review(RC.TRUNCATED_LOGO_REQUIRES_REVIEW, 10) if i.is_truncated is True else None


def rule_11_reject(i: RouterInput, cfg) -> Verdict | None:
    if (i.brand_from_audio and i.vlm_decision in ("partial_logo", "uncertain") and i.ocr_text == ""
            and not i.strong_visual_match):
        return Verdict(REJECT, RC.WEAK_VISUAL_CROP_RESCUED_BY_AUDIO_HINT, 11)
    return None


def rule_11_review(i: RouterInput, cfg) -> Verdict | None:
    return _review(RC.AUDIO_ONLY_BRAND_PREDICTION, 11) if i.audio_only_prediction else None


def rule_12(i: RouterInput, cfg) -> Verdict | None:
    if not i.brand_from_faiss:
        return None
    t = cfg.router.thresholds
    if i.ocr_fuzzy_score < t.rule12_ocr_fuzzy and not i.req("faiss_agrees_with_vlm"):
        return _review(RC.FAISS_ONLY_OR_FAISS_CONFLICTING_PREDICTION, 12)
    if i.req("faiss_top1_score") < cfg.faiss.rule12_top1_threshold and i.ocr_fuzzy_score < t.rule12_ocr_fuzzy_second:
        return _review(RC.FAISS_SCORE_NOT_DISCRIMINATIVE_ENOUGH, 12)
    return None


def rule_13(i: RouterInput, cfg) -> Verdict | None:
    return _review(RC.OCR_VLM_BRAND_CONFLICT, 13) if i.has_conflicting_ocr else None


def rule_14(i: RouterInput, cfg) -> Verdict | None:
    if (real_brand(i.faiss_brand_c) and real_brand(i.vlm_brand_c) and i.faiss_brand_c != i.vlm_brand_c
            and i.req("faiss_top1_score") >= cfg.faiss.rule14_top1_threshold):
        return _review(RC.FAISS_VLM_BRAND_CONFLICT, 14)
    return None


def rule_5_branch(i: RouterInput, cfg) -> str | None:
    """T49's branch for a channel-brand row (``None`` when the brand is not a channel brand); the
    engine logs it as ``rule5_branch`` for T71."""
    if not i.is_channel_brand:
        return None
    ocr_supported = (i.ocr_fuzzy_score >= cfg.router.thresholds.rule5_ocr_fuzzy
                     and _same_real_brand(i.ocr_brand_c, i.vlm_brand_c))
    if i.watermark_overlap is None:
        return ("no_regions_ocr_supported" if i.ocr_fuzzy_score >= cfg.router.thresholds.rule5_ocr_fuzzy
                else "no_regions_needs_check")
    if i.watermark_overlap is True:
        return "inside"
    return "outside_ocr_supported" if ocr_supported else "outside_weak_ocr"


def rule_5_review(i: RouterInput, cfg) -> Verdict | None:
    branch = rule_5_branch(i, cfg)
    if branch == "no_regions_needs_check":
        return _review(RC.CHANNEL_BRAND_NEEDS_WATERMARK_CHECK, 5, payload={"rule5_branch": branch})
    if branch == "outside_weak_ocr":
        return _review(RC.CHANNEL_BRAND_OUTSIDE_WATERMARK_REGION, 5, payload={
            "rule5_branch": branch, "ocr_text": i.ocr_text, "ocr_fuzzy_score": i.ocr_fuzzy_score,
            "faiss_top1": i.faiss_top1, "faiss_top1_score": i.faiss_top1_score,
            "brand_from_audio": i.brand_from_audio})
    return None


# -- Block 4: UNKNOWN_LOGO -------------------------------------------------------

def rule_15(i: RouterInput, cfg) -> Verdict | None:
    if (i.vlm_decision == "logo" and not real_brand(i.vlm_brand_c)
            and _enhanced_ok(i, cfg, identifier=True, brand_types=("brand", "unknown"))):
        return Verdict(UNKNOWN_LOGO, RC.LOGO_LIKE_BUT_BRAND_UNKNOWN, 15)
    return None


def rule_16(i: RouterInput, cfg) -> Verdict | None:
    if (i.vlm_decision == "logo" and i.ocr_text != "" and i.ocr_fuzzy_score < cfg.router.thresholds.rule16_ocr_fuzzy
            and _enhanced_ok(i, cfg, identifier=False, brand_types=("brand", "unknown"))):
        return Verdict(UNKNOWN_LOGO, RC.READABLE_BRAND_LIKE_TEXT_NOT_IN_DATABASE, 16)
    return None


# -- The ordered rule table ---------------------------------------------------------

@dataclass(frozen=True)
class RuleSpec:
    name: str                                   # "1", "1-fallback", "5-review", ...
    key: str                                    # the router.rules.* enable flag
    block: int
    fn: Callable[[RouterInput, object], Verdict | None]
    requires: tuple[str, ...] = field(default_factory=tuple)   # RouterInput fields no stage may fail to produce


_FAISS = ("faiss_top1_score", "faiss_agrees_with_vlm", "brand_from_faiss")
_OCR = ("ocr_text", "ocr_fuzzy_score")

RULES: tuple[RuleSpec, ...] = (
    RuleSpec("1", "rule_1", 1, rule_1, ("not_logo_confidence",)),
    RuleSpec("1-fallback", "rule_1_fallback_logprob", 1, rule_1_fallback, ("logo_probability_scored",)),
    RuleSpec("2", "rule_2", 1, rule_2, ("brand_type",)),
    RuleSpec("3", "rule_3", 1, rule_3, ("is_brand_identifier",)),
    RuleSpec("4", "rule_4", 1, rule_4),
    RuleSpec("5-reject", "rule_5_reject", 1, rule_5_reject, _OCR),
    RuleSpec("6", "rule_6", 2, rule_6, _OCR),
    RuleSpec("7", "rule_7", 2, rule_7, _FAISS),
    RuleSpec("8", "rule_8", 2, rule_8, ("logo_probability_scored",) + _OCR),
    RuleSpec("9", "rule_9", 3, rule_9, _OCR),
    RuleSpec("10", "rule_10", 3, rule_10),
    RuleSpec("11-reject", "rule_11_reject", 3, rule_11_reject, _OCR),
    RuleSpec("11-review", "rule_11_review", 3, rule_11_review, _OCR),
    RuleSpec("12", "rule_12", 3, rule_12, _FAISS + _OCR),
    RuleSpec("13", "rule_13", 3, rule_13, _OCR),
    RuleSpec("14", "rule_14", 3, rule_14, _FAISS),
    RuleSpec("5-review", "rule_5_review", 3, rule_5_review, _OCR),
    RuleSpec("15", "rule_15", 4, rule_15),
    RuleSpec("16", "rule_16", 4, rule_16, _OCR),
)
