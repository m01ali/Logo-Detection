"""MM5 (spec docs/11 Part C): one test per rule branch, block ordering,
first-match-wins, the preempted-rule log, the enhanced-field switch and the
strict-input behaviour."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from logo_pipeline.stages.router import rules
from logo_pipeline.stages.router.engine import check_rules_producible, enabled_rules
from logo_pipeline.stages.router.inputs import MissingRouterInput
from mm5_helpers import BRANDS, decide, inp, make_cfg, route, row, rules_on  # noqa: E402

FAISS_STRONG = dict(faiss_top1="nike", faiss_top1_brand_canonical="nike", faiss_agrees_with_vlm=True,
                    faiss_top1_score=0.98, faiss_score_gap=0.10)
CHANNEL = frozenset({"rai sport"})


def code(d):
    return d["reason_code"]


# -- Block 1 ---------------------------------------------------------------------

def test_rule_1_not_logo_high_confidence_rejects():
    d = decide(row(vlm_decision="not_logo", is_logo="not_logo", canonical_brand=None, confidence=85))
    assert (d["router_decision"], code(d), d["router_rule"]) == ("REJECT", "high_confidence_not_logo", 1)


def test_rule_1_mid_confidence_not_logo_lands_in_default():
    d = decide(row(vlm_decision="not_logo", is_logo="not_logo", canonical_brand=None, confidence=79))
    assert (d["router_decision"], code(d), d["router_rule"]) == (
        "REVIEW", "default_uncertain_or_insufficient_evidence", None)
    assert d["review_priority"] == "normal"


def test_rule_1_reads_not_logo_confidence_as_confidence_over_100():
    i = inp(row(vlm_decision="not_logo", confidence=85))
    assert i.not_logo_confidence == 0.85
    assert inp(row(vlm_decision="logo", confidence=85)).not_logo_confidence is None


def test_rule_1_fallback_needs_scores_and_fires_below_threshold():
    on = {"vlm": {"scoring": {"enabled": True}}, **rules_on(rule_1_fallback_logprob=True)}
    d = decide(row(vlm_decision="not_logo", canonical_brand=None, confidence=50, logo_probability_scored=12.0), on)
    assert (d["router_decision"], code(d), d["router_rule"]) == ("REJECT", "low_logo_probability_not_logo", 1)
    d = decide(row(vlm_decision="not_logo", canonical_brand=None, confidence=50, logo_probability_scored=40.0), on)
    assert d["router_decision"] == "REVIEW"


def test_rule_2_and_3_fire_only_with_enhanced_fields():
    on = {"router": {"enhanced_vlm_fields_available": True, "rules": {"rule_2": True, "rule_3": True}}}
    d = decide(row(brand_type="slogan", is_brand_identifier=True), on)
    assert (d["router_decision"], code(d)) == ("REJECT", "non_brand_content_slogan")
    d = decide(row(brand_type="brand", is_brand_identifier=False, vlm_decision="partial_logo"), on)
    assert (d["router_decision"], code(d)) == ("REJECT", "not_a_brand_identifier")
    # Same row, default (v1) config: the enhanced fields are not looked at and the rules cannot fire.
    d = decide(row(vlm_decision="partial_logo"))
    assert code(d) == "partial_logo_default_review"


def test_rule_2_enabled_without_enhanced_fields_refuses_to_start():
    with pytest.raises(MissingRouterInput) as e:
        check_rules_producible(make_cfg(rules_on(rule_2=True)))
    assert e.value.field == "brand_type"


def test_rule_4_oversized_without_tight_box_rejects_but_tight_box_continues():
    d = decide(row(box_area_ratio=0.5))
    assert (d["router_decision"], code(d), d["router_rule"]) == (
        "REJECT", "oversized_scene_region_without_tight_logo", 4)
    d = decide(row(box_area_ratio=0.5, has_valid_tight_bbox=True))
    assert code(d) == "default_uncertain_or_insufficient_evidence"


def test_rule_5_reject_is_locked_off_and_never_fires_in_the_engine():
    from logo_pipeline.config import ConfigError
    with pytest.raises(ConfigError):
        make_cfg(rules_on(rule_5_reject=True))
    r = row(canonical_brand="rai sport", brand_from_audio=True, audio_brands_canonical=["rai sport"],
            watermark_overlap=False, ocr_fuzzy_score=10, hinted_brand_in_faiss_hints=False)
    d = decide(r, known=CHANNEL)
    assert d["router_decision"] != "REJECT"
    # The rule function itself implements the condition (it is only config-disabled).
    v = rules.rule_5_reject(inp(r, known=CHANNEL), make_cfg())
    assert v is not None and v.reason_code.value == "likely_channel_audio_contamination"


# -- Block 2 ---------------------------------------------------------------------

def test_rule_6_strong_ocr_match_accepts():
    d = decide(row(ocr_text="nike", ocr_fuzzy_score=96, ocr_brand_match="nike"))
    assert (d["router_decision"], code(d), d["router_rule"]) == ("ACCEPT", "clear_logo_with_strong_ocr_match", 6)


@pytest.mark.parametrize("kw", [
    dict(ocr_fuzzy_score=94, ocr_brand_match="nike"),                       # below the T72 threshold
    dict(ocr_fuzzy_score=96, ocr_brand_match="ikea"),                       # OCR names another brand
    dict(ocr_fuzzy_score=96, ocr_brand_match="nike", is_truncated=True),    # truncated
    dict(ocr_fuzzy_score=96, ocr_brand_match="nike", vlm_decision="partial_logo"),
])
def test_rule_6_negative_branches(kw):
    assert decide(row(ocr_text="x", **kw))["router_decision"] != "ACCEPT"


def test_rule_6_reads_its_own_threshold_key():
    off = {"router": {"thresholds": {"rule6_ocr_fuzzy_threshold": 99}}}
    assert decide(row(ocr_text="nike", ocr_fuzzy_score=96, ocr_brand_match="nike"), off)["router_decision"] != "ACCEPT"


def test_rule_6_uses_canonicalised_ocr_brand():
    d = decide(row(canonical_brand="verisure", ocr_text="very sure", ocr_fuzzy_score=97, ocr_brand_match="very sure"))
    assert code(d) == "clear_logo_with_strong_ocr_match"


def test_rule_7_accepts_on_strong_visual_match_when_enabled():
    on = rules_on(rule_7=True)
    d = decide(row(**FAISS_STRONG, hinted_brand_in_faiss_hints=False), on)
    assert (d["router_decision"], code(d)) == ("ACCEPT", "clear_logo_with_strong_visual_database_match")
    assert decide(row(**FAISS_STRONG))["router_decision"] == "REVIEW"        # disabled by default
    weak = dict(FAISS_STRONG, faiss_score_gap=0.05)
    assert decide(row(**weak), on)["router_decision"] != "ACCEPT"


def test_rule_8_is_locked_off_but_implemented():
    from logo_pipeline.config import ConfigError
    with pytest.raises(ConfigError):
        make_cfg(rules_on(rule_8=True))
    v = rules.rule_8(inp(row(logo_probability_scored=97.0)), make_cfg())
    assert v is not None and v.reason_code.value == "high_confidence_visual_logo_no_conflicts"
    assert rules.rule_8(inp(row(logo_probability_scored=90.0)), make_cfg()) is None


# -- Block 3 ---------------------------------------------------------------------

def test_rule_9a_and_9b_partial_logo():
    d = decide(row(vlm_decision="partial_logo", ocr_text="nike", ocr_fuzzy_score=96, ocr_brand_match="nike"))
    assert (d["router_decision"], code(d), d["review_priority"]) == ("REVIEW", "partial_logo_with_strong_ocr_support", "low")
    d = decide(row(vlm_decision="partial_logo"))
    assert (code(d), d["review_priority"]) == ("partial_logo_default_review", "normal")


def test_partial_logo_with_unknown_brand_is_rule_9_not_15():
    d = decide(row(vlm_decision="partial_logo", canonical_brand="UNKNOWN"))
    assert d["router_rule"] == 9 and d["router_decision"] == "REVIEW"
    assert d["router_decision"] != "UNKNOWN_LOGO"


def test_rule_10_truncated_and_9_before_10():
    d = decide(row(is_truncated=True))
    assert (code(d), d["router_rule"]) == ("truncated_logo_requires_review", 10)
    d = decide(row(vlm_decision="partial_logo", is_truncated=True))
    assert d["router_rule"] == 9                                              # numeric order within the block
    assert d["rules_matched_preempted"] == [10]


def test_rule_11_reject_and_review():
    audio = dict(canonical_brand="verisure", brand_from_audio=True, audio_brands_canonical=["verisure"])
    d = decide(row(vlm_decision="uncertain", ocr_text="", **audio))
    assert (d["router_decision"], code(d)) == ("REJECT", "weak_visual_crop_rescued_by_audio_hint")
    d = decide(row(ocr_text="", ocr_fuzzy_score=10, **audio))
    assert (d["router_decision"], code(d)) == ("REVIEW", "audio_only_brand_prediction")
    # Strong visual agreement lifts the audio-only flag.
    d = decide(row(**audio, **dict(FAISS_STRONG, faiss_top1_brand_canonical="verisure", faiss_top1="verisure")))
    assert code(d) != "audio_only_brand_prediction"


def test_rule_11_reject_is_evaluated_before_review_variant():
    audio = dict(canonical_brand="verisure", brand_from_audio=True, audio_brands_canonical=["verisure"])
    d = decide(row(vlm_decision="uncertain", ocr_text="", **audio))
    assert d["firing"]["deciding_rule"] == "11-reject"
    assert [m["rule"] for m in d["firing"]["rules_matched_preempted"]] == ["11-review"]


def test_rule_12a_and_12b():
    d = decide(row(hinted_brand_in_faiss_hints=True, faiss_top1="ikea", faiss_top1_brand_canonical="ikea",
                   faiss_agrees_with_vlm=False, faiss_top1_score=0.5, ocr_fuzzy_score=10))
    assert code(d) == "faiss_only_or_faiss_conflicting_prediction"
    d = decide(row(hinted_brand_in_faiss_hints=True, **dict(FAISS_STRONG, faiss_top1_score=0.93), ocr_fuzzy_score=50))
    assert code(d) == "faiss_score_not_discriminative_enough"
    d = decide(row(hinted_brand_in_faiss_hints=True, **dict(FAISS_STRONG, faiss_top1_score=0.99), ocr_fuzzy_score=50))
    assert code(d) != "faiss_score_not_discriminative_enough"


def test_rule_13_ocr_vlm_conflict_needs_score_at_least_85():
    d = decide(row(ocr_text="ikea", ocr_fuzzy_score=88, ocr_brand_match="ikea"))
    assert (d["router_decision"], code(d)) == ("REVIEW", "ocr_vlm_brand_conflict")
    assert code(decide(row(ocr_text="ikea", ocr_fuzzy_score=84, ocr_brand_match="ikea"))) != "ocr_vlm_brand_conflict"


def test_rule_14_faiss_vlm_conflict():
    d = decide(row(faiss_top1="ikea", faiss_top1_brand_canonical="ikea", faiss_agrees_with_vlm=False,
                   faiss_top1_score=0.92, faiss_score_gap=0.1))
    assert (d["router_decision"], code(d)) == ("REVIEW", "faiss_vlm_brand_conflict")
    d = decide(row(faiss_top1="ikea", faiss_top1_brand_canonical="ikea", faiss_agrees_with_vlm=False,
                   faiss_top1_score=0.85, faiss_score_gap=0.1))
    assert code(d) != "faiss_vlm_brand_conflict"


CH_ROW = dict(canonical_brand="rai sport", brand_in_database=True)


def test_rule_5_review_branches():
    # No regions, weak OCR -> needs a watermark check.
    d = decide(row(**CH_ROW, watermark_overlap=None, ocr_fuzzy_score=20), known=CHANNEL)
    assert code(d) == "channel_brand_needs_watermark_check" and d["firing"]["rule5_branch"] == "no_regions_needs_check"
    # Inside the region -> Rule 5 makes no decision (normal handling: here the default).
    d = decide(row(**CH_ROW, watermark_overlap=True, ocr_fuzzy_score=20), known=CHANNEL)
    assert code(d) == "default_uncertain_or_insufficient_evidence" and d["firing"]["rule5_branch"] == "inside"
    # Outside but OCR supports it -> no decision, branch logged for T71.
    d = decide(row(**CH_ROW, vlm_decision="uncertain", watermark_overlap=False, ocr_text="rai sport",
                   ocr_fuzzy_score=92, ocr_brand_match="rai sport"), known=CHANNEL)
    assert code(d) == "default_uncertain_or_insufficient_evidence"
    assert d["firing"]["rule5_branch"] == "outside_ocr_supported"
    # Outside with weak OCR -> review, payload carries what the review screen needs.
    d = decide(row(**CH_ROW, watermark_overlap=False, ocr_text="", ocr_fuzzy_score=10), known=CHANNEL)
    assert code(d) == "channel_brand_outside_watermark_region"
    assert set(d["firing"]["reason_payload"]) >= {"ocr_text", "ocr_fuzzy_score", "faiss_top1", "faiss_top1_score",
                                                   "brand_from_audio"}
    # A non-channel brand never reaches Rule 5.
    assert decide(row(watermark_overlap=False), known=CHANNEL)["firing"]["rule5_branch"] is None


def test_rule_5_disabled_review_is_skipped():
    d = decide(row(**CH_ROW, watermark_overlap=False), rules_on(rule_5_review=False), CHANNEL)
    assert code(d) == "default_uncertain_or_insufficient_evidence"


# -- Block 4 and 6 ----------------------------------------------------------------

def test_rule_15_and_16_unknown_logo():
    d = decide(row(canonical_brand="UNKNOWN", brand="UNKNOWN"))
    assert (d["router_decision"], code(d)) == ("UNKNOWN_LOGO", "logo_like_but_brand_unknown")
    d = decide(row(canonical_brand=None, brand=None))
    assert code(d) == "logo_like_but_brand_unknown"
    d = decide(row(ocr_text="tonno rio", ocr_fuzzy_score=40))
    assert (d["router_decision"], code(d)) == ("UNKNOWN_LOGO", "readable_brand_like_text_not_in_database")
    # partial_logo is no longer covered by Rule 16 (T39).
    assert decide(row(vlm_decision="partial_logo", ocr_text="tonno", ocr_fuzzy_score=40))["router_decision"] == "REVIEW"


def test_default_and_unparsed_rows():
    d = decide(row())
    assert (d["router_decision"], code(d), d["router_rule"]) == (
        "REVIEW", "default_uncertain_or_insufficient_evidence", None)
    d = decide(row(vlm_parse_ok=False, vlm_decision=None, is_logo=None, canonical_brand=None, confidence=None,
                   is_truncated=None))
    assert (d["router_decision"], code(d)) == ("REVIEW", "vlm_output_unparsed")


# -- Ordering ----------------------------------------------------------------------

def test_first_match_wins_and_lists_preempted_rules_only_if_they_held():
    # Rule 6 fires; Rule 16's condition (fuzzy < 80) cannot hold at 96, so it is not listed.
    d = decide(row(ocr_text="nike", ocr_fuzzy_score=96, ocr_brand_match="nike"))
    assert d["router_rule"] == 6 and d["rules_matched_preempted"] == []
    # Truncated (Rule 10) beats a Rule 16 whose condition also holds; 16 is listed as pre-empted.
    d = decide(row(is_truncated=True, ocr_text="abc", ocr_fuzzy_score=10))
    assert d["router_rule"] == 10 and d["rules_matched_preempted"] == [16]
    assert d["firing"]["rules_matched_preempted"] == [
        {"rule": "16", "reason_code": "readable_brand_like_text_not_in_database"}]


def test_block_order_reject_before_accept():
    # Oversized without a tight box (block 1) beats a Rule 6 accept (block 2).
    d = decide(row(box_area_ratio=0.6, ocr_text="nike", ocr_fuzzy_score=98, ocr_brand_match="nike"))
    assert d["router_rule"] == 4 and d["rules_matched_preempted"] == [6]


def test_disabled_rule_is_skipped_never_defaulted():
    d = decide(row(ocr_text="nike", ocr_fuzzy_score=98, ocr_brand_match="nike"), rules_on(rule_6=False))
    assert d["router_rule"] != 6


# -- Strict inputs -----------------------------------------------------------------

def test_missing_key_raises():
    r = row()
    del r["ocr_fuzzy_score"]
    with pytest.raises(MissingRouterInput) as e:
        inp(r)
    assert e.value.field == "ocr_fuzzy_score"


def test_rule_1_with_null_confidence_on_not_logo_row_raises():
    with pytest.raises(MissingRouterInput) as e:
        decide(row(vlm_decision="not_logo", canonical_brand=None, confidence=None))
    assert e.value.field == "not_logo_confidence"


def test_engine_refuses_to_start_when_fallback_enabled_without_scores():
    with pytest.raises(MissingRouterInput) as e:
        route([row()], rules_on(rule_1_fallback_logprob=True))
    assert e.value.field == "logo_probability_scored"


def test_engine_refuses_to_start_when_faiss_rules_lack_faiss():
    with pytest.raises(MissingRouterInput):
        route([row()], {"signals": {"faiss": {"compute_and_log": False, "include_in_vlm_prompt": False}}})
    ok = {"signals": {"faiss": {"compute_and_log": False, "include_in_vlm_prompt": False}},
          **rules_on(rule_12=False, rule_14=False)}
    assert route([row()], ok)[0][0]["router_decision"] == "REVIEW"


def test_enabled_rules_follow_yaml():
    names = [r.name for r in enabled_rules(make_cfg())]
    assert "7" not in names and "8" not in names and "5-reject" not in names and "1-fallback" not in names
    assert names.index("6") < names.index("9") < names.index("15")


def test_router_disabled_writes_null_decisions():
    outs, firing = route([row()], {"router": {"enabled": False}})
    assert outs[0]["router_decision"] is None and firing == []


# -- Enhanced-field switch ------------------------------------------------------------

def test_enhanced_switch_adds_brand_type_clauses_to_rule_6():
    strong = dict(ocr_text="nike", ocr_fuzzy_score=98, ocr_brand_match="nike", is_brand_identifier=True)
    assert decide(row(brand_type="generic_word", **strong))["router_rule"] == 6          # v1: clause omitted
    on = {"router": {"enhanced_vlm_fields_available": True}}
    assert decide(row(brand_type="generic_word", **strong), on)["router_rule"] != 6      # clause enforced
    assert decide(row(brand_type="brand", **strong), on)["router_rule"] == 6
