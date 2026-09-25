"""MM5 end to end on synthetic stage outputs: canonicalise -> watermark ->
router_input -> router, the firing log, the T65 ordering checks, the reason-code
document and router replay determinism."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


from eval.ordering_checks import CHECK_1, CHECK_2, check_firing_log, read_firing_log
from logo_pipeline.stages.canonicalise import CanonicaliseStage
from logo_pipeline.stages.router.engine import RouterStage, route_rows
from logo_pipeline.stages.router.reason_codes import CODE_DOCS, ReasonCode, render_markdown
from logo_pipeline.stages.router_input import RouterInputStage
from logo_pipeline.stages.watermark import WatermarkStage
from helpers import make_ctx  # noqa: E402


def _rep(det_id, frame_idx, box):
    return {"det_id": det_id, "frame_idx": frame_idx, "frame_width": 200, "frame_height": 100, "loose_bbox": box,
            "box_xyxy": box, "det_score": 0.9, "box_area_ratio": 0.05, "box_touches_edge": False,
            "detection_sources": ["gdino"], "merge_count": 0, "merged_raw_det_ids": []}


def _build(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    boxes = {0: [0, 0, 40, 20], 1: [100, 50, 140, 70], 2: [2, 2, 42, 22], 3: [0, 0, 40, 20]}
    frames = {0: 0, 1: 0, 2: 10, 3: 10}
    ctx.write_jsonl("merged_detections.jsonl", [_rep(d, frames[d], boxes[d]) for d in boxes])
    vlm = {
        0: {"is_logo": "logo", "brand": "Nike", "is_truncated": False, "confidence": 92, "logo_probability": 90},
        1: {"is_logo": "not_logo", "brand": None, "is_truncated": False, "confidence": 95, "logo_probability": 3},
        2: {"is_logo": "logo", "brand": "IKEA", "is_truncated": False, "confidence": 80, "logo_probability": 80},
        3: {"is_logo": "logo", "brand": "Ikea", "is_truncated": False, "confidence": 85, "logo_probability": 85},
    }
    ctx.write_jsonl("vlm/hinted.jsonl", [{"det_id": d, "parsed": p, "vlm_parse_ok": True, "enabled_signals": []}
                                          for d, p in vlm.items()])
    ctx.write_jsonl("ocr.jsonl", [
        {"det_id": 0, "ocr_text": "nike", "ocr_text_raw": "NIKE", "ocr_source": "tight_crop",
         "ocr_max_confidence": 0.99, "ocr_brand_match": "nike", "ocr_fuzzy_score": 98, "ocr_match_list": "canonical",
         "ocr_match_mode": "fuzzy"}])
    ctx.write_json("audio/audio_hints.json", {"status": "ok", "brands": []})
    for stage in (CanonicaliseStage(), WatermarkStage(), RouterInputStage(), RouterStage()):
        stage.run(ctx)
    return ctx


def test_full_router_chain_decisions(tmp_path, repo_root):
    ctx = _build(tmp_path, repo_root)
    out = {r["det_id"]: r for r in ctx.read_jsonl("router_output.jsonl")}
    assert (out[0]["router_decision"], out[0]["reason_code"], out[0]["router_rule"]) == (
        "ACCEPT", "clear_logo_with_strong_ocr_match", 6)
    assert (out[1]["router_decision"], out[1]["reason_code"], out[1]["router_rule"]) == (
        "REJECT", "high_confidence_not_logo", 1)
    # det 2 and 3 are the same brand, same frame, near-identical boxes: one survives.
    assert {out[2]["router_decision"], out[3]["router_decision"]} == {"DEDUPED", "REVIEW"}
    assert out[0]["trust_score"] is not None and 0.0 <= out[0]["router_confidence"] <= 1.0


def test_router_input_carries_the_t43_flags(tmp_path, repo_root):
    ctx = _build(tmp_path, repo_root)
    rows = {r["det_id"]: r for r in ctx.read_jsonl("router_input.jsonl")}
    assert rows[0]["brand_evidence"] == "ocr_confirmed"
    assert rows[0]["has_conflicting_ocr"] is False
    assert rows[2]["brand_evidence"] == "none"


def test_firing_log_shape_and_ordering_checks_pass(tmp_path, repo_root):
    ctx = _build(tmp_path, repo_root)
    firing = read_firing_log(ctx.run_dir)
    assert len(firing) == 4
    f0 = next(f for f in firing if f["det_id"] == 0)
    assert set(f0) >= {"det_id", "frame_idx", "deciding_rule", "reason_code", "decision", "rules_matched_preempted",
                       "dedup"}
    assert f0["deciding_rule"] == "6"
    assert check_firing_log(firing) == {CHECK_1: [], CHECK_2: []}
    assert ctx.path("reports/router_decisions.md").exists()


def test_ordering_check_detects_violations():
    bad = [
        {"det_id": 1, "vlm_decision": "partial_logo", "decision": "UNKNOWN_LOGO", "deciding_rule": "15",
         "rules_matched": [], "pre_dedup": None},
        {"det_id": 2, "vlm_decision": "logo", "decision": "UNKNOWN_LOGO", "deciding_rule": "16",
         "rules_matched": [{"rule": "6", "reason_code": "x"}, {"rule": "16", "reason_code": "y"}], "pre_dedup": None},
        {"det_id": 3, "vlm_decision": "logo", "decision": "DEDUPED", "deciding_rule": "17",
         "rules_matched": [{"rule": "7", "reason_code": "x"}, {"rule": "16", "reason_code": "y"}],
         "pre_dedup": {"deciding_rule": "16", "decision": "UNKNOWN_LOGO", "reason_code": "y"}},
    ]
    assert check_firing_log(bad) == {CHECK_1: [1], CHECK_2: [2, 3]}


def test_router_replay_is_deterministic_and_rereads_thresholds(tmp_path, repo_root):
    ctx = _build(tmp_path, repo_root)
    rows = ctx.read_jsonl("router_input.jsonl")
    first = route_rows(rows, ctx.cfg, ctx.brands)
    assert first == route_rows(rows, ctx.cfg, ctx.brands)
    from mm5_helpers import make_cfg
    strict = make_cfg({"router": {"thresholds": {"rule6_ocr_fuzzy_threshold": 99}}})
    replay = {o["det_id"]: o for o in route_rows(rows, strict, ctx.brands)[0]}
    assert replay[0]["router_rule"] != 6            # the stored flags do not pin the decision


def test_router_stage_output_matches_replay(tmp_path, repo_root):
    ctx = _build(tmp_path, repo_root)
    stage_out = ctx.read_jsonl("router_output.jsonl")
    replay, _ = route_rows(ctx.read_jsonl("router_input.jsonl"), ctx.cfg, ctx.brands)
    assert stage_out == replay


# -- reason codes ---------------------------------------------------------------------

def test_reason_code_enum_matches_the_spec_list():
    spec = """high_confidence_not_logo low_logo_probability_not_logo non_brand_content_generic_word
    non_brand_content_ingredient non_brand_content_slogan non_brand_content_ui_icon non_brand_content_sign
    not_a_brand_identifier oversized_scene_region_without_tight_logo channel_brand_outside_watermark_region
    channel_brand_needs_watermark_check likely_channel_audio_contamination clear_logo_with_strong_ocr_match
    clear_logo_with_strong_visual_database_match high_confidence_visual_logo_no_conflicts
    partial_logo_with_strong_ocr_support partial_logo_default_review truncated_logo_requires_review
    weak_visual_crop_rescued_by_audio_hint audio_only_brand_prediction faiss_only_or_faiss_conflicting_prediction
    faiss_score_not_discriminative_enough ocr_vlm_brand_conflict faiss_vlm_brand_conflict
    logo_like_but_brand_unknown readable_brand_like_text_not_in_database duplicate_same_brand_box
    subset_box_same_brand oversized_container_replaced_by_tight_box default_uncertain_or_insufficient_evidence
    vlm_output_unparsed""".split()
    assert sorted(c.value for c in ReasonCode) == sorted(spec)
    assert set(CODE_DOCS) == set(ReasonCode)


def test_committed_reason_codes_document_is_current(repo_root):
    committed = (Path(repo_root) / "configs" / "reason_codes.md").read_text(encoding="utf-8")
    assert committed == render_markdown(), "run: python scripts/export_reason_codes.py"


def test_every_code_documented_with_fields_and_reviewer_check():
    for code, doc in CODE_DOCS.items():
        assert doc.condition and doc.check, code
        assert f"`{code.value}`" in render_markdown()
