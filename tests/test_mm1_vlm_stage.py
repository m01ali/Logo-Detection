"""21_test_plan.md — MM1 groundwork: stages/vlm.py's signal-filtering (T10
two-switch), hint-composition (T34 shape) and JSON-parsing logic, with the
model mocked out so this runs without loading a 7B checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.stages.canonicalise import CanonicaliseStage
from logo_pipeline.stages.merge import MergeStage
from logo_pipeline.stages.vlm import VlmStage


def _prep(ctx, faiss_rows=None, ocr_rows=None, audio_brands=None):
    ctx.write_jsonl("raw_detections.jsonl", [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [10, 10, 60, 60],
         "score": 0.8, "label": "a logo", "class_conf": None, "frame_width": 200, "frame_height": 100},
    ], header=ctx.header("detect_gdino"))
    MergeStage().run(ctx)
    ctx.write_jsonl("refinement.jsonl", [
        {"det_id": 0, "refined_bbox": None, "refinement_method": "none", "refinement_status": "no_candidate",
         "selected_bbox": [10, 10, 60, 60], "has_valid_tight_bbox": False, "refinement_area_ratio": None},
    ], header=ctx.header("refine"))
    ctx.write_jsonl("ocr.jsonl", ocr_rows or [
        {"det_id": 0, "ocr_text": "", "ocr_text_raw": "", "ocr_source": None, "ocr_max_confidence": None,
         "ocr_lines": [], "ocr_brand_match": None, "ocr_fuzzy_score": 0, "ocr_match_list": None,
         "ocr_match_mode": None},
    ], header=ctx.header("ocr"))
    ctx.write_jsonl("faiss_hints.jsonl", faiss_rows or [
        {"det_id": 0, "faiss_top1": None, "faiss_top1_score": None, "faiss_top5_brands": [],
         "faiss_top5_scores": [], "faiss_score_gap": None},
    ], header=ctx.header("faiss_query"))
    ctx.write_json("audio/audio_hints.json", {"status": "ok" if audio_brands else "empty",
                                              "brands": audio_brands or []})


def _mock_generation(reply_json: str):
    """Patches everything VlmStage._get_model / generate touches so run()
    exercises real signal-selection + parsing logic against a scripted reply."""
    model = MagicMock()
    model.device = "cpu"
    model.dtype = None
    model.generate.return_value = [[1, 2, 3, 4, 5]]

    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT_TEXT"
    processor.batch_decode.return_value = [reply_json]
    inputs = MagicMock()
    inputs.input_ids = [[1]]
    inputs.to.return_value = inputs
    processor.return_value = inputs

    patches = [
        patch("logo_pipeline.stages.vlm._get_model", return_value=(model, processor)),
        patch("qwen_vl_utils.process_vision_info", return_value=([], [])),
    ]
    return patches, processor


def test_vlm_stage_respects_two_switch_and_parses_json(tmp_path, repo_root):
    overrides = {"signals": {"ocr": {"compute_and_log": True, "include_in_vlm_prompt": False},
                             "faiss": {"compute_and_log": True, "include_in_vlm_prompt": True}}}
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(200, 100), overrides=overrides)
    _prep(ctx, faiss_rows=[{"det_id": 0, "faiss_top1": "rai", "faiss_top1_score": 0.9,
                            "faiss_top5_brands": ["rai"], "faiss_top5_scores": [0.9], "faiss_score_gap": None}],
          ocr_rows=[{"det_id": 0, "ocr_text": "rai sport", "ocr_text_raw": "RAI SPORT", "ocr_source": "tight_crop",
                    "ocr_max_confidence": 0.8, "ocr_lines": [{"text": "RAI SPORT", "confidence": 0.8}],
                    "ocr_brand_match": None, "ocr_fuzzy_score": 0, "ocr_match_list": None, "ocr_match_mode": None}])
    CanonicaliseStage().run(ctx)

    reply = '{"is_logo": "logo", "brand": "rai", "is_truncated": false, "logo_probability": 90, ' \
            '"confidence": 80, "signals_used": ["Signal 1", "faiss"], "reasoning": "matches"}'
    patches, processor = _mock_generation(reply)
    for p in patches:
        p.start()
    try:
        result = VlmStage().run(ctx)
    finally:
        for p in patches:
            p.stop()

    assert result.counts["calls"] == 1
    rows = ctx.read_jsonl("vlm/hinted.jsonl")
    assert len(rows) == 1
    assert rows[0]["vlm_parse_ok"] is True
    assert rows[0]["parsed"]["brand"] == "rai"
    assert rows[0]["parsed"]["signals_used"] == ["tight_crop", "faiss_matches"]   # normalised
    # ocr_text disabled in the prompt -> not in enabled_signals; faiss enabled -> is.
    assert "ocr_text" not in rows[0]["enabled_signals"]
    assert "faiss_matches" in rows[0]["enabled_signals"]

    comp = ctx.read_jsonl("vlm/hint_composition.jsonl")
    sources = {item["source"] for item in comp[0]["hint_composition"]}
    assert "faiss" in sources
    assert "ocr" not in sources   # disabled signal contributes nothing to the composition log either


def test_vlm_stage_unparseable_reply_sets_parse_ok_false(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(200, 100))
    _prep(ctx)
    CanonicaliseStage().run(ctx)
    patches, _ = _mock_generation("not json at all")
    for p in patches:
        p.start()
    try:
        VlmStage().run(ctx)
    finally:
        for p in patches:
            p.stop()
    rows = ctx.read_jsonl("vlm/hinted.jsonl")
    assert rows[0]["vlm_parse_ok"] is False
    assert rows[0]["parsed"] is None


def test_vlm_stage_skips_when_hinted_pass_disabled(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, overrides={"vlm": {"hinted_pass": False}})
    _prep(ctx)
    with patch("logo_pipeline.stages.vlm._get_model") as m:
        result = VlmStage().run(ctx)
    m.assert_not_called()
    assert result.counts["calls"] == 0
    assert ctx.read_jsonl("vlm/hinted.jsonl") == []
