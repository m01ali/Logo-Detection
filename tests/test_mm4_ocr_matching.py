"""spec docs/10 — MM4: OCR verification pass and brand matching (T25-T29).
Pure-function tests for build_candidate_set/match_ocr_text need no heavy
deps; the stage integration test mocks EasyOCR the same way test_mm3_refine
mocks it."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.brands import BrandEntry, build_brand_list, load_brand_list, normalise
from logo_pipeline.config import OcrMatchingCfg
from logo_pipeline.stages.ocr import OcrStage, build_candidate_set, get_scorer, match_ocr_text

MCFG = OcrMatchingCfg()   # v1 defaults: short_name_cutoff 4, general_fuzzy_threshold 90, scorer WRatio
SCORER = get_scorer(MCFG.scorer)


def _small_brands():
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    return load_brand_list(fixtures_dir / "brands_small.yaml", require_hash_match=False)


def _brands_with_nurofen_loreal():
    entries = [BrandEntry(canonical="nurofen", aliases=[]), BrandEntry(canonical="loreal", aliases=["l oreal"])]
    return build_brand_list("2026-01-01", entries)


def _match(lines, brands, audio_brands=(), match_lists=("canonical", "audio")):
    candidates = build_candidate_set(brands, list(audio_brands), list(match_lists))
    return match_ocr_text([normalise(x) for x in lines], candidates, SCORER, MCFG)


# ---------------------------------------------------------------------------
# Definition-of-done unit tests
# ---------------------------------------------------------------------------

def test_nurofen_fuzzy_100():
    r = _match(["NUROFEN"], _brands_with_nurofen_loreal())
    assert r["ocr_match_mode"] == "fuzzy"
    assert r["ocr_fuzzy_score"] == 100
    assert r["ocr_brand_match"] == "nurofen"


def test_hp_exact_short_100():
    r = _match(["hp"], _small_brands())
    assert r["ocr_match_mode"] == "exact_short"
    assert r["ocr_fuzzy_score"] == 100
    assert r["ocr_brand_match"] == "hp"


def test_hb_vs_hp_scores_zero():
    r = _match(["hb"], _small_brands())
    assert r["ocr_match_mode"] == "exact_short"
    assert r["ocr_fuzzy_score"] == 0
    assert r["ocr_brand_match"] is None


def test_rai_sport_hd_matches_via_alias():
    r = _match(["RAI SPORT HD"], _small_brands())
    assert r["ocr_brand_match"] == "rai sport"
    assert r["ocr_match_list"] == "canonical"


def test_very_sure_from_audio_resolves_to_canonical_alias():
    """'very sure' is both an alias of the canonical 'verisure' AND (in this
    scenario) an audio-extracted brand string; T26's tie-break prefers
    canonical, so the alias wins even though the string came from audio."""
    r = _match(["very sure"], _small_brands(), audio_brands=["very sure"])
    assert r["ocr_brand_match"] == "verisure"
    assert r["ocr_match_list"] == "canonical"


def test_accent_stripping_loreal():
    r = _match(["L'Oréal"], _brands_with_nurofen_loreal())
    assert r["ocr_brand_match"] == "loreal"
    assert r["ocr_match_mode"] == "fuzzy"
    assert r["ocr_fuzzy_score"] == 100


# ---------------------------------------------------------------------------
# T28 — no auto-quarantine: score kept, candidate/query logged, even below threshold
# ---------------------------------------------------------------------------

def test_below_threshold_fuzzy_keeps_score_and_candidate_but_no_match():
    # "nurofin" is close to "nurofen" but likely won't hit 100; regardless, whatever
    # the best score is, brand_match should be null unless it clears the threshold.
    r = _match(["completely unrelated text"], _brands_with_nurofen_loreal())
    assert r["ocr_match_candidate"] is not None    # diagnostic: still records what it scored against
    assert r["ocr_match_query"] == "completely unrelated text"
    if r["ocr_fuzzy_score"] < MCFG.general_fuzzy_threshold:
        assert r["ocr_brand_match"] is None
    assert r["ocr_fuzzy_score"] >= 0


def test_no_lines_returns_all_null():
    r = _match([], _small_brands())
    assert r == {"ocr_brand_match": None, "ocr_fuzzy_score": 0, "ocr_match_list": None,
                 "ocr_match_mode": None, "ocr_match_candidate": None, "ocr_match_query": None}


def test_match_lists_can_exclude_audio():
    r = _match(["very sure"], _small_brands(), audio_brands=["very sure"], match_lists=("canonical",))
    assert r["ocr_match_list"] == "canonical"   # unchanged, since audio was never a candidate anyway
    candidates = build_candidate_set(_small_brands(), ["very sure"], ["canonical"])
    assert all(c["list"] == "canonical" for c in candidates)


def test_concatenation_query_matches_wordmark_split_across_lines():
    r = _match(["rai", "sport"], _small_brands())
    assert r["ocr_brand_match"] == "rai sport"
    assert r["ocr_match_query"] == "rai sport"


# ---------------------------------------------------------------------------
# Stage integration
# ---------------------------------------------------------------------------

class _FakeReader:
    def __init__(self, results):
        self._results = results

    def readtext(self, img_array):
        return self._results


def _patched_reader(results):
    return patch("logo_pipeline.stages.ocr.get_reader", return_value=_FakeReader(results))


def test_ocr_stage_end_to_end_populates_match_fields(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(1920, 1080))
    ctx.write_jsonl("merged_detections.jsonl", [{
        "det_id": 0, "frame_idx": 0, "frame_width": 1920, "frame_height": 1080,
        "loose_bbox": [100.0, 100.0, 300.0, 300.0], "box_xyxy": [100.0, 100.0, 300.0, 300.0],
        "box_area_ratio": 0.02, "box_touches_edge": False, "det_score": 0.9,
        "detection_sources": ["gdino"], "merge_count": 0, "merged_raw_det_ids": [], "seed_raw_det_id": 0,
        "gdino_label": "a logo", "yolo_class": None, "yolo_class_conf": None,
    }], header=ctx.header("merge"))
    ctx.write_jsonl("refinement.jsonl", [{
        "det_id": 0, "refined_bbox": None, "refinement_method": "none", "refinement_status": "no_candidate",
        "selected_bbox": [100.0, 100.0, 300.0, 300.0], "has_valid_tight_bbox": False,
        "refinement_area_ratio": None,
    }], header=ctx.header("refine"))

    with _patched_reader([([[0, 0], [50, 0], [50, 20], [0, 20]], "NIKE", 0.95)]):
        OcrStage().run(ctx)

    row = ctx.read_jsonl("ocr.jsonl")[0]
    assert row["ocr_text"] == "nike"
    assert row["ocr_text_raw"] == "NIKE"
    assert row["ocr_source"] == "tight_crop"
    assert row["ocr_brand_match"] == "nike"
    assert row["ocr_match_list"] == "canonical"
    assert row["ocr_fuzzy_score"] == 100


def test_ocr_stage_no_text_leaves_match_fields_null(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(1920, 1080))
    ctx.write_jsonl("merged_detections.jsonl", [{
        "det_id": 0, "frame_idx": 0, "frame_width": 1920, "frame_height": 1080,
        "loose_bbox": [100.0, 100.0, 300.0, 300.0], "box_xyxy": [100.0, 100.0, 300.0, 300.0],
        "box_area_ratio": 0.02, "box_touches_edge": False, "det_score": 0.9,
        "detection_sources": ["gdino"], "merge_count": 0, "merged_raw_det_ids": [], "seed_raw_det_id": 0,
        "gdino_label": "a logo", "yolo_class": None, "yolo_class_conf": None,
    }], header=ctx.header("merge"))
    ctx.write_jsonl("refinement.jsonl", [{
        "det_id": 0, "refined_bbox": None, "refinement_method": "none", "refinement_status": "no_candidate",
        "selected_bbox": [100.0, 100.0, 300.0, 300.0], "has_valid_tight_bbox": False,
        "refinement_area_ratio": None,
    }], header=ctx.header("refine"))

    with _patched_reader([]):
        OcrStage().run(ctx)

    row = ctx.read_jsonl("ocr.jsonl")[0]
    assert row["ocr_text"] == ""
    assert row["ocr_brand_match"] is None
    assert row["ocr_fuzzy_score"] == 0
