"""spec docs/09 — MM3: heuristic 1 (provenance), heuristic 2 (OCR-guided
tightening) and heuristic 5 (oversized-box guard). Synthetic frames, real
brand list, EasyOCR mocked via a fake reader so this runs without loading
the model."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.brands import load_brand_list
from logo_pipeline.config import Heuristic2Cfg, OcrCfg
from logo_pipeline.geometry import box_area_ratio
from logo_pipeline.stages.refine import RefineStage, _match_audio, ocr_guided_candidate, passes_safety

FRAME_W, FRAME_H = 1920, 1080

H2CFG = Heuristic2Cfg()   # v1 defaults: ocr_min_confidence 0.40, brand_match_min_score 80, padding_ratio 0.10,
                          # min_area_ratio_vs_loose 0.05, max_area_ratio_vs_loose 0.95, min_side_px 12, max_aspect 8.0
OCFG = OcrCfg()


class _FakeReader:
    def __init__(self, results):
        self._results = results

    def readtext(self, img_array):
        return self._results


def _patched_reader(results):
    return patch("logo_pipeline.stages.refine.get_reader", return_value=_FakeReader(results))


def _merged_row(det_id, loose, frame_idx=0, w=FRAME_W, h=FRAME_H, score=0.9, source="gdino"):
    return {
        "det_id": det_id, "frame_idx": frame_idx, "frame_width": w, "frame_height": h,
        "loose_bbox": loose, "box_xyxy": loose, "box_area_ratio": round(box_area_ratio(loose, w, h), 6),
        "box_touches_edge": False, "det_score": score, "detection_sources": [source], "merge_count": 0,
        "merged_raw_det_ids": [], "seed_raw_det_id": det_id, "gdino_label": "a logo", "yolo_class": None,
        "yolo_class_conf": None,
    }


# ---------------------------------------------------------------------------
# Pure-function tests: ocr_guided_candidate / passes_safety / _match_audio
# ---------------------------------------------------------------------------

def _brands():
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    return load_brand_list(fixtures_dir / "brands_small.yaml", require_hash_match=False)


def _crop(w=200, h=200):
    from PIL import Image
    return Image.new("RGB", (w, h), (30, 30, 30))


def test_ocr_guided_candidate_matches_brand_and_pads_and_offsets():
    loose = [100.0, 100.0, 300.0, 300.0]   # 200x200
    quad = [[20, 20], [100, 20], [100, 60], [20, 60]]   # 80x40, crop-local
    reader = _FakeReader([(quad, "nike", 0.9)])
    box, reason, log = ocr_guided_candidate(reader, _crop(200, 200), loose, H2CFG, OCFG, _brands(), [])
    assert box is not None
    assert reason.startswith("matched:nike:")
    assert log[0]["matched_brand"] == "nike"
    # padded + offset into frame coords, still inside the loose box
    assert loose[0] <= box[0] and box[2] <= loose[2]
    assert loose[1] <= box[1] and box[3] <= loose[3]


def test_ocr_guided_candidate_low_confidence_text_is_ignored():
    loose = [0.0, 0.0, 200.0, 200.0]
    quad = [[20, 20], [100, 20], [100, 60], [20, 60]]
    reader = _FakeReader([(quad, "nike", 0.1)])   # below ocr_min_confidence 0.40
    box, reason, log = ocr_guided_candidate(reader, _crop(200, 200), loose, H2CFG, OCFG, _brands(), [])
    assert box is None
    assert reason == "no_matching_text"
    assert log[0]["matched_brand"] is None


def test_ocr_guided_candidate_no_text_returns_none():
    reader = _FakeReader([])
    box, reason, log = ocr_guided_candidate(reader, _crop(), [0.0, 0.0, 200.0, 200.0], H2CFG, OCFG, _brands(), [])
    assert box is None
    assert reason == "no_matching_text"
    assert log == []


def test_ocr_guided_candidate_falls_back_to_audio_brand():
    loose = [0.0, 0.0, 200.0, 200.0]
    quad = [[20, 20], [100, 20], [100, 60], [20, 60]]
    reader = _FakeReader([(quad, "acme corp", 0.9)])   # not in the brand list at all
    box, reason, log = ocr_guided_candidate(reader, _crop(200, 200), loose, H2CFG, OCFG, _brands(), ["acme corp"])
    assert box is not None
    assert log[0]["match_mode"] == "audio"


def test_match_audio_short_string_needs_exact_match():
    assert _match_audio("hp", ["hp", "nike"], 90, 4) == ("hp", 100)
    assert _match_audio("hx", ["hp", "nike"], 90, 4) == (None, 0)


def test_passes_safety_rejects_tiny_area_ratio():
    loose = [0.0, 0.0, 200.0, 200.0]        # area 40000
    candidate = [10.0, 10.0, 26.0, 26.0]     # 16x16 = 256, ratio 0.0064 < 0.05
    ok, checks = passes_safety(candidate, loose, H2CFG)
    assert ok is False
    assert checks["area_ratio_vs_loose"] is False


def test_passes_safety_accepts_well_formed_candidate():
    loose = [0.0, 0.0, 200.0, 200.0]
    candidate = [20.0, 20.0, 120.0, 70.0]    # 100x50 = 5000, ratio 0.125
    ok, checks = passes_safety(candidate, loose, H2CFG)
    assert ok is True
    assert all(checks.values())


# ---------------------------------------------------------------------------
# Stage integration: the four Definition-of-done synthetic scenarios
# ---------------------------------------------------------------------------

def test_refine_applied_when_text_box_inside_loose(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(FRAME_W, FRAME_H))
    loose = [100.0, 100.0, 300.0, 300.0]
    ctx.write_jsonl("merged_detections.jsonl", [_merged_row(0, loose)], header=ctx.header("merge"))
    quad = [[20, 20], [100, 20], [100, 60], [20, 60]]
    with _patched_reader([(quad, "nike", 0.9)]):
        RefineStage().run(ctx)
    row = ctx.read_jsonl("refinement.jsonl")[0]
    assert row["refinement_status"] == "applied"
    assert row["refinement_method"] == "ocr_guided"
    assert row["has_valid_tight_bbox"] is True
    assert row["selected_bbox"] == row["refined_bbox"]
    assert row["refinement_area_ratio"] is not None


def test_refine_rejected_safety_check_when_candidate_too_small(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(FRAME_W, FRAME_H))
    loose = [100.0, 100.0, 300.0, 300.0]   # 200x200
    ctx.write_jsonl("merged_detections.jsonl", [_merged_row(0, loose)], header=ctx.header("merge"))
    quad = [[10, 10], [26, 10], [26, 26], [10, 26]]   # tiny -> candidate area ratio ~1-2%
    with _patched_reader([(quad, "nike", 0.9)]):
        RefineStage().run(ctx)
    row = ctx.read_jsonl("refinement.jsonl")[0]
    assert row["refinement_status"] == "rejected_safety_check"
    assert row["refinement_method"] == "ocr_guided"
    assert row["refined_bbox"] is None
    assert row["has_valid_tight_bbox"] is False
    assert row["selected_bbox"] == loose
    assert row["safety_results"]["area_ratio_vs_loose"] is False


def test_refine_oversized_region_when_60pct_frame_box_has_no_text(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(FRAME_W, FRAME_H))
    loose = [0.0, 0.0, 1920.0, 648.0]   # exactly 60% of the frame area
    ctx.write_jsonl("merged_detections.jsonl", [_merged_row(0, loose)], header=ctx.header("merge"))
    with _patched_reader([]):
        RefineStage().run(ctx)
    row = ctx.read_jsonl("refinement.jsonl")[0]
    assert row["oversized"] is True
    assert row["refinement_status"] == "oversized_region"
    assert row["refinement_method"] == "oversized_guard"
    assert row["has_valid_tight_bbox"] is False
    assert row["selected_bbox"] == loose


def test_refine_no_candidate_when_no_text_and_not_oversized(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(FRAME_W, FRAME_H))
    loose = [100.0, 100.0, 300.0, 300.0]
    ctx.write_jsonl("merged_detections.jsonl", [_merged_row(0, loose)], header=ctx.header("merge"))
    with _patched_reader([]):
        RefineStage().run(ctx)
    row = ctx.read_jsonl("refinement.jsonl")[0]
    assert row["refinement_status"] == "no_candidate"
    assert row["refinement_method"] == "none"
    assert row["refined_bbox"] is None
    assert row["selected_bbox"] == loose
    assert row["refinement_area_ratio"] is None


def test_refine_oversized_wins_over_rejected_safety_check(tmp_path, repo_root):
    """T22: 'the guard wins over rejected/no_candidate' — even with a
    (too-small) OCR candidate, an oversized loose box reports
    oversized_region, not rejected_safety_check."""
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(FRAME_W, FRAME_H))
    loose = [0.0, 0.0, 1920.0, 648.0]   # 60% of frame -> oversized
    ctx.write_jsonl("merged_detections.jsonl", [_merged_row(0, loose)], header=ctx.header("merge"))
    quad = [[10, 10], [26, 10], [26, 26], [10, 26]]   # tiny -> fails safety
    with _patched_reader([(quad, "nike", 0.9)]):
        RefineStage().run(ctx)
    row = ctx.read_jsonl("refinement.jsonl")[0]
    assert row["refinement_status"] == "oversized_region"
    assert row["refinement_method"] == "ocr_guided"   # per T22 pseudocode: only status is overridden


def test_refine_box_area_ratio_matches_merge(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(FRAME_W, FRAME_H))
    loose = [50.0, 50.0, 400.0, 200.0]
    rep = _merged_row(0, loose)
    ctx.write_jsonl("merged_detections.jsonl", [rep], header=ctx.header("merge"))
    with _patched_reader([]):
        RefineStage().run(ctx)
    row = ctx.read_jsonl("refinement.jsonl")[0]
    assert row["box_area_ratio"] == rep["box_area_ratio"]


def test_refine_disabled_produces_no_candidate_for_everything(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(FRAME_W, FRAME_H),
                    overrides={"refinement": {"enabled": False}})
    loose = [0.0, 0.0, 1920.0, 648.0]   # would be oversized if the guard ran
    ctx.write_jsonl("merged_detections.jsonl", [_merged_row(0, loose)], header=ctx.header("merge"))
    RefineStage().run(ctx)   # no reader patch needed: refinement.enabled false never calls get_reader
    row = ctx.read_jsonl("refinement.jsonl")[0]
    assert row["refinement_status"] == "no_candidate"
    assert row["selected_bbox"] == loose
