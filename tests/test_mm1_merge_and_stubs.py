"""21_test_plan.md — MM1: merge pass-through, refine/canonicalise/watermark/
router stubs, and router_input's join (synthetic fixtures, no heavy deps)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.schema import CSV_COLUMNS
from logo_pipeline.stages.base import dumps_line
from logo_pipeline.stages.canonicalise import CanonicaliseStage
from logo_pipeline.stages.merge import MergeStage
from logo_pipeline.stages.ocr import OcrStage
from logo_pipeline.stages.refine import RefineStage
from logo_pipeline.stages.router.engine import RouterStage
from logo_pipeline.stages.router_input import RouterInputStage
from logo_pipeline.stages.watermark import WatermarkStage


def _write_raw_detections(ctx, dets: list[dict]) -> None:
    header = ctx.header("detect_gdino", detector_config={})
    ctx.write_jsonl("raw_detections.jsonl", dets, header=header)


def test_merge_pass_through_one_rep_per_box(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2, frame_size=(200, 100))
    dets = [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [10, 10, 30, 30],
         "score": 0.9, "label": "a logo", "class_conf": None, "frame_width": 200, "frame_height": 100},
        {"raw_det_id": 1, "frame_idx": 0, "source": "gdino", "box_xyxy": [50, 50, 70, 70],
         "score": 0.5, "label": "a brand", "class_conf": None, "frame_width": 200, "frame_height": 100},
        {"raw_det_id": 2, "frame_idx": 10, "source": "yolo", "box_xyxy": [0, 0, 20, 20],
         "score": 0.7, "label": "rai", "class_conf": 0.7, "frame_width": 200, "frame_height": 100},
    ]
    _write_raw_detections(ctx, dets)

    result = MergeStage().run(ctx)
    assert result.counts["representatives"] == 3
    reps = ctx.read_jsonl("merged_detections.jsonl")
    assert len(reps) == 3
    assert all(r["merge_count"] == 0 for r in reps)
    assert all(r["merged_raw_det_ids"] == [] for r in reps)
    assert {r["det_id"] for r in reps} == {0, 1, 2}
    assert sorted(r["detection_sources"] for r in reps) == [["gdino"], ["gdino"], ["yolo"]]

    log_rows = ctx.read_jsonl("merge_log.jsonl")
    assert log_rows == []   # T16: empty log on the false/pass-through path

    counters = ctx.read_json("merge_counters.json")
    assert counters["video"]["boxes_total_raw"] == 3
    assert counters["video"]["boxes_remaining"] == 3
    assert counters["video"]["pairs_evaluated"] == 0
    assert counters["video"]["vlm_calls_saved"] == 0


def test_merge_det_id_order_is_frame_then_raw_det_id(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2)
    dets = [
        {"raw_det_id": 5, "frame_idx": 10, "source": "gdino", "box_xyxy": [0, 0, 5, 5],
         "score": 0.1, "label": "x", "class_conf": None, "frame_width": 200, "frame_height": 100},
        {"raw_det_id": 1, "frame_idx": 0, "source": "gdino", "box_xyxy": [0, 0, 5, 5],
         "score": 0.1, "label": "x", "class_conf": None, "frame_width": 200, "frame_height": 100},
    ]
    _write_raw_detections(ctx, dets)
    MergeStage().run(ctx)
    reps = ctx.read_jsonl("merged_detections.jsonl")
    # frame_idx 0 (raw_det_id 1) sorts before frame_idx 10 (raw_det_id 5) regardless of raw id order.
    assert [r["frame_idx"] for r in reps] == [0, 10]
    assert [r["det_id"] for r in reps] == [0, 1]


def test_no_legacy_filter_constants_in_merge_stage():
    src = Path("logo_pipeline/stages/merge.py").read_text()
    assert "500" not in src
    assert re_no_bare_half(src)


def re_no_bare_half(src: str) -> bool:
    import re
    # 0.5 must not appear as a bare area-ratio-style literal (area_aspect_filter's max_area_ratio).
    return not re.search(r"(?<![\w.])0\.5(?![\w.])", src)


def test_refine_stub_selected_equals_loose(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    _write_raw_detections(ctx, [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [1, 2, 3, 4],
         "score": 0.9, "label": "a", "class_conf": None, "frame_width": 200, "frame_height": 100}])
    MergeStage().run(ctx)
    RefineStage().run(ctx)
    rows = ctx.read_jsonl("refinement.jsonl")
    assert rows[0]["refinement_status"] == "no_candidate"
    assert rows[0]["refinement_method"] == "none"
    assert rows[0]["has_valid_tight_bbox"] is False
    assert rows[0]["selected_bbox"] == [1, 2, 3, 4]


def test_canonicalise_stub_carries_real_brand_list_provenance(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    _write_raw_detections(ctx, [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [1, 2, 3, 4],
         "score": 0.9, "label": "a", "class_conf": None, "frame_width": 200, "frame_height": 100}])
    MergeStage().run(ctx)
    CanonicaliseStage().run(ctx)
    rows = ctx.read_jsonl("canonical.jsonl")
    assert rows[0]["canonical_brand"] is None
    assert rows[0]["brand_list_version"] == ctx.brands.version
    assert rows[0]["brand_list_hash"] == ctx.brands.content_hash
    assert rows[0]["brand_list_hash"].startswith("sha256:")


def test_watermark_stub_reports_no_regions(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    _write_raw_detections(ctx, [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [1, 2, 3, 4],
         "score": 0.9, "label": "a", "class_conf": None, "frame_width": 200, "frame_height": 100}])
    MergeStage().run(ctx)
    WatermarkStage().run(ctx)
    rows = ctx.read_jsonl("watermark.jsonl")
    assert rows[0]["watermark_regions_available"] is False
    assert rows[0]["watermark_overlap"] is None


def test_router_stub_decision_is_null(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    _write_raw_detections(ctx, [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [1, 2, 3, 4],
         "score": 0.9, "label": "a", "class_conf": None, "frame_width": 200, "frame_height": 100}])
    MergeStage().run(ctx)
    RefineStage().run(ctx)
    CanonicaliseStage().run(ctx)
    WatermarkStage().run(ctx)
    RouterInputStage().run(ctx)
    RouterStage().run(ctx)
    out = ctx.read_jsonl("router_output.jsonl")
    assert out[0]["router_decision"] is None
    assert out[0]["router_rule"] is None
    firing = ctx.read_jsonl("firing_log.jsonl")
    assert firing == []


def test_router_input_covers_every_csv_column(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1)
    _write_raw_detections(ctx, [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [1, 2, 30, 40],
         "score": 0.8, "label": "a logo", "class_conf": None, "frame_width": 200, "frame_height": 100}])
    MergeStage().run(ctx)
    RefineStage().run(ctx)
    CanonicaliseStage().run(ctx)
    WatermarkStage().run(ctx)
    ctx.write_jsonl("ocr.jsonl", [{"det_id": 0, "ocr_text": "", "ocr_text_raw": "", "ocr_source": None,
                                   "ocr_max_confidence": None, "ocr_brand_match": None, "ocr_fuzzy_score": 0,
                                   "ocr_match_list": None, "ocr_match_mode": None}], header=ctx.header("ocr"))
    ctx.write_jsonl("faiss_hints.jsonl", [{"det_id": 0, "faiss_top1": None, "faiss_top1_score": None,
                                           "faiss_top5_brands": [], "faiss_top5_scores": [],
                                           "faiss_score_gap": None}], header=ctx.header("faiss_query"))
    ctx.write_json("audio/audio_hints.json", {"status": "empty", "brands": []})
    RouterInputStage().run(ctx)

    rows = ctx.read_jsonl("router_input.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == set(CSV_COLUMNS)
    assert row["video_id"] == "synthetic"
    assert row["brand_evidence"] == "none"
    assert row["config_hash"] == ctx.config_hash


def test_jsonl_header_written_once_and_round_trips(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1)
    header = ctx.header("detect_gdino", detector_config={"box_threshold": 0.2})
    ctx.write_jsonl("raw_detections.jsonl", [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [1.0, 2.0, 3.0, 4.0],
         "score": 0.5, "label": "x", "class_conf": None, "frame_width": 10, "frame_height": 10},
    ], header=header)
    text = (ctx.run_dir / "raw_detections.jsonl").read_text()
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["_header"] is True
    h, rows = ctx.read_jsonl("raw_detections.jsonl", with_header=True)
    assert h["stage"] == "detect_gdino"
    assert len(rows) == 1
    assert rows[0]["raw_det_id"] == 0


def test_jsonl_appender_resumes_skipping_done_ids(tmp_path):
    from logo_pipeline.stages.base import JsonlAppender
    path = tmp_path / "x.jsonl"
    header = {"_header": True, "stage": "x"}
    with JsonlAppender(path, header, key="det_id") as app:
        app.write({"det_id": 0, "v": "a"})
        app.write({"det_id": 1, "v": "b"})
    with JsonlAppender(path, header, key="det_id") as app2:
        assert app2.done == {0, 1}
        app2.write({"det_id": 2, "v": "c"})
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4  # 1 header + 3 records
