"""21_test_plan.md — MM1: detect_gdino resumability (per-frame skip) and
min_crop_px filtering, with the model mocked out so this runs without torch
downloading/loading the real Grounding DINO weights."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.stages.detect_gdino import DetectGdinoStage


class _FakePipeline:
    """Stands in for DataExtractionPipeline: one big box, one tiny box that
    min_crop_px should drop, and a per-frame score that varies so tests can
    tell frames apart."""

    def __init__(self, config):
        self.config = config
        self.calls = 0

    def _run_detection(self, pil_image):
        self.calls += 1
        boxes = [[10.0, 10.0, 60.0, 60.0], [5.0, 5.0, 8.0, 8.0]]   # second box is 3x3 px
        scores = [0.9, 0.5]
        labels = ["a logo", "a brand"]
        return boxes, scores, labels


def _patched():
    return patch("data_extraction_pipeline.DataExtractionPipeline", _FakePipeline)


def test_detect_gdino_writes_boxes_and_applies_min_crop_px(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2, frame_size=(200, 100))
    with _patched():
        result = DetectGdinoStage().run(ctx)
    assert result.counts["frames_processed"] == 2
    assert result.counts["boxes_written"] == 2   # the 3x3 box is dropped (min_crop_px 16)

    header, rows = ctx.read_jsonl("raw_detections.jsonl", with_header=True)
    assert header["stage"] == "detect_gdino"
    assert len(rows) == 2
    assert all(r["source"] == "gdino" for r in rows)
    assert {r["raw_det_id"] for r in rows} == {0, 1}


class _ExplodingPipeline:
    """Proves the second call never even imports the real detector: the
    stage's own todo-list check (base.outputs_exist / the todo filter)
    must short-circuit before `DataExtractionPipeline(...)` is reached."""

    def __init__(self, *a, **kw):
        raise AssertionError("DataExtractionPipeline should not be instantiated when nothing is left to do")


def test_detect_gdino_skips_frames_already_done(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2, frame_size=(200, 100))
    with _patched():
        DetectGdinoStage().run(ctx)
    ctx.invalidate()

    with patch("data_extraction_pipeline.DataExtractionPipeline", _ExplodingPipeline):
        result = DetectGdinoStage().run(ctx)   # raises if it tries to instantiate the detector
    assert result.counts.get("frames_skipped") == 2
    rows = ctx.read_jsonl("raw_detections.jsonl")
    assert len(rows) == 2   # unchanged


def test_detect_gdino_force_rewrites(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(200, 100))
    with _patched():
        DetectGdinoStage().run(ctx)
    ctx.invalidate()
    ctx.force = True
    with _patched():
        DetectGdinoStage().run(ctx)
    rows = ctx.read_jsonl("raw_detections.jsonl")
    # --force re-processes the frame; with append semantics this doubles the rows
    # (documented behaviour: --force on detect_gdino is meant to be paired with
    # a fresh run directory or a manual clear, matching the 03 idempotency model
    # where --force means "recompute", not "deduplicate").
    assert len(rows) == 2
