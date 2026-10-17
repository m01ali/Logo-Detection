"""spec docs/08 — MM2: T12 (YOLO detection) and T13-T19 (pooled greedy
merge). Pure-function tests for evaluate_pair/merge_frame need no heavy
deps; MergeStage/DetectYoloStage integration tests use the synthetic
RunContext fixture with ultralytics mocked out (like test_mm1_detect_gdino
mocks DataExtractionPipeline)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.config import MergeCfg
from logo_pipeline.stages.detect_yolo import DetectYoloStage
from logo_pipeline.stages.merge import MergeStage, evaluate_pair, merge_frame

# ---------------------------------------------------------------------------
# T13 — evaluate_pair / merge_frame pure functions
# ---------------------------------------------------------------------------

CFG = MergeCfg()   # v1 defaults: iou 0.70, containment 0.90, area_ratio 0.80, min_merge_area 0.0004


def _det(raw_det_id, box, score=0.9, source="gdino", label="a logo", class_conf=None):
    return {"raw_det_id": raw_det_id, "source": source, "box_xyxy": box, "score": score,
            "label": label, "class_conf": class_conf}


def test_identical_boxes_merge():
    a = _det(0, [100, 100, 200, 200])
    b = _det(1, [100, 100, 200, 200])
    m = evaluate_pair(a, b, 1920, 1080, CFG)
    assert m["outcome"] == "merged"
    assert m["iou"] == 1.0
    assert m["pair_type"] == "intra"


def test_tight_vs_padded_area_ratio_0_4_does_not_merge():
    # b fully contained in a (containment 1.0, passes overlap), but area_ratio ~0.4 < 0.80.
    a = _det(0, [0, 0, 100, 100])            # area 10000
    b = _det(1, [0, 0, 63, 64])               # area 4032, ratio 0.4032
    m = evaluate_pair(a, b, 1920, 1080, CFG)
    assert m["contain"] >= CFG.containment_threshold
    assert 0.35 < m["area_ratio"] < 0.45
    assert m["outcome"] == "failed_size_precondition"


def test_contained_tiny_box_area_ratio_0_05_does_not_merge():
    a = _det(0, [0, 0, 1000, 1000])           # area 1,000,000 on a big frame
    b = _det(1, [0, 0, 224, 223])              # area 49952, ratio ~0.05
    m = evaluate_pair(a, b, 2000, 2000, CFG)   # smaller_fraction well above the floor
    assert m["smaller_fraction"] >= CFG.min_merge_area
    assert m["contain"] >= CFG.containment_threshold
    assert m["area_ratio"] < 0.06
    assert m["outcome"] == "failed_size_precondition"


def test_floor_skips_tiny_pair_on_1920x1080():
    a = _det(0, [0, 0, 15, 15])
    b = _det(1, [0, 0, 15, 15])
    m = evaluate_pair(a, b, 1920, 1080, CFG)
    assert m["smaller_fraction"] < CFG.min_merge_area
    assert m["outcome"] == "skipped_area_floor"


def test_evaluate_pair_logs_metrics_even_when_not_merged():
    a = _det(0, [0, 0, 100, 100])
    b = _det(1, [500, 500, 600, 600])   # far away, no overlap, both above the area floor
    m = evaluate_pair(a, b, 1920, 1080, CFG)
    assert m["outcome"] == "no_overlap"
    for key in ("iou", "contain", "area_ratio", "smaller_fraction", "pair_type"):
        assert key in m


def test_merge_frame_greedy_order_is_deterministic_regardless_of_input_order():
    dets = [
        _det(2, [10, 10, 60, 60], score=0.4),
        _det(0, [12, 12, 58, 58], score=0.95),   # highest score -> seed of cluster 1
        _det(1, [500, 500, 560, 560], score=0.6),
    ]
    reps_a, _ = merge_frame(dets, 1920, 1080, CFG)
    reps_b, _ = merge_frame(list(reversed(dets)), 1920, 1080, CFG)
    assert [r["seed_raw_det_id"] for r in reps_a] == [r["seed_raw_det_id"] for r in reps_b]
    assert reps_a[0]["seed_raw_det_id"] == 0   # score 0.95 wins the first seed slot
    assert reps_a[0]["merge_count"] == 1
    assert set(reps_a[0]["merged_raw_det_ids"]) == {2}
    assert reps_a[1]["seed_raw_det_id"] == 1


def test_merge_frame_keeps_tighter_box_as_representative():
    loose = _det(0, [0, 0, 100, 100], score=0.9)     # bigger box, higher score -> seed; area 10000
    tight = _det(1, [5, 5, 95, 95], score=0.5)        # contained, area 8100, ratio 0.81 >= 0.80
    reps, _ = merge_frame([loose, tight], 1920, 1080, CFG)
    assert len(reps) == 1
    assert reps[0]["loose_bbox"] == tight["box_xyxy"]   # tighter box wins, not the seed
    assert reps[0]["det_score"] == 0.5                  # its own score, not the seed's
    assert reps[0]["seed_raw_det_id"] == 0


def test_cross_source_merge_records_both_sources():
    g = _det(0, [10, 10, 60, 60], score=0.9, source="gdino", label="a logo")
    y = _det(1, [12, 12, 58, 58], score=0.8, source="yolo", label="rai", class_conf=0.8)
    reps, log = merge_frame([g, y], 1920, 1080, CFG)
    assert len(reps) == 1
    assert reps[0]["detection_sources"] == ["gdino", "yolo"]
    assert log[0]["pair_type"] == "cross"
    assert log[0]["outcome"] == "merged"


# ---------------------------------------------------------------------------
# T17 — no legacy filters; only min_crop_px at detection, not in the merge
# ---------------------------------------------------------------------------

def test_small_and_large_boxes_both_survive_merge_stage(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(1920, 1080))
    dets = [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [0, 0, 30, 30],
         "score": 0.9, "label": "small", "class_conf": None, "frame_width": 1920, "frame_height": 1080},
        {"raw_det_id": 1, "frame_idx": 0, "source": "gdino", "box_xyxy": [0, 0, 1600, 700],
         "score": 0.8, "label": "big", "class_conf": None, "frame_width": 1920, "frame_height": 1080},
    ]
    header = ctx.header("detect_gdino", detector_config={})
    ctx.write_jsonl("raw_detections.jsonl", dets, header=header)

    MergeStage().run(ctx)
    reps = ctx.read_jsonl("merged_detections.jsonl")
    assert len(reps) == 2   # neither the 30x30 nor the ~60%-of-frame box was filtered out


def test_no_legacy_filter_constants_in_merge_stage():
    import re
    src = Path("logo_pipeline/stages/merge.py").read_text()
    assert "500" not in src
    assert not re.search(r"(?<![\w.])0\.5(?![\w.])", src)


# ---------------------------------------------------------------------------
# T18 — merge counters (real algorithm, default merge.enabled: true)
# ---------------------------------------------------------------------------

def test_merge_counters_sum_correctly(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(1920, 1080))
    dets = [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [10, 10, 60, 60],
         "score": 0.9, "label": "a logo", "class_conf": None, "frame_width": 1920, "frame_height": 1080},
        {"raw_det_id": 1, "frame_idx": 0, "source": "yolo", "box_xyxy": [12, 12, 58, 58],
         "score": 0.8, "label": "rai", "class_conf": 0.8, "frame_width": 1920, "frame_height": 1080},
        {"raw_det_id": 2, "frame_idx": 0, "source": "gdino", "box_xyxy": [1000, 800, 1050, 850],
         "score": 0.5, "label": "far away", "class_conf": None, "frame_width": 1920, "frame_height": 1080},
    ]
    header = ctx.header("detect_gdino", detector_config={})
    ctx.write_jsonl("raw_detections.jsonl", dets, header=header)

    MergeStage().run(ctx)
    counters = ctx.read_json("merge_counters.json")
    v = counters["video"]
    assert v["boxes_total_raw"] == 3
    assert v["boxes_remaining"] == 2   # the two overlapping boxes merged into one
    assert v["boxes_total_raw"] - v["boxes_remaining"] == v["vlm_calls_saved"]
    assert v["pairs_merged_cross"] == 1   # gdino+yolo pair
    assert v["remaining_by_sources"]["gdino+yolo"] == 1
    assert v["remaining_by_sources"]["gdino"] == 1
    # every sampled frame appears, including ones with zero boxes
    assert counters["frames"]["0"]["sampled"] is True


def test_merge_log_failed_pairs_false_only_persists_merges(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(1920, 1080),
                    overrides={"merge": {"log_failed_pairs": False}})
    dets = [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [10, 10, 60, 60],
         "score": 0.9, "label": "a", "class_conf": None, "frame_width": 1920, "frame_height": 1080},
        {"raw_det_id": 1, "frame_idx": 0, "source": "gdino", "box_xyxy": [1000, 800, 1050, 850],
         "score": 0.5, "label": "b", "class_conf": None, "frame_width": 1920, "frame_height": 1080},
    ]
    header = ctx.header("detect_gdino", detector_config={})
    ctx.write_jsonl("raw_detections.jsonl", dets, header=header)
    MergeStage().run(ctx)
    log_rows = ctx.read_jsonl("merge_log.jsonl")
    assert log_rows == []   # the only pair evaluated was no_overlap, not persisted
    counters = ctx.read_json("merge_counters.json")
    assert counters["video"]["pairs_evaluated"] == 1   # counters still see it


# ---------------------------------------------------------------------------
# T12 — YOLO detection stage
# ---------------------------------------------------------------------------

class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self._xyxy = xyxy
        self._conf = conf
        self._cls = cls

    @property
    def xyxy(self):
        return [MagicMock(tolist=lambda: self._xyxy)]

    @property
    def conf(self):
        return [self._conf]

    @property
    def cls(self):
        return [self._cls]


class _FakeResult:
    def __init__(self, boxes, orig_shape):
        self.boxes = boxes
        self.orig_shape = orig_shape


class _FakeYolo:
    """Stands in for ultralytics.YOLO: one big box (class 0), one 3x3 box
    (class 1) that min_crop_px should drop."""

    calls = []

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.names = {0: "logo", 1: "brand"}

    def predict(self, source, imgsz, conf, iou, agnostic_nms, verbose):
        _FakeYolo.calls.append({"source": source, "imgsz": imgsz, "conf": conf, "iou": iou,
                                 "agnostic_nms": agnostic_nms, "verbose": verbose})
        boxes = [_FakeBox([10.0, 10.0, 60.0, 60.0], 0.9, 0), _FakeBox([5.0, 5.0, 8.0, 8.0], 0.5, 1)]
        return [_FakeResult(boxes, (100, 200))]   # orig_shape is (h, w)


def _patched_yolo(yolo_cls=_FakeYolo):
    """``ultralytics`` isn't installed on this Mac (only confirmed-absent GPU
    dep), so patch via sys.modules rather than ``patch("ultralytics.YOLO")``,
    which would need to import the real package just to find the target."""
    _FakeYolo.calls = []
    fake_module = types.SimpleNamespace(YOLO=yolo_cls)
    return patch.dict(sys.modules, {"ultralytics": fake_module})


def test_detect_yolo_writes_boxes_and_applies_min_crop_px(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2, frame_size=(200, 100))
    with _patched_yolo():
        result = DetectYoloStage().run(ctx)
    assert result.counts["frames_processed"] == 2
    assert result.counts["boxes_written"] == 2   # the 3x3 box is dropped (min_crop_px 16, shared w/ gdino)

    header, rows = ctx.read_jsonl("raw_detections.jsonl", with_header=True)
    assert header["stage"] == "detect_yolo"
    assert len(rows) == 2
    assert all(r["source"] == "yolo" for r in rows)
    assert all(r["label"] == "logo" for r in rows)          # names[0]
    assert all(r["class_conf"] == 0.9 for r in rows)


def test_detect_yolo_calls_predict_class_agnostic(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(200, 100))
    with _patched_yolo():
        DetectYoloStage().run(ctx)
    assert len(_FakeYolo.calls) == 1
    call = _FakeYolo.calls[0]
    assert call["agnostic_nms"] is True
    assert call["iou"] == ctx.cfg.detectors.yolo.iou_nms
    assert call["conf"] == ctx.cfg.detectors.yolo.conf
    assert call["imgsz"] == ctx.cfg.detectors.yolo.imgsz


def test_detect_yolo_records_config_in_run_manifest(tmp_path, repo_root):
    """detect_gdino always runs before detect_yolo in DEFAULT_STAGES and both
    append to the same raw_detections.jsonl, whose single header only ever
    keeps the first writer's detector_config — so YOLO's settings must be
    recoverable from run_manifest.json instead."""
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(200, 100))
    with _patched_yolo():
        DetectYoloStage().run(ctx)
    manifest = ctx.read_json("run_manifest.json")
    assert manifest["detectors"]["yolo"]["weights"] == ctx.cfg.detectors.yolo.weights
    assert manifest["detectors"]["yolo"]["conf"] == ctx.cfg.detectors.yolo.conf


def test_detect_yolo_log_predicted_class_false_nulls_label_and_conf(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(200, 100),
                    overrides={"detectors": {"yolo": {"log_predicted_class": False}}})
    with _patched_yolo():
        DetectYoloStage().run(ctx)
    rows = ctx.read_jsonl("raw_detections.jsonl")
    assert rows
    assert all(r["label"] is None and r["class_conf"] is None for r in rows)


def test_detect_yolo_skips_frames_already_done(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2, frame_size=(200, 100))
    with _patched_yolo():
        DetectYoloStage().run(ctx)
    ctx.invalidate()

    class _Exploding:
        def __init__(self, *a, **kw):
            raise AssertionError("YOLO should not be instantiated when nothing is left to do")

    with _patched_yolo(_Exploding):
        result = DetectYoloStage().run(ctx)
    assert result.counts.get("frames_skipped") == 2
    rows = ctx.read_jsonl("raw_detections.jsonl")
    assert len(rows) == 2


def test_detect_yolo_force_rewrites(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, frame_size=(200, 100))
    with _patched_yolo():
        DetectYoloStage().run(ctx)
    ctx.invalidate()
    ctx.force = True
    with _patched_yolo():
        DetectYoloStage().run(ctx)
    rows = ctx.read_jsonl("raw_detections.jsonl")
    # --force replaces this detector's earlier boxes: 1 frame x 1 surviving box (the 3x3 box is
    # dropped by min_crop_px), not one copy per pass.
    assert len(rows) == 1


def test_detect_yolo_frame_with_no_boxes_is_not_reprocessed(tmp_path, repo_root):
    """The Windows video_1 run re-ran one YOLO frame on every resume: a frame with zero boxes
    wrote no line, so outputs_exist() never saw it as done."""
    from logo_pipeline.stages.base import mark_frame_detected
    ctx = make_ctx(tmp_path, repo_root, n_frames=2, frame_size=(200, 100))
    ctx.write_jsonl("raw_detections.jsonl", [
        {"raw_det_id": 0, "frame_idx": 0, "source": "yolo", "box_xyxy": [1, 1, 30, 30], "score": 0.5,
         "label": "x", "class_conf": 0.5, "frame_width": 200, "frame_height": 100}],
        header=ctx.header("detect_yolo"))
    assert not DetectYoloStage().outputs_exist(ctx)      # frame 10 has no line and no mark
    mark_frame_detected(ctx, "yolo", 10)
    assert DetectYoloStage().outputs_exist(ctx)


# ---------------------------------------------------------------------------
# T12 Check — real video_3-trimmed vs the M2 YOLO predict label files
# ---------------------------------------------------------------------------

M2_YOLO_LABELS = Path("runs") / "detect" / "yolo results" / "video_3-trimmed" / "labels"


def _m2_label_box_count(labels_dir: Path, frame_idx: int) -> int:
    """Boxes M2's predict run found on 0-based ``frame_idx``. Ultralytics numbers a video's label
    files from 1 (``<stem>_<n>.txt`` is frame n-1) and writes none for a frame without boxes."""
    label_path = labels_dir / f"video_3-trimmed_{frame_idx + 1}.txt"
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


@pytest.mark.skipif(not Path("video_3-trimmed.mp4").exists(), reason="video_3-trimmed.mp4 not present")
@pytest.mark.skipif(not Path("weights/best.pt").exists(), reason="weights/best.pt not present")
@pytest.mark.skipif(not M2_YOLO_LABELS.is_dir(),
                    reason="M2 YOLO label files not present (runs/ is git-ignored; copy them to run this check)")
def test_detect_yolo_matches_m2_label_counts_on_video_3_trimmed(tmp_path, repo_root):
    pytest.importorskip("ultralytics")
    from logo_pipeline.brands import load_brand_list
    from logo_pipeline.config import resolve_config
    from logo_pipeline.logging_utils import StageLogger
    from logo_pipeline.stages.base import RunContext
    from logo_pipeline.stages.ingest import IngestStage
    from logo_pipeline.stages.sampling import SamplingStage
    from logo_pipeline.stages.shots import ShotsStage

    resolved = resolve_config(
        str(repo_root / "configs" / "pipeline.yaml"), str(repo_root / "configs" / "videos" / "video_3-trimmed.yaml"))
    brands = load_brand_list(repo_root / "configs" / "brands" / "roboflow_2026-04-28.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = StageLogger(run_dir, log_every_n=1000)
    ctx = RunContext(run_dir, resolved, brands, log, repo_root=repo_root)

    IngestStage().run(ctx); ctx.invalidate()
    ShotsStage().run(ctx); ctx.invalidate()
    SamplingStage().run(ctx); ctx.invalidate()
    DetectYoloStage().run(ctx)
    log.close()

    rows = ctx.read_jsonl("raw_detections.jsonl")
    labels_dir = repo_root / M2_YOLO_LABELS
    for f in ctx.sampled_frames():
        frame_idx = f["frame_idx"]
        got = sum(1 for r in rows if r["frame_idx"] == frame_idx)
        want = _m2_label_box_count(labels_dir, frame_idx)
        assert got == want, f"frame {frame_idx}: got {got} boxes, M2 label file has {want}"
