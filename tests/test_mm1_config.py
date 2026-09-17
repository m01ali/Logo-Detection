"""21_test_plan.md — MM1: config loading, hashing, two-switch validation."""

from __future__ import annotations

import pytest

from logo_pipeline.config import ConfigError, default_pipeline_dict, resolve_config


def _min_video():
    return {"video_id": "v1", "source": "v1.mp4"}


def test_unknown_key_fails():
    bad = default_pipeline_dict()
    bad["not_a_real_key"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        resolve_config(bad, _min_video())


def test_unknown_nested_key_fails():
    bad = default_pipeline_dict()
    bad["merge"]["not_a_real_key"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        resolve_config(bad, _min_video())


def test_hash_changes_with_any_value():
    base = resolve_config(default_pipeline_dict(), _min_video())
    changed = default_pipeline_dict()
    changed["merge"]["min_merge_area"] = 0.001
    other = resolve_config(changed, _min_video())
    assert base.config_hash != other.config_hash


def test_hash_stable_for_identical_config():
    r1 = resolve_config(default_pipeline_dict(), _min_video())
    r2 = resolve_config(default_pipeline_dict(), _min_video())
    assert r1.config_hash == r2.config_hash


def test_hash_ignores_run_id_and_log_cadence():
    r1 = resolve_config(default_pipeline_dict(), _min_video(), run_id="run-a")
    r2 = resolve_config(default_pipeline_dict(), _min_video(), run_id="run-b")
    assert r1.config_hash == r2.config_hash


def test_per_video_override_merges():
    video = {"video_id": "v1", "source": "v1.mp4", "overrides": {"merge": {"min_merge_area": 0.002}}}
    r = resolve_config(default_pipeline_dict(), video)
    assert r.pipeline.merge.min_merge_area == 0.002
    assert r.pipeline.merge.iou_threshold == 0.70  # untouched sibling key keeps its default


def test_two_switch_validation_rejects_prompt_without_compute():
    bad = default_pipeline_dict()
    bad["signals"]["ocr"] = {"compute_and_log": False, "include_in_vlm_prompt": True}
    with pytest.raises(ConfigError, match="compute_and_log"):
        resolve_config(bad, _min_video())


def test_two_switch_allows_compute_without_prompt():
    cfg = default_pipeline_dict()
    cfg["signals"]["ocr"] = {"compute_and_log": True, "include_in_vlm_prompt": False}
    r = resolve_config(cfg, _min_video())
    assert r.pipeline.signals.ocr.compute_and_log is True
    assert r.pipeline.signals.ocr.include_in_vlm_prompt is False


def test_frames_mode_requires_dir():
    video = {"video_id": "v14", "overrides": {"ingest": {"mode": "frames"}}}
    with pytest.raises(ConfigError, match="frames.dir"):
        resolve_config(default_pipeline_dict(), video)


def test_frames_mode_forces_content_detector():
    r = resolve_config(default_pipeline_dict(), {"video_id": "v14"}, frames_dir="fixed-images")
    assert r.pipeline.ingest.mode == "frames"
    assert r.effective.get("shots.detector") == "content"


def test_locked_decisions_enforced():
    bad = default_pipeline_dict()
    bad["merge"]["legacy_filters_enabled"] = True
    with pytest.raises(ConfigError, match="legacy_filters_enabled"):
        resolve_config(bad, _min_video())

    bad2 = default_pipeline_dict()
    bad2["router"]["rules"]["rule_8"] = True
    with pytest.raises(ConfigError, match="rule_8"):
        resolve_config(bad2, _min_video())


def test_video_id_required():
    with pytest.raises(ConfigError, match="video_id"):
        resolve_config(default_pipeline_dict(), {"source": "x.mp4"})


def test_video_source_required_in_video_mode():
    with pytest.raises(ConfigError, match="source"):
        resolve_config(default_pipeline_dict(), {"video_id": "v1"})
