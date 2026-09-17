"""21_test_plan.md — MM1: sampling. interval on video_1 reproduces the 102
M2 indices; segment on video_3-trimmed reproduces the 4 M2 indices;
per_shot returns one frame per scene."""

from __future__ import annotations

from pathlib import Path

import pytest

from logo_pipeline.stages.sampling import (_interval_frames_positions, _interval_video_indices,
                                            _segment_indices, _shot_id_for)


def test_interval_video_matches_video_1_m2_indices():
    # extraction_output/video_1/manifest.json: fps=30.00004944791404,
    # total_frames=6067, sample_every_n_seconds=2.0 -> 102 M2 indices.
    indices = _interval_video_indices(total=6067, fps=30.00004944791404, interval_s=2.0)
    assert len(indices) == 102
    assert indices[:3] == [0, 60, 120]
    assert indices[-1] == 6060


def test_segment_matches_video_3_trimmed_m2_indices():
    # extraction_output/video_3-trimmed/manifest.json: fps=30.0, total=340,
    # segment_duration_seconds=3.0 -> the 4 M2 sampled frame_idx values.
    indices = _segment_indices(total=340, fps=30.0, duration_s=3.0)
    assert indices == [44, 134, 224, 304]


def test_interval_frames_positions_matches_02_examples():
    # 02 §3.1 / 03 §5.2: default (2 s, 1 fps) processes every second ordinal;
    # interval_s 1.0 at assumed_fps 1.0 processes every ordinal.
    assert _interval_frames_positions(n=46, assumed_fps=1.0, interval_s=2.0) == list(range(0, 46, 2))
    assert _interval_frames_positions(n=46, assumed_fps=1.0, interval_s=1.0) == list(range(46))


def test_shot_id_for_finds_containing_scene():
    scenes = [{"shot_id": 0, "start_frame": 0, "end_frame": 49},
              {"shot_id": 1, "start_frame": 50, "end_frame": 99}]
    assert _shot_id_for(25, scenes) == 0
    assert _shot_id_for(75, scenes) == 1
    assert _shot_id_for(500, scenes) is None


@pytest.mark.skipif(not Path("video_3-trimmed.mp4").exists(), reason="video_3-trimmed.mp4 not present")
def test_sampling_stage_end_to_end_reproduces_m2_frames(tmp_path, repo_root):
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

    IngestStage().run(ctx)
    ctx.invalidate()
    ShotsStage().run(ctx)
    ctx.invalidate()
    SamplingStage().run(ctx)
    log.close()

    sampled = ctx.read_json("sampled_frames.json")
    frame_indices = sorted(f["frame_idx"] for f in sampled)
    assert frame_indices == [44, 134, 224, 304]
    for f in sampled:
        assert (run_dir / f["path"]).exists()
        for rel in f["temporal_frames"].values():
            if rel is not None:
                assert (run_dir / rel).exists()
