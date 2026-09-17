"""21_test_plan.md — MM1: idempotency (a stage with unchanged outputs and
config_hash is skipped unless --force) and stage-selection slicing."""

from __future__ import annotations

from pathlib import Path

import pytest

from logo_pipeline.runner import run_pipeline, select_stages


def test_select_stages_from_to():
    all_stages = ["ingest", "shots", "sampling", "merge", "output"]
    assert select_stages(all_stages, "shots", "merge") == ["shots", "sampling", "merge"]
    assert select_stages(all_stages, None, "sampling") == ["ingest", "shots", "sampling"]
    assert select_stages(all_stages, "merge", None) == ["merge", "output"]


def test_select_stages_explicit_subset_preserves_pipeline_order():
    all_stages = ["ingest", "shots", "sampling", "merge", "output"]
    assert select_stages(all_stages, None, None, only=["output", "ingest"]) == ["ingest", "output"]


def test_select_stages_rejects_unknown():
    with pytest.raises(ValueError):
        select_stages(["ingest", "shots"], "not_a_stage", None)


@pytest.mark.skipif(not Path("fixed-images").is_dir(), reason="fixed-images/ not present")
def test_ingest_stage_is_idempotent_and_force_reruns(tmp_path, repo_root):
    kwargs = dict(
        pipeline_yaml=str(repo_root / "configs" / "pipeline.yaml"), video_yaml=None,
        repo_root=repo_root, frames_dir="fixed-images", video_id="14",
        only_stages=["ingest"], run_id="idempotency-test",
    )
    ctx1 = run_pipeline(**kwargs)
    status1 = (ctx1.run_dir / "stage_status.json").read_text()
    mtime1 = (ctx1.run_dir / "run_manifest.json").stat().st_mtime

    ctx2 = run_pipeline(**kwargs)   # same config_hash, outputs already exist -> skipped
    mtime2 = (ctx2.run_dir / "run_manifest.json").stat().st_mtime
    assert mtime1 == mtime2
    assert ctx1.run_dir == ctx2.run_dir

    kwargs_forced = dict(kwargs, force=True)
    import time
    time.sleep(0.01)
    ctx3 = run_pipeline(**kwargs_forced)
    mtime3 = (ctx3.run_dir / "run_manifest.json").stat().st_mtime
    assert mtime3 > mtime1   # --force actually re-ran the stage
