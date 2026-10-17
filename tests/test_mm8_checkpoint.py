"""MM8 / T77 (spec docs/14): the checkpoint self-check, the reports bundle, the
client package, and provenance on every row (21 "Regression and provenance")."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mm5_router_stage import _build  # noqa: E402

from logo_pipeline.checkpoint import checklist, compare_csv, package, render_checklist, run_reports
from logo_pipeline.stages.output import OutputStage
from logo_pipeline.stages.router_input import RouterInputStage


def _run(tmp_path, repo_root, git_sha="abc123"):
    ctx = _build(tmp_path, repo_root)
    ctx.update_manifest(git_sha=git_sha, brand_list_version=ctx.brands.version, brand_list_hash=ctx.brands.content_hash)
    ctx.invalidate()
    RouterInputStage().run(ctx)          # re-join so rows carry the manifest's provenance
    from logo_pipeline.stages.router.engine import RouterStage
    RouterStage().run(ctx)
    OutputStage().run(ctx)
    return ctx


def test_git_sha_config_and_brand_hash_on_every_row(tmp_path, repo_root):
    ctx = _run(tmp_path, repo_root)
    df = pd.read_csv(ctx.path("rows.csv"))
    for col in ("config_hash", "brand_list_version", "brand_list_hash", "git_sha"):
        assert df[col].notna().all() and (df[col].astype(str) != "").all(), col
    assert set(df["git_sha"]) == {"abc123"}


def test_checklist_reports_each_check(tmp_path, repo_root):
    ctx = _run(tmp_path, repo_root)
    items = {c: (ok, d) for c, ok, d in checklist(ctx.run_dir)}
    assert items["Every sampled frame has an annotated image"][0]
    assert items["rows.csv loads without warnings"][0]
    assert items["No null router_decision; unparsed rows show vlm_output_unparsed"][0]
    assert items["The two T65 ordering checks pass"][0]
    assert items["Config hash, brand list version/hash and git SHA on every row"][0]
    # The synthetic fixture has neither an UNKNOWN_LOGO row nor a channel-brand row: reported, not hidden.
    assert not items["At least one channel_brand_* row"][0]
    md = render_checklist(ctx.run_dir, checklist(ctx.run_dir), None)
    assert "FAIL" in md and "m3-checkpoint" in md


def test_checklist_flags_missing_git_sha(tmp_path, repo_root):
    ctx = _run(tmp_path, repo_root, git_sha=None)
    items = {c: ok for c, ok, _ in checklist(ctx.run_dir)}
    assert not items["Config hash, brand list version/hash and git SHA in the manifest"]


def test_compare_csv(tmp_path, repo_root):
    ctx = _run(tmp_path, repo_root)
    copy = tmp_path / "copy.csv"
    copy.write_bytes(ctx.path("rows.csv").read_bytes())
    assert compare_csv(ctx.path("rows.csv"), copy) == (True, "byte-identical")
    copy.write_text("x")
    assert compare_csv(ctx.path("rows.csv"), copy)[0] is False


def test_reports_bundle_and_package(tmp_path, repo_root):
    ctx = _run(tmp_path, repo_root)
    ctx.update_manifest(config=ctx.resolved.resolved, video_config={"video_id": "synthetic", "source": "s.mp4"})
    results = run_reports(ctx.run_dir, repo_root)
    for name in ("metrics", "ordering_checks", "acceptance_evidence"):
        assert results[name] == "ok", results
    for f in ("metrics.md", "ordering_checks.md", "acceptance_evidence.md"):
        assert ctx.path(f"reports/{f}").exists(), f
    out = package(ctx.run_dir, tmp_path / "client", repo_root)
    for f in ("rows.csv", "annotated.mp4", "CHECKPOINT_NOTE.md", "configs/pipeline.yaml", "configs/reason_codes.md",
              "reports/metrics.md"):
        assert (out / f).exists(), f
    df = pd.read_csv(out / "rows.csv")
    assert (out / df.loc[0, "crop_path"]).exists()          # images travel with the CSV
    note = (out / "CHECKPOINT_NOTE.md").read_text()
    assert "What is not yet in this run" in note and "No blind pass" in note
