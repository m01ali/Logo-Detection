"""T60 (``spec docs/13``): the fresh-video run harness.

``python -m logo_pipeline harness --config … --video-config …`` runs the
pipeline up to the merge (``--to-stage merge``; ``--full`` for every stage on
the GPU machine), then the T19 merge ablation, and writes
``reports/harness.md``: frame counts, run time per stage, the T18 counters per
video and per frame, the T19 summary, and — when the GPU stages ran — the
decision distribution. No labels are needed; the recall figure from labelled
fresh frames is ``eval/iou_match.py``'s job (MM11).

On the Mac it runs from a copied run directory whose raw detections already
exist (the detector stages are skipped as done).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from .stages.base import (MERGE_COUNTERS_FILE, MERGE_LOG_FILE, ROUTER_OUTPUT_FILE, RUN_MANIFEST, SAMPLED_FRAMES_FILE,
                          STAGE_STATUS, RunContext)


def _eval_module(repo_root: Path, name: str):
    path = str(Path(repo_root) / "eval")
    if path not in sys.path:
        sys.path.insert(0, path)
    return __import__(name)


def render(ctx: RunContext, ablation: dict | None) -> str:
    import json
    status = json.loads(ctx.path(STAGE_STATUS).read_text()) if ctx.path(STAGE_STATUS).exists() else {}
    manifest = ctx.read_json(RUN_MANIFEST)
    sampled = ctx.read_json(SAMPLED_FRAMES_FILE) if ctx.path(SAMPLED_FRAMES_FILE).exists() else []
    counters = ctx.read_json(MERGE_COUNTERS_FILE) if ctx.path(MERGE_COUNTERS_FILE).exists() else {}
    lines = [f"# Harness report — `{manifest.get('video_id')}` / `{manifest.get('run_id')}`", "",
             f"- Source mode `{manifest.get('mode')}`, {manifest.get('frame_count')} frames at "
             f"{manifest.get('fps')} fps{' (assumed)' if manifest.get('fps_is_assumed') else ''}; "
             f"{len(sampled)} sampled ({ctx.cfg.sampling.mode}).",
             f"- Config hash `{manifest.get('config_hash')}`, git `{manifest.get('git_sha')}`.", "",
             "## Run time per stage", "", "| stage | status | seconds | counts |", "|---|---|---|---|"]
    for name in ctx.cfg.run.stages:
        st = status.get(name)
        if st:
            lines.append(f"| {name} | {st.get('status')} | {st.get('elapsed_s', '—')} | `{st.get('counts', {})}` |")
    video = counters.get("video", {})
    lines += ["", "## T18 merge counters (video)", "", "| counter | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in video.items()]
    frames = counters.get("frames", {})
    if frames:
        keys = sorted({k for c in frames.values() for k in c if k != "sampled"})
        lines += ["", "## T18 merge counters per frame", "", "| frame | " + " | ".join(keys) + " |",
                  "|" + "---|" * (len(keys) + 1)]
        for fi in sorted(frames, key=int):
            lines.append(f"| {fi} | " + " | ".join(str(frames[fi].get(k, 0)) for k in keys) + " |")
    if ablation is not None:
        lines += ["", "## T19 distributions", "",
                  f"- Pairs evaluated {ablation['n_pairs_evaluated']}; failed size precondition "
                  f"{ablation['n_failed_size_precondition']}; skipped by the area floor {ablation['n_skipped_area_floor']} "
                  f"({ablation['floor_share_of_overlapping']:.1%} of overlapping pairs); extra VLM calls the floor causes "
                  f"{ablation['n_floor_would_have_merged']}.",
                  "- Histograms: `reports/merge_distributions.md` / `.csv`."]
    if ctx.path(ROUTER_OUTPUT_FILE).exists():
        dec = Counter(r.get("router_decision") for r in ctx.read_jsonl(ROUTER_OUTPUT_FILE))
        lines += ["", "## Decision distribution", "", "| decision | rows |", "|---|---|"]
        lines += [f"| {d} | {n} |" for d, n in dec.most_common()]
    return "\n".join(lines) + "\n"


def write_report(ctx: RunContext) -> Path:
    """Runs the T19 ablation over the run's merge log and writes ``reports/harness.md``."""
    ablation = None
    if ctx.path(MERGE_LOG_FILE).exists():
        ma = _eval_module(ctx.repo_root, "merge_ablation")
        header, rows = ma.read_merge_log(ctx.run_dir)
        ablation = ma.analyse(header, rows)
        reports = ctx.path("reports")
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "merge_distributions.md").write_text(ma.render_markdown(ctx.run_dir, ablation), encoding="utf-8")
        ma.write_csv(reports / "merge_distributions.csv", ablation)
    out = ctx.path("reports/harness.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(ctx, ablation), encoding="utf-8")
    return out
