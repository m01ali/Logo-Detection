"""The stage runner: idempotency, ``stage_status.json``, resumability (03
§3). Used by ``cli.py``'s ``run`` command and importable directly for tests.
"""

from __future__ import annotations

import time
from pathlib import Path

from .config import DEFAULT_STAGES, ResolvedConfig, resolve_config
from .logging_utils import StageLogger
from .stages import get_stage
from .stages.base import RUN_MANIFEST, STAGE_STATUS, RunContext, utcnow_iso


def make_run_id(resolved: ResolvedConfig) -> str:
    return resolved.pipeline.run.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{resolved.hash8}"


def run_dir_for(repo_root: Path, resolved: ResolvedConfig, run_id: str) -> Path:
    return repo_root / resolved.pipeline.run.output_root / resolved.video.video_id / run_id


def select_stages(all_stages: list[str], from_stage: str | None, to_stage: str | None,
                   only: list[str] | None = None) -> list[str]:
    if only:
        unknown = [s for s in only if s not in all_stages]
        if unknown:
            raise ValueError(f"unknown stage(s) in --stages: {unknown}")
        return [s for s in all_stages if s in only]
    i0 = all_stages.index(from_stage) if from_stage else 0
    i1 = all_stages.index(to_stage) + 1 if to_stage else len(all_stages)
    if from_stage and from_stage not in all_stages:
        raise ValueError(f"unknown --from-stage {from_stage!r}")
    if to_stage and to_stage not in all_stages:
        raise ValueError(f"unknown --to-stage {to_stage!r}")
    return all_stages[i0:i1]


def load_stage_status(run_dir: Path) -> dict:
    p = run_dir / STAGE_STATUS
    if p.exists():
        import json
        return json.loads(p.read_text())
    return {}


def save_stage_status(run_dir: Path, status: dict) -> None:
    import json
    (run_dir / STAGE_STATUS).write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
    pipeline_yaml: str, video_yaml: str | None, *, repo_root: Path, frames_dir: str | None = None,
    video_id: str | None = None, source: str | None = None, from_stage: str | None = None,
    to_stage: str | None = None, only_stages: list[str] | None = None, force: bool = False,
    run_id: str | None = None, brands_path: str | None = None,
) -> RunContext:
    resolved = resolve_config(pipeline_yaml, video_yaml, frames_dir=frames_dir, video_id=video_id,
                              source=source, run_id=run_id)
    rid = make_run_id(resolved)
    rdir = run_dir_for(repo_root, resolved, rid)
    rdir.mkdir(parents=True, exist_ok=True)

    from .brands import load_brand_list
    bpath = brands_path or resolved.pipeline.brands.list_path
    bpath_full = Path(bpath) if Path(bpath).is_absolute() else repo_root / bpath
    brand_list = load_brand_list(bpath_full, resolved.pipeline.brands.require_hash_match)

    log = StageLogger(rdir, resolved.pipeline.run.log_every_n)
    ctx = RunContext(rdir, resolved, brand_list, log, force=force, repo_root=repo_root)
    # run_id must be resolvable via ctx.manifest before ingest writes it (idempotent resume).
    if not ctx.path(RUN_MANIFEST).exists():
        ctx.write_json(RUN_MANIFEST, {"video_id": resolved.video.video_id, "run_id": rid,
                                       "config_hash": resolved.config_hash, "created_at": utcnow_iso()})

    stage_order = list(resolved.pipeline.run.stages) or list(DEFAULT_STAGES)
    plan = select_stages(stage_order, from_stage, to_stage, only_stages)
    status = load_stage_status(rdir)

    for name in plan:
        stage = get_stage(name)
        prior = status.get(name, {})
        if (not force and prior.get("status") == "done" and prior.get("config_hash") == ctx.config_hash
                and stage.outputs_exist(ctx)):
            log.stage_skipped(name, "already done at this config_hash")
            continue
        log.stage_start(name, stage.device)
        t0 = time.perf_counter()
        try:
            result = stage.run(ctx)
        except Exception:
            status[name] = {"status": "failed", "config_hash": ctx.config_hash, "finished": utcnow_iso()}
            save_stage_status(rdir, status)
            log.close()
            raise
        elapsed = time.perf_counter() - t0
        status[name] = {"status": "done", "config_hash": ctx.config_hash, "started": utcnow_iso(),
                        "finished": utcnow_iso(), "elapsed_s": round(elapsed, 2), "counts": result.counts}
        save_stage_status(rdir, status)
        log.stage_end(name, result.counts)
        ctx.invalidate()

    log.close()
    return ctx
