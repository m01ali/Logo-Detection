"""``python -m logo_pipeline`` entry points (07 MM1 Step 1.4, T5/T8).

``run`` executes the configured stage plan with ``--from-stage``,
``--to-stage``, ``--force``, ``--replay``, ``--blind-only``, ``--frames-dir`` and
``--video-id`` (the last two override the per-video file so ``fixed-images``
runs need no per-video YAML). ``brands validate|hash|add`` are the T5 commands.

M3 additions: ``router-only`` (T61), ``harness`` (T60), ``vlm-spike`` (T30),
``reports`` / ``checkpoint-check`` / ``package`` (T77), ``faiss selfmatch|rebenchmark``
(T51/T54) and ``eval <script> ...`` (any ``eval/<script>.py`` with its own flags).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_pipeline

    only = args.stages.split(",") if args.stages else None
    options = {"vlm_pass": "blind"} if args.blind_only else {}
    if args.blind_only and only is None and not (args.from_stage or args.to_stage):
        only = ["vlm"]                       # --blind-only alone means "add the blind pass" (15 T33)
    ctx = run_pipeline(
        args.config, args.video_config, repo_root=_repo_root(), frames_dir=args.frames_dir,
        video_id=args.video_id, source=args.source, from_stage=args.from_stage, to_stage=args.to_stage,
        only_stages=only, force=args.force, run_id=args.run_id, brands_path=args.brands, replay=args.replay,
        options=options,
    )
    print(f"run_dir: {ctx.run_dir}")
    return 0


def cmd_router_only(args: argparse.Namespace) -> int:
    from .router_only import replay, write
    run_dir = Path(args.run)
    outputs, firing, resolved = replay(run_dir, args.config, args.video_config, args.input, _repo_root())
    out = Path(args.out) if args.out else run_dir / "reports" / "router_replay"
    info = write(run_dir, out, outputs, firing, resolved)
    from collections import Counter
    print(f"wrote {out}: {info['rows']} rows; " + ", ".join(
        f"{d}={n}" for d, n in Counter(o["router_decision"] for o in outputs).most_common()))
    return 0


def cmd_harness(args: argparse.Namespace) -> int:
    from .harness import write_report
    from .runner import run_pipeline
    ctx = run_pipeline(args.config, args.video_config, repo_root=_repo_root(), frames_dir=args.frames_dir,
                       video_id=args.video_id, source=args.source, to_stage=None if args.full else "merge",
                       run_id=args.run_id)
    print(f"wrote {write_report(ctx)}")
    return 0


def cmd_vlm_spike(args: argparse.Namespace) -> int:
    from .runner import open_context
    from .stages.vlm_scoring import SPIKE_REPORT, run_spike
    ctx = open_context(args.config, args.video_config, repo_root=_repo_root(), frames_dir=args.frames_dir,
                       video_id=args.video_id, run_id=args.run_id)
    verdict = run_spike(ctx, n=args.n, seed=args.seed)
    print(f"{'PASS' if verdict['passed'] else 'FAIL'}: wrote {ctx.path(SPIKE_REPORT)}")
    return 0 if verdict["passed"] else 1


def cmd_reports(args: argparse.Namespace) -> int:
    from .checkpoint import run_reports
    for name, res in run_reports(Path(args.run), _repo_root()).items():
        print(f"{name:20s} {res}")
    return 0


def cmd_checkpoint_check(args: argparse.Namespace) -> int:
    from .checkpoint import checklist, compare_csv, render_checklist
    run_dir = Path(args.run)
    items = checklist(run_dir)
    replay = compare_csv(run_dir / "rows.csv", Path(args.compare_csv)) if args.compare_csv else None
    out = run_dir / "reports" / "checkpoint_checklist.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_checklist(run_dir, items, replay), encoding="utf-8")
    for c, ok, d in items + ([("CPU replay rows.csv identical", *replay)] if replay else []):
        print(f"{'PASS' if ok else 'FAIL'}  {c}  ({d})")
    return 0 if all(ok for _, ok, _ in items) and (replay is None or replay[0]) else 1


def cmd_package(args: argparse.Namespace) -> int:
    from .checkpoint import package
    out = package(Path(args.run), Path(args.out), _repo_root(), args.video_config, args.config)
    print(f"packaged into {out}")
    return 0


def _run_eval_script(name: str, rest: list[str]) -> int:
    path = _repo_root() / "eval" / f"{name}.py"
    if not path.exists():
        print(f"no eval script {path}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(path.parent))
    return __import__(name).main(rest) or 0


def cmd_eval(args: argparse.Namespace) -> int:
    return _run_eval_script(args.script, args.rest)


def cmd_faiss(args: argparse.Namespace) -> int:
    return _run_eval_script({"selfmatch": "faiss_selfmatch", "rebenchmark": "faiss_rebenchmark"}[args.action],
                            args.rest)


def cmd_brands_validate(args: argparse.Namespace) -> int:
    from .brands import load_brand_list

    try:
        bl = load_brand_list(args.list, require_hash_match=False)
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(bl.canonical)} brands, {len(bl.channel_brands)} channel brands, "
          f"{len(bl.short_names)} short names (<=4 chars)")
    print(f"version={bl.version} content_hash={bl.content_hash}")
    if bl.hash_matches is False:
        print(f"WARNING: stored content_hash ({bl.stored_hash}) does not match recomputed value; "
              f"run 'brands hash' to refresh it.")
    for w in bl.warnings:
        print(f"WARNING: {w}")
    return 0


def cmd_brands_hash(args: argparse.Namespace) -> int:
    from .brands import rewrite_hash

    bl = rewrite_hash(args.list)
    print(bl.content_hash)
    return 0


def cmd_brands_add(args: argparse.Namespace) -> int:
    from .brands import add_brand

    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    bl = add_brand(args.list, args.canonical, aliases, channel=args.channel, version=args.version)
    print(f"added {args.canonical!r}; version={bl.version} content_hash={bl.content_hash}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m logo_pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the configured stage plan.")
    r.add_argument("--config", default="configs/pipeline.yaml")
    r.add_argument("--video-config", default=None, help="configs/videos/<video_id>.yaml")
    r.add_argument("--video-id", default=None, help="Overrides the per-video file's video_id.")
    r.add_argument("--source", default=None, help="Overrides the per-video file's source path (video mode).")
    r.add_argument("--frames-dir", default=None, help="Sets ingest.mode=frames and this directory.")
    r.add_argument("--from-stage", default=None)
    r.add_argument("--to-stage", default=None)
    r.add_argument("--stages", default=None, help="Comma-separated explicit subset, overrides from/to.")
    r.add_argument("--force", action="store_true")
    r.add_argument("--run-id", default=None)
    r.add_argument("--brands", default=None, help="Overrides brands.list_path.")
    r.add_argument("--replay", action="store_true",
                   help="Re-run every planned stage but reuse the per-record caches (no GPU work repeated).")
    r.add_argument("--blind-only", action="store_true",
                   help="Add the blind VLM pass to an existing run without repeating the hinted one (T33).")
    r.set_defaults(func=cmd_run)

    ro = sub.add_parser("router-only", help="Replay the router on a run's router_input.jsonl (T61).")
    ro.add_argument("--config", default="configs/pipeline.yaml")
    ro.add_argument("--run", required=True)
    ro.add_argument("--video-config", default=None, help="Default: the copy in run_manifest.json.")
    ro.add_argument("--input", default="router_input.jsonl")
    ro.add_argument("--out", default=None, help="Default: <run>/reports/router_replay")
    ro.set_defaults(func=cmd_router_only)

    h = sub.add_parser("harness", help="Fresh-video harness: run to the merge + merge ablation (T60).")
    for a, d in (("--config", "configs/pipeline.yaml"), ("--video-config", None), ("--video-id", None),
                 ("--source", None), ("--frames-dir", None), ("--run-id", None)):
        h.add_argument(a, default=d)
    h.add_argument("--full", action="store_true", help="Run every stage (GPU machine), not just to the merge.")
    h.set_defaults(func=cmd_harness)

    vs = sub.add_parser("vlm-spike", help="T30 log-prob spike over a stratified sample of an existing run.")
    for a, d in (("--config", "configs/pipeline.yaml"), ("--video-config", None), ("--video-id", None),
                 ("--frames-dir", None)):
        vs.add_argument(a, default=d)
    vs.add_argument("--run-id", required=True)
    vs.add_argument("--n", type=int, default=50)
    vs.add_argument("--seed", type=int, default=42)
    vs.set_defaults(func=cmd_vlm_spike)

    rp = sub.add_parser("reports", help="Every report that applies to a run (T77 step 4).")
    rp.add_argument("--run", required=True)
    rp.set_defaults(func=cmd_reports)

    cc = sub.add_parser("checkpoint-check", help="T77 self-checks -> reports/checkpoint_checklist.md.")
    cc.add_argument("--run", required=True)
    cc.add_argument("--compare-csv", default=None, help="rows.csv of the CPU replay to byte-compare (step 3).")
    cc.set_defaults(func=cmd_checkpoint_check)

    pk = sub.add_parser("package", help="Client package for the checkpoint (T77 step 5).")
    pk.add_argument("--run", required=True)
    pk.add_argument("--out", required=True)
    pk.add_argument("--config", default="configs/pipeline.yaml")
    pk.add_argument("--video-config", default=None)
    pk.set_defaults(func=cmd_package)

    ev = sub.add_parser("eval", help="Run eval/<script>.py with its own arguments.")
    ev.add_argument("script")
    ev.add_argument("rest", nargs=argparse.REMAINDER)
    ev.set_defaults(func=cmd_eval)

    fa = sub.add_parser("faiss", help="FAISS diagnostics (T51 selfmatch, T54 rebenchmark).")
    fa.add_argument("action", choices=("selfmatch", "rebenchmark"))
    fa.add_argument("rest", nargs=argparse.REMAINDER)
    fa.set_defaults(func=cmd_faiss)

    b = sub.add_parser("brands", help="Canonical brand list commands (T5).")
    bsub = b.add_subparsers(dest="brands_command", required=True)

    bv = bsub.add_parser("validate")
    bv.add_argument("--list", default="configs/brands/roboflow_2026-04-28.yaml")
    bv.set_defaults(func=cmd_brands_validate)

    bh = bsub.add_parser("hash")
    bh.add_argument("--list", default="configs/brands/roboflow_2026-04-28.yaml")
    bh.set_defaults(func=cmd_brands_hash)

    ba = bsub.add_parser("add")
    ba.add_argument("--list", default="configs/brands/roboflow_2026-04-28.yaml")
    ba.add_argument("canonical")
    ba.add_argument("--aliases", default="", help="Comma-separated alias list.")
    ba.add_argument("--channel", action="store_true")
    ba.add_argument("--version", default=None, help="Defaults to the file's current version.")
    ba.set_defaults(func=cmd_brands_add)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
