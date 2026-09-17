"""``python -m logo_pipeline`` entry points (07 MM1 Step 1.4, T5/T8).

``run`` executes the configured stage plan with ``--from-stage``,
``--to-stage``, ``--force``, ``--frames-dir`` and ``--video-id`` (the last
two override the per-video file so ``fixed-images`` runs need no per-video
YAML). ``brands validate|hash|add`` are the T5 commands MM0's definition of
done names explicitly.
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
    ctx = run_pipeline(
        args.config, args.video_config, repo_root=_repo_root(), frames_dir=args.frames_dir,
        video_id=args.video_id, source=args.source, from_stage=args.from_stage, to_stage=args.to_stage,
        only_stages=only, force=args.force, run_id=args.run_id, brands_path=args.brands,
    )
    print(f"run_dir: {ctx.run_dir}")
    return 0


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
    r.set_defaults(func=cmd_run)

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
