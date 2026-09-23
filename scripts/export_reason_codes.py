#!/usr/bin/env python3
"""T44 (``spec docs/11`` Part D): renders ``configs/reason_codes.md`` from the
enum in ``logo_pipeline/stages/router/reason_codes.py`` so the reference
document cannot drift from the code. ``tests/test_mm5_reason_codes.py`` fails
when the committed file is stale.

Usage::

    python scripts/export_reason_codes.py [--out configs/reason_codes.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logo_pipeline.stages.router.reason_codes import ReasonCode, render_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs/reason_codes.md")
    args = ap.parse_args()
    Path(args.out).write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {args.out} ({len(ReasonCode)} reason codes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
