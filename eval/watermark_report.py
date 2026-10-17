#!/usr/bin/env python3
"""T71 (``spec docs/17``): channel-brand predictions outside the watermark region
with strong OCR support.

Reads ``rule5_branch`` from ``firing_log.jsonl`` (written for every row whose
``canonical_brand`` is a channel brand) and the OCR score from the rows; writes
``reports/watermark_report.md``: counts per branch (``no_regions``, ``inside``,
``outside_ocr_supported``, ``outside_weak_ocr``) and the share of channel-brand
predictions outside the region with ``ocr_fuzzy_score ≥ 90``.

Usage::

    python eval/watermark_report.py --run runs/video_1/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import by_det, joined_rows, md_table, pct, read_jsonl, reports_dir  # noqa: E402

BRANCHES = ("no_regions", "inside", "outside_ocr_supported", "outside_weak_ocr")
STRONG_OCR = 90


def analyse(firing: list[dict], rows: list[dict]) -> dict:
    rows_by = by_det(rows)
    chan = [f for f in firing if f.get("rule5_branch")]
    detail = Counter(f["rule5_branch"] for f in chan)
    branches = Counter("no_regions" if b.startswith("no_regions") else b for b in
                       (f["rule5_branch"] for f in chan))
    outside = [f for f in chan if f["rule5_branch"].startswith("outside")]
    strong = [f for f in outside if (rows_by.get(f["det_id"], {}).get("ocr_fuzzy_score") or 0) >= STRONG_OCR]
    return {"n_channel": len(chan), "branches": branches, "detail": detail, "n_outside": len(outside),
            "n_outside_strong": len(strong),
            "decisions": Counter(rows_by.get(f["det_id"], {}).get("reason_code") for f in chan)}


def render(run_dir: Path, a: dict) -> str:
    lines = [f"# Rule 5 / watermark report (T71) — `{run_dir}`", "",
             f"{a['n_channel']} channel-brand predictions.", ""]
    lines += md_table(["rule5_branch", "rows", "share"],
                      [(b, a["branches"].get(b, 0), pct(a["branches"].get(b, 0), a["n_channel"])) for b in BRANCHES])
    if a["detail"].get("no_regions_ocr_supported") or a["detail"].get("no_regions_needs_check"):
        lines += ["", f"`no_regions` splits into needs-check {a['detail'].get('no_regions_needs_check', 0)} and "
                  f"OCR-supported {a['detail'].get('no_regions_ocr_supported', 0)}."]
    lines += ["", f"**Outside the region with OCR ≥ {STRONG_OCR}:** {a['n_outside_strong']} of {a['n_outside']} "
              f"outside predictions ({pct(a['n_outside_strong'], a['n_outside'])}); "
              f"{pct(a['n_outside_strong'], a['n_channel'])} of all channel-brand predictions.", "",
              "## Reason codes on channel-brand rows", ""]
    lines += md_table(["reason code", "rows"], [(k, c) for k, c in a["decisions"].most_common()])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    a = analyse(read_jsonl(run_dir, "firing_log.jsonl"), joined_rows(run_dir))
    out = reports_dir(run_dir, args.out)
    (out / "watermark_report.md").write_text(render(run_dir, a), encoding="utf-8")
    print(f"wrote {out / 'watermark_report.md'} ({a['n_channel']} channel-brand rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
