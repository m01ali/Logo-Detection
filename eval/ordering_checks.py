#!/usr/bin/env python3
"""T65 (``spec docs/11`` Part G, ``17``): the two ordering checks over a run's
``firing_log.jsonl``, with the offending ``det_id``s.

1. No row with ``vlm_decision == partial_logo`` carries an UNKNOWN_LOGO decision
   (Rule 16 no longer covers partial_logo; Rule 9 must pre-empt Rule 15).
2. No row that satisfied an enabled accept-rule condition (6, 7, 8 as enabled)
   received a Rule 16 decision.

A row later turned into DEDUPED is judged on the decision it had before dedup
(``pre_dedup``). Exit status 1 when either check fails.

Usage::

    python eval/ordering_checks.py --run runs/<video_id>/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHECK_1 = "partial_logo_never_unknown_logo"
CHECK_2 = "accept_condition_never_rule_16"
ACCEPT_RULES = ("6", "7", "8")


def check_firing_log(rows: list[dict]) -> dict[str, list[int]]:
    v1: list[int] = []
    v2: list[int] = []
    for r in rows:
        decided = r.get("pre_dedup") or {"deciding_rule": r["deciding_rule"], "decision": r["decision"]}
        if r["vlm_decision"] == "partial_logo" and decided["decision"] == "UNKNOWN_LOGO":
            v1.append(r["det_id"])
        if decided["deciding_rule"] == "16" and any(m["rule"] in ACCEPT_RULES for m in r["rules_matched"]):
            v2.append(r["det_id"])
    return {CHECK_1: v1, CHECK_2: v2}


def read_firing_log(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "firing_log.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if not obj.get("_header"):
                rows.append(obj)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing firing_log.jsonl")
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    results = check_firing_log(read_firing_log(run_dir))
    failed = False
    lines = [f"# Ordering checks (T65) — `{run_dir}`", "", "| check | result | offending det_ids |", "|---|---|---|"]
    for name, dets in results.items():
        print(f"{'FAIL' if dets else 'PASS'}  {name}" + (f"  det_ids={dets}" if dets else ""))
        lines.append(f"| {name} | {'FAIL' if dets else 'pass'} | {', '.join(map(str, dets)) or '—'} |")
        failed = failed or bool(dets)
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ordering_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import csv
    with open(reports / "ordering_checks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "passed", "violating_det_id"])
        for name, dets in results.items():
            w.writerows([(name, False, d) for d in dets] or [(name, True, "")])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
