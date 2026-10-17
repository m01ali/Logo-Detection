#!/usr/bin/env python3
"""T32 (``spec docs/15``): the distribution check that gates any rule from
consuming the scored field.

Reads ``vlm/scores.jsonl`` and the hinted decisions; writes
``reports/score_distribution.md`` + ``.csv``: 100-bin histograms of
``logo_probability_scored`` and ``not_logo_confidence`` stratified by
``is_logo``, the number of distinct values, the share of ``logo``/
``partial_logo`` rows above 95, and the same summary for the generated
``logo_probability`` (which took only 0/95/100 in M2).

Pass criterion (stated in the report): no single 1-point bin holds more than
50 % of the positive rows, and there are at least 30 distinct values. On pass
the report prints the two YAML switches to make; on fail the router stays on
the generated field and the report says so. The script never edits the YAML.

Usage::

    python eval/score_distribution.py --run runs/<video_id>/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import by_det, md_table, pct, read_jsonl, reports_dir, write_csv  # noqa: E402

POSITIVE = ("logo", "partial_logo")
STRATA = ("logo", "partial_logo", "not_logo", "unparsed")


def bin100(v: float) -> int:
    """1-point bins over 0-100: [0,1), [1,2), ..., [99,100]."""
    return min(int(v), 99)


def analyse(scores: list[dict], hinted: dict[int, dict], max_bin_share: float = 0.5, min_distinct: int = 30) -> dict:
    rows = []
    for s in scores:
        parsed = (hinted.get(s["det_id"]) or {}).get("parsed") or {}
        rows.append({"det_id": s["det_id"], "is_logo": parsed.get("is_logo") or "unparsed",
                     "lps": s.get("logo_probability_scored"), "nlc": s.get("not_logo_confidence"),
                     "generated": parsed.get("logo_probability")})
    hist = {(field, stratum): Counter() for field in ("lps", "nlc") for stratum in STRATA}
    for r in rows:
        if r["lps"] is not None:
            hist[("lps", r["is_logo"] if r["is_logo"] in STRATA else "unparsed")][bin100(r["lps"])] += 1
        if r["nlc"] is not None:
            hist[("nlc", r["is_logo"] if r["is_logo"] in STRATA else "unparsed")][bin100(100 * r["nlc"])] += 1

    positive = [r for r in rows if r["is_logo"] in POSITIVE and r["lps"] is not None]
    pos_bins = Counter(bin100(r["lps"]) for r in positive)
    top_bin, top_count = (pos_bins.most_common(1)[0] if pos_bins else (None, 0))
    distinct = len({round(r["lps"], 2) for r in rows if r["lps"] is not None})
    gen_values = Counter(int(r["generated"]) for r in rows if isinstance(r["generated"], (int, float)))
    max_share = top_count / len(positive) if positive else 1.0
    checks = {f"no 1-point bin holds > {int(max_bin_share * 100)}% of positive rows": max_share <= max_bin_share,
              f">= {min_distinct} distinct values": distinct >= min_distinct}
    return {
        "n": len(rows), "n_positive": len(positive), "top_bin": top_bin, "top_bin_share": max_share,
        "distinct": distinct, "above_95_positive": sum(1 for r in positive if r["lps"] > 95),
        "generated_distinct": len(gen_values), "generated_values": dict(gen_values.most_common(10)),
        "hist": hist, "checks": checks, "passed": bool(rows) and all(checks.values()),
    }


def render(run_dir: Path, a: dict) -> str:
    lines = ["# T32 score distribution check", "", f"Run: `{run_dir}`. {a['n']} scored detections, "
             f"{a['n_positive']} positive (hinted `logo`/`partial_logo`).", "",
             f"**Result: {'PASS' if a['passed'] else 'FAIL'}.**", ""]
    if a["passed"]:
        lines += ["Switch, in `configs/pipeline.yaml` (Rule 8 stays locked off regardless):", "", "```yaml",
                  "router:", "  not_logo_confidence_source: scored", "  rules:", "    rule_1_fallback_logprob: true",
                  "```", ""]
    else:
        lines += ["The router stays on the generated field (`router.not_logo_confidence_source: generated`); "
                  "Rule 1-fallback stays off. State this in the acceptance evidence.", ""]
    lines += md_table(["criterion", "result"], [(k, "pass" if v else "FAIL") for k, v in a["checks"].items()])
    lines += ["",
              f"- Most-populated 1-point bin among positive rows: `[{a['top_bin']}, {(a['top_bin'] or 0) + 1})` "
              f"holding {pct(int(round(a['top_bin_share'] * a['n_positive'])), a['n_positive'])}.",
              f"- Distinct `logo_probability_scored` values (2 dp): {a['distinct']}.",
              f"- Positive rows above 95: {a['above_95_positive']} ({pct(a['above_95_positive'], a['n_positive'])}).",
              f"- Generated `logo_probability` for comparison: {a['generated_distinct']} distinct values; "
              f"most common {a['generated_values']}.", ""]
    for field, title in (("lps", "logo_probability_scored"), ("nlc", "not_logo_confidence × 100")):
        lines += [f"## `{title}` by hinted `is_logo` (non-empty 1-point bins)", ""]
        bins = sorted({b for s in STRATA for b in a["hist"][(field, s)]})
        lines += md_table(["bin"] + list(STRATA), [[f"[{b},{b + 1})"] + [a["hist"][(field, s)].get(b, 0) for s in STRATA]
                                                   for b in bins])
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="reports directory (default <run>/reports)")
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    scores = read_jsonl(run_dir, "vlm/scores.jsonl")
    hinted = by_det(read_jsonl(run_dir, "vlm/hinted.jsonl"))
    a = analyse(scores, hinted)
    out = reports_dir(run_dir, args.out)
    (out / "score_distribution.md").write_text(render(run_dir, a), encoding="utf-8")
    write_csv(out / "score_distribution.csv", ["field", "is_logo", "bin", "count"],
              [(f, s, b, c) for (f, s), cnt in a["hist"].items() for b, c in sorted(cnt.items())])
    print(f"{'PASS' if a['passed'] else 'FAIL'}: wrote {out / 'score_distribution.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
