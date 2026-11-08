#!/usr/bin/env python3
"""19 A12 follow-up: two ``vlm-ab`` variants on the same sampled boxes, box by box.

``vlm-ab`` samples with a fixed seed, so two tags run on one run directory call the
same boxes. Their per-group shares hide which boxes moved; this pairs the two CSVs
on ``det_id`` and lists the boxes on which the variants disagree.

Reads ``reports/vlm_ab_<a>.csv``, ``reports/vlm_ab_<b>.csv`` and the run's
``rows.csv``; writes ``reports/vlm_ab_compare_<a>_vs_<b>.md`` + ``.csv`` and
``vlm_ab_compare_<a>_vs_<b>.png`` (the disagreeing crops):

* per group, how many boxes each variant gets right (rescue and held: called logo
  or partial; negative: left ``not_logo``), how many only A or only B gets right,
  and an exact two-sided sign test on those two counts;
* every disagreeing box with both calls, both brands and both reasonings.

Usage::

    python eval/vlm_ab_compare.py --run runs/<video_id>/<run_id> --a <tag> --b <tag>
"""

from __future__ import annotations

import argparse
import csv
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import md_table, reports_dir, write_csv  # noqa: E402
from rejected_true_logos import contact_sheet  # noqa: E402

GROUPS = ("rescue", "held", "negative")
LOGO_LIKE = ("logo", "partial_logo")
WANTED = {"rescue": "logo or partial", "held": "logo or partial", "negative": "not_logo"}
SHORT = {"logo": "logo", "partial_logo": "partial", "not_logo": "not", "uncertain": "unc"}


def right(group: str, call: str) -> bool:
    """The outcome a fix should produce: true logos called logos, unlabelled boxes left ``not_logo``."""
    return call not in LOGO_LIKE if group == "negative" else call in LOGO_LIKE


def sign_test(only_a: int, only_b: int) -> float:
    """Exact two-sided sign (McNemar) test on the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n)


def _read(run_dir: Path, tag: str) -> dict[str, dict]:
    path = Path(run_dir) / "reports" / f"vlm_ab_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `vlm-ab --tag {tag}` on this run first")
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["det_id"]: r for r in csv.DictReader(fh)}


def load(run_dir: Path, a: str, b: str) -> tuple[list[dict], int, int]:
    """``(paired rows, boxes only in A, boxes only in B)``; each paired row carries both calls and ``crop_path``."""
    import pandas as pd
    ra, rb = _read(run_dir, a), _read(run_dir, b)
    crops = pd.read_csv(Path(run_dir) / "rows.csv", usecols=["det_id", "crop_path"]).set_index("det_id")["crop_path"]
    paired = []
    for d in sorted(set(ra) & set(rb), key=int):
        x, y = ra[d], rb[d]
        if x["group"] != y["group"]:
            raise ValueError(f"det_id {d} is in group {x['group']} under {a} but {y['group']} under {b}")
        paired.append({"det_id": int(d), "group": x["group"], "gold_class": x["gold_class"],
                       "a_call": x["variant_is_logo"], "b_call": y["variant_is_logo"],
                       "a_brand": x["variant_brand"], "b_brand": y["variant_brand"],
                       "a_canonical": x["variant_brand_canonical"], "b_canonical": y["variant_brand_canonical"],
                       "a_brand_correct": x["variant_brand_correct"] == "True",
                       "b_brand_correct": y["variant_brand_correct"] == "True",
                       "a_reasoning": x["variant_reasoning"], "b_reasoning": y["variant_reasoning"],
                       "crop_path": crops.get(int(d), "")})
    return paired, len(set(ra) - set(rb)), len(set(rb) - set(ra))


def summarise(paired: list[dict]) -> dict[str, dict]:
    stats = {}
    for g in GROUPS:
        rows = [r for r in paired if r["group"] == g]
        ok_a = [right(g, r["a_call"]) for r in rows]
        ok_b = [right(g, r["b_call"]) for r in rows]
        s = {"n": len(rows), "a": sum(ok_a), "b": sum(ok_b),
             "only_a": sum(x and not y for x, y in zip(ok_a, ok_b)),
             "only_b": sum(y and not x for x, y in zip(ok_a, ok_b))}
        s["p"] = sign_test(s["only_a"], s["only_b"])
        both_logo = [r for r in rows if r["a_call"] in LOGO_LIKE and r["b_call"] in LOGO_LIKE and r["gold_class"]]
        s["both_logo"] = len(both_logo)
        s["brand_a"] = sum(r["a_brand_correct"] for r in both_logo)
        s["brand_b"] = sum(r["b_brand_correct"] for r in both_logo)
        stats[g] = s
    return stats


def disagreements(paired: list[dict]) -> list[dict]:
    return [r for r in paired if right(r["group"], r["a_call"]) != right(r["group"], r["b_call"])]


def render(run_dir: Path, a: str, b: str, stats: dict, diff: list[dict], only: tuple[int, int],
           n_sheet: int) -> str:
    def text(s, n=220):
        s = str(s or "").replace("|", "/").replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    def share(k, n):
        return f"{k} ({100 * k / n:.1f}%)" if n else "—"

    lines = [f"# VLM A/B `{a}` (A) vs `{b}` (B) — `{run_dir}`", "",
             "Both variants called the same sampled boxes. \"Right\" is what the change is for: true logos called "
             "`logo` or `partial_logo` (rescue, held), unlabelled boxes left `not_logo` (negative). The sign test "
             "uses only the boxes on which A and B disagree.", ""]
    lines += md_table(["group", "boxes", "wanted", "A right", "B right", "only A right", "only B right",
                       "sign test p"],
                      [(g, s["n"], WANTED[g], share(s["a"], s["n"]), share(s["b"], s["n"]), s["only_a"], s["only_b"],
                        f"{s['p']:.3f}") for g, s in stats.items()])
    lines += [""]
    for g in ("rescue", "held"):
        s = stats[g]
        if s["both_logo"]:
            lines.append(f"- {g}: of {s['both_logo']} boxes both call a logo, the brand is right on {s['brand_a']} "
                         f"under A and {s['brand_b']} under B.")
    if any(only):
        lines.append(f"- Not paired: {only[0]} boxes only in A, {only[1]} only in B (different sample settings).")
    lines += ["", f"## Boxes on which A and B disagree ({len(diff)})", "",
              f"Crops: `vlm_ab_compare_{a}_vs_{b}.png` ({n_sheet}), labelled det_id, group and gold class, then "
              "`A <call> B <call>`.", ""]
    lines += md_table(["det_id", "group", "gold", "A call", "A brand", "B call", "B brand", "A reasoning",
                       "B reasoning"],
                      [(r["det_id"], r["group"], r["gold_class"] or "—", r["a_call"], text(r["a_brand"], 40) or "—",
                        r["b_call"], text(r["b_brand"], 40) or "—", text(r["a_reasoning"]), text(r["b_reasoning"]))
                       for r in diff]) if diff else ["None."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--a", required=True, help="first vlm-ab tag")
    ap.add_argument("--b", required=True, help="second vlm-ab tag")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    paired, only_a, only_b = load(run_dir, args.a, args.b)
    stats, diff = summarise(paired), disagreements(paired)
    out = reports_dir(run_dir, args.out)
    stem = f"vlm_ab_compare_{args.a}_vs_{args.b}"
    n_sheet = contact_sheet(run_dir, [dict(r, label_top=f"{r['det_id']} {r['group']} {r['gold_class']}",
                                           label_bottom=f"A {SHORT.get(r['a_call'], r['a_call'])} "
                                                        f"B {SHORT.get(r['b_call'], r['b_call'])} "
                                                        f"{r['a_brand'] or r['b_brand'] or ''}")
                                      for r in diff], out / f"{stem}.png")
    (out / f"{stem}.md").write_text(render(run_dir, args.a, args.b, stats, diff, (only_a, only_b), n_sheet),
                                    encoding="utf-8")
    write_csv(out / f"{stem}.csv", ["det_id", "group", "gold_class", "a_call", "b_call", "a_brand", "b_brand",
                                    "a_canonical", "b_canonical", "a_reasoning", "b_reasoning"],
              [(r["det_id"], r["group"], r["gold_class"], r["a_call"], r["b_call"], r["a_brand"], r["b_brand"],
                r["a_canonical"], r["b_canonical"], r["a_reasoning"], r["b_reasoning"]) for r in diff])
    print(f"wrote {out / f'{stem}.md'} ({len(paired)} paired boxes, {len(diff)} disagreements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
