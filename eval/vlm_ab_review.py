#!/usr/bin/env python3
"""19 A12 follow-up: what a ``vlm-ab`` variant's flips are.

Reads ``reports/vlm_ab_<tag>.csv`` and the run's ``rows.csv``; writes
``reports/vlm_ab_<tag>_review.md`` + ``.csv`` and ``vlm_ab_<tag>_negatives.png``:

* every ``negative``-group box the variant turned into a logo (the cost side of the
  change), with its brand, confidence and reasoning, and a contact sheet of the
  crops — so a reader can tell logos the labeller did not draw from real false
  positives;
* every rescued box whose brand did not match the gold class, the gold class
  against the variant's brand string and its canonical form — which separates
  alias-table gaps from wrong brands.

Usage::

    python eval/vlm_ab_review.py --run runs/<video_id>/<run_id> --tag <tag>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import md_table, reports_dir, write_csv  # noqa: E402
from rejected_true_logos import contact_sheet  # noqa: E402

LOGO_LIKE = ("logo", "partial_logo")


def load(run_dir: Path, tag: str) -> tuple[list[dict], list[dict]]:
    """``(negative flips, rescued boxes with a non-matching brand)``, each with its ``crop_path``."""
    import pandas as pd
    path = Path(run_dir) / "reports" / f"vlm_ab_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `vlm-ab --tag {tag}` on this run first")
    with open(path, newline="", encoding="utf-8") as fh:
        ab = list(csv.DictReader(fh))
    crops = pd.read_csv(Path(run_dir) / "rows.csv", usecols=["det_id", "crop_path"]).set_index("det_id")["crop_path"]
    for r in ab:
        r["crop_path"] = crops.get(int(r["det_id"]), "")
    flips = [r for r in ab if r["group"] == "negative" and r["variant_is_logo"] in LOGO_LIKE]
    misses = [r for r in ab if r["group"] == "rescue" and r["variant_is_logo"] in LOGO_LIKE
              and r["variant_brand_correct"] != "True"]
    return flips, misses


def render(run_dir: Path, tag: str, flips: list[dict], misses: list[dict], n_sheet: int) -> str:
    def text(s):
        return str(s or "").replace("|", "/").replace("\n", " ")

    lines = [f"# VLM A/B `{tag}` review — `{run_dir}`", "",
             f"## Negative-group boxes the variant calls a logo ({len(flips)})", "",
             "Boxes with IoU < 0.1 against every gold box. Each is a false positive only if the crop is not a "
             f"logo: see `vlm_ab_{tag}_negatives.png` ({n_sheet} crops, labelled det_id and the variant's brand).", ""]
    lines += md_table(["det_id", "variant call", "brand", "confidence", "reasoning"],
                      [(r["det_id"], r["variant_is_logo"], text(r["variant_brand"]), r["variant_confidence"],
                        text(r["variant_reasoning"])) for r in flips]) if flips else ["None."]
    by_pair = Counter((r["gold_class"], r["variant_brand"] or "—", r["variant_brand_canonical"] or "—") for r in misses)
    lines += ["", f"## Rescued boxes whose brand does not match the gold class ({len(misses)})", "",
              "A brand string that names the gold brand but canonicalises to something else is an alias-table gap "
              "(`brands add`), not a model error.", ""]
    lines += md_table(["gold class", "variant brand", "canonical", "boxes"],
                      [(*k, n) for k, n in by_pair.most_common()]) if misses else ["None."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    flips, misses = load(run_dir, args.tag)
    out = reports_dir(run_dir, args.out)
    n_sheet = contact_sheet(run_dir, [dict(r, label_top=f"{r['det_id']} -> {r['variant_brand'] or '?'}",
                                           label_bottom=f"{r['variant_is_logo']} conf {r['variant_confidence']}")
                                      for r in flips], out / f"vlm_ab_{args.tag}_negatives.png")
    (out / f"vlm_ab_{args.tag}_review.md").write_text(render(run_dir, args.tag, flips, misses, n_sheet),
                                                      encoding="utf-8")
    write_csv(out / f"vlm_ab_{args.tag}_review.csv", ["kind", "det_id", "gold_class", "variant_is_logo",
                                                      "variant_brand", "variant_brand_canonical"],
              [("negative_flip", r["det_id"], "", r["variant_is_logo"], r["variant_brand"],
                r["variant_brand_canonical"]) for r in flips]
              + [("rescued_brand_miss", r["det_id"], r["gold_class"], r["variant_is_logo"], r["variant_brand"],
                  r["variant_brand_canonical"]) for r in misses])
    print(f"wrote {out / f'vlm_ab_{args.tag}_review.md'} ({len(flips)} negative flips, {len(misses)} brand misses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
