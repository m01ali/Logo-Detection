"""Compare open-set vs closed-set VLM brand-assignment experiments.

Usage:
    python scripts/compare_open_closed.py \
        --extraction-dir extraction_output/video_3-trimmed \
        --open-exp open_baseline \
        --closed-exp closed_watermarkfix
"""
import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--extraction-dir", default="extraction_output/video_3-trimmed")
parser.add_argument("--open-exp", default="open_baseline")
parser.add_argument("--closed-exp", default="closed_watermarkfix")
args = parser.parse_args()

root = Path(args.extraction_dir)
o = pd.read_csv(root / f"vlm_predictions__{args.open_exp}.csv")
c = pd.read_csv(root / f"vlm_predictions__{args.closed_exp}.csv")

print(f"=== counts ({args.open_exp} vs {args.closed_exp}) ===")
print("open  :", o.is_logo.value_counts().to_dict())
print("closed:", c.is_logo.value_counts().to_dict())
print()

print("=== open brands (logo/partial) ===")
print(o[o.is_logo.isin(["logo", "partial_logo"])].brand_canonical.value_counts())
print()

print("=== closed brands (logo/partial) ===")
print(c[c.is_logo.isin(["logo", "partial_logo"])].brand_canonical.value_counts())
print()

print("UNKNOWN — open:", (o.brand_canonical == "UNKNOWN").sum(),
      "| closed:", (c.brand_canonical == "UNKNOWN").sum())
print()

m = o.merge(c, on="det_id", suffixes=("_open", "_closed"))
print("=== is_logo agreement matrix (rows=open, cols=closed) ===")
print(pd.crosstab(m.is_logo_open, m.is_logo_closed))
print()

flip_to_not = m[(m.is_logo_open.isin(["logo", "partial_logo"])) & (m.is_logo_closed == "not_logo")]
print(f"open=logo -> closed=not_logo: {len(flip_to_not)}")
if len(flip_to_not):
    print(flip_to_not[["det_id", "brand_canonical_open", "confidence_open"]].to_string())
print()

flip_to_logo = m[(m.is_logo_closed.isin(["logo", "partial_logo"])) & (m.is_logo_open == "not_logo")]
print(f"closed=logo -> open=not_logo: {len(flip_to_logo)}")
if len(flip_to_logo):
    print(flip_to_logo[["det_id", "brand_canonical_closed", "confidence_closed"]].to_string())
print()

both_logo = m[(m.is_logo_open.isin(["logo", "partial_logo"])) & (m.is_logo_closed.isin(["logo", "partial_logo"]))]
print(f"both say logo: {len(both_logo)}")
if len(both_logo):
    brand_match = both_logo.brand_canonical_open.str.lower() == both_logo.brand_canonical_closed.str.lower()
    print("same brand assigned:", brand_match.sum(), "/", len(both_logo))
    if (~brand_match).any():
        print(both_logo[~brand_match][
            ["det_id", "brand_canonical_open", "brand_canonical_closed", "confidence_open", "confidence_closed"]
        ].to_string())
    print()
    print("avg confidence when logo (open):  ", both_logo.confidence_open.mean())
    print("avg confidence when logo (closed):", both_logo.confidence_closed.mean())
