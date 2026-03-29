"""
build_siglip2_index.py
----------------------
Batch-populate the SigLIP2 FAISS index from a CSV + logo image folder.

Usage
-----
  python build_siglip2_index.py \
      --csv   /path/to/logos.csv \
      --dir   /path/to/logo/images \
      --augments 5          # augmented copies per image (0 = no augmentation)
      --batch 8             # images per embed batch (lower if OOM)

CSV format
----------
Required columns : filename, brand
Optional columns : source, correction, postfilter_verdict, family
Rows with a "correction" of "false_positive" are skipped automatically.

The script resumes safely: if logo_index_siglip2.faiss already exists it
appends to it rather than overwriting, so you can run it in chunks.
"""

import argparse
import os
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── locate project root so models/ is importable ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.faiss_db import LogoDatabaseSigLIP2

# ── augmentation pipeline (same as the Streamlit app) ──────────────────────
AUGMENTATIONS = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ColorJitter(hue=0.05, saturation=0.1, p=0.4),
    A.Resize(224, 224),
])


def parse_args():
    p = argparse.ArgumentParser(description="Build SigLIP2 FAISS index from a logo folder + CSV.")
    p.add_argument("--csv",      required=True,  help="Path to CSV with 'filename' and 'brand' columns.")
    p.add_argument("--dir",      required=True,  help="Directory containing the logo image files.")
    p.add_argument("--augments", type=int, default=5,
                   help="Augmented copies per image (0 = originals only). Default: 5.")
    p.add_argument("--batch",    type=int, default=8,
                   help="Embed batch size. Reduce if you hit OOM. Default: 8.")
    p.add_argument("--faiss-dir", default=str(PROJECT_ROOT / "FAISS"),
                   help="Folder where logo_index_siglip2.faiss will be saved.")
    p.add_argument("--skip-false-positives", action="store_true", default=True,
                   help="Skip rows whose 'correction' column equals 'false_positive' (default: on).")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    img_dir  = Path(args.dir)
    faiss_dir = Path(args.faiss_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    index_path = faiss_dir / "logo_index_siglip2.faiss"
    meta_path  = faiss_dir / "metadata_siglip2.json"

    # ── load CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if not {"filename", "brand"}.issubset(df.columns):
        sys.exit("CSV must have at least 'filename' and 'brand' columns.")

    # Skip false-positives if column exists
    if args.skip_false_positives and "correction" in df.columns:
        before = len(df)
        df = df[df["correction"].fillna("").str.lower().str.strip() != "false_positive"]
        print(f"Skipped {before - len(df)} false-positive rows.")

    # Keep only rows whose image file actually exists
    df["filepath"] = df["filename"].apply(lambda fn: img_dir / fn)
    missing = df[~df["filepath"].apply(lambda p: p.exists())]
    if not missing.empty:
        print(f"Warning: {len(missing)} files not found in '{img_dir}' — skipping.")
        print(missing["filename"].tolist()[:10], "…" if len(missing) > 10 else "")
    df = df[df["filepath"].apply(lambda p: p.exists())].reset_index(drop=True)

    if df.empty:
        sys.exit("No valid images found. Check --dir path.")

    print(f"\nImages to embed : {len(df):,}")
    print(f"Augments/image  : {args.augments}  (total vectors ≈ {len(df) * (1 + args.augments):,})")
    print(f"SigLIP2 index   : {index_path}\n")

    # ── load (or create) SigLIP2 DB ───────────────────────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP2 on {device} …")
    db = LogoDatabaseSigLIP2(
        index_path=str(index_path),
        metadata_path=str(meta_path),
        device=device,
        batch_size=args.batch,
    )
    print(f"Index already contains {db.index.ntotal:,} vectors.\n")

    # ── process in chunks of 64 originals at a time ───────────────────────
    CHUNK = 64
    total_added = 0

    for start in tqdm(range(0, len(df), CHUNK), desc="Chunks", unit="chunk"):
        chunk = df.iloc[start : start + CHUNK]
        images, brands = [], []
        for _, row in chunk.iterrows():
            try:
                img = Image.open(row["filepath"]).convert("RGB")
                images.append(img)
                brands.append(str(row["brand"]).strip())
            except Exception as e:
                print(f"  Could not open {row['filename']}: {e}")

        if not images:
            continue

        db.add_logos(
            images=images,
            brand_names=brands,
            augmentations=AUGMENTATIONS if args.augments > 0 else None,
            num_augments=args.augments,
        )
        total_added += len(images)

    # ── save ──────────────────────────────────────────────────────────────
    db.save()
    print(f"\nDone. Added {total_added:,} logos ({db.index.ntotal:,} total vectors).")
    print(f"Saved → {index_path}")
    print(f"        {meta_path}")


if __name__ == "__main__":
    main()
