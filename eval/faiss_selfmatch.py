#!/usr/bin/env python3
"""T51/T52/T53 (``spec docs/16``): self-match diagnostic of the FAISS index on
the Roboflow gold boxes.

For each label file, every ``cls cx cy w h`` box is denormalised against the
``fixed-images`` frame of the same stem, the crop is cut (boxes with a side
below 16 px are skipped, as detection would) and embedded with the production
embedder (CLIP ViT-L/14 through ``LogoDatabaseNew._embed_batch``). Each crop
is queried against the index: the raw top-1 vector similarity (``1 − d²/4``,
the production convention) and the brand-deduplicated top-5 (over-fetch ×3).

Writes ``reports/faiss_selfmatch.md`` (histogram of raw top-1 similarity,
counts at ≥ 0.999 / 0.99–0.999 / below, the matched brands of the ≥ 0.999
cases, the near-duplicate caveat, ``rai sport`` and ``verisure`` broken out),
``reports/faiss_selfmatch_excluded.csv`` (gold ids at ≥ 0.999, T53),
``reports/faiss_gold_embeddings.npy`` + ``faiss_gold_boxes.csv`` (the cache
T54 reuses). The report ends with T52's statement; a rebuild is never started.

Usage::

    python eval/faiss_selfmatch.py --frames-dir fixed-images \\
        --labels-dir ~/Downloads/logo2.yolov11/train/labels --names ~/Downloads/logo2.yolov11/data.yaml \\
        --index FAISS/ [--videos 14,1] [--limit N] [--out reports]
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # faiss + torch OpenMP runtimes (macOS), as the FAISS app

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runio import md_table  # noqa: E402

SELF_MATCH = 0.999
NEAR_DUP = 0.99
MIN_SIDE = 16
FOCUS_CLASSES = ("rai sport", "verisure")
_STEM_RE = re.compile(r"^(?P<video>\d+)_frame_(?P<n>\d+)")


@dataclass(frozen=True)
class GoldBox:
    gold_id: str
    video: str
    frame_path: Path
    class_name: str
    box: tuple[float, float, float, float]     # absolute pixels
    width: int
    height: int


def load_names(path: str | Path) -> list[str]:
    import yaml
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    names = data["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def frame_stem(label_file: Path) -> str | None:
    """``14_frame_0001_png.rf.<hash>.txt`` -> ``14_frame_0001``."""
    m = _STEM_RE.match(label_file.name)
    return f"{m.group('video')}_frame_{m.group('n')}" if m else None


def denormalise(cx: float, cy: float, w: float, h: float, width: int, height: int) -> tuple[float, ...]:
    return ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)


def iter_gold_boxes(labels_dir: Path, frames_dir: Path, names: list[str], videos: set[str] | None = None,
                    size_of=None):
    """Yields every labelled box with its frame; ``size_of(path)`` defaults to opening the PNG header."""
    from PIL import Image
    size_of = size_of or (lambda p: Image.open(p).size)
    for lf in sorted(Path(labels_dir).expanduser().glob("*.txt")):
        stem = frame_stem(lf)
        if stem is None:
            continue
        video = stem.split("_frame_")[0]
        if videos and video not in videos:
            continue
        frame = Path(frames_dir) / f"{stem}.png"
        if not frame.exists():
            continue
        width, height = size_of(frame)
        for line_no, line in enumerate(lf.read_text(encoding="utf-8").splitlines()):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
            yield GoldBox(f"{stem}#{line_no}", video, frame, names[cls], denormalise(cx, cy, w, h, width, height),
                          width, height)


def crop(gb: GoldBox, image=None):
    from PIL import Image
    img = image or Image.open(gb.frame_path).convert("RGB")
    x1, y1, x2, y2 = (int(round(v)) for v in gb.box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(gb.width, x2), min(gb.height, y2)
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return img.crop((x1, y1, x2, y2))


def query(index, metadata: list[str], embeddings: np.ndarray, top_k: int = 5, overfetch: int = 3) -> list[dict]:
    """Raw top-1 and the production brand-deduplicated top-k for each embedding."""
    distances, indices = index.search(embeddings.astype("float32"), top_k * overfetch)
    out = []
    for dists, idxs in zip(distances, indices):
        seen: dict[str, float] = {}
        for d, i in zip(dists, idxs):
            if i < 0 or i >= len(metadata):
                continue
            sim = 1.0 - float(d) / 4.0
            b = metadata[i]
            if b not in seen or sim > seen[b]:
                seen[b] = sim
        top = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        i0 = int(idxs[0])
        out.append({"raw_top1_sim": 1.0 - float(dists[0]) / 4.0, "raw_top1_index": i0,
                    "raw_top1_brand": metadata[i0] if 0 <= i0 < len(metadata) else None,
                    "top_brands": [b for b, _ in top], "top_sims": [round(s, 6) for _, s in top]})
    return out


def histogram(values, lo=0.0, hi=1.0, n=20) -> list[tuple[float, float, int]]:
    width = (hi - lo) / n
    counts = [0] * n
    for v in values:
        counts[min(max(int((v - lo) / width), 0), n - 1)] += 1
    return [(round(lo + i * width, 3), round(lo + (i + 1) * width, 3), c) for i, c in enumerate(counts)]


def render(rows: list[dict], n_skipped: int) -> str:
    sims = [r["raw_top1_sim"] for r in rows]
    selfm = [r for r in rows if r["raw_top1_sim"] >= SELF_MATCH]
    near = [r for r in rows if NEAR_DUP <= r["raw_top1_sim"] < SELF_MATCH]
    lines = ["# T51 FAISS self-match diagnostic", "",
             f"{len(rows)} gold crops queried ({n_skipped} skipped with a side below {MIN_SIDE} px).", "",
             f"- Self-match (raw top-1 similarity ≥ {SELF_MATCH}): **{len(selfm)}**",
             f"- Near-duplicate band ({NEAR_DUP}–{SELF_MATCH}): {len(near)}",
             f"- Below {NEAR_DUP}: {len(rows) - len(selfm) - len(near)}", "",
             "**Caveat.** Adjacent-frame crops of the same bug (a static channel logo) score very high without "
             "being the same crop, so the near-duplicate band is not evidence of leakage; only ≥ 0.999 is treated "
             "as a self-match.", "", "## Raw top-1 similarity", ""]
    lines += md_table(["bin", "count"], [(f"[{lo}, {hi})", c) for lo, hi, c in histogram(sims, 0.0, 1.0)])
    if sims and min(sims) >= 0.8:
        lines += ["", "Zoomed to [0.8, 1.0]:", ""]
        lines += md_table(["bin", "count"], [(f"[{lo}, {hi})", c) for lo, hi, c in histogram(sims, 0.8, 1.0)])
    if selfm:
        from collections import Counter
        lines += ["", "## Matched brand of the self-match cases", ""]
        lines += md_table(["gold class", "matched index brand", "n"],
                          [(a, b, n) for (a, b), n in Counter((r["class_name"], r["raw_top1_brand"])
                                                              for r in selfm).most_common()])
    for cls in FOCUS_CLASSES:
        sub = [r for r in rows if r["class_name"] == cls]
        if sub:
            s2 = [r["raw_top1_sim"] for r in sub]
            lines += ["", f"## `{cls}`", "",
                      f"{len(sub)} crops; self-match {sum(v >= SELF_MATCH for v in s2)}, near-duplicate "
                      f"{sum(NEAR_DUP <= v < SELF_MATCH for v in s2)}; median raw top-1 "
                      f"{float(np.median(s2)):.4f}."]
    lines += ["", "## Conclusion (T52)", ""]
    if selfm:
        lines.append(f"{len(selfm)} crops self-match (listed in `faiss_selfmatch_excluded.csv`); recommend excluding "
                     "them from the benchmark; a rebuild is not proposed.")
    else:
        lines.append("No crop self-matches; the benchmark proceeds on the existing index.")
    return "\n".join(lines) + "\n"


def run(gold: list[GoldBox], embed, index, metadata: list[str], out: Path, batch: int = 64,
        image_loader=None) -> list[dict]:
    """Crops, embeds (``embed(list[PIL]) -> np.ndarray``), queries and writes the T51/T53 outputs."""
    out.mkdir(parents=True, exist_ok=True)
    kept, crops, skipped = [], [], 0
    cache: dict[Path, object] = {}
    for gb in gold:
        if image_loader is not None:
            img = image_loader(gb)
        else:
            if gb.frame_path not in cache:
                from PIL import Image
                cache = {gb.frame_path: Image.open(gb.frame_path).convert("RGB")}
            img = cache[gb.frame_path]
        c = crop(gb, img)
        if c is None:
            skipped += 1
            continue
        kept.append(gb)
        crops.append(c)
    embs = []
    for i in range(0, len(crops), batch):
        e = np.asarray(embed(crops[i:i + batch]))
        if e.ndim != 2 or len(e) != len(crops[i:i + batch]):      # _embed_batch returns [] on error
            raise RuntimeError(f"embedding failed for crops {i}-{i + batch - 1}; rows would misalign")
        embs.append(e)
    emb = np.concatenate(embs, axis=0) if embs else np.zeros((0, index.d), dtype="float32")
    results = query(index, metadata, emb) if len(emb) else []
    rows = [{"gold_id": gb.gold_id, "video": gb.video, "class_name": gb.class_name,
             "box": [round(v, 2) for v in gb.box], **r} for gb, r in zip(kept, results)]

    np.save(out / "faiss_gold_embeddings.npy", emb.astype("float32"))
    with open(out / "faiss_gold_boxes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "gold_id", "video", "class_name", "x1", "y1", "x2", "y2"])
        for i, r in enumerate(rows):
            w.writerow([i, r["gold_id"], r["video"], r["class_name"], *r["box"]])
    with open(out / "faiss_selfmatch_excluded.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gold_id", "class_name", "raw_top1_sim", "raw_top1_brand", "raw_top1_index"])
        for r in rows:
            if r["raw_top1_sim"] >= SELF_MATCH:
                w.writerow([r["gold_id"], r["class_name"], round(r["raw_top1_sim"], 6), r["raw_top1_brand"],
                            r["raw_top1_index"]])
    (out / "faiss_selfmatch.md").write_text(render(rows, skipped), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", default="fixed-images")
    ap.add_argument("--labels-dir", default="~/Downloads/logo2.yolov11/train/labels")
    ap.add_argument("--names", default="~/Downloads/logo2.yolov11/data.yaml")
    ap.add_argument("--index", default="FAISS/", help="directory with logo_index.faiss + metadata.json")
    ap.add_argument("--videos", default=None, help="comma-separated video ids (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="first N boxes only (smoke runs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    from generate_hints import load_faiss_db
    idx_dir = Path(args.index)
    db = load_faiss_db(str(idx_dir / "logo_index.faiss"), str(idx_dir / "metadata.json"))
    names = load_names(args.names)
    videos = set(args.videos.split(",")) if args.videos else None
    gold = list(iter_gold_boxes(Path(args.labels_dir), Path(args.frames_dir), names, videos))
    if args.limit:
        gold = gold[:args.limit]
    rows = run(gold, db._embed_batch, db.index, db.metadata, Path(args.out), args.batch)
    print(f"{len(rows)} crops queried; {sum(r['raw_top1_sim'] >= SELF_MATCH for r in rows)} self-match; "
          f"wrote {Path(args.out) / 'faiss_selfmatch.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
