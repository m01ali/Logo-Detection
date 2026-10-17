#!/usr/bin/env python3
"""T54 (``spec docs/16``): FAISS re-benchmark on the gold crops, two
populations, k ∈ {1, 3, 5}. Gates any FAISS threshold work and the Rule 7
decision.

Reuses T51's cache (``reports/faiss_gold_embeddings.npy`` +
``faiss_gold_boxes.csv``) and drops T53's exclusions. A crop is **present**
when its class (or any alias of it in the brand snapshot) equals a normalised
index brand string, **absent** otherwise.

Writes ``reports/faiss_rebenchmark.md`` + ``faiss_present_brands.csv``:

* accuracy at k = 1, 3, 5 on ``present`` (brand-deduplicated top-k contains the
  true brand), with counts and MRR; on ``absent`` it is zero by construction,
  so the top-1 similarity distribution there is given instead (the abstain
  evidence);
* top-1 similarity for correct vs incorrect predictions on ``present``
  (histogram, percentiles) and a threshold sweep with the overlap statement;
* everything twice: all classes and without ``evaluation.channel_classes``;
* the recommendation block: whether Rule 7's 0.97 / gap 0.08 is supported,
  whether ``faiss.abstain_threshold`` should be set and to what, and the kill
  criterion when nothing separates. Rule 7 stays disabled without the client.

Usage::

    python eval/faiss_rebenchmark.py --index FAISS/ [--reports reports] [--config configs/pipeline.yaml]
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # faiss + torch OpenMP runtimes (macOS), as the FAISS app

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runio import md_table  # noqa: E402
from faiss_selfmatch import histogram, query  # noqa: E402

from benchmarks.evaluation.metrics import mean_recall_at_k, mean_reciprocal_rank, threshold_sensitivity  # noqa: E402
from logo_pipeline.brands import BrandList, normalise  # noqa: E402

KS = (1, 3, 5)
RULE7_TOP1, RULE7_GAP = 0.97, 0.08
TARGET_PRECISION = 0.95
SWEEP = [round(0.80 + 0.005 * i, 3) for i in range(41)]


def brand_key(s: str, brands: BrandList) -> str:
    """One comparable key per brand: the canonical name when the alias table knows it, else the normalised string."""
    return brands.map_exact(s) or normalise(s)


def class_keys(cls: str, brands: BrandList) -> set[str]:
    canon = brands.map_exact(cls)
    keys = {normalise(cls)}
    if canon:
        keys.add(normalise(canon))
        entry = brands.entry(canon)
        if entry:
            keys |= {normalise(a) for a in entry.aliases}
    return keys


def populations(rows: list[dict], index_keys: set[str], brands: BrandList) -> tuple[list[dict], list[dict]]:
    present, absent = [], []
    for r in rows:
        (present if class_keys(r["class_name"], brands) & index_keys else absent).append(r)
    return present, absent


def present_brands(rows: list[dict], metadata: list[str], brands: BrandList) -> list[tuple[str, int]]:
    counts = Counter(normalise(m) for m in metadata)
    out = {}
    for cls in {r["class_name"] for r in rows}:
        n = sum(counts.get(k, 0) for k in class_keys(cls, brands))
        if n:
            out[cls] = n
    return sorted(out.items())


def percentiles(v: list[float]) -> dict:
    if not v:
        return {}
    a = np.asarray(v)
    return {p: round(float(np.percentile(a, p)), 4) for p in (5, 25, 50, 75, 95)}


def evaluate(present: list[dict], absent: list[dict], brands: BrandList) -> dict:
    truth = [brand_key(r["class_name"], brands) for r in present]
    retrieved = [[brand_key(b, brands) for b in r["top_brands"]] for r in present]
    sims = [r["top_sims"] for r in present]
    acc = {k: mean_recall_at_k(retrieved, truth, k) for k in KS}
    correct = [bool(rb) and rb[0] == t for rb, t in zip(retrieved, truth)]
    top1 = [s[0] if s else 0.0 for s in sims]
    gaps = [(s[0] - s[1]) if len(s) >= 2 else (s[0] if s else 0.0) for s in sims]
    sweep = threshold_sensitivity(sims, retrieved, truth, SWEEP, k=1) if present else {}
    curve = []
    for t in SWEEP:
        sel = [c for c, s in zip(correct, top1) if s >= t]
        curve.append((t, len(sel), (sum(sel) / len(sel)) if sel else None, sum(sel) / len(present) if present else 0))
    sep = next(((t, p, r) for t, n, p, r in curve if p is not None and p >= TARGET_PRECISION and n >= 10), None)
    r7 = [c for c, s, g in zip(correct, top1, gaps) if s >= RULE7_TOP1 and g >= RULE7_GAP]
    absent_top1 = [r["top_sims"][0] for r in absent if r["top_sims"]]
    abstain = float(np.percentile(absent_top1, 95)) if absent_top1 else None
    kept = [c for c, s in zip(correct, top1) if abstain is not None and s >= abstain]
    return {
        "n_present": len(present), "n_absent": len(absent), "accuracy": acc,
        "mrr": mean_reciprocal_rank(retrieved, truth) if present else 0.0,
        "correct_top1": [s for s, c in zip(top1, correct) if c], "incorrect_top1": [s for s, c in zip(top1, correct) if not c],
        "absent_top1": absent_top1, "curve": curve, "sweep": sweep, "separation": sep,
        "rule7_n": len(r7), "rule7_precision": (sum(r7) / len(r7)) if r7 else None,
        "rule7_recall": (sum(r7) / len(present)) if present else 0.0,
        "abstain": abstain, "abstain_recall": (sum(kept) / len(present)) if present and abstain is not None else None,
    }


def render_section(title: str, e: dict) -> list[str]:
    lines = [f"## {title}", "", f"Present {e['n_present']} crops, absent {e['n_absent']}.", ""]
    lines += md_table(["k", "accuracy (present)", "n"], [(k, f"{e['accuracy'][k]:.3f}", e["n_present"]) for k in KS])
    lines += ["", f"MRR (present): {e['mrr']:.3f}. On `absent` accuracy is 0 by construction (the true brand is not in "
              "the index); its top-1 similarity distribution is the abstain evidence:", ""]
    lines += md_table(["population", "n", "p5", "p25", "p50", "p75", "p95"],
                      [(name, len(v), *percentiles(v).values()) if v else (name, 0, "—", "—", "—", "—", "—")
                       for name, v in (("present, correct top-1", e["correct_top1"]),
                                       ("present, incorrect top-1", e["incorrect_top1"]),
                                       ("absent, top-1", e["absent_top1"]))])
    lines += ["", "Top-1 similarity histogram, [0.8, 1.0]:", ""]
    hc = histogram(e["correct_top1"], 0.8, 1.0, 20)
    hi = histogram(e["incorrect_top1"], 0.8, 1.0, 20)
    ha = histogram(e["absent_top1"], 0.8, 1.0, 20)
    lines += md_table(["bin", "present correct", "present incorrect", "absent"],
                      [(f"[{a[0]}, {a[1]})", a[2], b[2], c[2]) for a, b, c in zip(hc, hi, ha)])
    lines += ["", "Threshold sweep (present, top-1):", ""]
    lines += md_table(["threshold", "predictions ≥ t", "precision", "recall"],
                      [(t, n, f"{p:.3f}" if p is not None else "—", f"{r:.3f}") for t, n, p, r in e["curve"][::4]])
    sep = e["separation"]
    lines += ["", (f"**Overlap:** a threshold of {sep[0]} yields precision {sep[1]:.3f} at recall {sep[2]:.3f}."
                   if sep else f"**Overlap:** the distributions do not separate (no threshold reaches precision "
                   f"{TARGET_PRECISION} on at least 10 predictions)."), ""]
    return lines


def render(all_e: dict, no_ch: dict, excluded: int, n_brands: int) -> str:
    lines = ["# T54 FAISS re-benchmark", "",
             f"Gold crops from T51's cache; {excluded} self-match crops excluded (T53). {n_brands} gold classes are "
             "present in the index (`faiss_present_brands.csv`).", ""]
    lines += render_section("All classes", all_e)
    lines += render_section("Without channel classes (`evaluation.channel_classes`)", no_ch)
    e = no_ch if no_ch["n_present"] else all_e
    lines += ["## Recommendation", ""]
    if e["rule7_precision"] is None:
        lines.append(f"- Rule 7 ({RULE7_TOP1} / gap {RULE7_GAP}): no present crop reaches it; not supported.")
    else:
        ok = e["rule7_precision"] >= TARGET_PRECISION
        lines.append(f"- Rule 7 ({RULE7_TOP1} / gap {RULE7_GAP}): {e['rule7_n']} predictions, precision "
                     f"{e['rule7_precision']:.3f}, recall {e['rule7_recall']:.3f} — "
                     f"{'supported by this evidence' if ok else 'not supported'}.")
    if e["abstain"] is not None:
        lines.append(f"- `faiss.abstain_threshold`: the 95th percentile of absent top-1 similarity is "
                     f"{e['abstain']:.4f}; abstaining below it keeps {e['abstain_recall']:.3f} of present crops "
                     "correct at top-1. Set it only if that recall is acceptable.")
    if not e["separation"]:
        lines.append("- Kill criterion: correct and incorrect top-1 similarities do not separate, so FAISS should "
                     "stay a hint (Rules 12/14 review it) and never decide.")
    lines.append("- Rule 7 remains disabled unless the client agrees to a threshold from this evidence (T54).")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="FAISS/")
    ap.add_argument("--reports", default="reports", help="where T51 wrote its cache; outputs go here too")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args(argv)

    import faiss
    import yaml

    from logo_pipeline.brands import load_brand_list
    rep = Path(args.reports)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    brands = load_brand_list(cfg["brands"]["list_path"])
    channel = {normalise(c) for c in cfg.get("evaluation", {}).get("channel_classes", [])}
    idx_dir = Path(args.index)
    index = faiss.read_index(str(idx_dir / "logo_index.faiss"))
    metadata = json.loads((idx_dir / "metadata.json").read_text(encoding="utf-8"))

    emb = np.load(rep / "faiss_gold_embeddings.npy")
    with open(rep / "faiss_gold_boxes.csv", newline="", encoding="utf-8") as fh:
        boxes = list(csv.DictReader(fh))
    excl_path = rep / "faiss_selfmatch_excluded.csv"
    excluded = set()
    if excl_path.exists():
        with open(excl_path, newline="", encoding="utf-8") as fh:
            excluded = {r["gold_id"] for r in csv.DictReader(fh)}
    results = query(index, metadata, emb)
    rows = [{"gold_id": b["gold_id"], "class_name": b["class_name"], **r} for b, r in zip(boxes, results)
            if b["gold_id"] not in excluded]

    index_keys = {normalise(m) for m in metadata}
    pres_list = present_brands(rows, metadata, brands)
    with open(rep / "faiss_present_brands.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_name", "index_vectors"])
        w.writerows(pres_list)
    all_e = evaluate(*populations(rows, index_keys, brands), brands)
    no_ch_rows = [r for r in rows if normalise(r["class_name"]) not in channel]
    no_ch = evaluate(*populations(no_ch_rows, index_keys, brands), brands)
    (rep / "faiss_rebenchmark.md").write_text(render(all_e, no_ch, len(excluded), len(pres_list)), encoding="utf-8")
    print(f"present {all_e['n_present']}, absent {all_e['n_absent']}; wrote {rep / 'faiss_rebenchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
