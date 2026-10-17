#!/usr/bin/env python3
"""T64 (``spec docs/17``): blind-versus-hinted agreement and FAISS-origin
attribution, on a run with both VLM passes.

No labels needed: the agreement matrix of hinted × blind ``is_logo``; brand
agreement among rows where both say ``logo``/``partial_logo``
(``canonical_brand`` vs ``vlm_brand_blind_canonical``); the share of hinted rows
whose brand appears in the FAISS hints shown (``hinted_brand_in_faiss_hints``)
and, of those, the share where the blind brand differs (the hint plausibly
moved the answer); the same for the audio and OCR hints; how often blind says
UNKNOWN where hinted names a brand. In open mode the FAISS figure means "the
hinted brand matched a FAISS hint that was shown" (Revision 3 §10).

With ``--labels`` carrying ``true_brand``, a second section compares which of
the two passes named the right brand (route two).

Usage::

    python eval/blind_vs_hinted.py --run runs/video_1/<run_id> [--labels labels.csv]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runio import joined_rows, md_table, pct, reports_dir  # noqa: E402

DECISIONS = ("logo", "partial_logo", "not_logo", None)
POSITIVE = ("logo", "partial_logo")


def _real(b) -> bool:
    return b not in (None, "", "UNKNOWN")


def analyse(rows: list[dict], ocr_canon=None) -> dict:
    """``ocr_canon(row) -> canonical OCR brand``; defaults to the row's ``ocr_brand_match`` as is."""
    ocr_canon = ocr_canon or (lambda r: r.get("ocr_brand_match"))
    both = [r for r in rows if r.get("vlm_decision_blind") is not None or r.get("vlm_brand_blind") is not None]
    matrix = Counter((r.get("vlm_decision"), r.get("vlm_decision_blind")) for r in both)
    pos = [r for r in both if r.get("vlm_decision") in POSITIVE and r.get("vlm_decision_blind") in POSITIVE]
    brand_agree = sum(1 for r in pos if r.get("canonical_brand") == r.get("vlm_brand_blind_canonical"))
    hinted_named = [r for r in both if _real(r.get("canonical_brand"))]

    def attribution(pred):
        sub = [r for r in hinted_named if pred(r)]
        moved = [r for r in sub if r.get("vlm_brand_blind_canonical") != r.get("canonical_brand")]
        return len(sub), len(moved)

    return {
        "n": len(both), "n_total": len(rows), "matrix": matrix, "n_pos": len(pos), "brand_agree": brand_agree,
        "n_named": len(hinted_named),
        "faiss": attribution(lambda r: bool(r.get("hinted_brand_in_faiss_hints"))),
        "audio": attribution(lambda r: bool(r.get("brand_from_audio"))),
        "ocr": attribution(lambda r: ocr_canon(r) is not None and ocr_canon(r) == r.get("canonical_brand")),
        "blind_unknown": sum(1 for r in hinted_named if r.get("vlm_brand_blind_canonical") == "UNKNOWN"),
    }


def accuracy(rows: list[dict], labels: dict, brands) -> dict | None:
    from logo_pipeline.brands import normalise
    scored = []
    for r in rows:
        lab = labels.get((str(r.get("video_id")), str(r.get("run_id")), int(r["det_id"])))
        tb = (lab or {}).get("true_brand")
        if isinstance(tb, str) and tb.strip():
            truth = brands.map_exact(tb) or normalise(tb)
            scored.append((r.get("canonical_brand") == truth, r.get("vlm_brand_blind_canonical") == truth))
    if not scored:
        return None
    return {"n": len(scored), "hinted": sum(h for h, _ in scored), "blind": sum(b for _, b in scored),
            "only_hinted": sum(h and not b for h, b in scored), "only_blind": sum(b and not h for h, b in scored)}


def render(run_dir: Path, a: dict, acc: dict | None) -> str:
    lines = [f"# Blind vs hinted (T64) — `{run_dir}`", "",
             f"{a['n']} of {a['n_total']} rows have a blind call.", "", "## Decision agreement (rows: hinted, columns: blind)", ""]
    lines += md_table(["hinted \\ blind"] + [str(d) for d in DECISIONS],
                      [[str(h)] + [a["matrix"].get((h, b), 0) for b in DECISIONS] for h in DECISIONS])
    agree = sum(a["matrix"].get((d, d), 0) for d in DECISIONS)
    f_n, f_m = a["faiss"]
    au_n, au_m = a["audio"]
    o_n, o_m = a["ocr"]
    lines += ["", f"- Decision agreement: {agree} of {a['n']} ({pct(agree, a['n'])}).",
              f"- Brand agreement where both say logo/partial: {a['brand_agree']} of {a['n_pos']} "
              f"({pct(a['brand_agree'], a['n_pos'])}).", "",
              "## Hint attribution (rows where the hinted pass named a brand)", ""]
    lines += md_table(["hint", "hinted brand appears in the hint", "share of named", "of those, blind brand differs",
                       "share"],
                      [("FAISS (shown)", f_n, pct(f_n, a["n_named"]), f_m, pct(f_m, f_n)),
                       ("audio", au_n, pct(au_n, a["n_named"]), au_m, pct(au_m, au_n)),
                       ("OCR", o_n, pct(o_n, a["n_named"]), o_m, pct(o_m, o_n))])
    lines += ["", f"- Blind says UNKNOWN where hinted names a brand: {a['blind_unknown']} of {a['n_named']} "
              f"({pct(a['blind_unknown'], a['n_named'])}).",
              "- Open mode: the FAISS row means \"the hinted brand matched a FAISS hint that was shown\", the weaker "
              "statement Revision 3 §10 records.", "", "## Accuracy (route-two labels)", ""]
    if acc is None:
        lines.append("Not computed: no labels with `true_brand` were given.")
    else:
        lines += [f"- {acc['n']} labelled rows with a true brand.",
                  f"- Hinted correct: {acc['hinted']} ({pct(acc['hinted'], acc['n'])}); blind correct: {acc['blind']} "
                  f"({pct(acc['blind'], acc['n'])}).",
                  f"- Only hinted right: {acc['only_hinted']}; only blind right: {acc['only_blind']}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    import yaml

    from logo_pipeline.brands import load_brand_list
    from logo_pipeline.config import CanonicalisationCfg
    from logo_pipeline.stages.canonicalise import canonicalise
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    brands = load_brand_list(cfg["brands"]["list_path"])
    ccfg = CanonicalisationCfg()
    run_dir = Path(args.run)
    rows = joined_rows(run_dir)
    a = analyse(rows, ocr_canon=lambda r: canonicalise(r.get("ocr_brand_match"), brands, ccfg).canonical)
    acc = None
    if args.labels:
        from metrics import load_labels
        acc = accuracy(rows, load_labels(args.labels), brands)
    out = reports_dir(run_dir, args.out)
    (out / "blind_vs_hinted.md").write_text(render(run_dir, a, acc), encoding="utf-8")
    print(f"wrote {out / 'blind_vs_hinted.md'} ({a['n']} rows with both passes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
