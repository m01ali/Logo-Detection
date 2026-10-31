#!/usr/bin/env python3
"""``reports/acceptance_evidence.md`` (``spec docs/17``, deliverables table): the
caveats that must travel with every acceptance figure, stated from the run's
own resolved configuration and manifest rather than from memory. When T32 has
run, the ``not_logo_confidence`` caveat states its verdict and, if a
``router-only`` replay on the scored field sits under ``reports/``, what the
switch would do to the decisions.

Usage::

    python eval/acceptance_evidence.py --run runs/video_1/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import read_json, read_jsonl, reports_dir, write_csv  # noqa: E402


def t32_outcome(run_dir: Path) -> dict | None:
    """T32's verdict from ``reports/score_distribution.md`` and, when a ``router-only``
    replay under the scored field sits in ``reports/``, the decision counts it and the
    run's own router produced. ``None`` when T32 has not been run."""
    md = Path(run_dir) / "reports" / "score_distribution.md"
    if not md.exists():
        return None
    out = {"passed": "**Result: PASS.**" in md.read_text(encoding="utf-8"), "replay": None, "scored": None,
           "current": None}
    for p in sorted((Path(run_dir) / "reports").glob("*/router_output.jsonl")):
        with open(p, encoding="utf-8") as fh:
            head = json.loads(fh.readline() or "{}")
        if head.get("_header") and head.get("not_logo_confidence_source") == "scored":
            out["replay"] = p.parent.name
            out["scored"] = Counter(r.get("router_decision") for r in read_jsonl(p.parent, p.name))
            out["current"] = Counter(r.get("router_decision")
                                     for r in read_jsonl(run_dir, "router_output.jsonl", required=False))
            break
    return out


def _held_switch(t32: dict) -> str:
    if not t32["passed"]:
        return " T32 failed (`reports/score_distribution.md`), so the router stays on the generated field."
    text = " T32 passed (`reports/score_distribution.md`) but the switch is held:"
    if t32["scored"]:
        load = lambda c: c.get("REVIEW", 0) + c.get("UNKNOWN_LOGO", 0)
        text += (f" replayed on the scored field (`reports/{t32['replay']}`), the review load goes from "
                 f"{load(t32['current'])} to {load(t32['scored'])} rows and REJECT from "
                 f"{t32['current'].get('REJECT', 0)} to {t32['scored'].get('REJECT', 0)}.")
    return text + (" `rule1_not_logo_confidence` (0.80) was set for the generated field; its value on the scored "
                   "field is a client question (`19` A11).")


def caveats(manifest: dict, status: dict, has_blind: bool, has_scores: bool, t32: dict | None = None) -> list[str]:
    cfg = manifest.get("config") or {}
    router = cfg.get("router", {})
    rules = router.get("rules", {})
    vlm = cfg.get("vlm", {})
    out = [
        "**Rule 6 is the only auto-accept and rests on one signal pair**: the VLM's brand and an OCR read of the "
        f"same brand at fuzzy score ≥ {router.get('thresholds', {}).get('rule6_ocr_fuzzy_threshold', 95)}. "
        "A logo without readable text cannot be auto-accepted.",
        "**YOLO**: the 321-class YOLO11-L model runs class-agnostic (its class drives nothing) and was trained on a "
        "random split of the same Roboflow set the route-three gold boxes come from, so route-three detector "
        "figures are optimistic for frames that were in its training split.",
        "**video_1's M2 prompt variant is unknown** (the July run is lost); the M3 run uses the M3 prompt.",
    ]
    if not router.get("enhanced_vlm_fields_available", False):
        out.append("**Rules 2 and 3 are unreachable** and the `brand_type` / `is_brand_identifier` clauses of "
                   "Rules 6, 7, 8, 15 and 16 are omitted: the prompt does not produce those fields (`19` A1).")
    src = router.get("not_logo_confidence_source", "generated")
    if src != "generated":
        out.append(f"**`not_logo_confidence` source: `{src}`** — the T31 scored field, enabled after T32 passed.")
    elif t32 is None:
        out.append(f"**`not_logo_confidence` source: `{src}`** — derived from the generated `confidence / 100` on "
                   "`not_logo` rows until T32 passes (`19` A6).")
    else:
        out.append(f"**`not_logo_confidence` source: `{src}`** — the generated `confidence / 100` on `not_logo` "
                   "rows." + _held_switch(t32))
    off = [k for k in ("rule_7", "rule_8", "rule_5_reject", "rule_1_fallback_logprob") if not rules.get(k, False)]
    if off:
        out.append("**Disabled rules**: " + ", ".join(f"`{k}`" for k in off) +
                   " (Rule 8 and the Rule 5 REJECT variant are locked off for all of M3; Rule 7 waits on T54).")
    if manifest.get("mode") == "frames":
        out.append("**Audio is unavailable in frames mode**: Rule 11 cannot fire and OCR matches the canonical list "
                   "alone (`19` A8).")
    if not has_blind:
        out.append("**No blind pass in this run**: `vlm_decision_blind` and `vlm_brand_blind` are null; T64 needs the "
                   "MM9 corpus run.")
    if not has_scores:
        out.append("**No scored decision field in this run**: `logo_probability_scored`, `not_logo_confidence` and "
                   "`decision_scores` are null (scoring off or T30 not passed).")
    if not manifest.get("channel_brand"):
        out.append("**No channel declared for this video**: Rule 5 uses the snapshot's channel brands and no "
                   "watermark regions (`watermark_overlap` null).")
    vlm_s = (status.get("vlm") or {}).get("elapsed_s")
    if vlm_s:
        out.append(f"VLM stage time for this run: {vlm_s / 3600:.1f} h (`{vlm.get('model_id')}`, "
                   f"{vlm.get('quantisation')}).")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    manifest = read_json(run_dir, "run_manifest.json")
    status = read_json(run_dir, "stage_status.json", default={})
    has_blind = bool(read_jsonl(run_dir, "vlm/blind.jsonl", required=False))
    has_scores = bool(read_jsonl(run_dir, "vlm/scores.jsonl", required=False))
    lines = [f"# Acceptance evidence caveats — `{manifest.get('video_id')}` / `{manifest.get('run_id')}`", "",
             f"Config hash `{manifest.get('config_hash')}`; brand list `{manifest.get('brand_list_version')}` "
             f"(`{manifest.get('brand_list_hash')}`); git `{manifest.get('git_sha')}`.", ""]
    items = caveats(manifest, status, has_blind, has_scores, t32_outcome(run_dir))
    lines += [f"- {c}" for c in items]
    out = reports_dir(run_dir, args.out)
    (out / "acceptance_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(out / "acceptance_evidence.csv", ["caveat"], [(c.replace("**", "").replace("`", ""),) for c in items])
    print(f"wrote {out / 'acceptance_evidence.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
