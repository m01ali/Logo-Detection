#!/usr/bin/env python3
"""``reports/acceptance_evidence.md`` (``spec docs/17``, deliverables table): the
caveats that must travel with every acceptance figure, stated from the run's
own resolved configuration and manifest rather than from memory.

Usage::

    python eval/acceptance_evidence.py --run runs/video_1/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import read_json, read_jsonl, reports_dir  # noqa: E402


def caveats(manifest: dict, status: dict, has_blind: bool, has_scores: bool) -> list[str]:
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
    out.append(f"**`not_logo_confidence` source: `{src}`**" + (
        " — derived from the generated `confidence / 100` on `not_logo` rows until T32 passes (`19` A6)."
        if src == "generated" else " — the T31 scored field, enabled after T32 passed."))
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
    lines += [f"- {c}" for c in caveats(manifest, status, has_blind, has_scores)]
    out = reports_dir(run_dir, args.out)
    (out / "acceptance_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'acceptance_evidence.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
