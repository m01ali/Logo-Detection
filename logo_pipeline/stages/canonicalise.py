"""``stages/canonicalise.py`` (07 MM1 Step 5.5): pass-through stub.

The deterministic canonical-list mapping (T37) is MM5 (``spec docs/11``).
In MM1 every decision field is null, but ``brand_list_version`` and
``brand_list_hash`` are real (T5 provenance, 03 §8: "Written ... on every
row" is a standing requirement, not MM5 scope).
"""

from __future__ import annotations

from .base import CANONICAL_FILE, MERGED_DETECTIONS_FILE, RunContext, Stage, StageResult


class CanonicaliseStage(Stage):
    name = "canonicalise"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE]
    outputs = [CANONICAL_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        reps = ctx.read_jsonl(MERGED_DETECTIONS_FILE)
        rows = [{
            "det_id": r["det_id"],
            "canonical_brand": None, "canonical_match_mode": None, "canonical_match_score": None,
            "brand_list_version": ctx.brands.version, "brand_list_hash": ctx.brands.content_hash,
            "faiss_top1_brand_canonical": None, "faiss_agrees_with_vlm": None, "brand_in_database": None,
            "audio_brands_canonical": [], "brand_from_audio": False,
            "vlm_brand_blind_canonical": None, "blind_hinted_brand_agree": None,
            "hinted_brand_in_faiss_hints": None,
        } for r in reps]
        header = ctx.header("canonicalise", note="MM1 stub; the deterministic map is MM5 (spec docs/11, T37)")
        ctx.write_jsonl(CANONICAL_FILE, rows, header=header)
        return StageResult(counts={"rows": len(rows)})
