"""``stages/refine.py`` (07 MM1 Goal): pass-through stub.

Crop refinement heuristics (T20–T24) are MM3 (``spec docs/09``). In MM1
every representative gets ``refinement_status: "no_candidate"``,
``refinement_method: "none"``, ``refined_bbox: null`` and
``selected_bbox = loose_bbox`` (07 Step 5.2: "crops cut from selected_bbox,
which equals loose_bbox until MM3").
"""

from __future__ import annotations

from .base import MERGED_DETECTIONS_FILE, REFINEMENT_FILE, RunContext, Stage, StageResult


class RefineStage(Stage):
    name = "refine"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE]
    outputs = [REFINEMENT_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        reps = ctx.read_jsonl(MERGED_DETECTIONS_FILE)
        rows = []
        for r in reps:
            rows.append({
                "det_id": r["det_id"],
                "refined_bbox": None,
                "refinement_method": "none",
                "refinement_status": "no_candidate",
                "selected_bbox": r["loose_bbox"],
                "has_valid_tight_bbox": False,
                "refinement_area_ratio": None,
            })
        header = ctx.header("refine", note="MM1 stub; heuristics 1/2/5 are MM3 (spec docs/09, T20-T24)")
        ctx.write_jsonl(REFINEMENT_FILE, rows, header=header)
        return StageResult(counts={"rows": len(rows)})
