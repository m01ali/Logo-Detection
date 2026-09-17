"""``stages/router/engine.py`` (07 MM1 Step 5.5): pass-through stub.

The six-block, 18-rule engine (T38–T46) is MM5 (``spec docs/11``).
``router_decision`` is null on every row in MM1; ``router_input.jsonl``
already carries everything else the future engine will read.
"""

from __future__ import annotations

from ..base import FIRING_LOG_FILE, ROUTER_INPUT_FILE, ROUTER_OUTPUT_FILE, RunContext, Stage, StageResult


class RouterStage(Stage):
    name = "router"
    device = "cpu"
    inputs = [ROUTER_INPUT_FILE]
    outputs = [ROUTER_OUTPUT_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        rows = ctx.read_jsonl(ROUTER_INPUT_FILE)
        out = [{
            "det_id": r["det_id"], "router_decision": None, "router_reason": None, "reason_code": None,
            "router_rule": None, "router_confidence": None, "risk_score": None, "trust_score": None,
            "review_priority": None, "dedup_cluster_id": None, "dedup_representative": None,
            "rules_matched_preempted": [],
        } for r in rows]
        header = ctx.header("router", note="MM1 stub; Rules 1-18 are MM5 (spec docs/11, T38-T46)")
        ctx.write_jsonl(ROUTER_OUTPUT_FILE, out, header=header)
        ctx.write_jsonl(FIRING_LOG_FILE, [], header=header)
        return StageResult(counts={"rows": len(out)})
