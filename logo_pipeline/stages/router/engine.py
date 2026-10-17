"""``stages/router/engine.py`` (T38-T46, ``spec docs/11``): the six-block,
first-match-wins router.

``route_rows`` is a pure function of ``(router_input rows, config, brand list)``
so the router-only runner (T61) replays the very same engine. Blocks 1-4 and
the default (block 6) give every row a decision; block 5 (dedup, Rules 17-18)
then runs per frame and may turn a row into DEDUPED. Every enabled rule is
evaluated on every row so the firing log (T46) can list the rules that held
but lost to an earlier one; conditions are side-effect free, so this changes
nothing about the decision.
"""

from __future__ import annotations

from collections import Counter

from ..base import (FIRING_LOG_FILE, ROUTER_INPUT_FILE, ROUTER_OUTPUT_FILE, VLM_SCORES_FILE, RunContext, Stage,
                    StageResult)
from .dedup import DedupEntry, dedup_frame
from .inputs import MissingRouterInput, RouterInput, known_channel_brands
from .reason_codes import REVIEW, ReasonCode
from .rules import RULES, RuleSpec, Verdict, rule_5_branch
from .scoring import risk_score, router_confidence, trust_score



def enabled_rules(cfg) -> list[RuleSpec]:
    return [r for r in RULES if getattr(cfg.router.rules, r.key)]


def field_is_produced(name: str, cfg) -> bool:
    """Whether some stage produces ``name`` under this config (not merely whether a row has it)."""
    if name in ("brand_type", "is_brand_identifier"):
        return cfg.router.enhanced_vlm_fields_available
    if name == "logo_probability_scored":
        return cfg.vlm.scoring.enabled
    if name == "not_logo_confidence":
        return cfg.router.not_logo_confidence_source == "generated" or cfg.vlm.scoring.enabled
    if name in ("faiss_top1_score", "faiss_agrees_with_vlm", "brand_from_faiss"):
        return cfg.signals.faiss.compute_and_log
    if name in ("ocr_text", "ocr_fuzzy_score"):
        return cfg.signals.ocr.compute_and_log
    return True


def check_rules_producible(cfg) -> None:
    """27 July: refuse to start when an enabled rule reads a field no stage produces."""
    for spec in enabled_rules(cfg):
        for f in spec.requires:
            if not field_is_produced(f, cfg):
                raise MissingRouterInput(
                    f, f"router.rules.{spec.key} is enabled but no stage produces it under this config; "
                       f"disable the rule or enable its producer")


def evaluate_row(inp: RouterInput, cfg, rules: list[RuleSpec]) -> tuple[Verdict, list[tuple[RuleSpec, Verdict]]]:
    """Returns the deciding verdict and every enabled rule that held, in evaluation order."""
    matched = [(spec, v) for spec in rules if (v := spec.fn(inp, cfg)) is not None]
    if matched:
        return matched[0][1], matched
    code = ReasonCode.DEFAULT_UNCERTAIN_OR_INSUFFICIENT_EVIDENCE if inp.vlm_parse_ok else ReasonCode.VLM_OUTPUT_UNPARSED
    return Verdict(REVIEW, code, None, "normal"), matched


def _rule_no(name: str) -> int:
    return int(name.split("-")[0])


def null_decisions(det_id: int) -> dict:
    return {"det_id": det_id, "router_decision": None, "router_reason": None, "reason_code": None,
            "router_rule": None, "router_confidence": None, "risk_score": None, "trust_score": None,
            "review_priority": None, "dedup_cluster_id": None, "dedup_representative": None,
            "rules_matched_preempted": []}


def route_rows(rows: list[dict], cfg, brands, known_channels: frozenset[str] = frozenset(),
               ) -> tuple[list[dict], list[dict]]:
    """Blocks 1-6 over ``router_input`` rows. Returns ``(router_output rows, firing log rows)``."""
    if not cfg.router.enabled:
        return [null_decisions(r["det_id"]) for r in rows], []
    check_rules_producible(cfg)
    rules = enabled_rules(cfg)

    entries: dict[int, DedupEntry] = {}
    info: dict[int, dict] = {}
    for row in rows:
        inp = RouterInput.from_row(row, cfg, brands, known_channels)
        verdict, matched = evaluate_row(inp, cfg, rules)
        decider = matched[0][0].name if matched else None
        preempted = matched[1:]     # variants of the deciding rule (11-reject / 11-review) are listed too
        e = DedupEntry(inp, verdict.decision, verdict.reason_code.value, verdict.rule, verdict.review_priority)
        entries[inp.det_id] = e
        info[inp.det_id] = {
            "frame_idx": inp.frame_idx, "vlm_decision": inp.vlm_decision, "deciding_rule": decider,
            "matched": [{"rule": s.name, "reason_code": v.reason_code.value} for s, v in matched],
            "preempted": [{"rule": s.name, "reason_code": v.reason_code.value} for s, v in preempted],
            "preempted_nos": list(dict.fromkeys(_rule_no(s.name) for s, _ in preempted if _rule_no(s.name) != verdict.rule)),
            "rule5_branch": rule_5_branch(inp, cfg),
            "payload": verdict.payload,
        }

    by_frame: dict[int, list[DedupEntry]] = {}
    for det_id in sorted(entries):
        by_frame.setdefault(entries[det_id].inp.frame_idx, []).append(entries[det_id])
    next_cluster = 0
    for frame_idx in sorted(by_frame):
        next_cluster = dedup_frame(by_frame[frame_idx], cfg.router, next_cluster)

    outputs, firing = [], []
    for row in rows:
        det_id = row["det_id"]
        e, meta = entries[det_id], info[det_id]
        trust = risk = conf = None
        if cfg.router.scoring_layer.compute_trust_risk:
            trust, risk = trust_score(e.inp, cfg), risk_score(e.inp, cfg)
            conf = router_confidence(trust, risk)
        outputs.append({
            "det_id": det_id, "router_decision": e.decision, "router_reason": e.reason_code,
            "reason_code": e.reason_code, "router_rule": e.rule, "router_confidence": conf, "risk_score": risk,
            "trust_score": trust, "review_priority": e.review_priority, "dedup_cluster_id": e.cluster_id,
            "dedup_representative": e.representative, "rules_matched_preempted": meta["preempted_nos"],
        })
        if cfg.router.firing_log:
            before = None
            if e.pre_dedup is not None:
                before = {"deciding_rule": meta["deciding_rule"], "decision": e.pre_dedup[0],
                          "reason_code": e.pre_dedup[1]}
            firing.append({
                "det_id": det_id, "frame_idx": meta["frame_idx"], "vlm_decision": meta["vlm_decision"],
                "deciding_rule": str(e.rule) if e.deduped else meta["deciding_rule"],
                "reason_code": e.reason_code, "decision": e.decision,
                "rules_matched": meta["matched"], "rules_matched_preempted": meta["preempted"],
                "rule5_branch": meta["rule5_branch"], "reason_payload": meta["payload"],
                "pre_dedup": before,
                "dedup": {"cluster_id": e.cluster_id, "representative": e.representative},
            })
    return outputs, firing


def decision_report(outputs: list[dict]) -> str:
    by_decision = Counter(o["router_decision"] for o in outputs)
    by_code = Counter((o["router_decision"], o["reason_code"]) for o in outputs)
    lines = ["# Router decisions", "", f"{len(outputs)} detections.", "", "| decision | rows |", "|---|---|"]
    lines += [f"| {d} | {n} |" for d, n in sorted(by_decision.items(), key=lambda kv: -kv[1])]
    lines += ["", "| decision | reason code | rows |", "|---|---|---|"]
    lines += [f"| {d} | {c} | {n} |" for (d, c), n in sorted(by_code.items(), key=lambda kv: -kv[1])]
    return "\n".join(lines) + "\n"


class RouterStage(Stage):
    name = "router"
    device = "cpu"
    inputs = [ROUTER_INPUT_FILE]
    outputs = [ROUTER_OUTPUT_FILE, FIRING_LOG_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        rows = ctx.read_jsonl(ROUTER_INPUT_FILE)
        cfg = ctx.cfg
        if cfg.router.enabled and cfg.router.not_logo_confidence_source == "scored" \
                and not ctx.path(VLM_SCORES_FILE).exists():
            raise MissingRouterInput("not_logo_confidence", "router.not_logo_confidence_source is 'scored' "
                                     f"but {VLM_SCORES_FILE} does not exist (run the vlm_scoring stage)")
        channels = known_channel_brands(ctx.repo_root, ctx.video, ctx.brands, cfg.watermark)
        outputs, firing = route_rows(rows, cfg, ctx.brands, channels)

        header = ctx.header("router", not_logo_confidence_source=cfg.router.not_logo_confidence_source,
                            enhanced_vlm_fields_available=cfg.router.enhanced_vlm_fields_available,
                            enabled_rules=[r.name for r in enabled_rules(cfg)] if cfg.router.enabled else [])
        ctx.write_jsonl(ROUTER_OUTPUT_FILE, outputs, header=header)
        ctx.write_jsonl(FIRING_LOG_FILE, firing, header=header)
        report = ctx.path("reports/router_decisions.md")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(decision_report(outputs), encoding="utf-8")

        counts: dict = {"rows": len(outputs)}
        counts.update(Counter(o["router_decision"] for o in outputs if o["router_decision"]))
        if firing:
            from eval.ordering_checks import check_firing_log
            violations = check_firing_log(firing)
            counts["ordering_violations"] = sum(len(v) for v in violations.values())
            for name, dets in violations.items():
                if dets:
                    ctx.log.info("router: ordering check %s FAILED for det_ids %s", name, dets)
        return StageResult(counts=counts)
