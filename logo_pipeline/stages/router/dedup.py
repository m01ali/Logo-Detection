"""The shared dedup function (T45) and the router's per-frame dedup block
(Rules 17-18, ``spec docs/11`` block 5).

``cluster_boxes`` is the one greedy, seed-based clustering implementation.
``stages/merge.py`` calls it with the ``merge.*`` keys (brand-agnostic,
``min_area_fraction`` on); the router calls it with the ``router.dedup.*`` keys
on rows grouped by ``canonical_brand`` (``min_area_fraction`` off). Comparisons
are always against the seed, as in the merge (27 July C0): no transitive
chaining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ...geometry import area_ratio, box_area, containment, intersection_area, iou
from .inputs import RouterInput, real_brand
from .reason_codes import ACCEPT, DEDUPED, REJECT, REVIEW, UNKNOWN_LOGO, ReasonCode

Box = Sequence[float]


def evaluate_box_pair(a: Box, b: Box, iou_t: float, contain_t: float, area_ratio_t: float,
                      min_area_fraction: float | None = None, frame_area: float | None = None) -> dict:
    """Metrics and the outcome of one candidate pair. ``min_area_fraction`` (a fraction of
    ``frame_area``, tested on the smaller box) is the T14 floor; ``None`` disables it."""
    area_a, area_b = box_area(a), box_area(b)
    iou_v, contain_v, ratio_v = iou(a, b), containment(a, b), area_ratio(a, b)
    smaller_fraction = (min(area_a, area_b) / frame_area) if frame_area else 0.0
    if min_area_fraction is not None and smaller_fraction < min_area_fraction:
        outcome = "skipped_area_floor"
    elif not (iou_v >= iou_t or contain_v >= contain_t):
        outcome = "no_overlap"
    elif ratio_v < area_ratio_t:
        outcome = "failed_size_precondition"
    else:
        outcome = "merged"
    return {"iou": round(iou_v, 6), "contain": round(contain_v, 6), "area_ratio": round(ratio_v, 6),
            "smaller_fraction": round(smaller_fraction, 6), "intersection_area": round(intersection_area(a, b), 2),
            "outcome": outcome}


def cluster_boxes(boxes: Sequence[Box], iou_t: float, contain_t: float, area_ratio_t: float,
                  min_area_fraction: float | None = None, frame_area: float | None = None, *,
                  evaluate: Callable[[int, int], dict] | None = None,
                  on_pair: Callable[[int, int, dict], None] | None = None) -> list[list[int]]:
    """Greedy seed clustering over ``boxes`` in the given order (the caller sorts: best first).
    Returns index clusters, seed first. ``evaluate(seed, other)`` replaces the default pair test
    (the merge passes one that also carries its optional small-box branch and pair ids);
    ``on_pair`` sees every evaluated pair, merged or not (T19)."""
    remaining = list(range(len(boxes)))
    clusters: list[list[int]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        for other in list(remaining):
            m = (evaluate(seed, other) if evaluate is not None else
                 evaluate_box_pair(boxes[seed], boxes[other], iou_t, contain_t, area_ratio_t,
                                   min_area_fraction, frame_area))
            if on_pair is not None:
                on_pair(seed, other, m)
            if m["outcome"] == "merged":
                cluster.append(other)
                remaining.remove(other)
        clusters.append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Router block 5
# ---------------------------------------------------------------------------

DECISION_RANK = {ACCEPT: 0, REVIEW: 1, UNKNOWN_LOGO: 2, REJECT: 3}


@dataclass
class DedupEntry:
    """One row's verdict from blocks 1-4/6, mutated in place by the dedup pass."""
    inp: RouterInput
    decision: str
    reason_code: str
    rule: int | None
    review_priority: str | None = None
    cluster_id: int | None = None
    representative: bool | None = None
    pre_dedup: tuple[str, str] | None = None        # (decision, reason_code) before it became DEDUPED

    @property
    def deduped(self) -> bool:
        return self.decision == DEDUPED


def representative_key(e: DedupEntry, oversized_ratio: float) -> tuple:
    """18 July representative order; ``det_id`` last so the choice is deterministic."""
    i = e.inp
    area = box_area(i.selected_bbox)
    return (DECISION_RANK[e.decision], 0 if i.vlm_decision == "logo" else 1, 0 if i.is_truncated is False else 1,
            -i.ocr_fuzzy_score, -(i.confidence or 0), 0 if i.has_valid_tight_bbox else 1,
            -(area if i.box_area_ratio < oversized_ratio else 0.0), i.det_id)


def _mark(e: DedupEntry, reason: ReasonCode, rule: int, cluster_id: int) -> None:
    e.pre_dedup = (e.decision, e.reason_code)
    e.decision, e.reason_code, e.rule, e.review_priority = DEDUPED, reason.value, rule, None
    e.cluster_id, e.representative = cluster_id, False


def dedup_frame(entries: list[DedupEntry], router_cfg, next_cluster_id: int) -> int:
    """Rules 17 and 18 over one frame's entries, in place. Returns the next free cluster id."""
    dcfg, rules = router_cfg.dedup, router_cfg.rules
    oversized = router_cfg.thresholds.rule4_area_ratio
    by_brand: dict[str, list[DedupEntry]] = {}
    for e in entries:
        if real_brand(e.inp.vlm_brand_c):
            by_brand.setdefault(e.inp.vlm_brand_c, []).append(e)

    if rules.rule_17:
        for brand in sorted(by_brand):
            group = sorted(by_brand[brand], key=lambda e: representative_key(e, oversized))
            clusters = cluster_boxes([e.inp.selected_bbox for e in group], dcfg.dedup_iou_threshold,
                                     dcfg.dedup_containment_threshold, dcfg.dedup_area_ratio_threshold)
            for cl in clusters:
                if len(cl) < 2:
                    continue
                cid, next_cluster_id = next_cluster_id, next_cluster_id + 1
                group[cl[0]].cluster_id, group[cl[0]].representative = cid, True
                for k in cl[1:]:
                    _mark(group[k], ReasonCode.DUPLICATE_SAME_BRAND_BOX, 17, cid)

    if rules.rule_18:
        for brand in sorted(by_brand):
            group = sorted(by_brand[brand], key=lambda e: e.inp.det_id)
            for a in group:
                if a.deduped:
                    continue
                a_box = a.inp.selected_bbox
                containers = [b for b in group if b is not a and not b.deduped
                              and box_area(b.inp.selected_bbox) > box_area(a_box)
                              and containment(a_box, b.inp.selected_bbox) >= dcfg.dedup_containment_threshold]
                if not containers:
                    continue
                b = min(containers, key=lambda e: representative_key(e, oversized))
                if b.inp.box_area_ratio < oversized:
                    keep, drop, reason = b, a, ReasonCode.SUBSET_BOX_SAME_BRAND
                elif a.inp.has_valid_tight_bbox:
                    keep, drop, reason = a, b, ReasonCode.OVERSIZED_CONTAINER_REPLACED_BY_TIGHT_BOX
                else:
                    continue
                cid = keep.cluster_id if keep.cluster_id is not None else next_cluster_id
                if keep.cluster_id is None:
                    next_cluster_id += 1
                keep.cluster_id, keep.representative = cid, True
                _mark(drop, reason, 18, cid)
    return next_cluster_id
