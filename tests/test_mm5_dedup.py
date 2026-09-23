"""MM5 block 5 (Rules 17-18) and the shared clustering function (T45)."""

from __future__ import annotations

from logo_pipeline.stages import merge as merge_mod
from logo_pipeline.stages.merge import merge_frame
from logo_pipeline.stages.router.dedup import cluster_boxes
from tests.mm5_helpers import make_cfg, route, row

A = [0, 0, 100, 100]
B = [10, 10, 110, 110]          # IoU with A is about 0.68


def by_id(outs):
    return {o["det_id"]: o for o in outs}


def test_same_brand_overlapping_boxes_keep_one_representative():
    outs, firing = route([row(0, selected_bbox=A), row(1, selected_bbox=B)])
    o = by_id(outs)
    assert o[0]["router_decision"] == "REVIEW" and o[0]["dedup_representative"] is True
    assert (o[1]["router_decision"], o[1]["reason_code"], o[1]["router_rule"]) == (
        "DEDUPED", "duplicate_same_brand_box", 17)
    assert o[1]["dedup_cluster_id"] == o[0]["dedup_cluster_id"] is not None
    assert o[1]["dedup_representative"] is False
    assert firing[1]["pre_dedup"]["decision"] == "REVIEW"


def test_representative_prefers_accept_over_review():
    strong = dict(ocr_text="nike", ocr_fuzzy_score=98, ocr_brand_match="nike")
    outs, _ = route([row(0, selected_bbox=A), row(1, selected_bbox=B, **strong)])
    o = by_id(outs)
    assert o[1]["router_decision"] == "ACCEPT" and o[0]["router_decision"] == "DEDUPED"


def test_same_brand_far_apart_both_survive():
    far = [150, 0, 250, 100]
    outs, _ = route([row(0, selected_bbox=A), row(1, selected_bbox=far)])
    assert [o["router_decision"] for o in outs] == ["REVIEW", "REVIEW"]
    assert all(o["dedup_cluster_id"] is None for o in outs)


def test_different_brands_never_dedup_and_other_frames_are_independent():
    outs, _ = route([row(0, selected_bbox=A), row(1, selected_bbox=B, canonical_brand="ikea"),
                     row(2, frame_idx=1, selected_bbox=B)])
    assert {o["router_decision"] for o in outs} == {"REVIEW"}


def test_unknown_brand_is_never_grouped():
    outs, _ = route([row(0, selected_bbox=A, canonical_brand="UNKNOWN"),
                     row(1, selected_bbox=B, canonical_brand="UNKNOWN")])
    assert {o["router_decision"] for o in outs} == {"UNKNOWN_LOGO"}


def test_tight_box_inside_oversized_same_brand_box_dedupes_the_oversized_one():
    big = row(0, selected_bbox=[0, 0, 180, 90], box_area_ratio=0.6, has_valid_tight_bbox=True)
    tight = row(1, selected_bbox=[50, 20, 80, 40], box_area_ratio=0.03, has_valid_tight_bbox=True)
    outs, _ = route([big, tight])
    o = by_id(outs)
    assert (o[0]["router_decision"], o[0]["reason_code"], o[0]["router_rule"]) == (
        "DEDUPED", "oversized_container_replaced_by_tight_box", 18)
    assert o[1]["router_decision"] == "REVIEW" and o[1]["dedup_representative"] is True


def test_subset_box_inside_normal_box_is_deduped():
    outer = row(0, selected_bbox=[0, 0, 100, 100], box_area_ratio=0.2)
    inner = row(1, selected_bbox=[20, 20, 50, 50], box_area_ratio=0.03)
    outs, _ = route([outer, inner])
    o = by_id(outs)
    assert (o[1]["router_decision"], o[1]["reason_code"], o[1]["router_rule"]) == (
        "DEDUPED", "subset_box_same_brand", 18)
    assert o[0]["router_decision"] == "REVIEW"


def test_oversized_container_without_tight_inner_box_keeps_both():
    big = row(0, selected_bbox=[0, 0, 180, 90], box_area_ratio=0.6, has_valid_tight_bbox=True)
    inner = row(1, selected_bbox=[50, 20, 80, 40], box_area_ratio=0.03, has_valid_tight_bbox=False)
    outs, _ = route([big, inner])
    assert [o["router_decision"] for o in outs] == ["REVIEW", "REVIEW"]


def test_dedup_rules_can_be_disabled():
    off = {"router": {"rules": {"rule_17": False, "rule_18": False}}}
    outs, _ = route([row(0, selected_bbox=A), row(1, selected_bbox=B)], off)
    assert [o["router_decision"] for o in outs] == ["REVIEW", "REVIEW"]


def test_shared_function_gives_different_clusters_under_merge_and_router_keys():
    boxes = [A, B]
    merge_keys = make_cfg().merge
    router_keys = make_cfg().router.dedup
    at_merge = cluster_boxes(boxes, merge_keys.iou_threshold, merge_keys.containment_threshold,
                             merge_keys.area_ratio_threshold)
    at_router = cluster_boxes(boxes, router_keys.dedup_iou_threshold, router_keys.dedup_containment_threshold,
                              router_keys.dedup_area_ratio_threshold)
    assert at_merge == [[0], [1]] and at_router == [[0, 1]]


def test_merge_stage_clusters_through_the_shared_function(monkeypatch):
    calls = []
    real = merge_mod.cluster_boxes

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(merge_mod, "cluster_boxes", spy)
    raw = [{"raw_det_id": i, "source": "gdino", "box_xyxy": b, "score": 0.9 - i / 10, "label": "x", "class_conf": None}
           for i, b in enumerate([A, [1, 1, 99, 99]])]
    reps, log = merge_frame(raw, 200, 200, make_cfg().merge)
    assert calls and len(reps) == 1 and len(log) == 1


def test_min_area_fraction_floor_is_off_by_default_and_on_when_given():
    tiny = [[0, 0, 3, 3], [0, 0, 3, 3]]
    assert cluster_boxes(tiny, 0.5, 0.8, 0.8) == [[0, 1]]
    assert cluster_boxes(tiny, 0.5, 0.8, 0.8, min_area_fraction=0.01, frame_area=10000.0) == [[0], [1]]
