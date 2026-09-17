"""21_test_plan.md — MM1/MM2 groundwork: geometry helpers used by merge,
refine and the router (hand-computed pairs)."""

from __future__ import annotations

from logo_pipeline.geometry import (area_ratio, box_area, box_area_ratio, box_touches_edge, clamp_box,
                                     containment, enlarge_box, iou, rounded_int_box)


def test_iou_identical_boxes():
    a = [0, 0, 100, 100]
    assert iou(a, a) == 1.0


def test_iou_non_overlapping():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    a, b = [0, 0, 10, 10], [5, 0, 15, 10]
    assert abs(iou(a, b) - (50 / 150)) < 1e-9


def test_containment_fully_inside():
    outer, inner = [0, 0, 100, 100], [10, 10, 20, 20]
    assert containment(outer, inner) == 1.0  # inner fully inside outer: inter/min(area) = 1


def test_area_ratio_tight_vs_padded():
    tight, padded = [0, 0, 10, 10], [0, 0, 20, 20]  # areas 100 vs 400 -> ratio 0.25
    assert abs(area_ratio(tight, padded) - 0.25) < 1e-9


def test_box_area_ratio_and_touches_edge():
    assert abs(box_area_ratio([0, 0, 100, 200], 1000, 1000) - 0.02) < 1e-9
    assert box_touches_edge([0, 5, 50, 50], 1000, 1000) is True   # x1=0 touches left edge
    assert box_touches_edge([998, 5, 999, 50], 1000, 1000) is True  # x2 within tol of right edge
    assert box_touches_edge([10, 10, 50, 50], 1000, 1000) is False


def test_enlarge_box_matches_m2_formula():
    # M2 `_enlarge_box`: pad = (dim)*(factor-1)/2 per side, clamped to the image.
    box = [100, 100, 200, 200]
    out = enlarge_box(box, 1.5, 1000, 1000)
    assert out == [75.0, 75.0, 225.0, 225.0]


def test_enlarge_box_clamped_at_frame_edge():
    out = enlarge_box([0, 0, 20, 20], 1.5, 1000, 1000)
    assert out[0] == 0.0 and out[1] == 0.0


def test_clamp_box_sorts_and_clamps():
    assert clamp_box([50, 50, -10, -10], 100, 100) == [0.0, 0.0, 50.0, 50.0]
    assert clamp_box([10, 10, 200, 200], 100, 100) == [10.0, 10.0, 100.0, 100.0]


def test_rounded_int_box_matches_m2_crop_image():
    # M2 `_crop_image`: round, then sort each axis.
    assert rounded_int_box([10.6, 20.4, 5.2, 15.9]) == (5, 16, 11, 20)


def test_box_area_degenerate():
    assert box_area([10, 10, 10, 20]) == 0.0
