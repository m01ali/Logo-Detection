"""Box geometry helpers shared by every stage (03 §2 ``geometry.py``).

All boxes are ``[x1, y1, x2, y2]`` in absolute pixels of the frame they
belong to. Functions are pure and never read configuration.
"""

from __future__ import annotations

from typing import Sequence

Box = Sequence[float]


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def intersection_area(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a: Box, b: Box) -> float:
    """intersection / min(area_a, area_b)."""
    smaller = min(box_area(a), box_area(b))
    return intersection_area(a, b) / smaller if smaller > 0 else 0.0


def area_ratio(a: Box, b: Box) -> float:
    """min(area) / max(area)."""
    aa, ab = box_area(a), box_area(b)
    larger = max(aa, ab)
    return min(aa, ab) / larger if larger > 0 else 0.0


def centre(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def centre_distance(a: Box, b: Box) -> float:
    (ax, ay), (bx, by) = centre(a), centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def clamp_box(box: Box, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = (float(c) for c in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        min(max(0.0, x1), float(width)),
        min(max(0.0, y1), float(height)),
        min(max(0.0, x2), float(width)),
        min(max(0.0, y2), float(height)),
    ]


def enlarge_box(box: Box, factor: float, width: int, height: int) -> list[float]:
    """Expand symmetrically by ``factor`` and clamp (M2 ``_enlarge_box``)."""
    x1, y1, x2, y2 = (float(c) for c in box)
    pad_x = (x2 - x1) * (factor - 1) / 2
    pad_y = (y2 - y1) * (factor - 1) / 2
    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_y),
    ]


def box_area_ratio(box: Box, width: int, height: int) -> float:
    frame_area = float(width) * float(height)
    return box_area(box) / frame_area if frame_area > 0 else 0.0


def box_touches_edge(box: Box, width: int, height: int, tol: int = 2) -> bool:
    """True iff the box reaches any frame edge within ``tol`` px (task2 rule)."""
    if not box or width is None or height is None:
        return False
    x1, y1, x2, y2 = box
    return (x1 <= tol) or (y1 <= tol) or (x2 >= width - tol) or (y2 >= height - tol)


def rounded_int_box(box: Box) -> tuple[int, int, int, int]:
    """Integer crop window exactly as M2 ``_crop_image`` computed it."""
    x1, y1, x2, y2 = (int(round(float(c))) for c in box)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return x1, y1, x2, y2


def normalise_box(box: Box, width: int, height: int) -> list[float]:
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def denormalise_box(box: Box, width: int, height: int) -> list[float]:
    return [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
