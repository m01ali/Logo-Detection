"""Shared crop-cutting helper (03 §3: "Crops are cut on demand from frames/
using stored coordinates"). ``ocr``, ``faiss_query``, ``vlm`` and ``output``
all need a tight/enlarged crop image and none of them may depend on another
having written ``crops/`` first (stage order puts ``output`` last), so every
one cuts its own crop from the sampled frame in memory via this module.
"""

from __future__ import annotations

from PIL import Image

from .geometry import enlarge_box, rounded_int_box


def cut_crop(frame: Image.Image, box) -> Image.Image | None:
    x1, y1, x2, y2 = rounded_int_box(box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.width, x2), min(frame.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame.crop((x1, y1, x2, y2))


def tight_and_enlarged(frame: Image.Image, box, factor: float, width: int, height: int):
    """Returns ``(tight, enlarged, enlarged_box)``; ``enlarged`` falls back to
    ``tight`` when the padded crop is degenerate (M2 ``crop_enlarged or crop``)."""
    tight = cut_crop(frame, box)
    enlarged_box = enlarge_box(box, factor, width, height)
    enlarged = cut_crop(frame, enlarged_box) or tight
    return tight, enlarged, enlarged_box
