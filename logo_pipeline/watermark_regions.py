"""The keyed watermark-region store (T50) and the overlap measure (T48),
``spec docs/12``.

``configs/watermark_regions.yaml`` maps a key to ``{channel_brands, regions}``
with regions as normalised ``[x1, y1, x2, y2]``. A per-video config names one
key through ``watermark_regions_ref``. The overlap is containment
(``intersection / min(area(box), area(region))``, 27 July), not IoU, against
each region expanded by ``margin`` of the frame size; the best region counts.

A video without regions yields ``(None, None, False)`` on purpose: Rule 5 must
not read "no data" as "outside the region" (the 29 July defaults hazard).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .geometry import box_area, intersection_area

REGIONS_STORE = "configs/watermark_regions.yaml"
_ENTRY_KEYS = {"channel_brands", "regions"}


class RegionsError(ValueError):
    pass


@dataclass(frozen=True)
class RegionEntry:
    key: str
    channel_brands: tuple[str, ...]
    regions: tuple[tuple[float, float, float, float], ...]


def _validate_region(key: str, i: int, r) -> tuple[float, float, float, float]:
    if not isinstance(r, (list, tuple)) or len(r) != 4:
        raise RegionsError(f"{key}.regions[{i}]: expected [x1, y1, x2, y2], got {r!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in r)
    except (TypeError, ValueError):
        raise RegionsError(f"{key}.regions[{i}]: coordinates must be numbers, got {r!r}") from None
    if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
        raise RegionsError(f"{key}.regions[{i}]: coordinates must be within [0, 1] (normalised), got {r!r}")
    if not (x1 < x2 and y1 < y2):
        raise RegionsError(f"{key}.regions[{i}]: need x1 < x2 and y1 < y2, got {r!r}")
    return x1, y1, x2, y2


def parse_regions_store(data) -> dict[str, RegionEntry]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RegionsError("watermark regions store: top level must be a mapping of key -> entry")
    out: dict[str, RegionEntry] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            raise RegionsError(f"{key}: entry must be a mapping with channel_brands and regions")
        unknown = sorted(set(entry) - _ENTRY_KEYS)
        if unknown:
            raise RegionsError(f"{key}: unknown key(s) {unknown}; allowed: {sorted(_ENTRY_KEYS)}")
        brands = entry.get("channel_brands") or []
        if not isinstance(brands, list) or not all(isinstance(b, str) for b in brands):
            raise RegionsError(f"{key}.channel_brands: expected a list of strings")
        regions = entry.get("regions") or []
        if not isinstance(regions, list):
            raise RegionsError(f"{key}.regions: expected a list of [x1, y1, x2, y2]")
        out[str(key)] = RegionEntry(str(key), tuple(brands),
                                    tuple(_validate_region(key, i, r) for i, r in enumerate(regions)))
    return out


def load_regions_store(path: str | Path) -> dict[str, RegionEntry]:
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    return parse_regions_store(yaml.safe_load(p.read_text(encoding="utf-8")))


def regions_for_video(repo_root: str | Path, video, wcfg, store_path: str | Path | None = None) -> RegionEntry | None:
    """The store entry the video references, or ``None`` when it references none.

    A reference that names a missing key is an error (the loader "validates that every
    referenced key exists", T50); ``watermark.regions_ref_required`` makes a missing reference one.
    """
    ref = getattr(video, "watermark_regions_ref", None)
    if not ref:
        if wcfg.regions_ref_required:
            raise RegionsError(f"video {video.video_id!r} has no watermark_regions_ref but "
                               "watermark.regions_ref_required is true")
        return None
    path = Path(store_path) if store_path else Path(repo_root) / REGIONS_STORE
    store = load_regions_store(path)
    if ref not in store:
        raise RegionsError(f"watermark_regions_ref {ref!r} (video {video.video_id!r}) is not a key of {path}")
    return store[ref]


def expand_region(region, width: int, height: int, margin: float) -> list[float]:
    """Denormalise one region and grow it by ``margin`` of the frame size on every side."""
    x1, y1, x2, y2 = region
    mx, my = margin * width, margin * height
    return [x1 * width - mx, y1 * height - my, x2 * width + mx, y2 * height + my]


def region_overlap(box, regions, width: int, height: int, margin: float,
                   threshold: float) -> tuple[bool | None, float | None, bool]:
    """T48: ``(watermark_overlap, watermark_overlap_ratio, watermark_regions_available)``."""
    if not regions:
        return None, None, False
    best = 0.0
    for r in regions:
        rect = expand_region(r, width, height, margin)
        smaller = min(box_area(box), box_area(rect))
        if smaller <= 0:
            continue
        best = max(best, intersection_area(box, rect) / smaller)
    return best >= threshold, round(best, 4), True
