"""Shared builders for the MM5 tests: a config with overrides, the small brand
fixture and a full ``router_input``-shaped row (default: a clean, known-brand
logo that no rule fires on, i.e. it reaches the block-6 default)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from logo_pipeline.brands import load_brand_list
from logo_pipeline.config import resolve_config
from logo_pipeline.schema import Row
from logo_pipeline.stages.router.engine import route_rows
from logo_pipeline.stages.router.inputs import RouterInput

REPO = Path(__file__).resolve().parents[1]
BRANDS = load_brand_list(REPO / "tests" / "fixtures" / "brands_small.yaml")


def make_cfg(overrides: dict | None = None):
    video = {"video_id": "t", "source": "t.mp4", "overrides": overrides or {}}
    return resolve_config(str(REPO / "configs" / "pipeline.yaml"), video).pipeline


def rules_on(**flags) -> dict:
    return {"router": {"rules": flags}}


def row(det_id: int = 0, frame_idx: int = 0, **kw) -> dict:
    base = asdict(Row())
    base.update({
        "det_id": det_id, "frame_idx": frame_idx, "selected_bbox": [0, 0, 20, 10], "loose_bbox": [0, 0, 20, 10],
        "box_xyxy": [0, 0, 20, 10], "vlm_parse_ok": True, "vlm_decision": "logo", "is_logo": "logo",
        "brand": "Nike", "vlm_brand": "Nike", "canonical_brand": "nike", "is_truncated": False, "confidence": 90,
        "logo_probability": 90, "ocr_text": "", "ocr_fuzzy_score": 0, "ocr_brand_match": None,
        "audio_brands_canonical": [], "brand_from_audio": False, "hinted_brand_in_faiss_hints": False,
        "box_area_ratio": 0.05, "has_valid_tight_bbox": False, "faiss_top1": None, "faiss_top1_score": None,
        "faiss_score_gap": None, "faiss_agrees_with_vlm": None, "faiss_top1_brand_canonical": None,
        "watermark_overlap": None, "brand_in_database": True,
    })
    base.update(kw)
    return base


def route(rows, overrides: dict | None = None, known: frozenset = frozenset()):
    cfg = make_cfg(overrides)
    return route_rows(rows, cfg, BRANDS, known)


def decide(one_row: dict, overrides: dict | None = None, known: frozenset = frozenset()) -> dict:
    """Decision of a single row, merged with its firing-log line for convenience."""
    outs, firing = route([one_row], overrides, known)
    return {**outs[0], **{"firing": firing[0]}}


def inp(one_row: dict, overrides: dict | None = None, known: frozenset = frozenset()) -> RouterInput:
    return RouterInput.from_row(one_row, make_cfg(overrides), BRANDS, known)
