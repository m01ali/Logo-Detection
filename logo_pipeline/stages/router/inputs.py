"""``RouterInput``: every field the 18 rules read, built from one
``router_input.jsonl`` row (``spec docs/11`` Part B).

There are no ``.get()`` defaults (27 July). A key that is absent from the row,
or a field a rule *needs* that is ``None`` (read through ``RouterInput.req``),
raises ``MissingRouterInput(field)``. Fields that are legitimately null
(``watermark_overlap`` with no regions, ``ocr_brand_match``, ...) are plain
attributes the rules compare against ``None`` explicitly.

``derive_flags`` computes the section-K booleans (T43 and the 18 July §9
derived booleans) from the raw fields and the YAML thresholds; the
``router_input`` stage writes its result into ``router_input.jsonl`` and
``RouterInput.from_row`` recomputes it, so a router replay under different
thresholds (T61) never reads a stale flag.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ...brands import BrandList
from ..canonicalise import UNKNOWN, canonicalise


class MissingRouterInput(Exception):
    """An enabled rule needs ``field`` and it is absent or null (or no stage produces it)."""

    def __init__(self, field: str, detail: str = ""):
        self.field = field
        super().__init__(f"router input {field!r} is missing" + (f": {detail}" if detail else ""))


def real_brand(b: str | None) -> bool:
    return b not in (None, "", UNKNOWN)


def derive_flags(*, vlm_brand_c: str | None, ocr_brand_c: str | None, ocr_fuzzy_score: int,
                 faiss_agrees_with_vlm: bool | None, faiss_top1_score: float | None,
                 faiss_score_gap: float | None, brand_from_audio: bool, brand_from_faiss: bool, cfg) -> dict:
    """T43 + 18 July §9: the five section-K flags. ``cfg`` is the ``PipelineCfg``."""
    t, fc = cfg.router.thresholds, cfg.faiss
    has_conflicting_ocr = (real_brand(ocr_brand_c) and real_brand(vlm_brand_c) and ocr_brand_c != vlm_brand_c
                           and ocr_fuzzy_score >= t.rule13_ocr_fuzzy)
    strong_visual_match = bool(faiss_agrees_with_vlm and faiss_top1_score is not None
                               and faiss_score_gap is not None
                               and faiss_top1_score >= fc.rule7_top1_threshold
                               and faiss_score_gap >= fc.rule7_score_gap)
    audio_only_prediction = bool(brand_from_audio and ocr_fuzzy_score < t.rule11_ocr_fuzzy and not strong_visual_match)
    faiss_only_prediction = bool(brand_from_faiss and ocr_fuzzy_score < t.rule12_ocr_fuzzy
                                 and faiss_top1_score is not None and faiss_top1_score < fc.rule12_top1_threshold)
    brand_evidence = "none"
    if cfg.router.brand_evidence.compute:
        ocr_confirmed = (real_brand(vlm_brand_c) and ocr_brand_c == vlm_brand_c
                         and ocr_fuzzy_score >= cfg.ocr.matching.general_fuzzy_threshold)
        faiss_confirmed = bool(faiss_agrees_with_vlm and faiss_top1_score is not None
                               and faiss_top1_score >= fc.rule14_top1_threshold)
        brand_evidence = ("both" if ocr_confirmed and faiss_confirmed else "ocr_confirmed" if ocr_confirmed
                          else "faiss_confirmed" if faiss_confirmed else "none")
    return {"has_conflicting_ocr": has_conflicting_ocr, "strong_visual_match": strong_visual_match,
            "audio_only_prediction": audio_only_prediction, "faiss_only_prediction": faiss_only_prediction,
            "brand_evidence": brand_evidence}


def known_channel_brands(repo_root: Path, video, brands: BrandList) -> frozenset[str]:
    """``channel_brands`` of the referenced watermark-region entry when there is one (T50),
    else the ``channel: true`` brands of the snapshot (T49)."""
    ref = getattr(video, "watermark_regions_ref", None)
    store = Path(repo_root) / "configs" / "watermark_regions.yaml"
    if ref and store.exists():
        import yaml
        entry = (yaml.safe_load(store.read_text(encoding="utf-8")) or {}).get(ref)
        if entry and entry.get("channel_brands"):
            return frozenset(brands.map_exact(b) or b for b in entry["channel_brands"])
    return frozenset(brands.channel_brands)


def _need(row: dict, key: str) -> Any:
    if key not in row:
        raise MissingRouterInput(key, "not present on the router_input row")
    return row[key]


@dataclass(frozen=True)
class RouterInput:
    det_id: int
    frame_idx: int
    selected_bbox: tuple
    # VLM
    vlm_parse_ok: bool
    vlm_decision: str | None
    is_truncated: bool | None
    confidence: int | None
    logo_probability_scored: float | None
    not_logo_confidence: float | None
    brand_type: str | None                    # None unless router.enhanced_vlm_fields_available
    is_brand_identifier: bool | None          # same
    # brands (canonical forms)
    vlm_brand_c: str | None
    ocr_brand_c: str | None
    faiss_brand_c: str | None
    is_channel_brand: bool
    brand_in_database: bool | None
    # OCR
    ocr_text: str
    ocr_fuzzy_score: int
    # FAISS
    faiss_top1: str | None
    faiss_top1_score: float | None
    faiss_score_gap: float | None
    faiss_agrees_with_vlm: bool | None
    brand_from_faiss: bool
    # audio / geometry / watermark
    brand_from_audio: bool
    box_area_ratio: float
    has_valid_tight_bbox: bool
    watermark_overlap: bool | None
    # derived
    has_conflicting_ocr: bool
    strong_visual_match: bool
    audio_only_prediction: bool
    faiss_only_prediction: bool
    brand_evidence: str

    def req(self, name: str) -> Any:
        v = getattr(self, name)
        if v is None:
            raise MissingRouterInput(name, f"det_id {self.det_id}")
        return v

    @classmethod
    def from_row(cls, row: dict, cfg, brands: BrandList, known_channels: frozenset[str] = frozenset()) -> "RouterInput":
        ccfg = cfg.canonicalisation
        audio_c = _need(row, "audio_brands_canonical")
        vlm_brand_c = _need(row, "canonical_brand")
        ocr_brand_c = canonicalise(_need(row, "ocr_brand_match"), brands, ccfg, audio_c).canonical
        faiss_brand_c = _need(row, "faiss_top1_brand_canonical")
        faiss_top1_score = _need(row, "faiss_top1_score")
        faiss_score_gap = _need(row, "faiss_score_gap")
        agrees = _need(row, "faiss_agrees_with_vlm")
        brand_from_audio = _need(row, "brand_from_audio")
        brand_from_faiss = bool(_need(row, "hinted_brand_in_faiss_hints"))
        ocr_fuzzy = _need(row, "ocr_fuzzy_score")
        vlm_decision = _need(row, "vlm_decision")
        confidence = _need(row, "confidence")

        if cfg.router.not_logo_confidence_source == "scored":
            scores = _need(row, "decision_scores")
            not_logo_confidence = None if scores is None else scores.get("not_logo")
        else:
            not_logo_confidence = confidence / 100 if vlm_decision == "not_logo" and confidence is not None else None

        enhanced = cfg.router.enhanced_vlm_fields_available
        flags = derive_flags(vlm_brand_c=vlm_brand_c, ocr_brand_c=ocr_brand_c, ocr_fuzzy_score=ocr_fuzzy,
                             faiss_agrees_with_vlm=agrees, faiss_top1_score=faiss_top1_score,
                             faiss_score_gap=faiss_score_gap, brand_from_audio=brand_from_audio,
                             brand_from_faiss=brand_from_faiss, cfg=cfg)
        return cls(
            det_id=_need(row, "det_id"), frame_idx=_need(row, "frame_idx"),
            selected_bbox=tuple(_need(row, "selected_bbox")),
            vlm_parse_ok=_need(row, "vlm_parse_ok"), vlm_decision=vlm_decision,
            is_truncated=_need(row, "is_truncated"), confidence=confidence,
            logo_probability_scored=_need(row, "logo_probability_scored"), not_logo_confidence=not_logo_confidence,
            brand_type=_need(row, "brand_type") if enhanced else None,
            is_brand_identifier=_need(row, "is_brand_identifier") if enhanced else None,
            vlm_brand_c=vlm_brand_c, ocr_brand_c=ocr_brand_c, faiss_brand_c=faiss_brand_c,
            is_channel_brand=real_brand(vlm_brand_c) and vlm_brand_c in known_channels,
            brand_in_database=_need(row, "brand_in_database"),
            ocr_text=_need(row, "ocr_text"), ocr_fuzzy_score=ocr_fuzzy,
            faiss_top1=_need(row, "faiss_top1"), faiss_top1_score=faiss_top1_score, faiss_score_gap=faiss_score_gap,
            faiss_agrees_with_vlm=agrees, brand_from_faiss=brand_from_faiss,
            brand_from_audio=bool(brand_from_audio),
            box_area_ratio=_need(row, "box_area_ratio"), has_valid_tight_bbox=_need(row, "has_valid_tight_bbox"),
            watermark_overlap=_need(row, "watermark_overlap"),
            **flags,
        )


ROUTER_INPUT_FIELDS = tuple(f.name for f in fields(RouterInput))
