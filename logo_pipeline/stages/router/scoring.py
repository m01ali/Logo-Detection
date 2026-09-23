"""The trust / risk scoring layer (18 July §8, ``spec docs/11`` Part F).

Computed with the 18 July constants (``router.scoring_layer.trust`` /
``.risk``), written to ``trust_score`` / ``risk_score``, and folded into
``router_confidence = clip((trust - risk + 10) / 20, 0, 1)`` [ASSUMPTION, `19`
C6]. It is logged only and never decides (24 July). The ``brand_type`` /
``is_brand_identifier`` terms contribute only when
``router.enhanced_vlm_fields_available`` is true.
"""

from __future__ import annotations

from .inputs import RouterInput
from .rules import NON_BRAND_TYPES


def trust_score(i: RouterInput, cfg) -> int:
    w, enhanced = cfg.router.scoring_layer.trust, cfg.router.enhanced_vlm_fields_available
    s = 0
    s += w.vlm_logo if i.vlm_decision == "logo" else 0
    s += w.brand_type_brand if enhanced and i.brand_type == "brand" else 0
    s += w.is_brand_identifier if enhanced and i.is_brand_identifier is True else 0
    s += w.ocr_strong if i.ocr_fuzzy_score >= w.ocr_strong_threshold else 0
    s += w.brand_in_database if i.brand_in_database is True else 0
    s += w.faiss_strong_agreement if (i.faiss_agrees_with_vlm is True and i.faiss_top1_score is not None
                                      and i.faiss_top1_score >= cfg.faiss.rule7_top1_threshold) else 0
    s += w.small_box if i.box_area_ratio < w.small_box_area_ratio else 0
    s += w.tight_bbox if i.has_valid_tight_bbox else 0
    return s


def risk_score(i: RouterInput, cfg) -> int:
    w, enhanced = cfg.router.scoring_layer.risk, cfg.router.enhanced_vlm_fields_available
    s = 0
    s += w.partial_logo if i.vlm_decision == "partial_logo" else 0
    s += w.uncertain if i.vlm_decision == "uncertain" else 0
    s += w.truncated if i.is_truncated is True else 0
    s += w.non_brand_type if enhanced and i.brand_type in NON_BRAND_TYPES else 0
    s += w.audio_weak_ocr if i.brand_from_audio and i.ocr_fuzzy_score < w.weak_ocr_threshold else 0
    s += w.faiss_weak_ocr if i.brand_from_faiss and i.ocr_fuzzy_score < w.weak_ocr_threshold else 0
    s += w.conflicting_ocr if i.has_conflicting_ocr else 0
    s += w.faiss_conflict if (i.faiss_agrees_with_vlm is False and i.faiss_top1_score is not None
                              and i.faiss_top1_score >= cfg.faiss.rule14_top1_threshold) else 0
    s += w.oversized if i.box_area_ratio >= w.oversized_area_ratio else 0
    s += w.channel_outside_watermark if i.is_channel_brand and i.watermark_overlap is False else 0
    return s


def router_confidence(trust: int, risk: int) -> float:
    return round(min(1.0, max(0.0, (trust - risk + 10) / 20)), 4)
