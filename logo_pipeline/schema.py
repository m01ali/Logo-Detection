"""Frozen row schema (T9, 05) and the raw-detection record (T11).

``Row`` has every field of ``spec docs/05_row_schema.md`` in the document's
own order (sections A-L); ``CSV_COLUMNS`` is that same order and is what
``stages/output.py`` writes. ``scripts/export_schema_doc.py`` renders the
document from this file so the two cannot drift.

Enums are plain ``str`` subclasses (not ``Enum``) so a value round-trips
through JSON/CSV without a codec; validity is checked where a stage writes
the field, not by the type system alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Enums (plain string constants; see module docstring)
# ---------------------------------------------------------------------------

DETECTION_SOURCES = ("gdino", "yolo")
REFINEMENT_METHODS = ("none", "ocr_guided", "oversized_guard")
REFINEMENT_STATUSES = ("applied", "fallback_to_loose", "rejected_safety_check", "oversized_region", "no_candidate")
OCR_SOURCES = ("tight_crop", "enlarged_crop")
OCR_MATCH_LISTS = ("canonical", "audio")
OCR_MATCH_MODES = ("exact_short", "fuzzy")
AUDIO_SIGNAL_STATUSES = ("ok", "empty", "signal_unavailable", "disabled")
IS_LOGO_VALUES = ("logo", "partial_logo", "not_logo")
CANONICAL_MATCH_MODES = ("alias", "exact", "fuzzy", "audio", "unmatched", "llm_merge")
BRAND_EVIDENCE_VALUES = ("ocr_confirmed", "faiss_confirmed", "both", "none")
ROUTER_DECISIONS = ("ACCEPT", "REVIEW", "REJECT", "UNKNOWN_LOGO", "DEDUPED")
REVIEW_PRIORITIES = ("low", "normal", "high")

# ---------------------------------------------------------------------------
# Section tags, used by export_schema_doc.py to reconstruct 05's table layout
# ---------------------------------------------------------------------------

SECTION_TITLES: dict[str, str] = {
    "A": "Identity and geometry",
    "B": "Crop refinement (T20–T23)",
    "C": "Crop paths (review UI)",
    "D": "OCR verification pass (T25–T29)",
    "E": "FAISS",
    "F": "Audio",
    "G": "VLM, hinted pass",
    "H": "VLM, blind pass (T33)",
    "I": "Canonicalisation (T37)",
    "J": "Watermark (T47–T48)",
    "K": "Router (T38–T46)",
    "L": "Run provenance and review",
}


def _f(section: str, note: str = "", m2: bool = False, default: Any = None, default_factory=None):
    meta = {"section": section, "note": note, "m2": m2}
    if default_factory is not None:
        return field(default_factory=default_factory, metadata=meta)
    return field(default=default, metadata=meta)


@dataclass
class DetectionRecord:
    """A single raw detector box (T11); one line of ``raw_detections.jsonl``."""

    raw_det_id: int
    frame_idx: int
    source: str                       # DETECTION_SOURCES
    box_xyxy: list[float]
    score: float
    label: Optional[str]
    class_conf: Optional[float]
    frame_width: int
    frame_height: int


@dataclass
class Row:
    """One row of ``rows.csv`` / ``router_input.jsonl`` (05). One detection
    that survived the merge."""

    # -- A. Identity and geometry --------------------------------------------------
    video_id: str = _f("A", "From the per-video config (A7).")
    run_id: str = _f("A")
    det_id: int = _f("A", "Sequential over the run, assigned to representatives after the merge.", m2=True)
    frame_idx: int = _f("A", "Video frame index, or the ordinal in frames mode.", m2=True)
    timecode_s: float = _f("A", "frame_idx / fps; in frames mode (ordinal−1)/assumed_fps.", m2=True)
    shot_id: Optional[int] = _f("A", "Ordinal of the scene containing the frame (T57).")
    frame_width: int = _f("A")
    frame_height: int = _f("A")
    det_score: float = _f("A", "Score of the representative box (its own detector's score).", m2=True)
    box_xyxy: list[float] = _f("A", "loose_bbox of the representative, absolute pixels.", m2=True, default_factory=list)
    loose_bbox: list[float] = _f("A", "Identical to box_xyxy; the name the correspondence uses.", default_factory=list)
    box_area_ratio: float = _f("A", "box_area / frame_area. Rule 4 input.", default=0.0)
    box_touches_edge: bool = _f("A", "Any side within 2 px of the frame edge (from task2).", default=False)
    detection_sources: list[str] = _f("A", "Values from {gdino, yolo} (T15).", default_factory=list)
    merge_count: int = _f("A", "Boxes absorbed into this representative (T15). Merged ⇔ > 0.", default=0)
    merged_raw_det_ids: list[int] = _f("A", "Provenance; [] when merge_count = 0.", default_factory=list)
    gdino_label: Optional[str] = _f("A", "The GDINO phrase that produced the box (M2 label).")
    yolo_class: Optional[str] = _f("A", "Logged, drives nothing (T12).")
    yolo_class_conf: Optional[float] = _f("A")

    # -- B. Crop refinement ----------------------------------------------------------
    refined_bbox: Optional[list[float]] = _f("B", "Candidate tight box or null.")
    refinement_method: str = _f("B", "none | ocr_guided | oversized_guard", default="none")
    refinement_status: str = _f("B", "applied | fallback_to_loose | rejected_safety_check | oversized_region | no_candidate",
                                 default="no_candidate")
    selected_bbox: list[float] = _f("B", "refined_bbox when applied, else loose_bbox.", default_factory=list)
    has_valid_tight_bbox: bool = _f("B", "refinement_status == applied (T23). Rule 4 input.", default=False)
    refinement_area_ratio: Optional[float] = _f("B", "area(refined)/area(loose).")

    # -- C. Crop paths (review UI) ----------------------------------------------------
    crop_path: str = _f("C", "crops/det_XXXXXX/crop.jpg, cut from selected_bbox.", m2=True, default="")
    enlarged_crop_path: str = _f("C", "1.5x of selected_bbox, clamped.", m2=True, default="")
    loose_crop_path: str = _f("C", "Cut from loose_bbox when it differs from selected_bbox, else same file.", default="")
    frame_path: str = _f("C", "frames/frame_XXXXXX.jpg", m2=True, default="")
    annotated_frame_path: Optional[str] = _f("C", "annotated_frames/frame_XXXXXX.jpg", m2=True)

    # -- D. OCR verification pass ------------------------------------------------------
    ocr_text: str = _f("D", "Normalised, lines joined with ' | ' as in M2.", m2=True, default="")
    ocr_text_raw: str = _f("D", "Un-normalised lines.", default="")
    ocr_source: Optional[str] = _f("D", "tight_crop, enlarged_crop, null", m2=True)
    ocr_max_confidence: Optional[float] = _f("D", "Best EasyOCR line confidence.")
    ocr_brand_match: Optional[str] = _f("D", "Canonical spelling of the best match, or null (T28).")
    ocr_fuzzy_score: int = _f("D", "0-100; 0 when no text. Rule inputs.", default=0)
    ocr_match_list: Optional[str] = _f("D", "canonical, audio, null (T29).")
    ocr_match_mode: Optional[str] = _f("D", "exact_short, fuzzy (T27).")

    # -- E. FAISS ------------------------------------------------------------------------
    faiss_top1: Optional[str] = _f("E", "Brand string as stored in the index.", m2=True)
    faiss_top1_score: Optional[float] = _f("E", m2=True)
    faiss_top5_brands: list[str] = _f("E", m2=True, default_factory=list)
    faiss_top5_scores: list[float] = _f("E", m2=True, default_factory=list)
    faiss_top1_brand_canonical: Optional[str] = _f("E", "Mapped onto the canonical list where possible.")
    faiss_score_gap: Optional[float] = _f("E", "top1 − top2 (brand-deduplicated). Rule 7 input.")
    faiss_agrees_with_vlm: Optional[bool] = _f("E", "canonical(faiss_top1) == canonical_brand.")
    brand_in_database: Optional[bool] = _f("E", "canonical_brand (or an alias) appears among index brand strings.")

    # -- F. Audio -------------------------------------------------------------------------
    audio_brands: list[str] = _f("F", "Video-level list, repeated on every row (M2 convention).", m2=True, default_factory=list)
    audio_brands_canonical: list[str] = _f("F", "Mapped where possible.", default_factory=list)
    brand_from_audio: bool = _f("F", "canonical_brand in audio_brands_canonical. Rules 5, 11.", default=False)
    audio_signal_status: str = _f("F", "ok, empty, signal_unavailable (frames mode), disabled", default="disabled")

    # -- G. VLM, hinted pass ----------------------------------------------------------------
    is_logo: Optional[str] = _f("G", "logo, partial_logo, not_logo, null (unparsed).", m2=True)
    vlm_decision: Optional[str] = _f("G", "Same value as is_logo (the correspondence name).")
    brand: Optional[str] = _f("G", "Raw VLM brand string, UNKNOWN, or null.", m2=True)
    vlm_brand: Optional[str] = _f("G", "Same as brand.")
    is_truncated: Optional[bool] = _f("G", m2=True)
    logo_probability: Optional[int] = _f("G", "The generated 0-100 field (M2). Kept for the UI.", m2=True)
    confidence: Optional[int] = _f("G", "Generated 0-100.", m2=True)
    signals_used: list[str] = _f("G", m2=True, default_factory=list)
    signals_used_raw: list[str] = _f("G", m2=True, default_factory=list)
    reasoning: Optional[str] = _f("G", m2=True)
    vlm_parse_ok: bool = _f("G", "False when the JSON did not parse.", default=False)
    logo_probability_scored: Optional[float] = _f("G", "T31 continuous score, null until enabled.")
    not_logo_confidence: Optional[float] = _f("G", "T31, null until enabled. Rule 1 primary variant input.")
    decision_scores: Optional[dict] = _f("G", "{logo, partial_logo, not_logo, uncertain} renormalised.")

    # -- H. VLM, blind pass (T33) -------------------------------------------------------------
    vlm_decision_blind: Optional[str] = _f("H")
    vlm_brand_blind: Optional[str] = _f("H")
    vlm_brand_blind_canonical: Optional[str] = _f("H")
    blind_hinted_brand_agree: Optional[bool] = _f("H", "For T64.")
    hint_composition: list[dict] = _f("H", "[{item, source}], source in {audio, faiss, ocr, watermark}.", default_factory=list)
    hinted_brand_in_faiss_hints: Optional[bool] = _f("H", "For T64 FAISS-origin attribution.")

    # -- I. Canonicalisation (T37) --------------------------------------------------------------
    canonical_brand: Optional[str] = _f("I", "Deterministic map of vlm_brand; UNKNOWN stays UNKNOWN; null for not_logo.")
    brand_canonical: Optional[str] = _f("I", "Same value under the M2 column name.", m2=True)
    canonical_match_mode: Optional[str] = _f("I", "alias, exact, fuzzy, audio, unmatched, llm_merge")
    canonical_match_score: Optional[int] = _f("I")
    brand_list_version: str = _f("I", "From the snapshot (T5).", default="")
    brand_list_hash: str = _f("I", default="")

    # -- J. Watermark (T47-T48) -----------------------------------------------------------------
    channel: Optional[str] = _f("J", "Per-video config.")
    channel_brand_canonical: Optional[str] = _f("J")
    watermark_regions_available: bool = _f("J", "False when the video has no regions.", default=False)
    watermark_overlap: Optional[bool] = _f("J", "Null when no regions (rule reads null).")
    watermark_overlap_ratio: Optional[float] = _f("J", "The containment value.")

    # -- K. Router (T38-T46) --------------------------------------------------------------------
    brand_evidence: str = _f("K", "ocr_confirmed, faiss_confirmed, both, none (T43).", default="none")
    has_conflicting_ocr: bool = _f("K", "Derived per 18 July.", default=False)
    strong_visual_match: bool = _f("K", default=False)
    audio_only_prediction: bool = _f("K", default=False)
    faiss_only_prediction: bool = _f("K", default=False)
    router_decision: Optional[str] = _f("K", "ACCEPT, REVIEW, REJECT, UNKNOWN_LOGO, DEDUPED")
    router_reason: Optional[str] = _f("K", "A member of the reason-code enum (T44).")
    reason_code: Optional[str] = _f("K", "Same as router_reason (the name T73 uses for the filter).")
    router_rule: Optional[int] = _f("K", "1-18, or null for the default.")
    router_confidence: Optional[float] = _f("K")
    risk_score: Optional[int] = _f("K")
    trust_score: Optional[int] = _f("K")
    review_priority: Optional[str] = _f("K", "low, normal, high")
    dedup_cluster_id: Optional[int] = _f("K", "Rules 17/18.")
    dedup_representative: Optional[bool] = _f("K")
    rules_matched_preempted: list[int] = _f("K", "Rules whose condition held but were pre-empted (T46).", default_factory=list)

    # -- L. Run provenance and review ------------------------------------------------------------
    config_hash: str = _f("L", default="")
    git_sha: Optional[str] = _f("L", "Code revision of the run (run_manifest.json); 14 T77, 21 provenance.")
    experiment: str = _f("L", "Run label; defaults to run_id.", m2=True, default="")
    brand_mode: str = _f("L", "open", m2=True, default="open")
    enabled_signals: list[str] = _f("L", "Signals in the hinted prompt.", m2=True, default_factory=list)
    label: Optional[str] = _f("L", "TP, FP, FN, wrong_brand, null. Written by the review screen.", m2=True)
    enrolled_to_faiss: bool = _f("L", "Set by T75.", default=False)


CSV_COLUMNS: list[str] = [f.name for f in fields(Row)]

# JSON-serialised on the CSV (M2 convention: lists/dicts become JSON strings).
_LIST_OR_DICT_COLUMNS = {
    f.name for f in fields(Row)
    if f.type in ("list[float]", "list[str]", "list[int]", "list[dict]", "Optional[dict]", "Optional[list[float]]")
}


def row_to_csv_dict(row: Row) -> dict[str, Any]:
    """Serialise a ``Row`` for a CSV line: lists/dicts as JSON strings,
    booleans as Python's ``True``/``False`` text (pandas convention, M2)."""
    import json as _json

    out: dict[str, Any] = {}
    for f in fields(Row):
        v = getattr(row, f.name)
        if f.name in _LIST_OR_DICT_COLUMNS:
            out[f.name] = _json.dumps(v if v is not None else ([] if "list" in f.type else None), ensure_ascii=False)
        else:
            out[f.name] = v
    return out


def row_from_dict(d: dict[str, Any]) -> Row:
    known = {f.name for f in fields(Row)}
    return Row(**{k: v for k, v in d.items() if k in known})
