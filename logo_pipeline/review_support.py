"""Pure helpers behind the review screen's M3 additions (T73 queue filters and
panels, T75 Accept into FAISS), ``spec docs/13``.

``app/review_ui.py`` is a Streamlit script; everything here is plain pandas /
Python so it can be tested without a browser. Every new column is read through
``has``/``val`` so the M2 CSVs (no router columns) still load and filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

QUEUE_DEFAULT = ["REVIEW", "UNKNOWN_LOGO"]
DECISION_ORDER = ["ACCEPT", "REVIEW", "UNKNOWN_LOGO", "REJECT", "DEDUPED"]
UNROUTED = "unrouted"
UNPARSED = "unparsed"
DECISION_COLOURS = {"ACCEPT": "#2ecc71", "REVIEW": "#f39c12", "UNKNOWN_LOGO": "#3498db", "REJECT": "#e74c3c",
                    "DEDUPED": "#95a5a6"}
ENROLABLE = {"REVIEW", "UNKNOWN_LOGO"}
RULE5_FIELDS = ("ocr_text", "ocr_fuzzy_score", "faiss_top1", "faiss_top1_score", "brand_from_audio",
                "watermark_overlap", "watermark_overlap_ratio")


def has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def val(row, col: str, default=None):
    """A cell, with NaN/missing read as ``default``."""
    v = row.get(col, default) if hasattr(row, "get") else default
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def as_list(v) -> list:
    if isinstance(v, list):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    try:
        out = json.loads(v)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def decision_series(df: pd.DataFrame) -> pd.Series:
    if not has(df, "router_decision"):
        return pd.Series([UNROUTED] * len(df), index=df.index)
    return df["router_decision"].fillna(UNROUTED).astype(str)


def decision_options(df: pd.DataFrame) -> list[str]:
    present = set(decision_series(df))
    return [d for d in DECISION_ORDER if d in present] + sorted(present - set(DECISION_ORDER))


def default_decisions(df: pd.DataFrame) -> list[str]:
    """The REVIEW + UNKNOWN_LOGO queue when the CSV is routed, everything otherwise (M2 CSVs)."""
    opts = decision_options(df)
    queue = [d for d in QUEUE_DEFAULT if d in opts]
    return queue or opts


def reason_options(df: pd.DataFrame) -> list[str]:
    return sorted(df["reason_code"].dropna().astype(str).unique()) if has(df, "reason_code") else []


def filter_mask(df: pd.DataFrame, *, decisions=None, reason_codes=None, labels=None, is_logo=None,
                conf=(0, 100), logo_prob=(0, 100)) -> pd.Series:
    """The sidebar filters. An empty/None selection does not filter on that field."""
    mask = pd.Series(True, index=df.index)
    if decisions:
        mask &= decision_series(df).isin(decisions)
    if reason_codes and has(df, "reason_code"):
        mask &= df["reason_code"].astype(str).isin(reason_codes)
    if is_logo is not None and has(df, "is_logo"):
        mask &= df["is_logo"].fillna(UNPARSED).astype(str).isin(is_logo)
    if labels:
        lab = df["label"] if has(df, "label") else pd.Series([None] * len(df), index=df.index)
        unl = lab.isna() | (lab.astype(str) == "")
        chosen = lab.isin([x for x in labels if x != "unlabelled"])
        mask &= (unl & ("unlabelled" in labels)) | chosen
    if has(df, "confidence"):
        mask &= pd.to_numeric(df["confidence"], errors="coerce").fillna(50).between(*conf)
    if has(df, "logo_probability"):
        mask &= pd.to_numeric(df["logo_probability"], errors="coerce").fillna(50).between(*logo_prob)
    return mask


def counts_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows per decision and per (decision, reason code) for the metrics panel."""
    dec = decision_series(df).value_counts().rename_axis("decision").reset_index(name="rows")
    if has(df, "reason_code"):
        rc = (pd.DataFrame({"decision": decision_series(df), "reason_code": df["reason_code"].fillna("—")})
              .value_counts().reset_index(name="rows"))
    else:
        rc = pd.DataFrame(columns=["decision", "reason_code", "rows"])
    return dec, rc


def is_rule5_row(row, channel_brands: set[str]) -> bool:
    reason = str(val(row, "reason_code", "") or "")
    brand = val(row, "canonical_brand")
    channels = set(channel_brands)
    own = val(row, "channel_brand_canonical")
    if own:
        channels.add(own)
    return reason.startswith("channel_brand_") or (brand is not None and brand in channels)


def is_enrolable(row, include_accept: bool = False) -> bool:
    d = val(row, "router_decision")
    return d in ENROLABLE or (include_accept and d == "ACCEPT")


def default_enrol_brand(row) -> str:
    b = val(row, "canonical_brand")
    return "" if b in (None, "UNKNOWN") else str(b)


def build_enrolment_record(row, brand: str):
    """T75's provenance record for one reviewed crop."""
    from .faiss_store import EnrolmentRecord, utcnow
    video_id, run_id, det_id = val(row, "video_id"), val(row, "run_id"), val(row, "det_id")
    frame_idx = val(row, "frame_idx")
    return EnrolmentRecord(
        brand=brand, source_id=f"{video_id}/{run_id}/det_{int(det_id)}", enrolled_at=utcnow(),
        origin="review_ui.accept", enrolled_by="review_ui", video_id=video_id, run_id=run_id,
        frame_idx=int(frame_idx) if frame_idx is not None else None, det_id=int(det_id),
        crop_path=val(row, "crop_path"), review_label=val(row, "label"))


def commit_enrolments(db, pending: list[dict], resolve: Callable[[str], Path | None]) -> list[int]:
    """Enrols every queued crop in one batch and saves the index once (the file is large; saving per
    click is slow). ``pending`` items are ``{"row": dict, "brand": str}``; returns vector indices."""
    from PIL import Image

    from .faiss_store import enrol, save
    images, records = [], []
    for item in pending:
        row = item["row"]
        path = resolve(val(row, "crop_path"))
        if path is None:
            raise FileNotFoundError(f"crop for det_id {val(row, 'det_id')} not found: {val(row, 'crop_path')}")
        images.append(Image.open(path).convert("RGB"))
        records.append(build_enrolment_record(row, item["brand"]))
    indices = enrol(db, images, records)
    save(db)
    return indices


def load_channel_brands(repo_root: Path) -> set[str]:
    """Channel brands from the snapshot and the region store, for the Rule 5 panel. Never raises."""
    out: set[str] = set()
    try:
        import yaml

        from .brands import load_brand_list
        cfg = yaml.safe_load((repo_root / "configs" / "pipeline.yaml").read_text(encoding="utf-8")) or {}
        out |= set(load_brand_list(repo_root / cfg.get("brands", {}).get("list_path", "")).channel_brands)
    except Exception:
        pass
    try:
        from .watermark_regions import REGIONS_STORE, load_regions_store
        for e in load_regions_store(repo_root / REGIONS_STORE).values():
            out |= set(e.channel_brands)
    except Exception:
        pass
    return out
