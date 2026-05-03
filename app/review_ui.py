"""
Multi-Signal Logo Review UI
============================
Streamlit app for manually reviewing Qwen3-VL logo predictions and tracking
TP / FP / FN / wrong-brand labels.

Run:
    streamlit run app/review_ui.py -- --csv path/to/vlm_predictions.csv

Or set the CSV path directly in the sidebar.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logo Detection Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resolve CSV path from CLI arg (streamlit passes args after "--") ──────────
def _cli_csv() -> str | None:
    try:
        idx = sys.argv.index("--")
        p = argparse.ArgumentParser()
        p.add_argument("--csv", default=None)
        args, _ = p.parse_known_args(sys.argv[idx + 1:])
        return args.csv
    except ValueError:
        return None


# ── Load / save helpers ───────────────────────────────────────────────────────
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        df["label"] = None
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def _safe_img(path: str | None) -> Image.Image | None:
    if not path or not Path(path).exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _parse_list(val) -> list:
    if isinstance(val, list):
        return val
    if pd.isna(val) if not isinstance(val, (list, str)) else False:
        return []
    try:
        return json.loads(val)
    except Exception:
        return []


def _score_color(val) -> tuple[str, int]:
    try:
        v = int(float(val))
        return ("#2ecc71" if v >= 75 else "#f39c12" if v >= 45 else "#e74c3c"), v
    except (TypeError, ValueError):
        return "#aaaaaa", 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Logo Detection Review")

# CLI arg always wins — overwrite stale session state so restarting with a
# different --csv immediately takes effect without clearing the text box manually.
_cli_path = _cli_csv()
if _cli_path and st.session_state.get("_last_cli_csv") != _cli_path:
    st.session_state["csv_input"] = _cli_path
    st.session_state["_last_cli_csv"] = _cli_path

csv_path = st.sidebar.text_input(
    "CSV path",
    value=_cli_path or "",
    placeholder="/path/to/vlm_predictions.csv",
    key="csv_input",
)

if not csv_path or not Path(csv_path).exists():
    st.info("Enter a valid path to `vlm_predictions.csv` in the sidebar to begin.")
    st.stop()

# ── Path prefix remapping (for Kaggle-exported CSVs) ─────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("Path prefix remapping", expanded=False):
    remap_from = st.text_input(
        "Replace prefix",
        value="/kaggle/working/",
        key="remap_from",
    )
    remap_to = st.text_input(
        "With prefix",
        value=str(Path(__file__).resolve().parents[1]) + "/",
        key="remap_to",
    )
    do_remap = st.checkbox("Enable remapping", value=False, key="do_remap")

_cache_key = (csv_path, st.session_state.get("remap_from"), st.session_state.get("remap_to"), st.session_state.get("do_remap"))
if "df" not in st.session_state or st.session_state.get("_cache_key") != _cache_key:
    _df = load_csv(csv_path)
    if st.session_state.get("do_remap") and remap_from and remap_to:
        for _col in ("crop_path", "enlarged_crop_path", "frame_path"):
            if _col in _df.columns:
                _df[_col] = _df[_col].astype(str).str.replace(remap_from, remap_to, regex=False)
    st.session_state.df = _df
    st.session_state._cache_key = _cache_key

df: pd.DataFrame = st.session_state.df

# ── Filters ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

filter_label = st.sidebar.multiselect(
    "Review label",
    options=["unlabelled", "TP", "FP", "FN", "wrong_brand"],
    default=["unlabelled"],
)
filter_is_logo = st.sidebar.multiselect(
    "VLM prediction",
    options=["logo", "partial_logo", "not_logo"],
    default=["logo", "partial_logo", "not_logo"],
)
min_conf = st.sidebar.slider("Min confidence (brand)", 0, 100, 0)
max_conf = st.sidebar.slider("Max confidence (brand)", 0, 100, 100)
min_logo_prob = st.sidebar.slider("Min logo probability", 0, 100, 0)
max_logo_prob = st.sidebar.slider("Max logo probability", 0, 100, 100)

# ── Build filtered view ───────────────────────────────────────────────────────
mask = pd.Series([True] * len(df), index=df.index)

if filter_is_logo:
    mask &= df["is_logo"].isin(filter_is_logo)

if filter_label:
    if "unlabelled" in filter_label:
        unlabelled_mask = df["label"].isna() | (df["label"] == "")
        labelled_mask   = df["label"].isin([l for l in filter_label if l != "unlabelled"])
        mask &= unlabelled_mask | labelled_mask
    else:
        mask &= df["label"].isin(filter_label)

conf_col = pd.to_numeric(df["confidence"], errors="coerce")
mask &= conf_col.fillna(50).between(min_conf, max_conf)

if "logo_probability" in df.columns:
    logo_prob_col = pd.to_numeric(df["logo_probability"], errors="coerce")
    mask &= logo_prob_col.fillna(50).between(min_logo_prob, max_logo_prob)

view = df[mask].reset_index(drop=False)   # keep original index in "index" column

# ── Metrics panel ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Metrics")

labelled = df[df["label"].notna() & (df["label"] != "")]
total_labelled = len(labelled)
tp = (labelled["label"] == "TP").sum()
fp = (labelled["label"] == "FP").sum()
fn = (labelled["label"] == "FN").sum()
wb = (labelled["label"] == "wrong_brand").sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

st.sidebar.metric("Labelled", f"{total_labelled} / {len(df)}")
c1, c2 = st.sidebar.columns(2)
c1.metric("TP", tp)
c2.metric("FP", fp)
c3, c4 = st.sidebar.columns(2)
c3.metric("FN", fn)
c4.metric("Wrong brand", wb)
st.sidebar.metric("Precision", f"{precision:.1%}")
st.sidebar.metric("Recall",    f"{recall:.1%}")
st.sidebar.metric("F1",        f"{f1:.3f}")

# ── Save button ───────────────────────────────────────────────────────────────
if st.sidebar.button("💾 Save labels"):
    save_csv(st.session_state.df, csv_path)
    st.sidebar.success("Saved.")

# ── Export ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Export")

wb_as_fp = st.sidebar.checkbox(
    "Count wrong_brand as FP in export metrics",
    value=False,
    help="When enabled, wrong_brand detections are included as FP when computing "
         "precision/recall in the exported summary.",
)
export_scope = st.sidebar.radio(
    "Rows to export",
    options=["Labelled only (TP/FP/FN/wrong_brand)", "All detections"],
    index=0,
)

def _build_export_csv(full_df: pd.DataFrame, labelled_only: bool, wb_fp: bool) -> bytes:
    export_cols = [
        "det_id", "frame_idx", "timecode_s", "det_score",
        "crop_path", "enlarged_crop_path", "frame_path",
        "is_logo", "brand", "logo_probability", "confidence",
        "faiss_top1", "faiss_top1_score",
        "faiss_top5_brands", "faiss_top5_scores",
        "audio_brands", "signals_used", "reasoning",
        "label",
    ]
    present = [c for c in export_cols if c in full_df.columns]
    out = full_df[present].copy()

    if labelled_only:
        out = out[out["label"].notna() & (out["label"] != "")]

    # Derive effective_label for export: optionally remap wrong_brand → FP
    out["effective_label"] = out["label"]
    if wb_fp:
        out["effective_label"] = out["effective_label"].replace("wrong_brand", "FP")

    # Compute per-row metrics columns
    out["is_correct"] = out["label"].map({"TP": True, "FP": False, "FN": False, "wrong_brand": False})

    # Append summary rows
    _tp  = (out["effective_label"] == "TP").sum()
    _fp  = (out["effective_label"] == "FP").sum()
    _fn  = (out["effective_label"] == "FN").sum()
    _wb  = (out["label"] == "wrong_brand").sum() if not wb_fp else 0
    _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
    _rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
    _f1   = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0

    summary_rows = pd.DataFrame([
        {},  # blank separator
        {"det_id": "--- SUMMARY ---"},
        {"det_id": "TP",           "label": str(_tp)},
        {"det_id": "FP",           "label": str(_fp)},
        {"det_id": "FN",           "label": str(_fn)},
        {"det_id": "wrong_brand",  "label": str(_wb)},
        {"det_id": "Precision",    "label": f"{_prec:.4f}"},
        {"det_id": "Recall",       "label": f"{_rec:.4f}"},
        {"det_id": "F1",           "label": f"{_f1:.4f}"},
        {"det_id": "wb_counted_as_FP", "label": str(wb_fp)},
    ])

    combined = pd.concat([out, summary_rows], ignore_index=True)
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


_labelled_only = export_scope.startswith("Labelled")
_csv_bytes = _build_export_csv(st.session_state.df, _labelled_only, wb_as_fp)
_export_stem = Path(csv_path).stem
st.sidebar.download_button(
    label="⬇️ Download export CSV",
    data=_csv_bytes,
    file_name=f"{_export_stem}_export.csv",
    mime="text/csv",
)

# ── Navigation ────────────────────────────────────────────────────────────────
st.title("Logo Detection Review")

if view.empty:
    st.warning("No detections match the current filters.")
    st.stop()

total_in_view = len(view)
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
st.session_state.page_idx = min(st.session_state.page_idx, total_in_view - 1)

col_prev, col_counter, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("← Prev") and st.session_state.page_idx > 0:
        st.session_state.page_idx -= 1
        st.rerun()
with col_next:
    if st.button("Next →") and st.session_state.page_idx < total_in_view - 1:
        st.session_state.page_idx += 1
        st.rerun()
with col_counter:
    st.markdown(
        f"<div style='text-align:center;padding-top:6px'>"
        f"Detection <b>{st.session_state.page_idx + 1}</b> / {total_in_view}"
        f"</div>",
        unsafe_allow_html=True,
    )

row = view.iloc[st.session_state.page_idx]
orig_idx = row["index"]   # original DataFrame index for writing back

# ── Detection detail ──────────────────────────────────────────────────────────
det_id       = row.get("det_id", "?")
timecode     = row.get("timecode_s", "?")
det_score    = row.get("det_score", "?")
is_logo      = row.get("is_logo", "?")
brand        = row.get("brand", "?")
confidence   = row.get("confidence", "?")
logo_prob    = row.get("logo_probability")
reasoning    = row.get("reasoning", "")
signals      = _parse_list(row.get("signals_used", "[]"))
audio_br     = _parse_list(row.get("audio_brands", "[]"))
faiss_br     = _parse_list(row.get("faiss_top5_brands", "[]"))
faiss_sc     = _parse_list(row.get("faiss_top5_scores", "[]"))
current_label = row.get("label") if not pd.isna(row.get("label", float("nan"))) else None

# ── Signal reference legend ───────────────────────────────────────────────────
with st.expander("Signal reference — what each signal means", expanded=False):
    st.markdown(
        "| Signal name | # | What it is |\n"
        "|---|---|---|\n"
        "| `crop_visual` / `Signal 1` | 1 | **Tight crop** — bounding box at detector confidence |\n"
        "| `enlarged_crop` / `Signal 2` | 2 | **Enlarged crop** — same region padded with surrounding context |\n"
        "| `full_frame` / `Signal 3` | 3 | **Full frame** — entire video frame at detection timestamp |\n"
        "| `temporal_context` / `Signal 4` | 4 | **Temporal context** — adjacent frames at T−2, T−1, T+1, T+2 |\n"
        "| `audio_brands` / `Signal 5` | 5 | **Audio brands** — brand names from audio transcript (Whisper + Qwen3) |\n"
        "| `faiss_matches` / `Signal 6` | 6 | **FAISS matches** — CLIP visual similarity top-K from brand database |\n"
    )
    st.caption(
        "**logo_probability** = P(crop contains any logo), independent of brand. "
        "**confidence** = certainty about the specific brand name."
    )

st.markdown(f"### Detection `det_id={det_id}`  —  t={timecode}s  —  det_score={det_score}")

# ── Image columns ─────────────────────────────────────────────────────────────
img_col1, img_col2, img_col3 = st.columns(3)

crop_img     = _safe_img(row.get("crop_path"))
enlarged_img = _safe_img(row.get("enlarged_crop_path"))
frame_img    = _safe_img(row.get("frame_path"))

with img_col1:
    st.caption("Tight crop")
    if crop_img:
        st.image(crop_img, use_container_width=True)
    else:
        st.warning("crop not found")

with img_col2:
    st.caption("Enlarged crop")
    if enlarged_img:
        st.image(enlarged_img, use_container_width=True)
    else:
        st.warning("enlarged crop not found")

with img_col3:
    st.caption(f"Full frame  (t={timecode}s)")
    if frame_img:
        st.image(frame_img, use_container_width=True)
    else:
        st.warning("frame not found")

st.markdown("---")

# ── Prediction + signals ──────────────────────────────────────────────────────
pred_col, hint_col = st.columns([2, 2])

with pred_col:
    st.subheader("VLM Prediction")

    conf_color, conf_int   = _score_color(confidence)
    logop_color, logop_int = _score_color(logo_prob)

    verdict_color = {
        "logo":         "#2ecc71",
        "partial_logo": "#f39c12",
        "not_logo":     "#e74c3c",
    }.get(str(is_logo), "#aaaaaa")

    st.markdown(
        f"**Is logo:** <span style='color:{verdict_color};font-weight:bold'>{is_logo}</span>"
        f"&nbsp;&nbsp;&nbsp; **Brand:** `{brand}`",
        unsafe_allow_html=True,
    )

    _logo_prob_display = f"{logop_int}/100" if not pd.isna(logo_prob) and logo_prob is not None else "n/a"
    _conf_display = f"{conf_int}/100"
    st.markdown(
        f"**Logo probability** *(P any logo)*: "
        f"<span style='color:{logop_color};font-weight:bold'>{_logo_prob_display}</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"**Brand confidence**: "
        f"<span style='color:{conf_color};font-weight:bold'>{_conf_display}</span>",
        unsafe_allow_html=True,
    )

    if reasoning:
        st.markdown(f"> {reasoning}")
    if signals:
        st.markdown("**Signals used:** " + " · ".join(f"`{s}`" for s in signals))

with hint_col:
    st.subheader("Context Hints")

    st.markdown("**Audio brands:**")
    if audio_br:
        st.markdown(", ".join(f"`{b}`" for b in audio_br))
    else:
        st.caption("(none)")

    st.markdown("**FAISS top-5 visual matches:**")
    if faiss_br:
        rows_data = []
        for brand_name, score in zip(faiss_br, faiss_sc + [None] * 5):
            rows_data.append({"brand": brand_name, "similarity": f"{score:.3f}" if score is not None else "?"})
        st.dataframe(pd.DataFrame(rows_data), hide_index=True, use_container_width=True)
    else:
        st.caption("(none)")

st.markdown("---")

# ── Label buttons ─────────────────────────────────────────────────────────────
st.subheader("Label this detection")

label_cols = st.columns(5)
label_map = {
    "✅ True Positive":  "TP",
    "❌ False Positive": "FP",
    "🔍 False Negative": "FN",
    "🏷️ Wrong Brand":   "wrong_brand",
    "🗑️ Clear label":   None,
}

for col, (btn_text, btn_val) in zip(label_cols, label_map.items()):
    is_active = (current_label == btn_val) and btn_val is not None
    btn_style = "primary" if is_active else "secondary"
    with col:
        if st.button(btn_text, type=btn_style, use_container_width=True):
            st.session_state.df.at[orig_idx, "label"] = btn_val
            save_csv(st.session_state.df, csv_path)
            # Auto-advance to next unlabelled
            if st.session_state.page_idx < total_in_view - 1:
                st.session_state.page_idx += 1
            st.rerun()

if current_label:
    st.success(f"Current label: **{current_label}**")

# ── Thumbnail strip for the current frame ─────────────────────────────────────
with st.expander("All detections in this frame", expanded=False):
    same_frame = df[df["frame_idx"] == row.get("frame_idx", -1)]
    thumb_cols = st.columns(min(len(same_frame), 8))
    for col, (_, srow) in zip(thumb_cols, same_frame.iterrows()):
        thumb = _safe_img(srow.get("crop_path"))
        lbl   = srow.get("label") or "?"
        brand_s = srow.get("brand") or "?"
        with col:
            if thumb:
                st.image(thumb, use_container_width=True)
            st.caption(f"id={srow.get('det_id')}  {brand_s}\n[{lbl}]")
