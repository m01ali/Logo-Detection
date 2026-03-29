import streamlit as st
from PIL import Image
import io
import os
import albumentations as A
import numpy as np
import tempfile
from typing import List, Tuple
import json
import torch
import sys
from pathlib import Path
import re
import pandas as pd
import math

# 1) Compute your project root (one level up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)

# 2) Insert it on sys.path so `models` and `utils` become importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.faiss_db import LogoDatabaseNew, LogoDatabaseSigLIP2
from utils.video_annotations_to_crops import process_video_annotations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------
# Hardcoded augmentations (used for both insertion and TTA search)
# ------------------------------
AUGMENTATIONS = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ColorJitter(hue=0.05, saturation=0.1, p=0.4),
    A.Resize(224, 224)
])

# Use a dedicated FAISS directory (can be overridden via FAISS_DB_DIR env var)
DB_BASE_PATH = Path(os.getenv("FAISS_DB_DIR", str(PROJECT_ROOT / "FAISS"))).expanduser().resolve()
DB_BASE_PATH.mkdir(parents=True, exist_ok=True)
print(DB_BASE_PATH / "logo_index.faiss")
known_processed = 1216
st.set_page_config(page_title="Logo Search & Indexing", layout="wide")

# ------------------------------
# Initialize or load FAISS logo database
# ------------------------------
@st.cache_resource
def load_database():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return LogoDatabaseNew(
        index_path=str(DB_BASE_PATH/"logo_index.faiss"),
        metadata_path=str(DB_BASE_PATH/"metadata.json"),
        batch_size=64,
        device=device
    )

# Global FAISS DB instance (CLIP — default, used by all tabs except where overridden)
db = load_database()

# SigLIP2 database — loaded lazily only when the user first selects it
@st.cache_resource
def load_database_siglip2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return LogoDatabaseSigLIP2(
        index_path=str(DB_BASE_PATH / "logo_index_siglip2.faiss"),
        metadata_path=str(DB_BASE_PATH / "metadata_siglip2.json"),
        batch_size=8,
        device=device
    )

_EMBED_MODEL_OPTIONS = ["CLIP (default)", "SigLIP2"]

def get_db(model_label: str):
    """Return the correct DB instance for the selected embedding model."""
    if model_label == "SigLIP2":
        return load_database_siglip2()
    return db  # CLIP

# ------------------------------
# Helpers
# ------------------------------
def _inc(dct, key, n=1):
    dct[key] = dct.get(key, 0) + n

def _norm(s):
    return (s or "").strip().lower()

def _source_key(source_val: str):
    """Map CSV source values to metric dict keys"""
    if not isinstance(source_val, str):
        return None
    s = source_val.strip().lower()
    if s in ["top_2k_brands", "top2k", "top2k_brands"]:
        return "top2k_brands"
    if s in ["database", "internal_db"]:
        return "internal_db"
    if s in ["audio"]:
        return "audio"
    return None

def update_metrics(correction_type: str, count: int, source_val=None):
    """
    Centralized metrics update.
    correction_type ∈ {"false_positive","typo","wrong_attribution","confirm","known"}
    """
    gm = st.session_state.global_metrics
    # Always update global totals
    if correction_type == "false_positive":
        _inc(gm, "false_positives", count)
        _inc(gm, "total_processed", count)
    elif correction_type == "typo":
        _inc(gm, "typo_corrections", count)
        _inc(gm, "total_known", count)
        _inc(gm, "total_processed", count)
    elif correction_type == "wrong_attribution":
        _inc(gm, "wrong_attributions", count)
        _inc(gm, "total_known", count)
        _inc(gm, "total_processed", count)
    elif correction_type == "confirm":
        _inc(gm, "confirmed_attributions", count)
        _inc(gm, "total_processed", count)
    elif correction_type == "known":
        _inc(gm, "total_known", count)
        _inc(gm, "total_processed", count)

    # Per-source updates if source is known
    skey = _source_key(source_val)
    if skey:
        sm = gm[skey]
        _inc(sm, "processed", count)
        if correction_type == "false_positive":
            _inc(sm, "false_positives", count)
            _inc(sm, "errors", count)
        elif correction_type == "typo":
            _inc(sm, "typo_corrections", count)
            _inc(sm, "errors", count)
        elif correction_type == "wrong_attribution":
            _inc(sm, "wrong_attributions", count)
            _inc(sm, "errors", count)
        elif correction_type == "confirm":
            _inc(sm, "confirmed_attributions", count)
            
# ------------------------------
# Initialize session state for metrics
# ------------------------------
def initialize_metrics():
    return {
        "total_known": 0,
        "total_unknown": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "typo_corrections": 0,
        "wrong_attributions": 0,
        "confirmed_attributions": 0,
        "total_processed": 0,
        # NEW: manual global-only input
        "missed_logos_manual": 0,
        # Tracking: small logos and multiple logos in frame
        "small_logos": 0,
        "multiple_logos": 0,

        "top2k_brands": {
            "processed": 0, 
            "errors": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "typo_corrections": 0,
            "wrong_attributions": 0,
            "confirmed_attributions": 0
        },
        "internal_db": {
            "processed": 0, 
            "errors": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "typo_corrections": 0,
            "wrong_attributions": 0,
            "confirmed_attributions": 0
        },
        "audio": {
            "processed": 0, 
            "errors": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "typo_corrections": 0,
            "wrong_attributions": 0,
            "confirmed_attributions": 0
        }
    }


if "global_metrics" not in st.session_state:
    st.session_state.global_metrics = initialize_metrics()

# ------------------------------
# Streamlit UI Configuration
# ------------------------------
st.title("🔍 Logo Search & FAISS Indexing System")
st.markdown("""
A professional interface for inserting and searching logos using a FAISS-powered logo database.
""")

# ------------------------------
# Tabs for Add / Search / Save
# ------------------------------
tabs = st.tabs(["➕ Add Logo", "🔎 Search Logo", "💾 Save Database", "📹 Video Object Cropping with Annotations", 
                "Review Unknown Images", "Review Known Logos", "📊 Global Metrics"])

# ------------------------------
# TAB 1: Add Logo
# ------------------------------
with tabs[0]:
    st.subheader("Add a Logo to the Database")
    uploaded_file = st.file_uploader("Upload Logo Image", type=["png", "jpg", "jpeg"])
    brand_name = st.text_input("Enter Brand Name")
    logo_source = st.selectbox("Logo Source", ["Top 2k brands", "Internal database", "Audio"])
    add_embed_model = st.selectbox(
        "Embedding Model",
        _EMBED_MODEL_OPTIONS,
        key="add_embed_model",
        help="Choose which embedding model's FAISS index to add this logo to. "
             "CLIP and SigLIP2 use separate indexes — make sure you add and search with the same model."
    )
    use_aug = st.checkbox("Use augmentations (5 variants)", value=False)

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Logo", width=200)

    if st.button("Add to FAISS Database"):
        if not uploaded_file or not brand_name.strip():
            st.warning("Please upload an image and enter a valid brand name.")
        else:
            img = Image.open(uploaded_file).convert("RGB")
            _add_db = get_db(add_embed_model)
            _add_db.add_logos(images=[img], brand_names=[brand_name.strip()],
                        augmentations=AUGMENTATIONS if use_aug else None,
                        num_augments=5)

            # Update metrics
            st.session_state.global_metrics["total_known"] += 1

            # Update source-specific metrics
            source_key = ""
            if logo_source == "Top 2k brands":
                source_key = "top2k_brands"
            elif logo_source == "Internal database":
                source_key = "internal_db"
            elif logo_source == "Audio":
                source_key = "audio"

            if source_key:
                st.session_state.global_metrics[source_key]["processed"] += 1

            st.success(f"Successfully added logo for '{brand_name.strip()}' to the {add_embed_model} database.")

# ------------------------------
# TAB 2: Search Logo
# ------------------------------
with tabs[1]:
    st.subheader("Search for Similar Logos")
    query_file = st.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg"], key="query")
    threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.95, step=0.05, value=0.85)
    search_embed_model = st.selectbox(
        "Embedding Model",
        _EMBED_MODEL_OPTIONS,
        key="search_embed_model",
        help="Choose which embedding model's FAISS index to search. "
             "Must match the model used when the logos were added."
    )
    use_tta = st.checkbox("Use test-time augmentations (5 variants)", value=False)

    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        st.image(query_img, caption="Query Logo", width=200)

    if st.button("Search FAISS Database"):
        if not query_file:
            st.warning("Please upload a query image to search.")
        else:
            query_img = Image.open(query_file).convert("RGB")
            _search_db = get_db(search_embed_model)
            results = _search_db.search_logo(query_img, threshold=threshold, k=5,
                                             augmentations=AUGMENTATIONS if use_tta else None,
                                             num_augments=5)

            if not results:
                st.info(f"No matches found in the {search_embed_model} index.")
            else:
                st.success(f"Found {len(results)} matching logo(s) [{search_embed_model}]:")
                for match in results:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Similarity", f"{match['similarity']:.2f}")
                    with col2:
                        st.write(f"**Brand**: {match['brand_name']}")

# ------------------------------
# TAB 3: Save Database
# ------------------------------
with tabs[2]:
    st.subheader("Save Database")
    save_embed_model = st.selectbox(
        "Embedding Model",
        _EMBED_MODEL_OPTIONS,
        key="save_embed_model",
        help="Select which model's index to save to disk."
    )
    if st.button("Save FAISS Index & Metadata"):
        _save_db = get_db(save_embed_model)
        _save_db.save()
        _index_file = "logo_index_siglip2.faiss" if save_embed_model == "SigLIP2" else "logo_index.faiss"
        _meta_file  = "metadata_siglip2.json"    if save_embed_model == "SigLIP2" else "metadata.json"
        st.success(f"[{save_embed_model}] Database saved to disk ({_index_file} & {_meta_file}).")

# ------------------------------
# TAB 4: Export Video Annotations to Database
# ------------------------------
with tabs[3]:
    st.subheader("Extract Logos from Video with Annotations")
    
    # Upload video and annotation files
    video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
    json_file = st.file_uploader("Upload Label Studio JSON annotations", type=["json"])
    
    if "frame_results" not in st.session_state:
        st.session_state.frame_results = []
    
    use_aug = st.checkbox("Use augmentations for all logos (10 variants)", value=False)
    
    # Add to FAISS button
    if st.button("📦 Add to FAISS Database"):            
        if st.session_state.frame_results:
            try:
                logos = [item['crop'] for item in st.session_state.frame_results.values()]
                labels = [item['label'].strip() for item in st.session_state.frame_results.values()]
                # st.success(labels)
                
                db.add_logos(images=logos, brand_names=labels,
                            augmentations=AUGMENTATIONS if use_aug else None,
                            num_augments=5)
                
                # Update metrics
                st.session_state.global_metrics["total_known"] += len(logos)
                st.session_state.global_metrics["internal_db"]["processed"] += len(logos)
                
                st.success("All cropped logos successfully added to the FAISS index.")
            except Exception as e:
                st.error(f"Failed to add to FAISS DB: {e}")
        else:
            st.warning('No logos processed yet!')
            
    
    if video_file and json_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name

        annotations = json.load(json_file)
        
        
        if st.button("🚀 Process Video and Crop Objects"):
            st.info("Processing Video....")
            
            st.session_state.frame_results = process_video_annotations(video_path=video_path, annotations_json=annotations)
            
            st.success(f"Processed {len(st.session_state.frame_results)} logos.")
            for i, (id, item) in enumerate(st.session_state.frame_results.items()):
                st.markdown(f"**Label:** {item['label']} | **Frame #:** {item['frame_number']}")
                st.image(item['crop'], caption=f"{item['label']} (Frame {item['frame_number']})")
                
# ------------------------------
# TAB 5: Review Unknown Logos
# ------------------------------
def review_unknown_logos_tab(db, AUGMENTATIONS=None):
    """
    Review & label unknown logos from a single folder (no subfolders).

    Highlights:
      - Robust state (no post-instantiation widget key mutation)
      - Single folder path (images only)
      - Status filter: All / Reviewed / Not reviewed
      - Sorted by filename (ascending)
      - Pagination (adjustable; default 24) with decoupled internal state (no key collisions)
      - Select all on page (transition-aware)
      - Selected-on-page counter
      - Bulk label textbox + ONE combined bulk action:
          "✅ Bulk label, add to FAISS & mark reviewed (current page)"
        * Bulk add uses augmentation ONLY if "Use augmentation" is checked
      - Individual Add to FAISS
      - "Reviewed" = confirmed (added to FAISS): tracked as a pure set (not a widget key)
      - CSV export of confirmed unknowns (filename, filepath, brand, timestamp)
      - Shows global False Negatives metric
      - Images use use_container_width=True
    """
    st.subheader("🧠 Review & Label Unknown Logos")

    ss = st.session_state

    # ------------------ Session state initialization ------------------
    if "global_metrics" not in ss:
        ss.global_metrics = initialize_metrics()

    # Reviewed is a pure set (NOT tied to any widget key)
    ss.setdefault("unknown_reviewed_set", set())
    reviewed_set = ss.unknown_reviewed_set

    # CSV rows of confirmed unknowns
    ss.setdefault("unknown_confirmations", [])

    # Pagination & filtering state
    ss.setdefault("unknown_page_size_ui", 24)       # widget key for page size
    ss.setdefault("unknown_page_num", 1)            # INTERNAL page number (decoupled from any widget key)
    ss.setdefault("unknown_last_path", None)
    ss.setdefault("unknown_last_status_filter", None)

    # Select-all transition tracking
    ss.setdefault("unknown_select_all_prev", False)
    ss.setdefault("unknown_select_all_key", "")

    # Bulk label & augmentation UI
    ss.setdefault("unknown_bulk_label_ui", "")
    ss.setdefault("unknown_use_aug", False)         # unchecked by default

    # Deferred state changes to avoid post-instantiation mutations of selection keys
    ss.setdefault("unknown_pending_unselect", set())  # filenames to unselect next run
    ss.setdefault("unknown_pending_key_pops", set())  # keys to pop next run (after deletes)

    # ------------------ Inputs ------------------
    UNKNOWN_DIR = st.text_input("Unknowns folder path (images only)", value="unknowns")
    if not os.path.isdir(UNKNOWN_DIR):
        st.info("Unknowns folder not found at the provided path.")
        return


    _FRAME_LOGO_RE = re.compile(r"^frame(\d+)_logo(\d+)", re.IGNORECASE)

    def _list_images_sorted(dir_path: str):
        """
        Return image filenames sorted by (frame_no, logo_no) extracted from
        'frame{frame_no}_logo{logo_no}*.ext'. Non-matching files are placed after,
        sorted lexicographically.
        """
        # collect image files only
        files = [f for f in os.listdir(dir_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        parsed = []
        unmatched = []
        for f in files:
            # strip extension then parse numbers
            stem = Path(f).stem  # e.g., "frame1279_logo0"
            m = _FRAME_LOGO_RE.match(stem)
            if m:
                frame_no = int(m.group(1))
                logo_no = int(m.group(2))
                parsed.append((frame_no, logo_no, f))
            else:
                unmatched.append(f)

        # sort numeric-first
        parsed.sort(key=lambda t: (t[0], t[1]))          # by frame_no, then logo_no
        unmatched.sort(key=str.lower)                    # stable lexicographic for oddballs

        # return just filenames
        return [f for (_, _, f) in parsed] + unmatched


    status_filter = st.selectbox(
        "Show",
        ["All", "Reviewed", "Not reviewed"],
        key="unknown_status_filter",
        help="Filter images by review status."
    )

    # Reset pagination when path or status filter changes
    if ss.unknown_last_path != UNKNOWN_DIR or ss.unknown_last_status_filter != status_filter:
        ss.unknown_page_num = 1
        ss.unknown_last_path = UNKNOWN_DIR
        ss.unknown_last_status_filter = status_filter

    # Page size (state-driven via key)
    st.number_input(
        "Images per page",
        min_value=6, max_value=200, step=6,
        key="unknown_page_size_ui",
        help="Adjust page size to balance speed and visibility."
    )

    # ------------------ Build file list ------------------
    image_files = _list_images_sorted(UNKNOWN_DIR)
    if not image_files:
        st.info("No images found in the unknowns folder.")
        return

    all_paths = [str(Path(UNKNOWN_DIR) / fn) for fn in image_files]

    # ------------------ Filter by status (based on reviewed_set only) ------------------
    if status_filter == "Reviewed":
        filtered_paths = [p for p in all_paths if p in reviewed_set]
    elif status_filter == "Not reviewed":
        filtered_paths = [p for p in all_paths if p not in reviewed_set]
    else:
        filtered_paths = all_paths

    total_images = len(filtered_paths)
    if total_images == 0:
        st.info(f"No images to show for filter '{status_filter}'.")
        return

    # ------------------ Pagination (decoupled) ------------------
    page_size = int(ss.unknown_page_size_ui)
    total_pages = max(1, math.ceil(total_images / page_size))

    # PRE-RENDER HOUSEKEEPING (apply queued key changes before any widgets render)
    if ss.unknown_pending_unselect:
        for fn in list(ss.unknown_pending_unselect):
            key = f"u_sel_{fn}"
            if key in ss:
                ss[key] = False
            ss.unknown_pending_unselect.discard(fn)
    if ss.unknown_pending_key_pops:
        for key in list(ss.unknown_pending_key_pops):
            ss.pop(key, None)
            ss.unknown_pending_key_pops.discard(key)

    # NAV BUTTONS FIRST — update internal page number (NOT a widget key)
    nav_cols = st.columns([1, 2, 2, 2, 1])
    with nav_cols[0]:
        if st.button("⏮ First"):
            ss.unknown_page_num = 1
            st.rerun()
    with nav_cols[1]:
        if st.button("◀ Prev"):
            ss.unknown_page_num = max(1, ss.unknown_page_num - 1)
            st.rerun()
    # Page number INPUT WITHOUT KEY; it returns a value we copy into the internal page number
    with nav_cols[2]:
        # clamp current internal number before showing
        current_page_display = max(1, min(ss.unknown_page_num, total_pages))
        new_page = st.number_input("Page", min_value=1, max_value=total_pages, value=current_page_display, step=1)
        if int(new_page) != ss.unknown_page_num:
            ss.unknown_page_num = int(new_page)
            st.rerun()
    with nav_cols[3]:
        if st.button("Next ▶"):
            ss.unknown_page_num = min(total_pages, ss.unknown_page_num + 1)
            st.rerun()
    with nav_cols[4]:
        if st.button("Last ⏭"):
            ss.unknown_page_num = total_pages
            st.rerun()

    # Normalize again just in case
    ss.unknown_page_num = max(1, min(ss.unknown_page_num, total_pages))

    current_page = int(ss.unknown_page_num)
    start_idx = (current_page - 1) * page_size
    end_idx = min(total_images, start_idx + page_size)
    page_paths = filtered_paths[start_idx:end_idx]
    page_files = [Path(p).name for p in page_paths]

    st.caption(
        f"Showing {start_idx+1}–{end_idx} of {total_images} images "
        f"({status_filter.lower()}, page {current_page}/{total_pages})"
    )

    # ------------------ Counters ------------------
    reviewed_count = len([p for p in all_paths if p in reviewed_set])
    not_reviewed_count = len(all_paths) - reviewed_count
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Reviewed", reviewed_count)
    with m2: st.metric("Not Reviewed", not_reviewed_count)
    with m3: st.metric("False Negatives (global)", int(ss.global_metrics.get("false_negatives", 0)))

    st.markdown("---")

    # ------------------ CSV export ------------------
    export_confirmations = pd.DataFrame(ss.unknown_confirmations) if ss.unknown_confirmations else pd.DataFrame(
        columns=["filename", "filepath", "brand", "timestamp"]
    )
    st.download_button(
        "📤 Download CSV: Confirmed Unknowns",
        data=export_confirmations.to_csv(index=False).encode("utf-8"),
        file_name="confirmed_unknowns.csv",
        mime="text/csv",
        help="All unknowns manually confirmed as real logos with corrected names."
    )

    st.markdown("---")

    # ------------------ Bulk selection + combined bulk action ------------------
    # Ensure per-item selection KEYS exist for this page (BEFORE rendering widgets)
    for fn in page_files:
        sel_key = f"u_sel_{fn}"
        if sel_key not in ss:
            ss[sel_key] = False

    # Select all on page (transition-aware)
    selall_key = f"u_selall_{current_page}_{status_filter}"
    if selall_key not in ss:
        ss[selall_key] = False
    if ss.unknown_select_all_key != selall_key:
        ss.unknown_select_all_prev = ss[selall_key]
        ss.unknown_select_all_key = selall_key

    select_all_on_page = st.checkbox("Select all on page", key=selall_key)
    prev_val = ss.unknown_select_all_prev
    if select_all_on_page != prev_val:
        for fn in page_files:
            ss[f"u_sel_{fn}"] = select_all_on_page
        ss.unknown_select_all_prev = select_all_on_page

    # Selected count (on this page)
    selected_count = sum(1 for fn in page_files if ss.get(f"u_sel_{fn}", False))
    st.caption(f"**Selected on page:** {selected_count}")

    # Bulk label & augmentation controls
    cbl, caug = st.columns([3, 1.6])
    with cbl:
        st.text_input(
            "📝 Bulk label (used when an item's own label is blank)",
            key="unknown_bulk_label_ui",
            placeholder="Enter brand name for selected items with no per-item label"
        )
    with caug:
        st.checkbox("Use augmentation", key="unknown_use_aug",
                    help="Apply image augmentations during BULK add only.")

    # ONE combined bulk button:
    bulk_ok = st.button("✅ Bulk label, add to FAISS & mark reviewed (current page)")
    st.caption("For each selected item: use its per-item label if set; otherwise apply the bulk label. Items without any label are skipped.")

    st.markdown("---")

    # ------------------ Page grid ------------------
    GRID_COLS = 4
    cols = st.columns(GRID_COLS)

    for idx, p in enumerate(page_paths):
        fn = page_files[idx]
        image_path = Path(p)
        with cols[idx % GRID_COLS]:
            st.image(str(image_path), caption=fn, use_container_width=True)

            # Reviewed checkbox (NO key) bound to set value
            rev_checked = st.checkbox(f"Reviewed | {fn}", value=(p in reviewed_set))
            if rev_checked: reviewed_set.add(p)
            else: reviewed_set.discard(p)

            # Selection checkbox WITH key
            sel_key = f"u_sel_{fn}"
            st.checkbox("Select", key=sel_key)

            # Per-item label (KEYED)
            lbl_key = f"lbl_{fn}"
            if lbl_key not in ss:
                ss[lbl_key] = ""
            st.text_input(f"Label for {fn}", key=lbl_key)

            # Per-item actions
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑 Discard", key=f"discard_{fn}"):
                    try:
                        os.remove(image_path)
                        # Defer clearing keys to next run
                        ss.unknown_pending_key_pops.update({sel_key, lbl_key})
                        reviewed_set.discard(p)
                        st.success(f"Deleted {fn}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete {fn}: {e}")

            with c2:
                if st.button("✅ Add to FAISS", key=f"add_{fn}"):
                    brand_input = ss.get(lbl_key, "").strip()
                    if not brand_input:
                        st.warning("Please enter a brand label before adding to FAISS.")
                    else:
                        try:
                            img = Image.open(image_path).convert("RGB")
                            # Per-item add keeps original behavior (not tied to the bulk 'use augmentation' toggle)
                            db.add_logos(
                                [img], [brand_input],
                                augmentations=(AUGMENTATIONS if AUGMENTATIONS else None),
                                num_augments=(10 if AUGMENTATIONS else 0)
                            )
                            reviewed_set.add(p)
                            ss.unknown_confirmations.append({
                                "filename": fn,
                                "filepath": str(image_path),
                                "brand": brand_input,
                                "timestamp": pd.Timestamp.utcnow().isoformat()
                            })
                            ss.global_metrics["false_negatives"] = int(ss.global_metrics.get("false_negatives", 0)) + 1
                            ss.global_metrics["total_processed"] = int(ss.global_metrics.get("total_processed", 0)) + 1
                            st.success(f"Added to FAISS as '{brand_input}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to add to FAISS: {e}")

    # ------------------ Execute combined bulk action ------------------
    if bulk_ok:
        bulk_label = ss.unknown_bulk_label_ui.strip()
        selected_on_page = [fn for fn in page_files if ss.get(f"u_sel_{fn}", False)]

        if not selected_on_page:
            st.warning("Select at least one image on this page.")
        else:
            imgs, labels = [], []
            processed, skipped = 0, 0
            for fn in selected_on_page:
                p = str(Path(UNKNOWN_DIR) / fn)
                lbl_key = f"lbl_{fn}"
                label = ss.get(lbl_key, "").strip() or bulk_label
                if not label:
                    skipped += 1
                    continue
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    labels.append(label)
                    ss[lbl_key] = label  # persist label for UI
                    processed += 1
                except Exception:
                    skipped += 1
                    continue

            # BULK add to FAISS — use augmentation ONLY if checkbox ticked
            if imgs:
                if ss.unknown_use_aug and AUGMENTATIONS is not None:
                    db.add_logos(imgs, labels, augmentations=AUGMENTATIONS, num_augments=10)
                else:
                    db.add_logos(imgs, labels, augmentations=None, num_augments=0)

            # Mark reviewed, log CSV, update metrics, and queue unselect for next run
            now = pd.Timestamp.utcnow().isoformat()
            for fn in selected_on_page:
                p = str(Path(UNKNOWN_DIR) / fn)
                final_label = ss.get(f"lbl_{fn}", "").strip() or bulk_label
                if not final_label:
                    continue  # skipped above
                reviewed_set.add(p)
                ss.unknown_confirmations.append({
                    "filename": fn,
                    "filepath": p,
                    "brand": final_label,
                    "timestamp": now
                })
                ss.global_metrics["false_negatives"] = int(ss.global_metrics.get("false_negatives", 0)) + 1
                ss.global_metrics["total_processed"] = int(ss.global_metrics.get("total_processed", 0)) + 1
                ss.unknown_pending_unselect.add(fn)  # safely unselect on next run

            st.success(f"Bulk processed {processed} item(s) and added to FAISS. Skipped {skipped} without labels.")
            st.rerun()

    # --- Bottom navigation (4.4: next page button also at end of page) ---
    st.markdown("---")
    st.caption(f"Page {current_page} of {total_pages}")
    nav_bot = st.columns([1, 2, 2, 2, 1])
    with nav_bot[0]:
        if st.button("⏮ First", key="u_bot_first"):
            ss.unknown_page_num = 1
            st.rerun()
    with nav_bot[1]:
        if st.button("◀ Prev", key="u_bot_prev"):
            ss.unknown_page_num = max(1, ss.unknown_page_num - 1)
            st.rerun()
    with nav_bot[2]:
        new_page_bot = st.number_input(
            "Page", min_value=1, max_value=total_pages,
            value=max(1, min(ss.unknown_page_num, total_pages)),
            step=1, key="u_bot_page_input"
        )
        if int(new_page_bot) != ss.unknown_page_num:
            ss.unknown_page_num = int(new_page_bot)
            st.rerun()
    with nav_bot[3]:
        if st.button("Next ▶", key="u_bot_next"):
            ss.unknown_page_num = min(total_pages, ss.unknown_page_num + 1)
            st.rerun()
    with nav_bot[4]:
        if st.button("Last ⏭", key="u_bot_last"):
            ss.unknown_page_num = total_pages
            st.rerun()


# Call the function
with tabs[4]:
    review_unknown_logos_tab(db, AUGMENTATIONS=AUGMENTATIONS)


# ------------------------------
# TAB 6: Review Known Logos
# ------------------------------
def review_known_logos_tab(db, AUGMENTATIONS=None): 
    st.subheader("🔍 Review and Correct Known Logos")

    if "global_metrics" not in st.session_state:
        st.session_state.global_metrics = initialize_metrics()

    col1, col2 = st.columns(2)
    with col1:
        csv_file = st.file_uploader(
            "Upload CSV with known logos",
            type="csv",
            help="Must have columns: filename, brand. Optional: source, correction, postfilter_verdict"
        )
    with col2:
        known_base_dir = st.text_input(
            "Known logos directory",
            value="knowns",
            help="Folder containing the logo image files"
        )

    # --- Load CSV / directory and validate
    if csv_file is None and not os.path.isdir(known_base_dir):
        st.info("Please upload a CSV or specify a valid directory to begin.")
        return

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            st.error("Failed to read CSV. Please ensure it's formatted correctly.")
            return
        if not {"filename", "brand"}.issubset(df.columns):
            st.error("CSV must contain columns: filename, brand.")
            return
        has_source_col = "source" in df.columns
        has_correction_col = "correction" in df.columns
        has_postfilter_col = "postfilter_verdict" in df.columns
        has_family_col = "family" in df.columns
    else:
        df = pd.DataFrame(columns=["filename", "brand"])
        has_source_col = False
        has_correction_col = False
        has_postfilter_col = False
        has_family_col = False

    if os.path.isdir(known_base_dir):
        all_images = sorted([
            f for f in os.listdir(known_base_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ])
        if csv_file is None:
            df = pd.DataFrame({"filename": all_images, "brand": ["" for _ in all_images]})
            has_correction_col = False
            has_postfilter_col = False
            has_family_col = False
        else:
            df = df[df["filename"].isin(all_images)].reset_index(drop=True)
    else:
        st.error(f"Directory not found: {known_base_dir}")
        return

    if df.empty:
        st.info("No known logos found in the specified directory/CSV.")
        return

    # Materialize path column
    df["filepath"] = df["filename"].apply(lambda fn: Path(known_base_dir) / fn)

    # --- Session state setup
    ss = st.session_state
    ss.setdefault("known_brand_buffer", {})
    ss.setdefault("confirmed_attributions", {})
    ss.setdefault("correction_stats", {
        "typo_corrections": 0,
        "wrong_logo_corrections": 0,
        "false_positives": 0
    })
    ss.setdefault("bulk_selection", {})
    ss.setdefault("original_df", df.copy())
    ss.setdefault("corrections_map", {})  # filepath -> correction type
    ss.setdefault("known_family_buffer", {})   # filepath -> family name (4.1)
    ss.setdefault("small_logo_flags", {})      # filepath -> bool (4.2)
    ss.setdefault("multiple_logo_flags", {})   # filepath -> bool (4.2)

    # NEW: remember filters to reset selection when they change
    ss.setdefault("last_selected_brand", None)
    ss.setdefault("last_selected_postfilter", None)
    ss.setdefault("last_selected_review_status", None)

    # Helper: normalize correction label
    def _normalize_correction(val: str) -> str:
        if type(val) == float and math.isnan(val):
            return ""
        s = _norm(val)
        if s in ("typo", "typo_correction", "typo_corrections"):
            return "typo"
        if s in ("wrong", "wrong_attribution", "wrong_logo", "wrong_logo_attribution", "wrong_attributions"):
            return "wrong"
        if s in ("false_positive", "fp", "false positive", "falsepositives", "false_positives"):
            return "false_positive"
        if s in ("confirm", "confirmed", "confirmations", "confirmed_attribution"):
            return "confirm"
        return ""

    # Seed corrections_map from CSV if present
    if has_correction_col:
        for _, row in df.iterrows():
            c = _normalize_correction(row.get("correction", ""))
            if c:
                ss.corrections_map[str(row["filepath"])] = c

    # Keep 'total_known' aligned with uploaded known set
    st.session_state.global_metrics["total_known"] = len(df)

    # Recompute known-related metrics once if CSV carried corrections
    def _reset_known_metrics_from_corrections():
        gm = st.session_state.global_metrics
        gm["typo_corrections"] = 0
        gm["wrong_attributions"] = 0
        gm["false_positives"] = 0
        gm["confirmed_attributions"] = 0
        gm["total_processed"] = 0
        for s in ["top2k_brands", "internal_db", "audio"]:
            sm = gm.get(s, {})
            sm["processed"] = 0
            sm["errors"] = 0
            sm["false_positives"] = 0
            sm["typo_corrections"] = 0
            sm["wrong_attributions"] = 0
            sm["confirmed_attributions"] = 0
            gm[s] = sm
        for _, row in df.iterrows():
            path_str = str(row["filepath"])
            corr = ss.corrections_map.get(path_str, "")
            if not corr:
                continue
            src_key = _source_key(row["source"]) if has_source_col else None
            if corr == "typo":
                gm["typo_corrections"] += 1
            elif corr == "wrong":
                gm["wrong_attributions"] += 1
            elif corr == "false_positive":
                gm["false_positives"] += 1
            elif corr == "confirm":
                gm["confirmed_attributions"] += 1
            gm["total_processed"] += 1
            if src_key:
                sm = gm[src_key]
                sm["processed"] += 1
                if corr in ("typo", "wrong", "false_positive"):
                    sm["errors"] += 1
                if corr == "typo":
                    sm["typo_corrections"] += 1
                elif corr == "wrong":
                    sm["wrong_attributions"] += 1
                elif corr == "false_positive":
                    sm["false_positives"] += 1
                elif corr == "confirm":
                    sm["confirmed_attributions"] += 1
                gm[src_key] = sm

    if has_correction_col and ss.get("recomputed_from_corrections_once") is not True:
        _reset_known_metrics_from_corrections()
        ss.recomputed_from_corrections_once = True

    use_aug = st.checkbox(
        "Use augmentations for image (5 variants)",
        value=False,
        key='use_aug_known'
    )

    st.markdown("---")
    st.subheader("🔄 Bulk Correction & Confirmation")

    # ---------- Brand filter ----------
    unique_brands = sorted(df["brand"].unique())
    selected_brand = st.selectbox(
        "Select brand to filter",
        ["All brands"] + unique_brands,
        key="selected_brand_known"
    )

    # ---------- NEW: Postfilter verdict filter (if column exists) ----------
    if has_postfilter_col:
        # Prepare options (include only actually present values, e.g., Other/Incorrect/Correct/Skipped)
        unique_verdicts = sorted([v for v in df["postfilter_verdict"].dropna().unique()])
        selected_postfilter = st.selectbox(
            "Filter by postfilter verdict",
            ["All verdicts"] + unique_verdicts,
            key="selected_postfilter_known",
            help="Only shown if your CSV contains 'postfilter_verdict'. Helps evaluate metrics with/without postfilter."
        )
    else:
        selected_postfilter = "All verdicts"  # no-op default

    # ---------- NEW: Review status filter (4.4) ----------
    review_status_options = [
        "All",
        "Not reviewed yet",
        "Confirmed",
        "False positive",
        "Typo",
        "Wrong attribution",
    ]
    selected_review_status = st.selectbox(
        "Filter by review status",
        review_status_options,
        key="selected_review_status_known",
        help="Filter by what correction/confirmation has been applied in this session."
    )

    # Reset selection if any filter changed
    if (
        ss.get("last_selected_brand") != selected_brand
        or ss.get("last_selected_postfilter") != selected_postfilter
        or ss.get("last_selected_review_status") != selected_review_status
    ):
        ss.bulk_selection = {}
        ss.select_all_known_prev = False
        ss.last_selected_brand = selected_brand
        ss.last_selected_postfilter = selected_postfilter
        ss.last_selected_review_status = selected_review_status

    # Apply filters
    filtered_df = df if selected_brand == "All brands" else df[df["brand"] == selected_brand]
    if has_postfilter_col and selected_postfilter != "All verdicts":
        filtered_df = filtered_df[filtered_df["postfilter_verdict"] == selected_postfilter]
    # Review status filter
    if selected_review_status != "All":
        _status_map = {
            "Not reviewed yet": lambda p: ss.corrections_map.get(p, "") == "",
            "Confirmed":        lambda p: ss.corrections_map.get(p, "") == "confirm",
            "False positive":   lambda p: ss.corrections_map.get(p, "") == "false_positive",
            "Typo":             lambda p: ss.corrections_map.get(p, "") == "typo",
            "Wrong attribution": lambda p: ss.corrections_map.get(p, "") == "wrong",
        }
        _pred = _status_map.get(selected_review_status)
        if _pred:
            filtered_df = filtered_df[
                filtered_df["filepath"].apply(lambda fp: _pred(str(fp)))
            ]
    filtered_df = filtered_df.reset_index(drop=True)

    # --- Postfilter verdict summary (only if column exists) ---
    if has_postfilter_col:
        st.markdown("#### 📊 Postfilter Summary (current view)")
        # Count verdicts present in the *current filtered view*
        vc = (
            filtered_df["postfilter_verdict"]
            .fillna("—")
            .value_counts(dropna=False)
            .to_dict()
        )

        # Preferred display order; any extra values will be appended
        preferred = ["Correct", "Incorrect", "Skipped", "Other", "—"]
        ordered = [k for k in preferred if k in vc] + [k for k in vc if k not in preferred]

        # Build a compact metric row: Total + each verdict
        cols = st.columns(max(1, len(ordered) + 1))
        with cols[0]:
            st.metric("Total (view)", len(filtered_df))

        for i, k in enumerate(ordered, start=1):
            label = {
                "Correct": "✅ Correct",
                "Incorrect": "❌ Incorrect",
                "Skipped": "⚪ Skipped",
                "Other": "🟣 Other",
                "—": "—"
            }.get(k, k)
            with cols[i]:
                st.metric(label, int(vc.get(k, 0)))

        st.markdown("---")

    # --- Family statistics (4.1 / 4.2) ---
    st.markdown("#### 👨‍👩‍👧 Family Statistics")
    family_counts: dict = {}
    for _, _frow in df.iterrows():
        _fp = str(_frow["filepath"])
        _fam = ss.known_family_buffer.get(
            _fp,
            str(_frow.get("family", "") or "") if has_family_col else ""
        ).strip()
        if _fam:
            family_counts[_fam] = family_counts.get(_fam, 0) + 1
    if family_counts:
        _fam_df = (
            pd.DataFrame(list(family_counts.items()), columns=["Family", "Image Count"])
            .sort_values("Image Count", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(_fam_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No family names assigned yet. Use the **Family** field under each logo to assign one.")
    st.markdown("---")

    if not filtered_df.empty:

        # === Bulk controls ABOVE the images ===
        st.markdown("**Apply Bulk Correction**")
        correction_type = st.radio(
            "Correction type:",
            ["Typo in brand name","Wrong logo attribution","False positive"],
            horizontal=True
        )
        new_brand_name = st.text_input("Correct brand name (if applicable):", key="new_brand_name")

        bulk_apply_clicked = st.button("🔁 Apply Bulk Correction", key="apply_bulk_correction")

        # Merged: Confirm + Add to DB in one button
        confirm_bulk_clicked = st.button("✅ Confirm Bulk Attribution (also adds to FAISS)", key="confirm_bulk")
        
        # === Export Updated CSV controls (kept above the images) ===
        st.markdown("📤 **Export Updated CSV**")

        # Apply in-memory brand edits before exporting
        export_df = df.copy()
        for path_str, brand in st.session_state.known_brand_buffer.items():
            mask = export_df["filepath"].astype(str) == path_str
            if mask.any():
                export_df.loc[mask, "brand"] = brand

        # Create/update correction column
        if "correction" not in export_df.columns:
            export_df["correction"] = ""
        for path_str, corr in st.session_state.corrections_map.items():
            mask = export_df["filepath"].astype(str) == path_str
            if mask.any():
                export_df.loc[mask, "correction"] = corr

        # Add / update family column from buffer (4.1)
        if "family" not in export_df.columns:
            export_df["family"] = ""
        export_df["family"] = export_df["filename"].apply(
            lambda fn: ss.known_family_buffer.get(
                str(Path(known_base_dir) / fn),
                str(df.loc[df["filename"] == fn, "family"].iloc[0])
                if has_family_col and (df["filename"] == fn).any()
                else ""
            ) or ""
        )

        # Drop internal columns and prepare CSV
        export_df = export_df.drop(columns=["filepath"])
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Updated CSV",
            csv_data,
            "known_logos_updated.csv",
            "text/csv",
            help="Exports all brand edits and corrections applied so far."
        )

        st.markdown("---")
        
        st.subheader("🖼 Known Logos")

        # === Selection handling ===
        select_all = st.checkbox("Select all images", key="select_all_known")

        # Initialize/refresh selection flags for current filtered view
        for _, row in filtered_df.iterrows():
            path_str = str(row["filepath"])
            if select_all:
                # force-select all visible images
                st.session_state.bulk_selection[path_str] = True
            elif path_str not in st.session_state.bulk_selection:
                st.session_state.bulk_selection[path_str] = False
            elif not select_all and st.session_state.get("select_all_known_prev", True):
                # user just unchecked "Select all" -> clear current page's selections
                st.session_state.bulk_selection[path_str] = False

        st.session_state.select_all_known_prev = select_all

        # === Image grid (ALL images; scroll naturally) ===
        cols = st.columns(4)
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            with cols[i % 4]:
                st.image(str(row["filepath"]), caption=row["filename"], width=150)
                path_str = str(row["filepath"])

                # Show any recorded correction tag
                corr_val = ss.corrections_map.get(path_str, "")
                st.caption(f"Correction: {corr_val or 'none'}")

                # If postfilter_verdict exists, show it under each image for context
                if has_postfilter_col:
                    st.caption(f"Postfilter: {row.get('postfilter_verdict', '—')}")

                # per-item selection
                is_checked = ss.bulk_selection.get(path_str, False)
                ss.bulk_selection[path_str] = st.checkbox("Select", value=is_checked, key=f"bulk_{row['filename']}")

                # per-item brand edit
                current_brand = ss.known_brand_buffer.get(path_str, row["brand"])
                label = f"Brand for {row['filename']}"
                if has_source_col:
                    label += f" (Source: {row['source']})"
                new_brand = st.text_input(label, value=current_brand, key=f"brand_{row['filename']}").strip()
                if new_brand and _norm(new_brand) != _norm(current_brand):
                    ss.known_brand_buffer[path_str] = new_brand

                # Family name input (4.1)
                current_family = ss.known_family_buffer.get(
                    path_str,
                    str(row.get("family", "") or "") if has_family_col else ""
                )
                new_family = st.text_input(
                    f"Family for {row['filename']}",
                    value=current_family,
                    key=f"family_{row['filename']}",
                    placeholder="e.g. RAI, Sky…"
                ).strip()
                ss.known_family_buffer[path_str] = new_family

                # Tracking checkboxes (4.2)
                chk_c1, chk_c2 = st.columns(2)
                with chk_c1:
                    prev_small = ss.small_logo_flags.get(path_str, False)
                    is_small = st.checkbox(
                        "Small logo",
                        value=prev_small,
                        key=f"kl_small_{row['filename']}",
                        help="Logo occupies a very small area of the image"
                    )
                    ss.small_logo_flags[path_str] = is_small
                with chk_c2:
                    prev_multi = ss.multiple_logo_flags.get(path_str, False)
                    is_multi = st.checkbox(
                        "Multiple logos",
                        value=prev_multi,
                        key=f"kl_multi_{row['filename']}",
                        help="Frame shows 2+ logos (fully or partially)"
                    )
                    ss.multiple_logo_flags[path_str] = is_multi

                # per-item quick actions
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("Update", key=f"update_{row['filename']}"):
                        img = Image.open(row["filepath"]).convert("RGB")
                        brand_to_use = ss.known_brand_buffer.get(path_str, row["brand"])
                        mask = df["filepath"] == row["filepath"]
                        df.loc[mask, "brand"] = brand_to_use
                        db.add_logos([img], [brand_to_use],
                                     augmentations=(AUGMENTATIONS if use_aug else None), num_augments=5)
                        source_val = row["source"] if has_source_col else None
                        update_metrics("known", 1, source_val)
                        st.success(f"Updated FAISS entry for {brand_to_use}")

                with btn_col2:
                    if st.button("🚫 FALSE_LOGO", key=f"false_{row['filename']}"):
                        try:
                            img = Image.open(row["filepath"]).convert("RGB")
                            db.add_logos([img], ["FALSE_LOGO"],
                                         augmentations=(AUGMENTATIONS if use_aug else None), num_augments=5)
                            mask = df["filepath"] == row["filepath"]
                            df.loc[mask, "brand"] = "FALSE_LOGO"
                            ss.known_brand_buffer[path_str] = "FALSE_LOGO"
                            ss.confirmed_attributions.pop(path_str, None)
                            ss.bulk_selection.pop(path_str, None)
                            source_val = row["source"] if has_source_col else None
                            update_metrics("false_positive", 1, source_val)
                            ss.corrections_map[path_str] = "false_positive"
                            st.success("Marked as FALSE_LOGO and added to the database")
                        except Exception as e:
                            st.error(f"Failed to mark FALSE_LOGO: {e}")
                        st.rerun()

                with btn_col3:
                    if st.button("✓ Confirm", key=f"confirm_{row['filename']}"):
                        if not ss.confirmed_attributions.get(path_str):
                            ss.confirmed_attributions[path_str] = True
                            source_val = row["source"] if has_source_col else None
                            update_metrics("confirm", 1, source_val)
                            ss.corrections_map[path_str] = "confirm"
                            st.success("Attribution confirmed!")
                        st.rerun()

        # === Execute pending bulk actions AFTER the grid ===

        # Bulk Correction (typo / wrong / false positive)
        if bulk_apply_clicked:
            selected_paths = [p for p, v in ss.bulk_selection.items() if v]
            if not selected_paths:
                st.warning("Select at least one image")
            elif correction_type != "False positive" and not new_brand_name.strip():
                st.warning("Enter a valid brand name")
            else:
                images_to_update, brands_to_update = [], []
                n_fp = n_typo = n_wrong = 0

                for path_str in selected_paths:
                    mask = df["filepath"].astype(str) == path_str
                    if not mask.any():
                        continue
                    row = df[mask].iloc[0]
                    source_val = row["source"] if has_source_col else None

                    if correction_type == "False positive":
                        try:
                            img = Image.open(path_str).convert("RGB")
                            images_to_update.append(img)
                            brands_to_update.append("FALSE_LOGO")
                            df.loc[mask, "brand"] = "FALSE_LOGO"
                            ss.known_brand_buffer[path_str] = "FALSE_LOGO"
                            update_metrics("false_positive", 1, source_val)
                            ss.corrections_map[path_str] = "false_positive"
                            n_fp += 1
                        except Exception as e:
                            st.error(f"Failed processing {path_str}: {e}")
                    else:
                        current_brand = row["brand"]
                        target_brand = new_brand_name.strip()
                        if _norm(current_brand) == _norm(target_brand):
                            continue
                        df.loc[mask, "brand"] = target_brand
                        ss.known_brand_buffer[path_str] = target_brand
                        ss.confirmed_attributions.pop(path_str, None)
                        try:
                            img = Image.open(path_str).convert("RGB")
                            images_to_update.append(img)
                            brands_to_update.append(target_brand)
                        except Exception:
                            pass
                        if correction_type == "Typo in brand name":
                            update_metrics("typo", 1, source_val)
                            ss.corrections_map[path_str] = "typo"
                            n_typo += 1
                        else:
                            update_metrics("wrong_attribution", 1, source_val)
                            ss.corrections_map[path_str] = "wrong"
                            n_wrong += 1

                if images_to_update:
                    db.add_logos(
                        images_to_update, brands_to_update,
                        augmentations=(AUGMENTATIONS if use_aug else None), num_augments=5
                    )
                st.success(f"Applied corrections: {n_typo} typo, {n_wrong} wrong, {n_fp} false positives")
                ss.bulk_selection = {}
                st.rerun()

        # Merged: Confirm + Add to FAISS
        if confirm_bulk_clicked:
            selected_paths = [p for p, v in ss.bulk_selection.items() if v]
            if not selected_paths:
                st.warning("Select at least one image to confirm.")
            else:
                imgs, labels = [], []
                n_confirmed = 0
                for path_str in selected_paths:
                    mask = df["filepath"].astype(str) == path_str
                    if not mask.any():
                        continue
                    row = df[mask].iloc[0]
                    brand_to_use = ss.known_brand_buffer.get(path_str, row["brand"])
                    # add to FAISS
                    try:
                        img = Image.open(path_str).convert("RGB")
                        imgs.append(img)
                        labels.append(brand_to_use)
                    except Exception:
                        pass
                    # metrics + corrections
                    if not ss.confirmed_attributions.get(path_str):
                        ss.confirmed_attributions[path_str] = True
                        source_val = row["source"] if has_source_col else None
                        update_metrics("confirm", 1, source_val)
                        ss.corrections_map[path_str] = "confirm"
                        n_confirmed += 1

                if imgs:
                    db.add_logos(
                        imgs, labels,
                        augmentations=(AUGMENTATIONS if use_aug else None), num_augments=5
                    )
                st.success(f"Confirmed {n_confirmed} attribution(s) and added {len(imgs)} logo(s) to FAISS.")
                ss.bulk_selection = {}
                st.rerun()


# Call the function
with tabs[5]:
    review_known_logos_tab(db, AUGMENTATIONS=AUGMENTATIONS)
    
    
def global_metrics_tab():
    st.subheader("📊 Global Metrics Dashboard")
    if "global_metrics" not in st.session_state:
        st.session_state.global_metrics = initialize_metrics()
    metrics = st.session_state.global_metrics

    # Denominator for percentage tiles (kept as known + unknown)
    denom = max(0, int(metrics.get("total_known", 0) + metrics.get("total_unknown", 0)))
    def _fmt_with_pct(n: int) -> str:
        if denom <= 0:
            return f"{n} (0%)"
        pct = (n / denom) * 100.0
        return f"{n} ({pct:.0f}%)"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Known Logos", metrics["total_known"])
        st.metric("Total Unknown Logos", metrics["total_unknown"])
    with col2:
        st.metric("False Positives", _fmt_with_pct(metrics["false_positives"]))
        st.metric("False Negatives", _fmt_with_pct(metrics["false_negatives"]))
    with col3:
        st.metric("Typo Corrections", _fmt_with_pct(metrics["typo_corrections"]))
        st.metric("Wrong Attributions", _fmt_with_pct(metrics["wrong_attributions"]))
    with col4:
        st.metric("Confirmed Attributions", _fmt_with_pct(metrics["confirmed_attributions"]))
        st.metric("Total Processed", metrics["total_processed"])

    # Tracking metrics: small logos and multiple logos (4.2)
    # Computed live from session state flags set in the Known Logos tab
    small_logos_count = sum(1 for v in st.session_state.get("small_logo_flags", {}).values() if v)
    multiple_logos_count = sum(1 for v in st.session_state.get("multiple_logo_flags", {}).values() if v)
    st.markdown("---")
    st.subheader("🏷️ Tracking Metrics")
    tr_col1, tr_col2 = st.columns(2)
    with tr_col1:
        st.metric(
            "Small Logos",
            small_logos_count,
            help="Logos flagged as occupying a very small area of the image"
        )
    with tr_col2:
        st.metric(
            "Multiple Logos in Frame",
            multiple_logos_count,
            help="Frames where 2+ logos are visible (fully or partially)"
        )

    # NEW: Manual Missed Logos (Global-only) — bug-free single click
    st.markdown("---")
    st.subheader("🧮 Manual: Missed Logos (Global Only)")

    # 1) Initialize a dedicated UI state key once
    if "missed_logos_manual_ui" not in st.session_state:
        st.session_state.missed_logos_manual_ui = int(st.session_state.global_metrics.get("missed_logos_manual", 0))

    # 2) Bind widget to the state key (no 'value=' here)
    st.number_input(
        "Enter/update the number of logos missed by the pipeline (manual entry):",
        min_value=0,
        step=1,
        key="missed_logos_manual_ui",
        help="Global-only, manual counter—does not affect per-source metrics."
    )

    # 3) Sync UI value into metrics each run
    st.session_state.global_metrics["missed_logos_manual"] = int(st.session_state.missed_logos_manual_ui)

    # 4) Show the metric tile
    st.metric("Missed Logos (manual)", st.session_state.global_metrics["missed_logos_manual"])


    st.markdown("---")
    st.subheader("📈 Per-Source Metrics")
    sources = ["top2k_brands","internal_db","audio"]
    names = ["Top 2k Brands","Internal Database","Audio"]
    for s, n in zip(sources, names):
        st.markdown(f"**{n}**")
        sm = metrics.get(s, {})
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Processed", _fmt_with_pct(sm.get("processed", 0)))
        with col2: st.metric("Errors", _fmt_with_pct(sm.get("errors", 0)))
        with col3: st.metric("False Positives", _fmt_with_pct(sm.get("false_positives", 0)))
        with col4: st.metric("Confirmed", _fmt_with_pct(sm.get("confirmed_attributions", 0)))
        ec1, ec2 = st.columns(2)
        with ec1: st.metric("Typo Corrections", _fmt_with_pct(sm.get("typo_corrections", 0)))
        with ec2: st.metric("Wrong Attributions", _fmt_with_pct(sm.get("wrong_attributions", 0)))
        st.markdown("---")

    if st.button("🔄 Reset All Metrics"):
        st.session_state.global_metrics = initialize_metrics()
        # reset the UI control, too
        st.session_state.missed_logos_manual_ui = 0
        st.success("Metrics reset!")
        st.rerun()


# ------------------------------
# TAB 7: Global Metrics
# ------------------------------
with tabs[6]:
    global_metrics_tab()