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

from models.faiss_db import LogoDatabaseNew  # Assuming your class is in logo_database.py
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

DB_BASE_PATH = Path("D:\\milestone 2\\faiss_database_with_italian_logos") #Change path as needed
print(DB_BASE_PATH/'logo_index.faiss')
known_processed = 1216
st.set_page_config(page_title="Logo Search & Indexing", layout="wide")

# ------------------------------
# Initialize or load FAISS logo database
# ------------------------------
@st.cache_resource
def load_database():
    return LogoDatabaseNew(
        index_path=str(DB_BASE_PATH/"logo_index.faiss"),
        metadata_path=str(DB_BASE_PATH/"metadata.json"),
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

# Global FAISS DB instance
db = load_database()

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
        },
        "confirmed_attributions": 0,
        "total_processed": 0
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
    use_aug = st.checkbox("Use augmentations (5 variants)", value=False)
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Logo", width=200)

    if st.button("Add to FAISS Database"):
        if not uploaded_file or not brand_name.strip():
            st.warning("Please upload an image and enter a valid brand name.")
        else:
            img = Image.open(uploaded_file).convert("RGB")
            db.add_logos(images=[img], brand_names=[brand_name.strip()],
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
                
            st.success(f"Successfully added logo for '{brand_name.strip()}' to the database.")

# ------------------------------
# TAB 2: Search Logo
# ------------------------------
with tabs[1]:
    st.subheader("Search for Similar Logos")
    query_file = st.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg"], key="query")
    threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.95, step=0.05, value=0.85)
    use_tta = st.checkbox("Use test-time augmentations (5 variants)", value=False)
    
    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        st.image(query_img, caption="Query Logo", width=200)

    if st.button("Search FAISS Database"):
        if not query_file:
            st.warning("Please upload a query image to search.")
        else:
            query_img = Image.open(query_file).convert("RGB")
            results = db.search_logo(query_img, threshold=threshold, k=5,
                                     augmentations=AUGMENTATIONS if use_tta else None,
                                     num_augments=5)

            if not results:
                st.info("No matches found.")
            else:
                st.success(f"Found {len(results)} matching logo(s):")
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
    if st.button("Save FAISS Index & Metadata"):
        db.save()
        st.success("Database saved successfully to disk (logo_index.faiss & metadata.json).")

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
    IMAGES_PER_ROW = 3

    def list_images(folder: Path):
        return sorted([f.name for f in folder.iterdir()
                       if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    def load_image(p: Path):
        return Image.open(p).convert("RGB")

    st.subheader("🧠 Review and Label Unknown Logos")
    
    # Initialize false negative tracking
    if "false_negatives_count" not in st.session_state:
        st.session_state.false_negatives_count = 0

    # --- Folder input ---
    folder_input = st.text_input(
        "Enter path to folder containing UNKNOWN logos",
        placeholder="e.g. /home/user/unknown_logos"
    )
    if not folder_input:
        st.info("Please enter a folder path to begin.")
        return

    folder_path = Path(folder_input)
    if not folder_path.is_dir():
        st.error(f"❌ `{folder_input}` is not a valid folder.")
        return

    image_files = list_images(folder_path)
    if not image_files:
        st.info("No .jpg/.jpeg/.png files found in that folder.")
        return
        
    # Update total unknown count
    st.session_state.global_metrics["total_unknown"] = len(image_files)

    # --- Buffer for labels ---
    if "label_buffer" not in st.session_state:
        st.session_state.label_buffer = {}

    # --- Bulk selection for unknown logos ---
    if "unknown_bulk_selection" not in st.session_state:
        st.session_state.unknown_bulk_selection = {}
    
    use_aug = st.checkbox("Use augmentations for image (5 variants)", value=False)

    # --- Display metrics at the top ---
    st.markdown("---")
    st.subheader("📊 False Negative Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Unknown Logos", len(image_files))
    with col2:
        st.metric("False Negatives Identified", st.session_state.false_negatives_count)

    # --- Bulk processing section ---
    st.markdown("---")
    st.subheader("🔄 Bulk Processing")
    
    # Select all checkbox
    select_all = st.checkbox("Select all images", key="select_all_unknown")
    
    # Apply selection to all images
    if select_all:
        for filename in image_files:
            img_path = folder_path / filename
            st.session_state.unknown_bulk_selection[str(img_path)] = True
    else:
        # Only clear if we're not selecting all and the selection state is empty
        if not any(st.session_state.unknown_bulk_selection.values()):
            st.session_state.unknown_bulk_selection = {}

    # --- Bulk labeling option ---
    bulk_label = st.text_input("Bulk label for selected images", key="bulk_label_unknown")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Bulk Label", key="apply_bulk_label_unknown"):
            selected_paths = [path for path, selected in st.session_state.unknown_bulk_selection.items() 
                             if selected and Path(path).exists()]
            
            if not selected_paths:
                st.warning("Please select at least one image.")
            elif not bulk_label.strip():
                st.warning("Please enter a label.")
            else:
                # Add all selected images with the bulk label
                images_to_add = []
                labels_to_add = []
                
                for path_str in selected_paths:
                    img = load_image(Path(path_str))
                    images_to_add.append(img)
                    labels_to_add.append(bulk_label.strip())
                    
                    # Remove from folder after adding
                    if Path(path_str).exists():
                        os.remove(path_str)
                
                # Add to database
                db.add_logos(
                    images_to_add, labels_to_add,
                    augmentations=AUGMENTATIONS if use_aug else None,
                    num_augments=5
                )
                
                # Update metrics
                st.session_state.global_metrics["total_known"] += len(images_to_add)
                st.session_state.global_metrics["false_negatives"] += len(images_to_add)
                st.session_state.false_negatives_count += len(images_to_add)
                
                # Clear selection after processing
                for path_str in selected_paths:
                    st.session_state.unknown_bulk_selection[path_str] = False
                
                st.success(f"Added {len(images_to_add)} images with label '{bulk_label.strip()}'")
                st.rerun()

    # --- Render all images in rows of 3 ---
    for i in range(0, len(image_files), IMAGES_PER_ROW):
        row_files = image_files[i:i+IMAGES_PER_ROW]
        cols = st.columns(IMAGES_PER_ROW)
        for col, filename in zip(cols, row_files):
            img_path = folder_path / filename
            # sanitize key
            key_base = re.sub(r'\W+', '_', str(img_path))

            with col:
                # Check if file still exists (might have been deleted in another session)
                if not img_path.exists():
                    continue
                    
                # Checkbox for bulk selection
                is_selected = st.session_state.unknown_bulk_selection.get(str(img_path), False)
                checkbox_key = f"select_{key_base}"
                
                # Create checkbox and update selection state
                new_selection = st.checkbox("Select", value=is_selected, key=checkbox_key)
                st.session_state.unknown_bulk_selection[str(img_path)] = new_selection
                
                st.image(str(img_path), caption=filename)
                brand_input = st.text_input(
                    f"Label for {filename}",
                    key=f"label_{key_base}",
                    value=st.session_state.label_buffer.get(str(img_path), "")
                ).strip()

                # save typed-in label
                if brand_input:
                    st.session_state.label_buffer[str(img_path)] = brand_input
                else:
                    st.session_state.label_buffer.pop(str(img_path), None)

                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("🗑 Discard", key=f"discard_{key_base}"):
                        if img_path.exists():
                            os.remove(img_path)
                        st.session_state.label_buffer.pop(str(img_path), None)
                        st.session_state.unknown_bulk_selection.pop(str(img_path), None)
                        st.rerun()
                with btn2:
                    if st.button("✅ Add to FAISS", key=f"add_{key_base}"):
                        label = st.session_state.label_buffer.get(str(img_path), "")
                        if label and img_path.exists():
                            img = load_image(img_path)
                            db.add_logos(
                                [img],
                                [label],
                                augmentations=AUGMENTATIONS if use_aug else None,
                                num_augments=5,
                            )
                            # Increment false negative count when adding from unknown folder
                            st.session_state.false_negatives_count += 1
                            # Update global metrics
                            st.session_state.global_metrics["false_negatives"] += 1
                            st.session_state.global_metrics["total_known"] += 1
                            st.success(f"Added `{label}` to FAISS")
                            # Remove the file after adding to FAISS
                            os.remove(img_path)
                            # Remove from selection and label buffer
                            st.session_state.label_buffer.pop(str(img_path), None)
                            st.session_state.unknown_bulk_selection.pop(str(img_path), None)
                            st.rerun()

    # --- Add a button to reset the false negative count ---
    if st.button("🔄 Reset False Negative Counter"):
        st.session_state.false_negatives_count = 0
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
            help="Must have columns: filename, brand. Optional: source, correction"
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
    else:
        df = pd.DataFrame(columns=["filename", "brand"])
        has_source_col = False
        has_correction_col = False

    if os.path.isdir(known_base_dir):
        all_images = sorted([
            f for f in os.listdir(known_base_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ])
        if csv_file is None:
            df = pd.DataFrame({"filename": all_images, "brand": ["" for _ in all_images]})
            has_correction_col = False
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
    ss.setdefault("corrections_map", {})  # NEW: filepath -> correction type

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

    # >>>>>>> NEW: Seed corrections_map from CSV if "correction" exists
    if has_correction_col:
        for _, row in df.iterrows():
            c = _normalize_correction(row.get("correction", ""))
            if c:
                ss.corrections_map[str(row["filepath"])] = c
    # <<<<<<<

    # >>>>>>> NEW: Keep global 'total_known' in sync with uploaded known set
    st.session_state.global_metrics["total_known"] = len(df)
    # <<<<<<<

    # >>>>>>> NEW: Recompute known-related metrics from corrections (on upload)
    def _reset_known_metrics_from_corrections():
        gm = st.session_state.global_metrics

        # reset only the known-related top-level fields
        gm["typo_corrections"] = 0
        gm["wrong_attributions"] = 0
        gm["false_positives"] = 0
        gm["confirmed_attributions"] = 0
        gm["total_processed"] = 0  # here interpreted as "items corrected/confirmed"

        # reset per-source fields (keep dicts; zero their fields)
        for s in ["top2k_brands", "internal_db", "audio"]:
            sm = gm.get(s, {})
            sm["processed"] = 0
            sm["errors"] = 0
            sm["false_positives"] = 0
            sm["false_negatives"] = sm.get("false_negatives", 0)  # preserve FN per spec
            sm["typo_corrections"] = 0
            sm["wrong_attributions"] = 0
            sm["confirmed_attributions"] = 0
            gm[s] = sm

        # accumulate from corrections_map
        for _, row in df.iterrows():
            path_str = str(row["filepath"])
            corr = ss.corrections_map.get(path_str, "")
            if not corr:
                continue

            # map source (if any)
            src_key = _source_key(row["source"]) if has_source_col else None

            # top-level
            if corr == "typo":
                gm["typo_corrections"] += 1
            elif corr == "wrong":
                gm["wrong_attributions"] += 1
            elif corr == "false_positive":
                gm["false_positives"] += 1
            elif corr == "confirm":
                gm["confirmed_attributions"] += 1

            gm["total_processed"] += 1

            # per-source
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

    # Only recompute if we actually loaded a CSV that carried corrections
    if has_correction_col and ss.get("recomputed_from_corrections_once") is not True:
        _reset_known_metrics_from_corrections()
        ss.recomputed_from_corrections_once = True
    # <<<<<<<

    use_aug = st.checkbox(
        "Use augmentations for image (5 variants)",
        value=False,
        key='use_aug_known'
    )

    st.markdown("---")
    st.subheader("🔄 Bulk Correction")

    unique_brands = sorted(df["brand"].unique())
    selected_brand = st.selectbox(
        "Select brand to filter",
        ["All brands"] + unique_brands,
        key="selected_brand_known"
    )
    if ss.get("last_selected_brand") != selected_brand:
        ss.bulk_selection = {}
        ss.last_selected_brand = selected_brand

    filtered_df = df if selected_brand == "All brands" else df[df["brand"] == selected_brand]

    if not filtered_df.empty:
        st.write(f"Found {len(filtered_df)} images")
        select_all = st.checkbox("Select all images", key="select_all_known")
        for _, row in filtered_df.iterrows():
            path_str = str(row["filepath"])
            if select_all:
                ss.bulk_selection[path_str] = True
            else:
                ss.bulk_selection.setdefault(path_str, False)

        cols = st.columns(4)
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            if i < 12:
                with cols[i % 4]:
                    st.image(str(row["filepath"]), caption=row["filename"], width=150)
                    path_str = str(row["filepath"])
                    
                    #Display correction if any
                    corr_val = ss.corrections_map.get(str(row["filepath"]), "")
                    corr_display = corr_val if str(corr_val) != 'nan' else "none"
                    # print(corr_val)
                    st.caption(f"Correction: {corr_display}")
                    
                    is_checked = ss.bulk_selection.get(path_str, False)
                    ss.bulk_selection[path_str] = st.checkbox("Select", value=is_checked, key=f"bulk_{row['filename']}")
                    
                    #Display brand name and source
                    current_brand = ss.known_brand_buffer.get(path_str, row["brand"])
                    label = f"Brand for {row['filename']}"
                    if has_source_col:
                        label += f" (Source: {row['source']})"

                    new_brand = st.text_input(label, value=current_brand, key=f"brand_{row['filename']}").strip()
                    if new_brand and _norm(new_brand) != _norm(current_brand):
                        ss.known_brand_buffer[path_str] = new_brand

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
                            # NOTE: 'Update' alone is not tagged as a correction type

                    with btn_col2:
                        # Mark as FALSE_LOGO instead of deleting
                        if st.button("🗑 Discard", key=f"discard_{row['filename']}"):
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
                                ss.correction_stats["false_positives"] += 1
                                # >>>>>>> NEW: record correction for single action
                                ss.corrections_map[path_str] = "false_positive"
                                # <<<<<<<
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
                                # >>>>>>> NEW: record correction for single action
                                ss.corrections_map[path_str] = "confirm"
                                # <<<<<<<
                                st.success("Attribution confirmed!")
                            st.rerun()

        # Bulk correction
        st.markdown("---")
        st.subheader("Apply Bulk Correction")
        correction_type = st.radio(
            "Correction type:",
            ["Typo in brand name","Wrong logo attribution","False positive"]
        )
        new_brand_name = st.text_input("Correct brand name (if applicable):", key="new_brand_name")
        if st.button("Apply Bulk Correction", key="apply_bulk_correction"):
            selected_paths = [p for p,v in ss.bulk_selection.items() if v]
            if not selected_paths:
                st.warning("Select at least one image")
            elif correction_type != "False positive" and not new_brand_name.strip():
                st.warning("Enter a valid brand name")
            else:
                images_to_update, brands_to_update = [], []
                n_fp=n_typo=n_wrong=0
                for path_str in selected_paths:
                    mask = df["filepath"].astype(str)==path_str
                    if not mask.any():
                        continue
                    row = df[mask].iloc[0]
                    source_val = row["source"] if has_source_col else None
                    if correction_type=="False positive":
                        # add as FALSE_LOGO
                        try:
                            img = Image.open(path_str).convert("RGB")
                            images_to_update.append(img)
                            brands_to_update.append("FALSE_LOGO")
                            df.loc[mask, "brand"] = "FALSE_LOGO"
                            ss.known_brand_buffer[path_str] = "FALSE_LOGO"
                            update_metrics("false_positive", 1, source_val)
                            ss.correction_stats["false_positives"] += 1
                            # >>>>>>> NEW: record correction for bulk action
                            ss.corrections_map[path_str] = "false_positive"
                            # <<<<<<<
                            n_fp+=1
                        except Exception as e:
                            st.error(f"Failed processing {path_str}: {e}")
                    else:
                        current_brand=row["brand"]
                        target_brand=new_brand_name.strip()
                        if _norm(current_brand)==_norm(target_brand):
                            continue
                        df.loc[mask,"brand"]=target_brand
                        ss.known_brand_buffer[path_str]=target_brand
                        ss.confirmed_attributions.pop(path_str, None)
                        img=Image.open(path_str).convert("RGB")
                        images_to_update.append(img)
                        brands_to_update.append(target_brand)
                        if correction_type=="Typo in brand name":
                            update_metrics("typo",1,source_val)
                            ss.correction_stats["typo_corrections"]+=1
                            # >>>>>>> NEW: record correction for bulk action
                            ss.corrections_map[path_str] = "typo"
                            # <<<<<<<
                            n_typo+=1
                        else:
                            update_metrics("wrong_attribution",1,source_val)
                            ss.correction_stats["wrong_logo_corrections"]+=1
                            # >>>>>>> NEW: record correction for bulk action
                            ss.corrections_map[path_str] = "wrong"
                            # <<<<<<<
                            n_wrong+=1
                if images_to_update:
                    db.add_logos(
                        images_to_update, brands_to_update,
                        augmentations=(AUGMENTATIONS if use_aug else None), num_augments=5
                    )
                st.success(f"Applied corrections: {n_typo} typo, {n_wrong} wrong, {n_fp} false positives")
                ss.bulk_selection={}
                st.rerun()
                
        st.write("")
        st.write("")
        
        if st.button("Confirm Bulk Attribution", key="confirm_bulk"):
            selected_paths=[p for p,v in ss.bulk_selection.items() if v]
            new_confirms=[p for p in selected_paths if not ss.confirmed_attributions.get(p)]
            for p in new_confirms:
                ss.confirmed_attributions[p]=True
                if has_source_col:
                    src=df.loc[df["filepath"].astype(str)==p,"source"].iloc[0]
                else:
                    src=None
                update_metrics("confirm",1,src)
                # >>>>>>> NEW: record correction for bulk confirm
                ss.corrections_map[p] = "confirm"
                # <<<<<<<
            st.success(f"Confirmed {len(new_confirms)} attributions")
            ss.bulk_selection={}
            st.rerun()
            
        # ✅ Add to FAISS for selected logos (no metrics update)
        if st.button("✅ Add bulk to Database", key="add_bulk_to_faiss"):
            selected_paths = [p for p, v in ss.bulk_selection.items() if v]
            if not selected_paths:
                st.warning("Select at least one image to add.")
            else:
                images_to_add, brands_to_add = [], []
                for path_str in selected_paths:
                    if Path(path_str).exists():
                        img = Image.open(path_str).convert("RGB")
                        brand_to_use = ss.known_brand_buffer.get(path_str, 
                                        df.loc[df["filepath"].astype(str)==path_str, "brand"].iloc[0])
                        images_to_add.append(img)
                        brands_to_add.append(brand_to_use)
                if images_to_add:
                    db.add_logos(
                        images_to_add,
                        brands_to_add,
                        augmentations=(AUGMENTATIONS if use_aug else None),
                        num_augments=5
                    )
                    st.success(f"Added {len(images_to_add)} logos to FAISS database.")
                ss.bulk_selection = {}
                st.rerun()


    # Export
    st.markdown("---")
    export_df=df.copy()
    # Overlay brand edits from buffer
    for path_str,brand in ss.known_brand_buffer.items():
        mask=export_df["filepath"].astype(str)==path_str
        if mask.any():
            export_df.loc[mask,"brand"]=brand

    # >>>>>>> NEW: ensure correction column present and filled from corrections_map
    if "correction" not in export_df.columns:
        export_df["correction"] = ""
    # normalize any existing values
    export_df["correction"] = export_df["correction"].apply(lambda x: _normalize_correction(x))
    # overlay session corrections
    for path_str, corr in ss.corrections_map.items():
        mask = export_df["filepath"].astype(str)==path_str
        if mask.any():
            export_df.loc[mask, "correction"] = corr
    # <<<<<<<

    export_df=export_df.drop(columns=["filepath"])
    csv_data=export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Updated CSV", csv_data, "known_logos_updated.csv", "text/csv")



# Call the function
with tabs[5]:
    review_known_logos_tab(db, AUGMENTATIONS=AUGMENTATIONS)
    
    
def global_metrics_tab():
    st.subheader("📊 Global Metrics Dashboard")
    if "global_metrics" not in st.session_state:
        st.session_state.global_metrics=initialize_metrics()
    metrics=st.session_state.global_metrics

    # >>>>>>> NEW: percentage helper and denominator (known + unknown)
    denom = max(0, int(metrics.get("total_known", 0) + metrics.get("total_unknown", 0)))
    def _fmt_with_pct(n: int) -> str:
        if denom <= 0:
            return f"{n} (0%)"
        pct = (n / denom) * 100.0
        return f"{n} ({pct:.0f}%)"
    # <<<<<<<

    col1,col2,col3,col4=st.columns(4)
    with col1:
        # Totals: show absolute only
        st.metric("Total Known Logos", metrics["total_known"])
        st.metric("Total Unknown Logos", metrics["total_unknown"])
    with col2:
        # Non-total: include percentages
        st.metric("False Positives", _fmt_with_pct(metrics["false_positives"]))
        st.metric("False Negatives", _fmt_with_pct(metrics["false_negatives"]))
    with col3:
        st.metric("Typo Corrections", _fmt_with_pct(metrics["typo_corrections"]))
        st.metric("Wrong Attributions", _fmt_with_pct(metrics["wrong_attributions"]))
    with col4:
        st.metric("Confirmed Attributions", _fmt_with_pct(metrics["confirmed_attributions"]))
        # Total Processed: absolute only
        st.metric("Total Processed", metrics["total_processed"])

    st.markdown("---")
    st.subheader("📈 Per-Source Metrics")
    sources=["top2k_brands","internal_db","audio"]
    names=["Top 2k Brands","Internal Database","Audio"]
    for s,n in zip(sources,names):
        st.markdown(f"**{n}**")
        sm=metrics.get(s,{})
        col1,col2,col3,col4=st.columns(4)
        # Per-source: all shown with percentages (denominator = known + unknown)
        with col1: st.metric("Processed", _fmt_with_pct(sm.get("processed",0)))
        with col2: st.metric("Errors", _fmt_with_pct(sm.get("errors",0)))
        with col3: st.metric("False Positives", _fmt_with_pct(sm.get("false_positives",0)))
        with col4: st.metric("Confirmed", _fmt_with_pct(sm.get("confirmed_attributions",0)))
        ec1,ec2=st.columns(2)
        with ec1: st.metric("Typo Corrections", _fmt_with_pct(sm.get("typo_corrections",0)))
        with ec2: st.metric("Wrong Attributions", _fmt_with_pct(sm.get("wrong_attributions",0)))
        st.markdown("---")
    if st.button("🔄 Reset All Metrics"):
        st.session_state.global_metrics=initialize_metrics()
        st.success("Metrics reset!")
        st.rerun()

# ------------------------------
# TAB 7: Global Metrics
# ------------------------------
with tabs[6]:
    global_metrics_tab()