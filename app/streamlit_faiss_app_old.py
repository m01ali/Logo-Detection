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

DB_BASE_PATH = Path("D:\\milestone 2\\faiss_database_with_italian_logos")
print(DB_BASE_PATH/'logo_index.faiss')

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
# Streamlit UI Configuration
# ------------------------------
st.title("🔍 Logo Search & FAISS Indexing System")
st.markdown("""
A professional interface for inserting and searching logos using a FAISS-powered logo database.
""")

# ------------------------------
# Tabs for Add / Search / Save
# ------------------------------
tabs = st.tabs(["➕ Add Logo", "🔎 Search Logo", "💾 Save Database", "📹 Video Object Cropping with Annotations", "Review Unknown Images", "Review Known Logos"])

# ------------------------------
# TAB 1: Add Logo
# ------------------------------
with tabs[0]:
    st.subheader("Add a Logo to the Database")
    uploaded_file = st.file_uploader("Upload Logo Image", type=["png", "jpg", "jpeg"])
    brand_name = st.text_input("Enter Brand Name")
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
                # 
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
                

# def review_unknown_logos_tab(db, AUGMENTATIONS=None):
#     # --- Configuration ---
#     UNKNOWN_BASE_DIR = "unknowns"
#     IMAGES_PER_PAGE = 6

#     # --- Utilities ---
#     def list_unknown_folders(base_dir):
#         return sorted([f for f in os.listdir(base_dir) if (Path(base_dir) / f).is_dir()])

#     def list_images(folder):
#         return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

#     def load_image(image_path):
#         return Image.open(image_path).convert("RGB")

#     st.subheader("🧠 Review and Label Unknown Logos")

#     folders = list_unknown_folders(UNKNOWN_BASE_DIR)
#     if not folders:
#         st.info("No unknown logo folders found.")
#         return

#     selected_folder = st.selectbox("Select a folder to review", folders)
#     folder_path = Path(UNKNOWN_BASE_DIR) / selected_folder
#     image_files = list_images(folder_path)

#     if not image_files:
#         st.info("No images in the selected folder.")
#         return
    
#     def page_number_changed():
#         st.session_state['unknowns_page']=st.session_state['unknown_tab_page_no']
        
#     # --- Pagination state ---
#     total_pages = (len(image_files) - 1) // IMAGES_PER_PAGE + 1
#     if "unknowns_page" not in st.session_state:
#         st.session_state.unknowns_page = 1

#     col1, col2, col3 = st.columns([1, 3, 1])
#     with col1:
#         if st.button("⬅️ Previous") and st.session_state.unknowns_page > 1:
#             st.session_state.unknowns_page -= 1
#     with col2:
#         page_no = st.number_input("Jump to page:", 
#                                   min_value=1, 
#                                   max_value=total_pages, 
#                                   value=1,
#                                   key='unknown_tab_page_no',
#                                   on_change=page_number_changed,                                
#                                   step=1)
#         # st.session_state.unknowns_page = page_no
#     with col3:
#         if st.button("Next ➡️") and st.session_state.unknowns_page < total_pages:
#             st.session_state.unknowns_page += 1
#             print(st.session_state.unknowns_page)
    
#     # --- Get current page items ---
#     start = (st.session_state.unknowns_page - 1) * IMAGES_PER_PAGE
#     end = start + IMAGES_PER_PAGE
#     current_images = image_files[start:end]

#     st.markdown(f"### Page {st.session_state.unknowns_page} of {total_pages}")

#     # --- Caching label entries ---
#     if "label_buffer" not in st.session_state:
#         st.session_state.label_buffer = {}
        
#     use_aug = st.checkbox("Use augmentations for image (10 variants)", value=False)

#     # --- Display images in columns ---
#     cols = st.columns(3)
#     for idx, filename in enumerate(current_images):
#         col = cols[idx % 3]
#         image_path = folder_path / filename
#         with col:
#             st.image(str(image_path), caption=filename)
#             brand_input = st.text_input(f"Label for {filename}", key=f"label_{filename}")

#             if brand_input.strip():
#                 st.session_state.label_buffer[str(image_path)] = brand_input.strip()

#             col1, col2 = st.columns([1, 1])
#             with col1:
#                 if st.button("🗑 Discard", key=f"discard_{filename}"):
#                     os.remove(image_path)
#                     st.session_state.label_buffer.pop(str(image_path), None)
#                     st.rerun()

#             with col2:
#                 if st.button("✅ Add to FAISS", key=f"add_{filename}"):
#                     img = load_image(image_path)
#                     brand = brand_input.strip()
#                     if brand:
#                         db.add_logos([img], [brand], augmentations=AUGMENTATIONS if use_aug else None, num_augments=10)
#                         # os.remove(image_path)
#                         # st.session_state.label_buffer.pop(str(image_path), None)
#                         # st.success(f"Added {brand} to FAISS and removed {filename}")
#                         st.success(f"Added image of {brand} to FAISS")
#                         # st.rerun()

# def review_unknown_logos_tab(db, AUGMENTATIONS=None):
#     IMAGES_PER_PAGE = 6

#     def list_images(folder: Path):
#         return sorted([f.name for f in folder.iterdir()
#                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

#     def load_image(p: Path):
#         return Image.open(p).convert("RGB")

#     st.subheader("🧠 Review and Label Unknown Logos")

#     # 1) Let the user specify *any* folder path:
#     folder_input = st.text_input(
#         "Enter path to folder containing UNKNOWN logos",
#         placeholder="e.g. /home/user/unknown_logos"
#     )
#     if not folder_input:
#         st.info("Please enter a folder path to begin.")
#         return

#     folder_path = Path(folder_input)
#     if not folder_path.is_dir():
#         st.error(f"❌ `{folder_input}` is not a valid folder.")
#         return

#     # 2) List image files in that folder
#     image_files = list_images(folder_path)
#     if not image_files:
#         st.info("No .jpg/.jpeg/.png files found in that folder.")
#         return

#     # --- Pagination state ---
#     total_pages = (len(image_files) - 1) // IMAGES_PER_PAGE + 1
#     if "unknowns_page" not in st.session_state:
#         st.session_state.unknowns_page = 1

#     def _on_page_change():
#         st.session_state.unknowns_page = st.session_state.unknown_tab_page_no

#     col1, col2, col3 = st.columns([1, 3, 1])
#     with col1:
#         if st.button("⬅️ Previous") and st.session_state.unknowns_page > 1:
#             st.session_state.unknowns_page -= 1
#     with col2:
#         st.number_input(
#             "Jump to page:",
#             min_value=1,
#             max_value=total_pages,
#             value=st.session_state.unknowns_page,
#             key="unknown_tab_page_no",
#             on_change=_on_page_change
#         )
#     with col3:
#         if st.button("Next ➡️") and st.session_state.unknowns_page < total_pages:
#             st.session_state.unknowns_page += 1

#     start = (st.session_state.unknowns_page - 1) * IMAGES_PER_PAGE
#     end   = start + IMAGES_PER_PAGE
#     current_images = image_files[start:end]

#     st.markdown(f"### Page {st.session_state.unknowns_page} / {total_pages}")

#     if "label_buffer" not in st.session_state:
#         st.session_state.label_buffer = {}

#     use_aug = st.checkbox("Use augmentations for image (5 variants)", value=False)

#     # 3) Display the images in a 3-column grid
#     cols = st.columns(3)
#     for idx, filename in enumerate(current_images):
#         col = cols[idx % 3]
#         img_path = folder_path / filename
#         key_base = re.sub(r'\W+', '_', str(img_path))  # safe key

#         with col:
#             st.image(str(img_path), caption=filename)
#             brand_input = st.text_input(
#                 f"Label for {filename}",
#                 key=f"label_{key_base}"
#             )
#             # Store in session_state
#             if brand_input.strip():
#                 st.session_state.label_buffer[str(img_path)] = brand_input.strip()

#             c1, c2 = st.columns(2)
#             with c1:
#                 if st.button("🗑 Discard", key=f"discard_{key_base}"):
#                     os.remove(img_path)
#                     st.session_state.label_buffer.pop(str(img_path), None)
#                     st.experimental_rerun()
#             with c2:
#                 if st.button("✅ Add to FAISS", key=f"add_{key_base}"):
#                     img = load_image(img_path)
#                     brand = st.session_state.label_buffer.get(str(img_path), "").strip()
#                     if brand:
#                         db.add_logos(
#                             [img],
#                             [brand],
#                             augmentations=AUGMENTATIONS if use_aug else None,
#                             num_augments=5,
#                         )
#                         st.success(f"Added `{brand}` to FAISS")

    # # --- Add all labeled entries ---
    # if st.session_state.label_buffer:
    #     st.markdown("---")
    #     st.markdown("### ✅ Bulk Add All Labeled Logos")
    #     use_aug = st.checkbox("Use augmentations for all", value=False)

    #     if st.button("📦 Add All to FAISS"):
    #         added = 0
    #         for img_path, brand in st.session_state.label_buffer.items():
    #             if not brand or brand == "":
    #                 continue
                
    #             try:
    #                 img = load_image(img_path)
    #                 db.add_logos([img], [brand], augmentations=AUGMENTATIONS if use_aug else None, num_augments=10 if use_aug else 0)
    #                 os.remove(img_path)
    #                 added += 1
    #             except Exception as e:
    #                 st.error(f"Failed to add {img_path}: {str(e)}")
    #         st.success(f"Successfully added {added} logos to FAISS.")
    #         st.session_state.label_buffer.clear()
    #         st.experimental_rerun()
    # else:
    #     st.info("No labeled logos in current view.")

def review_unknown_logos_tab(db, AUGMENTATIONS=None):
    IMAGES_PER_ROW = 3

    def list_images(folder: Path):
        return sorted([f.name for f in folder.iterdir()
                       if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    def load_image(p: Path):
        return Image.open(p).convert("RGB")

    st.subheader("🧠 Review and Label Unknown Logos")

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

    # --- Buffer for labels ---
    if "label_buffer" not in st.session_state:
        st.session_state.label_buffer = {}

    use_aug = st.checkbox("Use augmentations for image (5 variants)", value=False)

    # --- Render all images in rows of 3 ---
    for i in range(0, len(image_files), IMAGES_PER_ROW):
        row_files = image_files[i:i+IMAGES_PER_ROW]
        cols = st.columns(IMAGES_PER_ROW)
        for col, filename in zip(cols, row_files):
            img_path = folder_path / filename
            # sanitize key
            key_base = re.sub(r'\W+', '_', str(img_path))

            with col:
                st.image(str(img_path), caption=filename)
                brand_input = st.text_input(
                    f"Label for {filename}",
                    key=f"label_{key_base}"
                ).strip()

                # save typed-in label
                if brand_input:
                    st.session_state.label_buffer[str(img_path)] = brand_input
                else:
                    st.session_state.label_buffer.pop(str(img_path), None)

                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("🗑 Discard", key=f"discard_{key_base}"):
                        os.remove(img_path)
                        st.session_state.label_buffer.pop(str(img_path), None)
                        st.experimental_rerun()
                with btn2:
                    if st.button("✅ Add to FAISS", key=f"add_{key_base}"):
                        label = st.session_state.label_buffer.get(str(img_path), "")
                        if label:
                            img = load_image(img_path)
                            db.add_logos(
                                [img],
                                [label],
                                augmentations=AUGMENTATIONS if use_aug else None,
                                num_augments=5,
                            )
                            st.success(f"Added `{label}` to FAISS")

with tabs[4]:
    review_unknown_logos_tab(db, AUGMENTATIONS=AUGMENTATIONS)

# def review_known_logos_tab(db, AUGMENTATIONS=None):
#     """
#     Streamlit tab for reviewing and correcting known logos.
#     Allows uploading a CSV of metadata and/or specifying a directory of logo images.
#     Enables brand name corrections, discarding unwanted images, updating FAISS, 
#     and downloading an updated CSV.
#     """
#     IMAGES_PER_PAGE = 6
#     st.subheader("🔍 Review and Correct Known Logos")

#     # --- Inputs: CSV uploader and directory selector ---
#     col1, col2 = st.columns(2)
#     with col1:
#         csv_file = st.file_uploader("Upload CSV with known logos", type="csv", help="Must have columns: filename, brand")
#     with col2:
#         known_base_dir = st.text_input("Known logos directory", value="knowns", help="Folder containing the logo image files")

#     # --- Validate inputs ---
#     if csv_file is None and not os.path.isdir(known_base_dir):
#         st.info("Please upload a CSV or specify a valid directory to begin.")
#         return

#     # --- Load metadata DataFrame ---
#     if csv_file is not None:
#         try:
#             df = pd.read_csv(csv_file)
#         except Exception:
#             st.error("Failed to read CSV. Please ensure it's formatted correctly.")
#             return
#         required_cols = {"filename", "brand"}
#         if not required_cols.issubset(df.columns):
#             st.error(f"CSV must contain columns: {', '.join(required_cols)}.")
#             return
#     else:
#         # no CSV: start with empty DataFrame
#         df = pd.DataFrame(columns=["filename", "brand"])
        
#     # print(df)

#     # --- Gather image files from directory ---
#     if os.path.isdir(known_base_dir):
#         all_images = sorted([
#             f for f in os.listdir(known_base_dir)
#             if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#         ])
#         if csv_file is None:
#             # initialize metadata if no CSV provided
#             df = pd.DataFrame({
#                 "filename": all_images,
#                 "brand": ["" for _ in all_images],
#             })
#         else:
#             image_suffixes = ('.jpg', '.png', '.jpeg')
#         #     # filter to files actually present
#         #     df = df[df["filename"].isin(all_images)].reset_index(drop=True)
#             df = df[df['filename'].str.lower().str.endswith(image_suffixes)]

#     else:
#         st.error(f"Directory not found: {known_base_dir}")
#         return

#     # --- Build full file paths ---
#     df["filepath"] = df["filename"].apply(lambda fn: Path(known_base_dir) / Path(fn).name)

#     if df.empty:
#         st.info("No known logos found in the specified directory/CSV.")
#         return

#     # --- Pagination setup ---
#     total_pages = (len(df) - 1) // IMAGES_PER_PAGE + 1
#     if "known_page" not in st.session_state:
#         st.session_state.known_page = 1

#     def _on_page_change():
#         st.session_state.known_page = st.session_state.known_tab_page_no

#     col_prev, col_page, col_next = st.columns([1, 3, 1])
#     with col_prev:
#         if st.button("⬅️ Previous", key='known_previous_btn') and st.session_state.known_page > 1:
#             st.session_state.known_page -= 1
#     with col_page:
#         st.number_input(
#             "Jump to page:", min_value=1, max_value=total_pages,
#             value=st.session_state.known_page, key="known_tab_page_no",
#             on_change=_on_page_change, step=1
#         )
#     with col_next:
#         if st.button("Next ➡️", key='known_next_btn') and st.session_state.known_page < total_pages:
#             st.session_state.known_page += 1

#     start_idx = (st.session_state.known_page - 1) * IMAGES_PER_PAGE
#     end_idx = start_idx + IMAGES_PER_PAGE
#     page_df = df.iloc[start_idx:end_idx]

#     st.markdown(f"### Page {st.session_state.known_page} of {total_pages}")

#     # --- Buffer for edited brands ---
#     if "known_brand_buffer" not in st.session_state:
#         st.session_state.known_brand_buffer = {}

#     use_aug = st.checkbox("Use augmentations for image (5 variants)", value=False, key='use_aug_known')

#     # --- Display images and controls ---
#     cols = st.columns(3)
#     for i, (_, row) in enumerate(page_df.iterrows()):
#         col = cols[i % 3]
#         with col:
#             st.image(str(row["filepath"]), caption=row["filename"])
#             brand_key = f"known_brand_{row['filename']}"
#             new_brand = st.text_input(
#                 f"Brand for {row['filename']}",
#                 value=row["brand"],
#                 key=brand_key
#             ).strip()

#             # track changes
#             if new_brand and new_brand != row["brand"]:
#                 st.session_state.known_brand_buffer[str(row["filepath"])] = new_brand
#             else:
#                 st.session_state.known_brand_buffer.pop(str(row["filepath"]), None)

#             # st.caption(f"Timestamp: {row['timestamp']} | Area: {row['percent_area']}")

#             btn_col1, btn_col2 = st.columns(2)
#             with btn_col1:
#                 if st.button("🗑 Discard", key=f"discard_known_{row['filename']}"):
#                     try:
#                         os.remove(row["filepath"])
#                     except OSError:
#                         st.error("Failed to delete file.")
#                     st.session_state.known_brand_buffer.pop(str(row["filepath"]), None)
#                     st.rerun()
#             with btn_col2:
#                 if st.button("✅ Update FAISS", key=f"update_known_{row['filename']}"):
#                     img = Image.open(row["filepath"]).convert("RGB")
#                     brand_to_use = st.session_state.known_brand_buffer.get(
#                         str(row["filepath"]), row["brand"]
#                     )
#                     db.add_logos(
#                         [img], [brand_to_use],
#                         augmentations=(AUGMENTATIONS if use_aug else None),
#                         num_augments=5
#                     )
#                     st.success(f"Updated FAISS entry for {brand_to_use}")

#     # --- Download updated CSV if any edits made ---
#     if st.session_state.known_brand_buffer:
#         if st.button("💾 Download Updated CSV"):
#             # apply buffered changes
#             for path_str, corrected in st.session_state.known_brand_buffer.items():
#                 mask = df["filepath"].astype(str) == path_str
#                 df.loc[mask, "brand"] = corrected
#             out_df = df.drop(columns=["filepath"])
#             csv_bytes = out_df.to_csv(index=False).encode("utf-8")
#             st.download_button(
#                 "Download corrected CSV",
#                 data=csv_bytes,
#                 file_name="known_logos_corrected.csv",
#                 mime="text/csv"
#             )

def review_known_logos_tab(db, AUGMENTATIONS=None): 
    """
    Streamlit tab for reviewing and correcting known logos.
    Allows uploading a CSV of metadata and/or specifying a directory of logo images.
    Enables brand name corrections, discarding unwanted images, updating FAISS, 
    and downloading an updated CSV.
    """
    IMAGES_PER_ROW = 3

    st.subheader("🔍 Review and Correct Known Logos")

    # --- Inputs: CSV uploader and directory selector ---
    col1, col2 = st.columns(2)
    with col1:
        csv_file = st.file_uploader(
            "Upload CSV with known logos", 
            type="csv", 
            help="Must have columns: filename, brand"
        )
    with col2:
        known_base_dir = st.text_input(
            "Known logos directory", 
            value="knowns", 
            help="Folder containing the logo image files"
        )

    # --- Validate inputs ---
    if csv_file is None and not os.path.isdir(known_base_dir):
        st.info("Please upload a CSV or specify a valid directory to begin.")
        return

    # --- Load metadata DataFrame ---
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            st.error("Failed to read CSV. Please ensure it's formatted correctly.")
            return
        if not {"filename", "brand"}.issubset(df.columns):
            st.error("CSV must contain columns: filename, brand.")
            return
    else:
        df = pd.DataFrame(columns=["filename", "brand"])

    # --- Gather image files from directory ---
    if os.path.isdir(known_base_dir):
        all_images = sorted([
            f for f in os.listdir(known_base_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if csv_file is None:
            df = pd.DataFrame({
                "filename": all_images,
                "brand":    ["" for _ in all_images],
            })
        else:
            # keep only rows pointing to existing image files
            df = df[df["filename"].isin(all_images)].reset_index(drop=True)
    else:
        st.error(f"Directory not found: {known_base_dir}")
        return

    if df.empty:
        st.info("No known logos found in the specified directory/CSV.")
        return

    # --- Build full file paths ---
    df["filepath"] = df["filename"].apply(lambda fn: Path(known_base_dir) / fn)

    # --- Buffer for edited brands ---
    if "known_brand_buffer" not in st.session_state:
        st.session_state.known_brand_buffer = {}

    use_aug = st.checkbox("Use augmentations for image (5 variants)", value=False, key='use_aug_known')

    # --- Display all images in a 3‑column grid (scrolling page) ---
    for i in range(0, len(df), IMAGES_PER_ROW):
        row = df.iloc[i:i+IMAGES_PER_ROW]
        cols = st.columns(IMAGES_PER_ROW)
        for col, (_, data) in zip(cols, row.iterrows()):
            with col:
                st.image(str(data["filepath"]), caption=data["filename"])
                key = f"known_brand_{data['filename']}"
                new_brand = st.text_input(
                    f"Brand for {data['filename']}",
                    value=data["brand"],
                    key=key
                ).strip()

                # --- Show source if present ---
                if "source" in df.columns:
                    st.text(f"Source: {data['source']}")

                # track changes
                path_str = str(data["filepath"])
                if new_brand and new_brand != data["brand"]:
                    st.session_state.known_brand_buffer[path_str] = new_brand
                else:
                    st.session_state.known_brand_buffer.pop(path_str, None)

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("🗑 Discard", key=f"discard_known_{data['filename']}"):
                        try:
                            os.remove(data["filepath"])
                        except OSError:
                            st.error("Failed to delete file.")
                        st.session_state.known_brand_buffer.pop(path_str, None)
                        st.experimental_rerun()
                with btn_col2:
                    if st.button("✅ Update FAISS", key=f"update_known_{data['filename']}"):
                        img = Image.open(data["filepath"]).convert("RGB")
                        brand_to_use = st.session_state.known_brand_buffer.get(path_str, data["brand"])
                        db.add_logos(
                            [img], [brand_to_use],
                            augmentations=(AUGMENTATIONS if use_aug else None),
                            num_augments=5
                        )
                        st.success(f"Updated FAISS entry for {brand_to_use}")

    # --- Download updated CSV if any edits made ---
    if st.session_state.known_brand_buffer:
        if st.button("💾 Download Updated CSV"):
            # apply buffered changes
            for path_str, corrected in st.session_state.known_brand_buffer.items():
                mask = df["filepath"].astype(str) == path_str
                df.loc[mask, "brand"] = corrected
            # out_df = df[["filename", "brand"]]
            out_df = df[["filename", "brand"] + (["source"] if "source" in df.columns else [])]
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download corrected CSV",
                data=csv_bytes,
                file_name="known_logos_corrected.csv",
                mime="text/csv"
            )

            
with tabs[5]:
    review_known_logos_tab(db, AUGMENTATIONS=AUGMENTATIONS)
