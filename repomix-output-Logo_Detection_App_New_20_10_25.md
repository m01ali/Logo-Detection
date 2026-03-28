This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where comments have been removed, empty lines have been removed, security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: __pycache__, **/__pycache__, **/__pycache__/**, app/streamlit_faiss_app_old.py, app/streamlit_faiss_app_18_10_25.py, app/streamlit_faiss_app_19_10_25.py, assets/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Code comments have been removed from supported file types
- Empty lines have been removed from all files
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
app/
  app_image.py
  logo_app.py
  streamlit_faiss_app.py
filtering/
  clip_filtering.py
  heuristic_filtering.py
  llm_filtering.py
  post_filtering.py
models/
  audio_model.py
  description_model.py
  detector_model.py
  faiss_db.py
  verification_model.py
pipelines/
  brand_recognition_chain.py
  pipeline.py
utils/
  fuzzy_matching.py
  prompts.py
  shot_detection.py
  video_annotations_to_crops.py
.gitignore
gradio_app.py
LICENSE
README.md
```

# Files

## File: app/app_image.py
```python
import gradio as gr
from models.detector_model import LogoDetector
from models.description_model import FlorenceModel
from models.verification_model import LLaVAModel, PaligemmaModel
import os
model = LogoDetector()
verify_model = None
def process_image(hf_key, image, thresh, nms_thresh):
  if hf_key:
      os.environ["HF_TOKEN"] = hf_key
  else:
      return "Please set a valid Huggingface API Key"
  global verify_model
  if not verify_model:
      verify_model = PaligemmaModel()
  annotated_image, potential_logos = model.process_image(image, thresh, nms_thresh)
  logos = [logo for logo in potential_logos if verify_model.verify_image(logo)]
  descriptions = [verify_model.run_example(logo, "What is the full brand name of this logo?") for logo in logos]
  return annotated_image, list(zip(logos, descriptions))
DESCRIPTION = "# Automated Logo Detection - Phase 1"
css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Logo Detection"):
        hf_key = gr.Textbox(label="Huggingface_API_KEY", placeholder="Huggingface API KEY", type="password")
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Frame", type="pil")
                submit_btn = gr.Button(value="Detect Logos")
                input_threshold = gr.Slider(
                  label="Threshold",
                  info="Larger value will detect fewer logos and vice versa.",
                  minimum=0.01,
                  maximum=0.2,
                  value=0.1,
                  step=0.01)
                input_nms_threshold = gr.Slider(
                  label="NMS_Threshold",
                  info="Larger value will detect fewer overlapping logos and vice versa.",
                  minimum=0.1,
                  maximum=0.9,
                  value=0.3,
                  step=0.1)
            with gr.Column():
                output_img = gr.Image(label="Output Image")
                output_logos = gr.Gallery(columns=1, label="Cropped Logos", preview=True, show_label=True)
        with gr.Accordion("Instructions"):
          gr.Markdown(
)
        submit_btn.click(process_image, [hf_key, input_img, input_threshold, input_nms_threshold], [output_img, output_logos])
demo.launch(debug=False)
```

## File: app/logo_app.py
```python
import gradio as gr
from models.description_model import FlorenceModel, GemmaModel, LLaMAModel
from models.verification_model import LLaVAModel, PaligemmaModel
import os
desc_model = FlorenceModel()
gemma_model = None
verify_model_A = LLaVAModel()
paligemma_model = None
llama_model = None
def process_image(hf_key, image, task, llava_prompt, pg_prompt):
  if hf_key:
      os.environ["HF_TOKEN"] = hf_key
  else:
      return "Please set a valid Huggingface API Key"
  try:
    global paligemma_model, gemma_model, llama_model
    if not paligemma_model:
        paligemma_model = PaligemmaModel()
    if not gemma_model:
        gemma_model = GemmaModel()
    if not llama_model:
        llama_model = LLaMAModel()
    if task=="Verify":
        result_A = verify_model_A.run_example(image, llava_prompt)
        result_B = paligemma_model.verify_image(image, pg_prompt)
        result =  f"LLaVAModel's Output: {result_A}\nPaligemmaModel's Output: {result_B}"
    else:
        desc = desc_model.generate_description(image)
        result_A = gemma_model.run_example(desc)
        result_B = paligemma_model.run_example(image, prompt="What is the full brand name of this logo?")
        result_C = llama_model.get_logo_name(image)
        result = f"FlorenceGemma's Output: {result_A}\nPaligemma's Output: {result_B}\nLLaMA 3.2's Output: {result_C}"
  except Exception as e:
      result = str(e)
  return result
DESCRIPTION = "# Automated Logo Detection - Phase 2"
css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""
VERIFY_PG_PROMPT = "Does this image represent a plain icon, plain text, branded logo, or something else?"
VERIFY_LLAVA_PROMPT = "Is this most likely an image of a single logo only? Just answer Yes or No."
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Logo Verification and Brand Recognition"):
        hf_key = gr.Textbox(label="Huggingface_API_KEY", placeholder="Huggingface API KEY", type="password")
        input_img = gr.Image(label="Input Frame", type="pil")
        task = gr.Radio(["Verify", "Describe"], label="Select Task")
        llava_prompt = gr.Textbox(value=VERIFY_LLAVA_PROMPT, label="Prompt for the LLaVA verification model only.")
        pg_prompt = gr.Textbox(value=VERIFY_PG_PROMPT, label="Prompt for the Paligemma verification model only.")
        submit_btn = gr.Button(value="Submit")
        output_text = gr.Textbox(value="", label="Model Output")
    with gr.Accordion("Instructions"):
        gr.Markdown(
)
    submit_btn.click(process_image, [hf_key, input_img, task, llava_prompt, pg_prompt], [output_text])
demo.launch(debug=True)
```

## File: app/streamlit_faiss_app.py
```python
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.faiss_db import LogoDatabaseNew
from utils.video_annotations_to_crops import process_video_annotations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
AUGMENTATIONS = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ColorJitter(hue=0.05, saturation=0.1, p=0.4),
    A.Resize(224, 224)
])
DB_BASE_PATH = Path("D:\\milestone 2\\faiss_database_with_italian_logos")
print(DB_BASE_PATH/'logo_index.faiss')
known_processed = 1216
st.set_page_config(page_title="Logo Search & Indexing", layout="wide")
@st.cache_resource
def load_database():
    return LogoDatabaseNew(
        index_path=str(DB_BASE_PATH/"logo_index.faiss"),
        metadata_path=str(DB_BASE_PATH/"metadata.json"),
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
db = load_database()
def _inc(dct, key, n=1):
    dct[key] = dct.get(key, 0) + n
def _norm(s):
    return (s or "").strip().lower()
def _source_key(source_val: str):
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
    gm = st.session_state.global_metrics
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
        "missed_logos_manual": 0,
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
st.title("🔍 Logo Search & FAISS Indexing System")
st.markdown("""
A professional interface for inserting and searching logos using a FAISS-powered logo database.
""")
tabs = st.tabs(["➕ Add Logo", "🔎 Search Logo", "💾 Save Database", "📹 Video Object Cropping with Annotations",
                "Review Unknown Images", "Review Known Logos", "📊 Global Metrics"])
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
            st.session_state.global_metrics["total_known"] += 1
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
with tabs[2]:
    st.subheader("Save Database")
    if st.button("Save FAISS Index & Metadata"):
        db.save()
        st.success("Database saved successfully to disk (logo_index.faiss & metadata.json).")
with tabs[3]:
    st.subheader("Extract Logos from Video with Annotations")
    video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
    json_file = st.file_uploader("Upload Label Studio JSON annotations", type=["json"])
    if "frame_results" not in st.session_state:
        st.session_state.frame_results = []
    use_aug = st.checkbox("Use augmentations for all logos (10 variants)", value=False)
    if st.button("📦 Add to FAISS Database"):
        if st.session_state.frame_results:
            try:
                logos = [item['crop'] for item in st.session_state.frame_results.values()]
                labels = [item['label'].strip() for item in st.session_state.frame_results.values()]
                db.add_logos(images=logos, brand_names=labels,
                            augmentations=AUGMENTATIONS if use_aug else None,
                            num_augments=5)
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
def review_unknown_logos_tab(db, AUGMENTATIONS=None):
    st.subheader("🧠 Review & Label Unknown Logos")
    ss = st.session_state
    if "global_metrics" not in ss:
        ss.global_metrics = initialize_metrics()
    ss.setdefault("unknown_reviewed_set", set())
    reviewed_set = ss.unknown_reviewed_set
    ss.setdefault("unknown_confirmations", [])
    ss.setdefault("unknown_page_size_ui", 24)
    ss.setdefault("unknown_page_num", 1)
    ss.setdefault("unknown_last_path", None)
    ss.setdefault("unknown_last_status_filter", None)
    ss.setdefault("unknown_select_all_prev", False)
    ss.setdefault("unknown_select_all_key", "")
    ss.setdefault("unknown_bulk_label_ui", "")
    ss.setdefault("unknown_use_aug", False)
    ss.setdefault("unknown_pending_unselect", set())
    ss.setdefault("unknown_pending_key_pops", set())
    UNKNOWN_DIR = st.text_input("Unknowns folder path (images only)", value="unknowns")
    if not os.path.isdir(UNKNOWN_DIR):
        st.info("Unknowns folder not found at the provided path.")
        return
    _FRAME_LOGO_RE = re.compile(r"^frame(\d+)_logo(\d+)", re.IGNORECASE)
    def _list_images_sorted(dir_path: str):
        files = [f for f in os.listdir(dir_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        parsed = []
        unmatched = []
        for f in files:
            stem = Path(f).stem
            m = _FRAME_LOGO_RE.match(stem)
            if m:
                frame_no = int(m.group(1))
                logo_no = int(m.group(2))
                parsed.append((frame_no, logo_no, f))
            else:
                unmatched.append(f)
        parsed.sort(key=lambda t: (t[0], t[1]))
        unmatched.sort(key=str.lower)
        return [f for (_, _, f) in parsed] + unmatched
    status_filter = st.selectbox(
        "Show",
        ["All", "Reviewed", "Not reviewed"],
        key="unknown_status_filter",
        help="Filter images by review status."
    )
    if ss.unknown_last_path != UNKNOWN_DIR or ss.unknown_last_status_filter != status_filter:
        ss.unknown_page_num = 1
        ss.unknown_last_path = UNKNOWN_DIR
        ss.unknown_last_status_filter = status_filter
    st.number_input(
        "Images per page",
        min_value=6, max_value=200, step=6,
        key="unknown_page_size_ui",
        help="Adjust page size to balance speed and visibility."
    )
    image_files = _list_images_sorted(UNKNOWN_DIR)
    if not image_files:
        st.info("No images found in the unknowns folder.")
        return
    all_paths = [str(Path(UNKNOWN_DIR) / fn) for fn in image_files]
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
    page_size = int(ss.unknown_page_size_ui)
    total_pages = max(1, math.ceil(total_images / page_size))
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
    nav_cols = st.columns([1, 2, 2, 2, 1])
    with nav_cols[0]:
        if st.button("⏮ First"):
            ss.unknown_page_num = 1
            st.rerun()
    with nav_cols[1]:
        if st.button("◀ Prev"):
            ss.unknown_page_num = max(1, ss.unknown_page_num - 1)
            st.rerun()
    with nav_cols[2]:
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
    reviewed_count = len([p for p in all_paths if p in reviewed_set])
    not_reviewed_count = len(all_paths) - reviewed_count
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Reviewed", reviewed_count)
    with m2: st.metric("Not Reviewed", not_reviewed_count)
    with m3: st.metric("False Negatives (global)", int(ss.global_metrics.get("false_negatives", 0)))
    st.markdown("---")
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
    for fn in page_files:
        sel_key = f"u_sel_{fn}"
        if sel_key not in ss:
            ss[sel_key] = False
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
    selected_count = sum(1 for fn in page_files if ss.get(f"u_sel_{fn}", False))
    st.caption(f"**Selected on page:** {selected_count}")
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
    bulk_ok = st.button("✅ Bulk label, add to FAISS & mark reviewed (current page)")
    st.caption("For each selected item: use its per-item label if set; otherwise apply the bulk label. Items without any label are skipped.")
    st.markdown("---")
    GRID_COLS = 4
    cols = st.columns(GRID_COLS)
    for idx, p in enumerate(page_paths):
        fn = page_files[idx]
        image_path = Path(p)
        with cols[idx % GRID_COLS]:
            st.image(str(image_path), caption=fn, use_container_width=True)
            rev_checked = st.checkbox(f"Reviewed | {fn}", value=(p in reviewed_set))
            if rev_checked: reviewed_set.add(p)
            else: reviewed_set.discard(p)
            sel_key = f"u_sel_{fn}"
            st.checkbox("Select", key=sel_key)
            lbl_key = f"lbl_{fn}"
            if lbl_key not in ss:
                ss[lbl_key] = ""
            st.text_input(f"Label for {fn}", key=lbl_key)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑 Discard", key=f"discard_{fn}"):
                    try:
                        os.remove(image_path)
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
                    ss[lbl_key] = label
                    processed += 1
                except Exception:
                    skipped += 1
                    continue
            if imgs:
                if ss.unknown_use_aug and AUGMENTATIONS is not None:
                    db.add_logos(imgs, labels, augmentations=AUGMENTATIONS, num_augments=10)
                else:
                    db.add_logos(imgs, labels, augmentations=None, num_augments=0)
            now = pd.Timestamp.utcnow().isoformat()
            for fn in selected_on_page:
                p = str(Path(UNKNOWN_DIR) / fn)
                final_label = ss.get(f"lbl_{fn}", "").strip() or bulk_label
                if not final_label:
                    continue
                reviewed_set.add(p)
                ss.unknown_confirmations.append({
                    "filename": fn,
                    "filepath": p,
                    "brand": final_label,
                    "timestamp": now
                })
                ss.global_metrics["false_negatives"] = int(ss.global_metrics.get("false_negatives", 0)) + 1
                ss.global_metrics["total_processed"] = int(ss.global_metrics.get("total_processed", 0)) + 1
                ss.unknown_pending_unselect.add(fn)
            st.success(f"Bulk processed {processed} item(s) and added to FAISS. Skipped {skipped} without labels.")
            st.rerun()
with tabs[4]:
    review_unknown_logos_tab(db, AUGMENTATIONS=AUGMENTATIONS)
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
    else:
        df = pd.DataFrame(columns=["filename", "brand"])
        has_source_col = False
        has_correction_col = False
        has_postfilter_col = False
    if os.path.isdir(known_base_dir):
        all_images = sorted([
            f for f in os.listdir(known_base_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ])
        if csv_file is None:
            df = pd.DataFrame({"filename": all_images, "brand": ["" for _ in all_images]})
            has_correction_col = False
            has_postfilter_col = False
        else:
            df = df[df["filename"].isin(all_images)].reset_index(drop=True)
    else:
        st.error(f"Directory not found: {known_base_dir}")
        return
    if df.empty:
        st.info("No known logos found in the specified directory/CSV.")
        return
    df["filepath"] = df["filename"].apply(lambda fn: Path(known_base_dir) / fn)
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
    ss.setdefault("corrections_map", {})
    ss.setdefault("last_selected_brand", None)
    ss.setdefault("last_selected_postfilter", None)
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
    if has_correction_col:
        for _, row in df.iterrows():
            c = _normalize_correction(row.get("correction", ""))
            if c:
                ss.corrections_map[str(row["filepath"])] = c
    st.session_state.global_metrics["total_known"] = len(df)
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
    unique_brands = sorted(df["brand"].unique())
    selected_brand = st.selectbox(
        "Select brand to filter",
        ["All brands"] + unique_brands,
        key="selected_brand_known"
    )
    if has_postfilter_col:
        unique_verdicts = sorted([v for v in df["postfilter_verdict"].dropna().unique()])
        selected_postfilter = st.selectbox(
            "Filter by postfilter verdict",
            ["All verdicts"] + unique_verdicts,
            key="selected_postfilter_known",
            help="Only shown if your CSV contains 'postfilter_verdict'. Helps evaluate metrics with/without postfilter."
        )
    else:
        selected_postfilter = "All verdicts"
    if ss.get("last_selected_brand") != selected_brand or ss.get("last_selected_postfilter") != selected_postfilter:
        ss.bulk_selection = {}
        ss.select_all_known_prev = False
        ss.last_selected_brand = selected_brand
        ss.last_selected_postfilter = selected_postfilter
    filtered_df = df if selected_brand == "All brands" else df[df["brand"] == selected_brand]
    if has_postfilter_col and selected_postfilter != "All verdicts":
        filtered_df = filtered_df[filtered_df["postfilter_verdict"] == selected_postfilter]
    filtered_df = filtered_df.reset_index(drop=True)
    if has_postfilter_col:
        st.markdown("#### 📊 Postfilter Summary (current view)")
        vc = (
            filtered_df["postfilter_verdict"]
            .fillna("—")
            .value_counts(dropna=False)
            .to_dict()
        )
        preferred = ["Correct", "Incorrect", "Skipped", "Other", "—"]
        ordered = [k for k in preferred if k in vc] + [k for k in vc if k not in preferred]
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
    if not filtered_df.empty:
        st.markdown("**Apply Bulk Correction**")
        correction_type = st.radio(
            "Correction type:",
            ["Typo in brand name","Wrong logo attribution","False positive"],
            horizontal=True
        )
        new_brand_name = st.text_input("Correct brand name (if applicable):", key="new_brand_name")
        bulk_apply_clicked = st.button("🔁 Apply Bulk Correction", key="apply_bulk_correction")
        confirm_bulk_clicked = st.button("✅ Confirm Bulk Attribution (also adds to FAISS)", key="confirm_bulk")
        st.markdown("📤 **Export Updated CSV**")
        export_df = df.copy()
        for path_str, brand in st.session_state.known_brand_buffer.items():
            mask = export_df["filepath"].astype(str) == path_str
            if mask.any():
                export_df.loc[mask, "brand"] = brand
        if "correction" not in export_df.columns:
            export_df["correction"] = ""
        for path_str, corr in st.session_state.corrections_map.items():
            mask = export_df["filepath"].astype(str) == path_str
            if mask.any():
                export_df.loc[mask, "correction"] = corr
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
        select_all = st.checkbox("Select all images", key="select_all_known")
        for _, row in filtered_df.iterrows():
            path_str = str(row["filepath"])
            if select_all:
                st.session_state.bulk_selection[path_str] = True
            elif path_str not in st.session_state.bulk_selection:
                st.session_state.bulk_selection[path_str] = False
            elif not select_all and st.session_state.get("select_all_known_prev", True):
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
    st.session_state.global_metrics["missed_logos_manual"] = int(st.session_state.missed_logos_manual_ui)
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
        st.session_state.missed_logos_manual_ui = 0
        st.success("Metrics reset!")
        st.rerun()
with tabs[6]:
    global_metrics_tab()
```

## File: filtering/clip_filtering.py
```python
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForZeroShotImageClassification, BitsAndBytesConfig
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import numpy as np
import os
import pandas as pd
from scipy.stats import entropy
from math import log
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path
from io import BytesIO
import hashlib
import uuid
def compute_image_hash(logo_image):
    try:
        buffer = BytesIO()
        logo_image.convert("RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return hashlib.md5(image_bytes).hexdigest()
    except Exception as e:
        return str(uuid.uuid4())
positive_prompts = ["This is an image of a single branded logo",
                    "This image represents a clear and distinct company logo",
                    "This is the logo of a well-known brand",
                    "This is a standalone corporate logo without extra elements",
                    "This is a minimalistic logo design representing a single brand",
                    "This is an image of a single professional logo with no distractions",
                    "This is a high-quality logo of a famous company or product",
                    "This image represents a single graphic design of a company logo",
                    "This is an isolated image of a logo used for branding purposes",
                    "This is the official logo of a brand displayed clearly",
                    "This is a focused image of one distinct logo with no additional logos or designs",
                    "This image showcases a single brand's logo without overlapping or extra elements",
                    ]
negative_prompts = ["This is not an image of a single branded logo",
                    "This is not an image of a single corporate logo",
                    "This is a random graphic or abstract image, not a logo",
                    "This is a photo of something other than a logo",
                    "This image contains multiple objects and not a single logo",
                    "This is not a clean or distinct design that represents a logo",
                    "This image shows random text or icons, not a logo",
                    "This photo represents a general illustration, not a company's branding",
                    "This is a photo or artwork unrelated to logos or branding",
                    "This image contains clutter or multiple elements, not a standalone logo",
                    "This image contains multiple logos or brand symbols, not a single distinct logo",
                    "This is a cluttered image with overlapping logos or designs from different brands",
                    ]
positive_hard_prompts = ["This is an image of a well-known logo",
                        "This is a logo of a globally recognized brand",
                        "This logo contains both design and text that represent a well-known brand",
                        "This is the logo of a popular company or product",
                        "This is a professional logo used by a globally established organization",
                        "This is a well-designed logo with a combination of text and design elements",
                        "This logo represents a highly recognizable global brand",
                        "This is an image of a widely known corporate logo",
                        "This logo is associated with a prominent multinational company",
                        "This is a famous brand logo with both visual and textual elements"]
negative_hard_prompts = ["This is a logo with only design and no text",
                        "This logo does not contain any recognizable brand name or text",
                        "This is not a logo of a globally known brand or company",
                        "This is a logo of a foreign or non-English brand that may not be widely recognized",
                        "This is an abstract design that does not represent a specific brand or company",
                        "This image lacks clear branding or association with a well-known logo",
                        "This is not a recognizable or globally known brand logo",
                        "This logo appears to belong to a small or local business, not a major brand",
                        "This is a simple graphic or design without any connection to a famous brand",
                        "This logo lacks identifiable characteristics of a globally established brand"]
def top_k_logit_sum_difference(df, k):
    pos_df = df[df['category'] == "POS"]
    neg_df = df[df['category'] == "NEG"]
    pos_sum = pos_df.head(k)['logits'].sum()
    neg_sum = neg_df.head(k)['logits'].sum()
    diff = pos_sum - neg_sum
    return pos_sum, neg_sum, diff
def rank_weighted_scores(df):
    pos_score = df[df['category'] == "POS"]['logits'].dot(df[df['category'] == "POS"]['Weight'])
    neg_score = df[df['category'] == "NEG"]['logits'].dot(df[df['category'] == "NEG"]['Weight'])
    return pos_score, neg_score, pos_score/neg_score
def rank_position_analysis(df):
    pos_df = df[df['category'] == "POS"]
    neg_df = df[df['category'] == "NEG"]
    avg_pos_rank = pos_df['Rank'].mean()
    avg_neg_rank = neg_df['Rank'].mean()
    return avg_pos_rank, avg_neg_rank
def positive_to_negative_rank_ratio(df, k):
    top_k = df.head(k)
    pos_count = len(top_k[top_k['category'] == "POS"])
    neg_count = len(top_k[top_k['category'] == "NEG"])
    ratio = pos_count / (neg_count + 1e-6)
    return ratio
def get_ranking_metrics_one(df, top_k):
  top_k_sum_diff = top_k_logit_sum_difference(df, top_k)[-1]
  rank_weighted_score = rank_weighted_scores(df)[-1]
  pos_rank, neg_rank = rank_position_analysis(df)
  pos_to_neg_rank = neg_rank/(pos_rank + 1e-6)
  pos_to_neg_ratio = positive_to_negative_rank_ratio(df, top_k)
  return {
      "logit_sum_diff": top_k_sum_diff,
      "rank_weighted_score": rank_weighted_score,
      "avg_pos_to_neg_rank": pos_to_neg_rank,
      "count_pos_to_neg_rank": pos_to_neg_ratio
  }
def cluster_hard_cases(results_df, metric_cols=['score', 'logit_sum_diff', 'rank_weighted_score', 'avg_pos_to_neg_rank', 'count_pos_to_neg_rank']):
    metric_cols_z = [col + '_z' for col in metric_cols]
    for col in metric_cols:
        results_df[col+'_z'] = (results_df[col] - results_df[col].mean()) / results_df[col].std()
    pca = PCA(n_components=1)
    pca_vals = pca.fit_transform(results_df[metric_cols_z]).squeeze()
    results_df['pca_score'] = pca_vals
    kmeans = KMeans(n_clusters=3, random_state=42, )
    results_df['cluster'] = kmeans.fit_predict(results_df[['pca_score']])
    sorted_clusters = kmeans.cluster_centers_.mean(axis=1).argsort()
    return results_df, sorted_clusters
class CLIPModel:
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
    def run_examples(self, images, prompts):
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
    def process_images(self, images, scores, positive_prompts=positive_prompts, negative_prompts=negative_prompts):
        pos_outputs = self.run_examples(images, positive_prompts)
        neg_outputs = self.run_examples(images, negative_prompts)
        pos_logits = pos_outputs.logits_per_image.cpu()
        neg_logits = neg_outputs.logits_per_image.cpu()
        combined_logits = np.concatenate([pos_logits.numpy(), neg_logits.numpy()], axis=1)
        combined_scores = torch.softmax(torch.tensor(combined_logits), dim=-1).numpy()
        combined_prompts = positive_prompts + negative_prompts
        combined_prompts_category = ['POS'] * len(positive_prompts) + ['NEG'] * len(negative_prompts)
        results = []
        for i, (image, score) in enumerate(zip(images, scores)):
          df = df = pd.DataFrame({'logits': combined_logits[i],
                          'scores': combined_scores[i],
                          'prompt': combined_prompts,
                          'category': combined_prompts_category
                          })
          df = df.sort_values(by='logits', ascending=False)
          df['Rank'] = np.arange(1, len(df) + 1)
          df['Weight'] = 1 / df['Rank']
          metrics = get_ranking_metrics_one(df, top_k=6)
          results.append({"image_path": i,
                          "score": score,
                          **metrics})
        results_df = pd.DataFrame(results)
        return results_df
def detect_brands_clip(logo_inputs, clip_model, clip_threshold=0.80):
    logo_ids, images = zip(*logo_inputs.items())
    top_brands_1000 = pd.read_csv("../Data/fortune1000_2024.csv")
    top_brands_2000 = pd.read_csv("../Data/Top2000CompaniesGlobally.csv")
    top_brands = pd.concat([top_brands_1000[['Company']], top_brands_2000['Company']])
    top_brands.drop_duplicates('Company', inplace=True)
    brand_names = [name + " logo" for name in top_brands["Company"].to_list()]
    brand_names += ["Other", "Not a logo"]
    clip_output = clip_model.run_examples(list(images), brand_names)
    logits = clip_output.logits_per_image.softmax(dim=-1)
    max_probs, max_indices = logits.max(dim=-1)
    clip_results = {}
    scraper_inputs = []
    for logo_id, prob, idx, image in zip(logo_ids, max_probs, max_indices, images):
        if prob.item() >= clip_threshold:
            clip_results[logo_id] = {"brand_name": brand_names[idx], "source": "CLIP", "prob": prob.item()}
        else:
            scraper_inputs.append({'id': logo_id, 'image': image})
    return clip_results, scraper_inputs
if __name__ == "__main__":
    clip_model = CLIPModel()
    logo_dir = Path("C:\\Users\\Admin\\Desktop\\milestone 2\\feedback\\feedback\\logos")
    logo_paths = list(logo_dir.glob('*.png'))
    logo_inputs = {}
    logo_to_paths ={}
    for path in logo_paths:
        try:
            img = Image.open(path)
            logo_id = compute_image_hash(img)
            logo_inputs[logo_id] = img
            logo_to_paths[logo_id] = path
        except Exception as e:
            print(f"Failed to load image {path}: {str(e)}")
    clip_results, scraper_inputs = detect_brands_clip(logo_inputs, clip_model, clip_threshold=0.80)
    for id, result in clip_results.items():
        print(logo_to_paths[id])
        print(result)
        print("------------")
```

## File: filtering/heuristic_filtering.py
```python
import cv2
import numpy as np
from PIL import Image, ImageStat
from typing import List, Callable, Optional
from pipelines.brand_recognition_chain import LogoImage
def area_aspect_filter(
    crop: Image.Image,
    min_area: int = 500,
    max_area_ratio: float = 0.5,
    max_aspect_ratio: float = 4.0,
    frame_area: Optional[int] = None
) -> bool:
    w, h = crop.size
    area = w * h
    if area < min_area:
        return False
    if frame_area and (area / frame_area) > max_area_ratio:
        return False
    aspect = max(w / (h + 1e-6), h / (w + 1e-6))
    if aspect > max_aspect_ratio:
        return False
    return True
def texture_filter(
    crop: Image.Image,
    min_std: float = 12.0
) -> bool:
    gray = crop.convert('L')
    std = ImageStat.Stat(gray).stddev[0]
    return std >= min_std
def edge_density_filter(
    crop: Image.Image,
    min_edge_frac: float = 0.05
) -> bool:
    arr = np.array(crop.convert('L'))
    edges = cv2.Canny(arr, 50, 150)
    edge_frac = np.mean(edges > 0)
    return edge_frac >= min_edge_frac
def color_variance_filter(
    crop: Image.Image,
    min_color_std: float = 10.0
) -> bool:
    arr = np.array(crop)
    stds = np.std(arr.reshape(-1, 3), axis=0)
    return np.mean(stds) >= min_color_std
class FilterPipeline:
    def __init__(self, filters: List[Callable[..., bool]]):
        self.filters = filters
    def apply(
        self,
        logo_images: List[LogoImage]
    ) -> List[LogoImage]:
        pruned: List[LogoImage] = []
        for li in logo_images:
            passed = True
            for fn in self.filters:
                try:
                    if fn.__code__.co_argcount >= 2 and 'frame_area' in fn.__code__.co_varnames:
                        if not fn(li.image, frame_area=li.metadata.get('frame_area', None)):
                            passed = False
                            break
                    else:
                        if not fn(li.image):
                            passed = False
                            break
                except Exception as e:
                    print(e)
                    passed = False
                    break
            if passed:
                pruned.append(li)
        return pruned
```

## File: filtering/llm_filtering.py
```python
from models.description_model import LLaMAModel
from tqdm import tqdm
from Levenshtein import ratio, distance
from itertools import combinations
import numpy as np
from enum import Enum
global_logo_prompts  = ["Is this an image of a well-known logo? Include Yes or No in your response at the start",
                    "Is this the logo of a globally recognized brand? Include Yes or No in your response at the start",
                    "Does this logo contain both design and text that represent a well-known brand? Include Yes or No in your response at the start",
                    "Is this the logo of a popular company or product? Include Yes or No in your response at the start",
                    "Is this a professional logo used by a globally established organization? Include Yes or No in your response at the start",
                    "Does this logo combine text and design elements in a way that represents a well-known brand? Include Yes or No in your response at the start",
                    "Is this logo associated with a highly recognizable global brand? Include Yes or No in your response at the start",
                    "Is this an image of a widely known corporate logo? Include Yes or No in your response at the start",
                    "Is this logo linked to a prominent multinational company? Include Yes or No in your response at the start",
                    "Is this a famous brand logo? Include Yes or No in your response at the start",
                    ]
logo_prompts = ["Is this an image of a single logo? Include Yes or No in your response at the start",
                  "Does this image represent a distinct company logo? Include Yes or No in your response at the start",
                  "Is this the logo of a brand or company or product, etc.? Include Yes or No in your response at the start",
                  "Does this image contain unique brand-specific shapes, colors, or text elements that would indicate a logo instead of a general symbol? Definitely include Yes or No in your response at the start!",
                  "Is this a logo design representing a single brand or product or company, etc.? Include Yes or No in your response at the start",
                  "Is this an image of a single professional logo? Include Yes or No in your response at the start",
                  "Is this potentially a logo of any company or product or brand, etc.? Include Yes or No in your response at the start",
                  "Does this image represent a single graphic design of a potential logo? Include Yes or No in your response at the start",
                  "Is there distinctive text placement or stylized fonts typical of a logo design? Include Yes or No in your response at the start",
                  "Would this image likely be used on a storefront, product packaging, or marketing material as a brand logo? Definitely include Yes or No in your response at the start!",
                  "Does this image contain artistic or stylized elements that suggest it is a logo? Include Yes or No in your response at the start",
                  "Could this image be used to represent a brand in marketing or advertising materials? Definitely include Yes or No in your response at the start!"]
case_prompts = {
    'stylized': "Does this image contain any stylized text typical of logos? Include Yes or No at start of your response",
    'multiple_logos': "Is this an image showing more than one logo? Include Yes or No at start of your response",
    'multiple_logos': "Yes or No: Does this image show multiple brand logos?",
    'multiple_logos': "Yes or No: Are there more than one company logos visible in this image?",
    'multiple_logos': "Yes or No: Does the image feature multiple company logos?",
    'poster': "Is this possibly an image of a movie poster? Include Yes or No at start of your response",
    'design': "Is this an image with design elements typical of logos but no text? Include Yes or No at start of your response",
}
class LLMResult(Enum):
    LLM = "Forward to LLM"
    API = "Forward to API"
    EXCLUDE = "EXCLUDE"
def process_image(llama_model, image):
    global_logo_df = llama_model.run_examples(global_logo_prompts, image, repetitions=1, num_beams=1, temperature=0.1)
    global_logo_prob = global_logo_df['yes'].sum() / global_logo_df['include'].sum()
    print("Global Prob", global_logo_prob)
    if global_logo_prob >= 0.8:
        names = [llama_model.run_example("What is the full brand name of this logo? Only mention the brand name and nothing else. No extra words!",
                                 image, num_beams=1, temperature=1.2) for _ in range(5)]
        names = [str.lower(name) for name in names]
        similarities = [ratio(a, b) for a, b in combinations(names, 2)]
        avg_sim = np.mean(similarities)
        if avg_sim >= 0.8:
            print("Model knows this logo very well, forward to LLM")
            return LLMResult.LLM
    logo_df = llama_model.run_examples(logo_prompts, image, num_beams=1, temperature=0.1)
    logo_prob = logo_df['yes'].sum() / logo_df['include'].sum()
    print("Logo Prob", logo_prob)
    case_df = llama_model.run_examples(case_prompts.values(), image, num_beams=1, temperature=0.1)
    case_df['prompt_type'] = list(case_prompts.keys())
    multiple_logos_prob = case_df.loc[case_df.prompt_type=='multiple_logos', 'yes'].sum() /  case_df.loc[case_df.prompt_type=='multiple_logos', 'include'].sum()
    print("Multiple Logos Prob:", multiple_logos_prob)
    case_df.set_index('prompt_type', inplace=True)
    if multiple_logos_prob >= 0.67:
        print("Multiple logos so exclude")
        return LLMResult.EXCLUDE
    if logo_prob >= 0.5 and not case_df.loc['poster', 'yes']:
        print("This is a potential logo. Forward to API")
        return LLMResult.API
    elif case_df.loc['poster', 'yes']:
        print("This is a poster so exclude")
        return LLMResult.EXCLUDE
    elif case_df.loc[['stylized', 'design'], 'yes'].any():
        print("Difficult Case. Forward to API")
        return LLMResult.API
    else:
        print("Exclude")
        return LLMResult.EXCLUDE
```

## File: filtering/post_filtering.py
```python
import os
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from PIL import Image
class PostFilterStrategy(ABC):
    name: str = "PostFilterStrategy"
    @abstractmethod
    def apply(
        self,
        detected_rows: List[dict],
        image_dir: str
    ) -> Tuple[List[dict], Dict[str, str]]:
        ...
class QwenCorrectnessStrategy(PostFilterStrategy):
    name = "Qwen2.5-VL-Correctness"
    def __init__(
        self,
        model,
        sources_to_filter: Optional[Set[str]] = None,
        prompt_template: str = "",
        image_size: Tuple[int, int] = (240, 240),
        batch_size: int = 8,
        model_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.sources_to_filter = set(sources_to_filter or {"Top_2K_Brands"})
        self.prompt_template = prompt_template
        self.image_size = image_size
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs or {"temperature": 0.1, "do_sample": True}
    @staticmethod
    def _normalize_verdict(s: str) -> str:
        s_clean = (s or "").strip().lower()
        if "other" in s_clean:
            return "Other"
        if "incorrect" in s_clean:
            return "Incorrect"
        if "correct" in s_clean:
            return "Correct"
        return "Incorrect"
    def apply(
        self,
        detected_rows: List[dict],
        image_dir: str
    ) -> Tuple[List[dict], Dict[str, str]]:
        idxs_to_check: List[int] = []
        prompts: List[str] = []
        images: List[Image.Image] = []
        for i, row in enumerate(detected_rows):
            src = str(row.get("source", "unknown"))
            if src in self.sources_to_filter:
                brand = str(row.get("brand", "")).strip()
                fname = row.get("filename")
                if not fname:
                    continue
                path = os.path.join(image_dir, fname)
                if not os.path.exists(path):
                    continue
                try:
                    img = Image.open(path).convert("RGB").resize(self.image_size)
                except Exception:
                    continue
                prompts.append(self.prompt_template.format(brand=brand))
                images.append(img)
                idxs_to_check.append(i)
        verdicts: List[str] = []
        if idxs_to_check:
            outputs: Sequence[str] = self.model.chunked_run_examples(
                prompts, images, batch_size=self.batch_size, **self.model_kwargs
            )
            verdicts = [self._normalize_verdict(o) for o in outputs]
        verdict_by_logo_id: Dict[str, str] = {}
        for row in detected_rows:
            row.setdefault("postfilter_method", self.name)
            row.setdefault("postfilter_applied", False)
            row.setdefault("postfilter_verdict", "Skipped")
        for local_idx, row_idx in enumerate(idxs_to_check):
            v = verdicts[local_idx] if local_idx < len(verdicts) else "Incorrect"
            row = detected_rows[row_idx]
            row["postfilter_applied"] = True
            row["postfilter_verdict"] = v
            logo_id = os.path.splitext(row.get("filename", ""))[0]
            if logo_id:
                verdict_by_logo_id[logo_id] = v
        return detected_rows, verdict_by_logo_id
#     Runs one or more post-filter strategies (in order).
#     Returns:
#       filtered_rows  : rows after removing items that failed *any* applied strategy
#       annotated_rows : every row with postfilter annotations (keep-all / diagnostics)
#       frame_results  : updated in-place with postfilter verdicts (non-correct -> UNKNOWN)
#     """
#             filtered_rows.append(row)
#     # Update frame_results so downstream steps (e.g., annotate_video) reflect filtering
#     for fr in frame_results:
#         results = fr.get("results", {})
#         for logo_id, info in results.items():
#             v = logo_verdicts.get(logo_id)
#             if v:
#                 # attach metadata
#                 info["postfilter"] = {
#                     "method": " | ".join(s.name for s in strategies),
#                     "verdict": v,
#                     "applied": True,
#                 }
#                 # if not Correct => blank it out so it's treated as unknown/ignored later
def apply_post_filters(
    detected_rows: List[dict],
    image_dir: str,
    frame_results: List[dict],
    strategies: Iterable[PostFilterStrategy],
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    annotated_rows = detected_rows
    logo_verdicts: Dict[str, str] = {}
    for strat in strategies:
        annotated_rows, vmap = strat.apply(annotated_rows, image_dir=image_dir)
        logo_verdicts.update(vmap)
    filtered_rows: List[dict] = []
    for row in annotated_rows:
        applied = row.get("postfilter_applied", False)
        verdict = row.get("postfilter_verdict", "Skipped")
        if applied:
            if verdict == "Correct":
                filtered_rows.append(row)
        else:
            filtered_rows.append(row)
    # Update frame_results so downstream steps (e.g., annotate_video) reflect filtering
    methods_str = " | ".join(s.name for s in strategies) if strategies else ""
    for fr in frame_results:
        results = fr.get("results", {})
        for logo_id, info in results.items():
            v = logo_verdicts.get(logo_id)
            if v:
                # attach metadata
                info["postfilter"] = {
                    "method": methods_str,
                    "verdict": v,
                    "applied": True,
                }
                # if not Correct => blank it out so it's treated as unknown/ignored later
                if v != "Correct":
                    info["brand"] = "UNKNOWN"
            else:
                info["postfilter"] = {
                    "method": methods_str,
                    "verdict": "Skipped",
                    "applied": False,
                }
    filtered_frame_results: List[dict] = []
    for fr in frame_results:
        results = fr.get("results", {})
        kept: Dict[str, dict] = {}
        for logo_id, info in results.items():
            pf = info.get("postfilter", {})
            applied = pf.get("applied", False)
            verdict = pf.get("verdict", "Skipped")
            if (applied and verdict == "Correct") or (not applied):
                kept[logo_id] = info
        filtered_frame_results.append({
            "frame_number": fr.get("frame_number"),
            "timestamp": fr.get("timestamp"),
            "results": kept
        })
    return filtered_rows, annotated_rows, frame_results, filtered_frame_results
```

## File: models/audio_model.py
```python
import os
import json
import argparse
from typing import List, Dict
import torch
from moviepy import AudioFileClip, VideoFileClip
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
import librosa
from pathlib import Path
#     Wrapper around HuggingFace Whisper v3 large for long-form
#     transcription and translation (Italian→English by default).
#     """
#         Initialize the WhisperModel.
#         Args:
#             model_id (str): HuggingFace model identifier.
#             chunk_length_s (int): Segment length (seconds) for chunked processing.
#             stride_length_s (int, optional): Overlap (seconds) between chunks.
#                 If None, defaults to chunk_length_s // 6.
#             device (str, optional): "cuda" or "cpu". If None, auto-detects.
#             dtype (torch.dtype, optional): torch.float16 or torch.float32.
#                 If None, uses float16 on GPU or float32 on CPU.
#         """
#         Transcribe and translate a long-form audio file.
#         Args:
#             audio_path (str): Path to the audio file (e.g., .mp3, .wav).
#         Returns:
#             str: The full English transcript.
#         """
import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
class WhisperModel:
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        chunk_length_s: int = 30,
        stride_length_s: int | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if "cuda" in self.device else torch.float32)
        self.chunk_length_s = int(chunk_length_s)
        self.stride_length_s = (
            int(stride_length_s) if stride_length_s is not None else max(1, self.chunk_length_s // 6)
        )
        self.batch_size = int(batch_size)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=(self.stride_length_s, self.stride_length_s),
            batch_size=self.batch_size,
            device=self.device,
            torch_dtype=self.dtype,
            return_timestamps=True,
        )
    def run_example(
        self,
        audio_path: str,
        mode: str = "translate",
        language: str | None = "english",
        return_language: bool = False,
    ):
        mode = (mode or "translate").lower()
        if mode not in ("translate", "transcribe"):
            raise ValueError(f"WhisperModel.run_example: invalid mode='{mode}', expected 'translate' or 'transcribe'.")
        gen_kwargs = {
            "task": mode,
            'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 0.1),
            'compression_ratio_threshold': 1.35,
            'logprob_threshold': -1.0,
        }
        if language:
            gen_kwargs["language"] = language.lower()
        try:
            result = self.asr_pipeline(
                audio_path,
                generate_kwargs=gen_kwargs,
                return_language=return_language,
            )
        except TypeError:
            result = self.asr_pipeline(
                audio_path,
                generate_kwargs=gen_kwargs,
            )
        transcript = result.get("text", "") if isinstance(result, dict) else ""
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        chunks_df = pd.DataFrame(chunks)
        lang = None
        if return_language:
            maybe_lang = result.get("language") if isinstance(result, dict) else None
            if isinstance(maybe_lang, str):
                lang = maybe_lang.lower()
            elif isinstance(maybe_lang, dict):
                lang = (
                    maybe_lang.get("language")
                    or maybe_lang.get("detected_language")
                    or maybe_lang.get("lang")
                )
                if isinstance(lang, str):
                    lang = lang.lower()
        if return_language:
            return transcript, chunks_df, lang
        return transcript, chunks_df
def convert_video_to_audio(video_path:str):
    with VideoFileClip(video_path) as video_clip:
        audio_clip = video_clip.audio
        audio_clip.write_audiofile('logo_audio.wav')
if __name__ == "__main__":
    video_path = Path('../videos/1.mp4')
    print('here')
    audio_path = '../logo_audio.wav'
    model = WhisperModel()
    transcript, chunk_df, lang = model.run_example(audio_path,
                                             mode='transcribe',
                                             language='italian',
                                             return_language=True)
    print(transcript, lang)
```

## File: models/description_model.py
```python
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import getpass
from qwen_vl_utils import process_vision_info
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
def fixed_get_imports(filename) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
class GemmaModel:
    def __init__(self, model_id="google/gemma-2-2b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    def run_example(self, text, prompt=None):
        if not prompt:
            prompt = f"{text}\nThe name of the logo given in the description (not OCR) is:\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=32)
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logo_name = outputs[len(prompt):].split('\n')[0].strip()
        return logo_name
class FlorenceModel:
    def __init__(self, model_id="microsoft/Florence-2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
    def run_example(self, task_prompt: str, text_input:str = None, image: Image =None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text,
                                                      task=task_prompt,
                                                      image_size=(image.width, image.height))
        return parsed_answer
    def describe_image(self, image: Image):
        prompt = "<MORE_DETAILED_CAPTION>"
        return self.run_example(prompt, image=image)[prompt]
    def extract_text(self, image: Image):
        prompt = "<OCR_WITH_REGION>"
        results = self.run_example(prompt, image=image)[prompt]
        labels = results['labels']
        labels = [str.strip(str.strip(label, "</s>")) for label in labels]
        text = " ".join(labels)
        return text
    def generate_description(self, image: Image):
        ocr = self.extract_text(image)
        desc = self.describe_image(image)
        result = "OCR: " + ocr + "\n" + "Description: " + desc
        return result
class LLaMAModel:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
    def run_example(self, task_prompt: str, image: Image, **kwargs):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": task_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device, self.model.dtype)
        output = self.model.generate(**inputs, max_new_tokens=25, **kwargs)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        logo_name = generated_text.strip().split('assistant')[-1].strip().strip('.')
        return logo_name
    def run_examples(self, prompts, image, repetitions=1, **kwargs):
        results = [{'prompt': prompt,
                    'answer': self.run_example(prompt, image, **kwargs)}
                   for prompt in prompts for _ in range(repetitions)]
        results_df = pd.DataFrame(results)
        results_df['yes'] = results_df['answer'].str.lower().str.contains('yes')
        results_df['no'] = results_df['answer'].str.lower().str.contains('no')
        results_df['include'] = (results_df['yes'] | results_df['no'])
        return results_df
    def get_logo_name(self, image: Image):
        prompt = "What is the full brand name of this logo? Only give the brand name (no special characters or extra words)"
        return self.run_example(prompt, image=image)
class Qwen2_5VLModel:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto",
            quantization_config=quantization_config).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left", max_pixels = 128 * 28 * 28)
    def run_example(self, task_prompt: str, image: Image, **kwargs):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": task_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device, self.model.dtype)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        output = output_text[0]
        return output_text
    def run_examples(self, prompts, images, **kwargs):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ]
            for prompt in prompts
        ]
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, self.model.dtype)
        generated_ids = self.model.generate(**inputs, max_new_tokens=256, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    def chunked_run_examples(self, prompts, images, batch_size=28, **kwargs):
        results = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            prompt_batch = prompts[i:i+batch_size]
            image_batch = images[i:i+batch_size]
            output = self.run_examples(prompt_batch, image_batch, **kwargs)
            results.extend(output)
        return results
    def get_logo_name(self, image: Image):
        prompt = "What is the full brand name of this logo? Only give the brand name (no special characters or extra words)"
        return self.run_example(prompt, image=image)
class Qwen2_5TextModel:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype="auto",
                        device_map="auto",
                        quantization_config=quantization_config,
                    ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    def run_example(self, task_prompt: str, system_prompt: str='You are a helpful assistant.', **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=768,
                            do_sample=True,
                            temperature=1.2,
                        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    qwen = Qwen2_5VLModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    logo_dir = Path("D:\milestone 2\logo_for_database\enel")
    logo_paths = [
        Path("D:\milestone 2\logo_for_database\enel\\1.png"),
        Path("D:\milestone 2\logo_for_database\coca-cola\\1.png"),
        Path("D:\milestone 2\logo_for_database\coca-cola\\2.png"),
        Path("D:\milestone 2\logo_for_database\coca-cola\\3.png"),
        Path("D:\milestone 2\logo_for_database\disney\\1.png"),
        Path("D:\milestone 2\logo_for_database\dropbox\\1.png"),
        Path("D:\milestone 2\logo_for_database\MSC\\1.png"),
        Path("D:\milestone 2\logo_for_database\mapei\\1.jpeg"),
        Path("D:\milestone 2\logo_for_database\mapei\\2.jpg"),
        Path("D:\milestone 2\logo_for_database\ENI\\2.png"),
        Path("D:\milestone 2\logo_for_database\ENI\\3.png"),
        Path("D:\milestone 2\logo_for_database\lotto\\1.png"),
    ]
    logos = [Image.open(path).convert('RGB') for path in logo_paths]
    #             Please consider potential false positives such as generic stylized text (in movie posters for example) and only output a brand name if your confidence level is sufficiently high.
    #             If you do not know, are unsure or the image is not a logo, just output: UNKNOWN"""
    prompt = """Extract the text given in this image. If no text is present, return an empty string. Do not include any special characters. Do NOT hallucinate!"""
    prompts = [prompt for _ in logos]
    logo_names = qwen.chunked_run_examples(prompts, logos, batch_size=16, temperature=0.2)
    print(logo_names)
if __name__ == "__main__":
    main()
```

## File: models/detector_model.py
```python
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import numpy as np
import supervision as sv
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
class LogoDetector:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
    def run_example(self, image: Image, threshold=0.1, nms_threshold=0.3):
        texts = [['a potential logo']]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )
        scores = results[0]['scores'].cpu().numpy().reshape(-1, 1)
        boxes = results[0]['boxes'].cpu().numpy()
        detections = np.hstack((boxes, scores))
        keep_idxs = sv.box_non_max_suppression(detections, iou_threshold=nms_threshold)
        return {
            'boxes': boxes[keep_idxs],
            'scores': scores.flatten()[keep_idxs],
            'labels': results[0]['labels'].cpu().numpy()[keep_idxs]
        }
    def run_examples(self, images, threshold=0.1, nms_threshold=0.3):
        texts = [['a potential logo'] * len(images)]
        inputs = self.processor(text=texts, images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.Tensor([img.size[::-1] for img in images]).to(self.device)
        batch_results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )
        all_results = []
        for result in batch_results:
            scores = result['scores'].cpu().numpy().reshape(-1, 1)
            boxes = result['boxes'].cpu().numpy()
            detections = np.hstack((boxes, scores))
            keep_idxs = sv.box_non_max_suppression(detections, iou_threshold=nms_threshold)
            all_results.append({
                'boxes': boxes[keep_idxs],
                'scores': scores.flatten()[keep_idxs],
                'labels': result['labels'].cpu().numpy()[keep_idxs]
            })
        return all_results
    def draw_boxes(self, image: Image, result: dict):
        boxes = result['boxes']
        scores = result['scores']
        logos = []
        for box in boxes:
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
            logos.append(image.crop((x1, y1, x2, y2)))
        return logos, scores, boxes
    def process_image(self, image: Image, threshold: float = 0.1, nms_threshold: float = 0.3):
        result = self.run_example(image, threshold=threshold, nms_threshold=nms_threshold)
        logos, scores, boxes = self.draw_boxes(image, result)
        return logos, scores, boxes
    def process_images(self, images, threshold=0.1, nms_threshold=0.3, batch_size=2):
        all_outputs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self.run_examples(batch, threshold=threshold, nms_threshold=nms_threshold)
            for img, res in zip(batch, batch_results):
                logos, scores, boxes = self.draw_boxes(img, res)
                all_outputs.append({
                    'logos': logos,
                    'scores': scores,
                    'boxes': boxes
                })
        return all_outputs
```

## File: models/faiss_db.py
```python
import os
import json
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import faiss
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import pandas as pd
import torch
import re
def apply_augmentation(image: Image.Image, augmenter) -> Image.Image:
    img_np = np.array(image)
    augmented = augmenter(image=img_np)['image']
    return Image.fromarray(augmented)
class LogoDatabase:
    def __init__(self,
                 index_path: str = "logo_index.faiss",
                 metadata_path: str = "metadata.json",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.device = device
        self.batch_size = batch_size
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large', use_fast=True)
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.index = None
        self.metadata = []
        self._init_index()
    def _init_index(self):
        combined_dim=768
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(combined_dim)
            self.metadata = []
        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        try:
            clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                clip_embeddings = self.clip_model.get_image_features(**clip_inputs)
            combined_embeddings = clip_embeddings
            combined_embeddings = combined_embeddings.cpu().numpy()
            norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
            combined_embeddings_normalized = combined_embeddings / norms
            return combined_embeddings_normalized
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return np.array([])
    def add_logos(self,
                 images: List[Image.Image],
                 brand_names: List[str],
                 duplicate_threshold: float = 0.95):
        images = list(images)
        if len(images) != len(brand_names):
            raise ValueError("Number of images and brand names must match")
        total_added = 0
        for i in tqdm(range(0, len(images), self.batch_size)):
            batch_images = images[i:i+self.batch_size]
            batch_brands = brand_names[i:i+self.batch_size]
            batch_embeddings = self._embed_batch(batch_images)
            if batch_embeddings.size == 0:
                continue
            if batch_embeddings.size > 0:
                self.index.add(batch_embeddings.astype('float32'))
                self.metadata.extend(batch_brands)
                total_added += len(batch_brands)
        print(f"Added {total_added} new logos (skipped {len(images)-total_added} duplicates)")
    def search_logo(self,
                   query_image: Image.Image,
                   threshold: float = 0.85,
                   k: int = 5) -> List[Tuple[str, float]]:
        batch_embed = self._embed_batch([query_image])
        if batch_embed.size == 0:
            return []
        query_embed = batch_embed[0].astype('float32')
        distances, indices = self.index.search(np.expand_dims(query_embed, 0), k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            similarity = 1 - dist / 4
            if similarity >= threshold and i < len(self.metadata):
                results.append({'index': i,
                                'brand_name': self.metadata[i],
                                'similarity': similarity
                               })
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    def save(self):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    def __len__(self):
        return len(self.metadata)
    @property
    def total_logos(self):
        return len(self.metadata)
class LogoDatabaseNew:
    def __init__(self,
                 index_path: str = "logo_index.faiss",
                 metadata_path: str = "metadata.json",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.device = device
        self.batch_size = batch_size
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.index = None
        self.metadata = []
        self._init_index()
    def _init_index(self):
        combined_dim=768
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(combined_dim)
            self.metadata = []
        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        try:
            clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                clip_embeddings = self.clip_model.get_image_features(**clip_inputs)
            combined_embeddings = clip_embeddings
            combined_embeddings = combined_embeddings.cpu().numpy()
            norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
            combined_embeddings_normalized = combined_embeddings / norms
            return combined_embeddings_normalized
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return np.array([])
    def add_logos(self,
                  images: List[Image.Image],
                  brand_names: List[str],
                  duplicate_threshold: float = 0.95,
                  augmentations = None,
                  num_augments: int = 10):
        if len(images) != len(brand_names):
            raise ValueError("Number of images and brand names must match")
        all_images = []
        all_brands = []
        for img, brand in zip(images, brand_names):
            all_images.append(img)
            all_brands.append(brand)
            if augmentations:
                for _ in range(num_augments):
                    aug_img = apply_augmentation(img, augmentations)
                    all_images.append(aug_img)
                    all_brands.append(brand)
        total_added = 0
        for i in tqdm(range(0, len(all_images), self.batch_size)):
            batch_images = all_images[i:i + self.batch_size]
            batch_brands = all_brands[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_images)
            if batch_embeddings.size == 0:
                continue
            self.index.add(batch_embeddings.astype('float32'))
            self.metadata.extend(batch_brands)
            total_added += len(batch_brands)
        print(f"Added {total_added} new logos including augmentations.")
    def search_logo(self,
                    query_image: Image.Image,
                    threshold: float = 0.85,
                    k: int = 5,
                    augmentations = None,
                    num_augments: int = 5):
        query_images = [query_image]
        if augmentations:
            img_np = np.array(query_image)
            for _ in range(num_augments):
                aug_np = augmentations(image=img_np)['image']
                query_images.append(Image.fromarray(aug_np))
        embeddings = self._embed_batch(query_images)
        if embeddings.size == 0:
            return []
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        avg_embedding = np.mean(normalized, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)
        distances, indices = self.index.search(np.expand_dims(avg_embedding.astype('float32'), 0), k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            similarity = 1 - dist / 4
            if similarity >= threshold and i < len(self.metadata):
                results.append({
                    'index': i,
                    'brand_name': self.metadata[i],
                    'similarity': similarity
                })
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    def search_logos(self,
                     images: List[Image.Image],
                     threshold: float = 0.85,
                     k: int = 5,
                     batch_size: int = 64,
                     augmentations = None,
                     num_augments: int = 5):
        all_results = []
        for img in images:
            variants = [img]
            if augmentations:
                img_np = np.array(img)
                for _ in range(num_augments):
                    aug_np = augmentations(image=img_np)['image']
                    variants.append(Image.fromarray(aug_np))
            emb = self._embed_batch(variants)
            if emb.size == 0:
                all_results.append([])
                continue
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            avg_emb = np.mean(emb, axis=0)
            avg_emb /= np.linalg.norm(avg_emb)
            distances, indices = self.index.search(np.expand_dims(avg_emb.astype('float32'), 0), k)
            image_results = []
            for i, dist in zip(indices[0], distances[0]):
                similarity = 1 - dist / 4
                if similarity >= threshold and i < len(self.metadata):
                    image_results.append({
                        'index': i,
                        'brand_name': self.metadata[i],
                        'similarity': similarity
                    })
            all_results.append(sorted(image_results, key=lambda x: x['similarity'], reverse=True))
        return all_results
    def save(self):
        index = self.index
        if self.device=='cuda':
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    def __len__(self):
        return len(self.metadata)
    @property
    def total_logos(self):
        return len(self.metadata)
if __name__ == '__main__':
    index_path = "D:\\milestone 2\\faiss_db\\logo_index.faiss"
    metadata_path = "D:\\milestone 2\\faiss_db\\metadata.json"
    db = LogoDatabase(index_path, metadata_path, device='cpu', batch_size=32)
    logos_path = Path("D:\\milestone 2\\faiss_db\\logo_for_database\\logo_for_database")
    logo = Image.open(logos_path/'frosta'/'1.png').convert('RGB')
    result = db.search_logo(logo,
                            threshold=0.85,
                            k=5)
    print(result[0])
```

## File: models/verification_model.py
```python
import torch
from PIL import Image
import os
import getpass
from transformers import BitsAndBytesConfig, pipeline, AutoProcessor, LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
class LLaVAModel:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
    def run_example(self, image, prompt=None):
        if prompt:
            prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        else:
            prompt =  "USER: <image>\nIs this most likely an image of a single logo only? Just answer Yes or No.\nASSISTANT:"
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 128})[0]["generated_text"]
        result = str.lower(str.strip(str.strip(str.split(output, "ASSISTANT:")[1]), '.'))
        if  "yes" in result:
            return True
        return False
    def verify_images(self, images):
        results = [image for image in images if self.run_example(image)]
        return results
class PaligemmaModel:
    def __init__(self, model_id="google/paligemma-3b-mix-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quantization_config).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
    def run_example(self, image, prompt=None):
        if prompt:
            prompt = f"<image>{prompt}<bos>"
        if not prompt:
            prompt = "<image>Does this image represent a plain icon, plain text, branded logo, or something else?<bos>"
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.model.dtype)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=128, do_sample=False, early_stopping=False, num_beams=3)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
    def verify_image(self, image, prompt=None):
            result = self.run_example(image, prompt)
            if 'logo' in str.lower(result):
                return True
            return False
    def verify_images(self, images):
        results = [image for image in images if self.run_example(image)]
        return results
def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    model = LLaVAModel()
    logo = Image.open("ARY_Digital_Logo.png")
    print(model.run_example(logo))
if __name__ == "__main__":
    main()
```

## File: pipelines/brand_recognition_chain.py
```python
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL import Image
import uuid
from filtering.clip_filtering import CLIPModel, positive_prompts, negative_prompts, detect_brands_clip
import pandas as pd
import re
from utils.fuzzy_matching import combine_easyocr_text_ordered, get_ocr_image, robust_fuzzy_match, robust_fuzzy_match_batch, smart_score
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Optional, Dict, Any
import albumentations as A
from models.description_model import Qwen2_5VLModel
from collections import Counter
try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
@dataclass
class LogoImage:
    id: str
    image: Image.Image
    metadata: Dict[str, Any] = field(default_factory=dict)
    @staticmethod
    def create(image: Image.Image, metadata: Dict[str, Any] = None) -> 'LogoImage':
        return LogoImage(
            id=str(uuid.uuid4()),
            image=image,
            metadata=metadata or {},
        )
class BrandRecognitionTechnique(ABC):
    @abstractmethod
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        pass
class BrandRecognizer:
    def __init__(self, techniques: List[BrandRecognitionTechnique]):
        self.techniques = techniques
    def recognize(self, logo_images: List[LogoImage]) -> Dict[str, Dict[str, Any]]:
        remaining_images = {logo.id: logo for logo in logo_images}
        final_results = {}
        for technique in self.techniques:
            if not remaining_images:
                break
            predictions = technique.predict(list(remaining_images.values()))
            processed = 0
            for image_id, prediction in predictions.items():
                if isinstance(prediction, str):
                    brand = prediction
                    extra_info = {}
                elif isinstance(prediction, dict):
                    brand = prediction.get('brand', 'UNKNOWN')
                    extra_info = {k: v for k, v in prediction.items() if k != 'brand'}
                else:
                    raise ValueError("Prediction must be either str or dict.")
                if brand != 'UNKNOWN':
                    logo = remaining_images[image_id]
                    final_results[image_id] = {
                        'brand': brand,
                        'technique': technique.__class__.__name__,
                        'image': logo.image,
                        'metadata': logo.metadata,
                        **extra_info
                    }
                    processed+= 1
            total_images = len(remaining_images)
            failed = total_images - processed
            print(f"[{technique.__class__.__name__}] Processed {processed} / {total_images} images successfully. {failed} remain unrecognized.")
            remaining_images = {
                img_id: logo for img_id, logo in remaining_images.items()
                if img_id not in final_results
            }
        for image_id, logo in remaining_images.items():
            final_results[image_id] = {
                'brand': 'UNKNOWN',
                'technique': None,
                'image': logo.image,
                'metadata': logo.metadata
            }
        return final_results
#         clip_model: CLIP model instance with run_examples method
#         clip_threshold: Probability threshold for accepting predictions
#         """
class ClipTechnique(BrandRecognitionTechnique):
    def __init__(
        self,
        clip_model,
        clip_threshold: float = 0.8,
        fuzzy_threshold: int = 90
    ):
        self.clip_model = clip_model
        self.clip_threshold = clip_threshold
        self.fuzzy_threshold = fuzzy_threshold
        base_df = pd.read_csv(r"D:\milestone 2\Data\Deduplicated_Companies.csv")
        base_df.rename(columns={'Brand':'Company'}, inplace=True)
        base_df.drop_duplicates('Company', inplace=True)
        self.base_brand_names = base_df["Company"].tolist()
        self.clip_brand_names = [
            name + " logo" for name in self.base_brand_names
        ] + ["Other", "Not a logo"]
        self.normalized_fuzzy_brand_names = [
            self._normalize(name) for name in self.base_brand_names
        ]
        self.llm = Qwen2_5VLModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
        self.llm_prompt = (
            "Extract the text given in this image. If no text is present, return "
            "an empty string. Do not include any special characters. Do NOT hallucinate!"
        )
        # You are given an image. Your job is to extract any legible text from it.
        # If you cannot confidently read any text—because it is too blurry, low resolution,
        # or otherwise unreadable, please return an EMPTY STRING!
        # Do not hallucinate any characters or words!!!
        # Do not include punctuation or special characters.
        # """
    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        results: Dict[str, str] = {}
        if not logo_images:
            return results
        images = [li.image for li in logo_images]
        ids = [li.id for li in logo_images]
        prompts = [self.llm_prompt] * len(images)
        ocr_texts = self.llm.chunked_run_examples(
            prompts, images, batch_size=32, temperature=0.2
        )
        norm_ocr = [self._normalize(txt) for txt in ocr_texts]
        long_idxs = [i for i, txt in enumerate(norm_ocr) if len(txt) >= 4]
        clip_candidates = set(range(len(images)))
        if long_idxs:
            queries = [norm_ocr[i] for i in long_idxs]
            print(queries)
            sim_mat = process.cdist(
                queries,
                self.normalized_fuzzy_brand_names,
                scorer=fuzz.WRatio
            )
            for q_i, sims in enumerate(sim_mat):
                orig_i = long_idxs[q_i]
                best_sim = sims.max()
                best_j = int(sims.argmax())
                if best_sim >= self.fuzzy_threshold:
                    assigned_brand = self.base_brand_names[best_j]
                    results[ids[orig_i]] = {
                        'brand': assigned_brand,
                        'source': "Top_2K_Brands",
                    }
                    clip_candidates.remove(orig_i)
        if clip_candidates:
            c_imgs = [images[i] for i in clip_candidates]
            c_ids = [ids[i] for i in clip_candidates]
            clip_out = self.clip_model.run_examples(
                c_imgs, self.clip_brand_names
            )
            logits = clip_out.logits_per_image.softmax(dim=-1)
            max_p, max_idx = logits.max(dim=-1)
            for lid, p, j in zip(c_ids, max_p, max_idx):
                if p.item() >= self.clip_threshold:
                    results[lid] = {
                        "brand": self.clip_brand_names[j],
                        "source": "Top_2K_Brands"
                    }
                else:
                    results[lid] = {"brand": "UNKNOWN"}
        return results
class OcrFuzzyTechnique(BrandRecognitionTechnique):
    def __init__(self, ocr_func, fuzzy_matcher, known_brands: List[str], fuzzy_threshold: float = 0.8):
        self.ocr_func = ocr_func
        self.fuzzy_matcher = fuzzy_matcher
        self.known_brands = [b.lower().strip() for b in known_brands]
        self.fuzzy_threshold = fuzzy_threshold
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        results = {}
        for logo in logo_images:
            ocr_text = self.ocr_func(logo.image)
            if not ocr_text or len(ocr_text) < 3:
                results[logo.id] = 'UNKNOWN'
                continue
            matched_brand, score = self.fuzzy_matcher(ocr_text, self.known_brands)
            results[logo.id] = {
                'brand': matched_brand,
                'fuzzy_score': score,
                'ocr_text': ocr_text,
                'source': 'audio',
            }
        return results
class OcrFuzzyBatchTechnique(BrandRecognitionTechnique):
    def __init__(
        self,
        ocr_func,
        fuzzy_matcher,
        known_brands: List[str],
        fuzzy_threshold: float = 0.8
    ):
        self.ocr_func = ocr_func
        self.fuzzy_matcher = fuzzy_matcher
        self.known_brands = [b.lower().strip() for b in known_brands]
        self.base_threshold = int(fuzzy_threshold * 100)
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        ocr_texts = []
        for logo in logo_images:
            raw = self.ocr_func(logo.image) or ""
            ocr_texts.append(raw.strip())
        matches = self.fuzzy_matcher(
            texts=ocr_texts,
            brand_list=self.known_brands,
            base_threshold=self.base_threshold
        )
        for logo, ocr_txt, (brand, score) in zip(logo_images, ocr_texts, matches):
            results[logo.id] = {
                "brand":       brand,
                "fuzzy_score": score,
                "ocr_text":    ocr_txt,
                "source": "audio",
            }
        return results
AUGMENTATIONS = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ColorJitter(hue=0.05, saturation=0.1, p=0.4),
    A.Resize(224, 224)
])
class DatabaseFaissTechnique(BrandRecognitionTechnique):
    def __init__(self, db, ocr_reader, base_threshold: int = 75, faiss_threshold: float = 0.85, min_text_len: int = 3):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len
    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results = {}
        for logo in logo_images:
            search_results = self.db.search_logo(logo.image, k=5, threshold=self.faiss_threshold)
            ocr_text = self._extract_clean_text(logo.image)
            if not search_results:
                results[logo.id] = {
                    'brand': 'UNKNOWN',
                    'ocr_text': ocr_text,
                    'matched_text': None,
                    'score': 0.0
                }
                continue
            top_result = search_results[0]
            db_text = top_result['brand_name'].lower().strip()
            db_score = top_result['similarity']
            if not ocr_text and not db_text and db_score >= 0.93:
                results[logo.id] = {
                    'brand': db_text,
                    'ocr_text': '',
                    'matched_text': db_text,
                    'score': db_score
                }
            else:
                matched_text = robust_fuzzy_match(ocr_text, db_text, self.base_threshold, self.min_text_len)
                brand = matched_text if matched_text != "UNKNOWN" else "UNKNOWN"
                results[logo.id] = {
                    'brand': brand,
                    'ocr_text': ocr_text,
                    'matched_text': db_text,
                    'score': db_score
                }
        return results
class DatabaseFaissParallelTechnique(BrandRecognitionTechnique):
    def __init__(self, db, ocr_reader, base_threshold: int = 75, faiss_threshold: float = 0.85, min_text_len: int = 3, num_workers: int = None):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len
        self.num_workers = num_workers if num_workers else max(1, mp.cpu_count() - 2)
    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z\\s]", "", text)
        return text
    def _process_logo(self, logo: LogoImage):
        search_results = self.db.search_logo(logo.image, k=5, threshold=self.faiss_threshold)
        ocr_text = self._extract_clean_text(logo.image)
        if not search_results:
            return logo.id, {
                'brand': 'UNKNOWN',
                'ocr_text': ocr_text,
                'matched_text': None,
                'score': 0.0
            }
        top_result = search_results[0]
        db_text = top_result['brand_name'].lower().strip()
        db_score = top_result['similarity']
        if not ocr_text and not db_text and db_score >= 0.93:
            return logo.id, {
                'brand': db_text,
                'ocr_text': '',
                'matched_text': db_text,
                'score': db_score
            }
        else:
            matched_text = robust_fuzzy_match(ocr_text, db_text, self.base_threshold, self.min_text_len)
            brand = matched_text if matched_text != "UNKNOWN" else "UNKNOWN"
            return logo.id, {
                'brand': brand,
                'ocr_text': ocr_text,
                'matched_text': db_text,
                'score': db_score
            }
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results = {}
        if not logo_images:
            return results
        with mp.Pool(self.num_workers) as pool:
            logo_result_pairs = pool.map(self._process_logo, logo_images)
        for logo_id, result in logo_result_pairs:
            results[logo_id] = result
        return results
class DatabaseFaissTechniqueBatch(BrandRecognitionTechnique):
    def __init__(
        self,
        db,
        ocr_reader,
        base_threshold: int = 75,
        faiss_threshold: float = 0.85,
        min_text_len: int = 3,
        visual_strict_threshold: float = 0.97,
        visual_consensus_threshold: float = 0.90,
        consensus_topk: int = 5,
        consensus_min_count: int = 4,
        text_min_threshold: int = 80,
        soft_boost_enable: bool = True,
        use_qwen_ocr: bool = False,
        qwen_batch_size: int = 32,
        qwen_temperature: float = 0.2,
        qwen_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len
        self.visual_strict_threshold = visual_strict_threshold
        self.visual_consensus_threshold = visual_consensus_threshold
        self.consensus_topk = consensus_topk
        self.consensus_min_count = consensus_min_count
        self.text_min_threshold = text_min_threshold
        self.soft_boost_enable = soft_boost_enable
        self.use_qwen_ocr = use_qwen_ocr
        self.qwen_batch_size = qwen_batch_size
        self.qwen_temperature = qwen_temperature
        self.qwen_model_id = qwen_model_id
        if self.use_qwen_ocr:
            self.llm = Qwen2_5VLModel(model_id=self.qwen_model_id)
            self.llm_prompt = (
                "Extract the text given in this image. If no text is present, return an empty string. "
                "Do not include any special characters. Do NOT hallucinate!"
            )
    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text
    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def _extract_ocr_batch(self, images: List[Image.Image]) -> List[str]:
        if self.use_qwen_ocr:
            prompts = [self.llm_prompt] * len(images)
            try:
                ocr_texts = self.llm.chunked_run_examples(
                    prompts, images, batch_size=self.qwen_batch_size, temperature=self.qwen_temperature
                )
                return [self._normalize(txt) for txt in ocr_texts]
            except Exception:
                return [self._extract_clean_text(img) for img in images]
        else:
            return [self._extract_clean_text(img) for img in images]
    def _consensus_stats(self, names: List[str]) -> Tuple[Optional[str], int, float]:
        if not names:
            return None, 0, 0.0
        counts = Counter([n.strip().lower() for n in names if n])
        best_name, best_cnt = max(counts.items(), key=lambda kv: kv[1])
        share = best_cnt / max(1, len(names))
        return best_name, best_cnt, share
    def _soft_boost(self, text: str, topk_names: List[str]) -> int:
        if not self.soft_boost_enable or not text or not topk_names:
            return 0
        if _HAS_RAPIDFUZZ:
            for nm in topk_names:
                if fuzz.token_set_ratio(text, nm.lower().strip()) >= 70:
                    return 5
            return 0
        else:
            tset = set(text.split())
            for nm in topk_names:
                nset = set(nm.lower().strip().split())
                inter = len(tset & nset)
            if inter >= max(1, 0.5 * max(len(tset), len(nset))):
                return 5
            return 0
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        images = [logo.image for logo in logo_images]
        batch_search = self.db.search_logos(
            images,
            threshold=self.faiss_threshold,
            k=5,
            batch_size=self.db.batch_size,
            augmentations=AUGMENTATIONS,
            num_augments=3,
        )
        ocr_texts = self._extract_ocr_batch(images)
        fuzzy_threshold = self.text_min_threshold
        min_ocr_len     = self.min_text_len
        for idx, logo in enumerate(logo_images):
            matches = batch_search[idx]
            ocr_txt = (ocr_texts[idx] or "").strip().lower()
            ocr_len = len(ocr_txt)
            if matches and matches[0]["brand_name"].lower().strip() == "false_logo" and matches[0]["similarity"] >= 0.93:
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }
                continue
            if not matches:
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }
                continue
            top1_sim = float(matches[0]["similarity"])
            top1_name = matches[0]["brand_name"]
            if top1_sim >= self.visual_strict_threshold:
                results[logo.id] = {
                    "brand":        top1_name,
                    "ocr_text":     ocr_txt,
                    "matched_text": top1_name,
                    "score":        top1_sim,
                    "source":       "database",
                }
                continue
            if top1_sim >= self.visual_consensus_threshold:
                names_topk = [m["brand_name"] for m in matches[:self.consensus_topk]]
                best_name, best_cnt, share = self._consensus_stats(names_topk)
                if (best_name and
                    best_cnt >= self.consensus_min_count and
                    share >= 0.4 and
                    best_name.lower().strip() == top1_name.lower().strip()):
                    results[logo.id] = {
                        "brand":        best_name,
                        "ocr_text":     ocr_txt,
                        "matched_text": best_name,
                        "score":        top1_sim,
                        "source":       "database",
                    }
                    continue
            if ocr_len < min_ocr_len:
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }
                continue
            names = [m["brand_name"] for m in matches]
            fuzzy_scores = [smart_score(ocr_txt, nm.lower().strip()) for nm in names]
            best_i       = int(np.argmax(fuzzy_scores))
            best_fuzzy   = float(fuzzy_scores[best_i])
            best_name    = names[best_i]
            boosted = best_fuzzy
            if boosted >= fuzzy_threshold:
                results[logo.id] = {
                    "brand":        best_name,
                    "ocr_text":     ocr_txt,
                    "matched_text": best_name,
                    "score":        boosted / 100.0,
                    "source":       "database",
                }
            else:
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }
        return results
```

## File: pipelines/pipeline.py
```python
from utils.shot_detection import ShotDetector
from models.detector_model import LogoDetector
from models.description_model import Qwen2_5VLModel, Qwen2_5TextModel
from filtering.llm_filtering import process_image, LLMResult
from filtering.clip_filtering import CLIPModel, positive_prompts, negative_prompts, cluster_hard_cases, detect_brands_clip
from filtering.heuristic_filtering import FilterPipeline, area_aspect_filter, texture_filter, edge_density_filter, color_variance_filter
from tqdm import tqdm
import numpy as np
import pandas as pd
import getpass
import math
from typing import List
from PIL import Image
import os
import re
import cv2
from utils.prompts import QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT, QWEN_TRANSCRIPT_LOGO_PROMPT
from models.audio_model import WhisperModel, convert_video_to_audio
from utils.fuzzy_matching import get_ocr_image, smart_fuzzy_brand_match, smart_fuzzy_brand_match_batch
from pipelines.brand_recognition_chain import BrandRecognizer, LogoImage, OcrFuzzyTechnique, OcrFuzzyBatchTechnique, ClipTechnique, DatabaseFaissTechnique, DatabaseFaissTechniqueBatch
from models.faiss_db import LogoDatabase, LogoDatabaseNew
from filtering.post_filtering import QwenCorrectnessStrategy, apply_post_filters
from pathlib import Path
def resize_images(images):
    resized = []
    for img in images:
        width, height = img.size
        if width < 28 or height < 28:
            scale = max(28 / width, 28 / height)
            new_size = (math.ceil(width * scale), math.ceil(height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        resized.append(img)
    return resized
def extract_audio_brands(video_path,
                         out_dir=None,
                         mode: str = "translate",
                         ):
    qwen_text_model = Qwen2_5TextModel()
    whisper = WhisperModel()
    if mode not in ("transcribe", "translate"):
        raise ValueError(f"Invalid mode '{mode}', expected 'transcribe' or 'translate'")
    out = Path(out_dir) if out_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    convert_video_to_audio(video_path)
    if mode == "transcribe":
        transcript, chunks_df, lang = whisper.run_example(
            'logo_audio.wav',
            mode="transcribe",
            language=None,
            return_language=True,
        )
        lang = lang or "orig"
        (out / f"transcript.{lang}.txt").write_text(transcript, encoding="utf-8")
    else:
        transcript, chunks_df, _ = whisper.run_example(
            'logo_audio.wav',
            mode="translate",
            language="english",
            return_language=True,
        )
        lang='english'
        (out / "transcript.en.txt").write_text(transcript, encoding="utf-8")
    brand_string = qwen_text_model.run_example(
        QWEN_TRANSCRIPT_LOGO_PROMPT(transcript),
        QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT
    )
    audio_brands = brand_string.split('\n')
    audio_brands = [name.lower().strip() for name in audio_brands]
    (out / f"brands_audio_{lang}.txt").write_text("\n".join(sorted(set(audio_brands))), encoding="utf-8")
    return audio_brands
def filter_logos_with_clip(logo_images: List[LogoImage], clip_model, positive_prompts: List[str], negative_prompts: List[str]) -> List[LogoImage]:
    if len(logo_images) == 0:
        return []
    raw_images = [li.image for li in logo_images]
    raw_scores = [li.metadata.get('score', 1.0) for li in logo_images]
    clip_filter_results_df = clip_model.process_images(
        raw_images,
        scores=raw_scores,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts
    )
    clip_filter_results_df, sorted_clusters = cluster_hard_cases(clip_filter_results_df)
    exclude_cluster = sorted_clusters[0]
    include_logo_indices = clip_filter_results_df.loc[
        clip_filter_results_df['cluster'] != exclude_cluster, 'image_path'
    ].values
    include_logo_indices = [int(idx) for idx in include_logo_indices]
    filtered_logo_images = [logo_images[i] for i in include_logo_indices]
    return filtered_logo_images
#     Draws bounding boxes, brand names, percent areas, and timestamps
#     onto each frame where logos were detected, and writes _all_ frames
#     (annotated or not) into a new video.
#     """
#     Draws bounding boxes, brand names, percent areas, and timestamps
#     onto each frame where logos were detected—and also on the +/- `window`
#     frames around each detection. Writes all frames into a new video.
#     """
#     Draws bounding boxes, brand names, percent areas, and timestamps
#     onto each frame where logos were detected—and also on the +/- `window`
#     frames around each detection.  Writes ONLY those annotated frames
#     (not the whole video) into a new video.
#     """
def annotate_video(video_path, frame_results, output_path, window=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    extended = {}
    for fr in frame_results:
        F = fr['frame_number']
        for f in range(max(0, F - window), min(len(frames), F + window + 1)):
            for info in fr['results'].values():
                extended.setdefault(f, []).append({
                    'bbox':         info['bbox'],
                    'brand':        info['brand'],
                    'percent_area': info['percent_area'],
                    'timestamp':    fr['timestamp']
                })
    if not extended:
        print("No detections found; no output video created.")
        return
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 127, 0),
        (127, 0, 255),
    ]
    brand_colors = {}
    next_color = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    annotated_count = 0
    for idx, frame in enumerate(frames):
        if idx not in extended:
            continue
        for det in extended[idx]:
            x1, y1, x2, y2 = map(int, det['bbox'])
            brand, pct     = det['brand'], det['percent_area']
            if brand == 'UNKNOWN':
                color = (128, 128, 128)
            else:
                if brand not in brand_colors:
                    brand_colors[brand] = palette[next_color % len(palette)]
                    next_color += 1
                color = brand_colors[brand]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if brand != 'UNKNOWN':
                label = f"{brand} {pct:.1f}%"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - baseline - 4),
                    (x1 + tw + 4, y1),
                    color,
                    thickness=-1
                )
                brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
                text_color = (0,0,0) if brightness > 128 else (255,255,255)
                cv2.putText(
                    frame, label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, text_color, 1
                )
            else:
                label = "unknown"
                cv2.putText(
                    frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (128, 128, 128), 1
                )
        ts = extended[idx][0]['timestamp']
        cv2.putText(
            frame, f"Time: {ts}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )
        out.write(frame)
        annotated_count += 1
    out.release()
    print(f"Saved {annotated_count} annotated frames to {output_path}")
ocr_check_prompt = """
You are given an image which may or may not be a logo. The proposed brand name by a machine learning model is: '{brand}'.
Please analyze the image and proposed brand name carefully. The categorize the image into one of the following categories:
1. Logo of the "{brand}" brand (need not be an exact match but should strongly represent it) - Correct
2. Logo of some other brand - Other
3. Not a Logo - Incorrect
- Use your best judgement.
- Do not hallucinate or make up stuff.
- If you are very unsure, just choose the last option i.e incorrect.
Give your answer as one of the following options only:
Correct | Other | Incorrect
Do not output anything else
""".strip()
def process_video(video_path,
                  threshold,
                  detector_threshold,
                  nms_threshold,
                  use_db=False,
                  use_translation=False,
                  use_llm_db=False,
                  use_qwen_filter=False,
                  ):
    print('use_db', use_db)
    print("video threshold", threshold)
    print("detector threshold", detector_threshold)
    print("nms threshold", nms_threshold)
    base = os.path.splitext(os.path.basename(video_path))[0]
    sanitized = re.sub(r'[^A-Za-z0-9_\-]', '_', base)
    output_root = os.path.join("results", sanitized)
    os.makedirs(output_root, exist_ok=True)
    unknown_dir = os.path.join(output_root, "unknowns")
    known_dir   = os.path.join(output_root, "knowns")
    os.makedirs(unknown_dir, exist_ok=True)
    os.makedirs(known_dir, exist_ok=True)
    audio_dir = os.path.join(output_root, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    detected_rows = []
    frame_results = []
    shot_detector   = ShotDetector(video_path)
    detector        = LogoDetector()
    clip_model      = CLIPModel()
    filter_pipeline = FilterPipeline([
        area_aspect_filter,
        texture_filter,
        edge_density_filter,
        color_variance_filter,
    ])
    db = LogoDatabaseNew(
        index_path="D:\\milestone 2\\faiss_database_with_italian_logos\\logo_index.faiss",
        metadata_path="D:\\milestone 2\\faiss_database_with_italian_logos\\metadata.json",
        batch_size=64
    )
    audio_brands = extract_audio_brands(video_path,
                                        out_dir=audio_dir,
                                        mode='translate' if use_translation else 'transcribe',
                                        )
    print(audio_brands)
    frames, frame_numbers, timestamps = shot_detector.process_video(quantile=threshold)
    print('Total Frames:', len(frames))
    detector_results = detector.process_images(
        frames,
        threshold=detector_threshold,
        nms_threshold=nms_threshold,
        batch_size=16
    )
    techniques = [
        ClipTechnique(clip_model, clip_threshold=0.80),
        OcrFuzzyBatchTechnique(get_ocr_image, smart_fuzzy_brand_match_batch, audio_brands, fuzzy_threshold=0.8),
    ]
    if use_db:
        techniques.insert(0, DatabaseFaissTechniqueBatch(db,
                                                         get_ocr_image,
                                                         use_qwen_ocr=use_llm_db,
                                                         ))
    recognizer = BrandRecognizer(techniques)
    for frame, result, timestamp, frame_no in tqdm(
        zip(frames, detector_results, timestamps, frame_numbers),
        total=len(detector_results),
        desc="Processing frames"
    ):
        logos, scores, boxes = result['logos'], result['scores'], result['boxes']
        if not logos:
            continue
        if isinstance(frame, Image.Image):
            fw, fh = frame.size
        else:
            fh, fw = frame.shape[:2]
        frame_area = fw * fh
        logo_images = []
        for idx, (crop_img, det_score, box) in enumerate(zip(logos, scores, boxes)):
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
            w, h = (crop_img.size if isinstance(crop_img, Image.Image)
                    else (crop_img.shape[1], crop_img.shape[0]))
            percent_area = (w * h) / frame_area * 100.0
            metadata = {
                'frame_number':    frame_no,
                'timestamp':       timestamp,
                'logo_index':      idx,
                'detection_score': det_score,
                'percent_area':    percent_area,
                'bbox':            (x1, y1, x2, y2),
            }
            logo_images.append(LogoImage.create(crop_img, metadata))
        if len(logo_images) >= 10:
            logo_images = filter_pipeline.apply(logo_images)
        if not logo_images:
            continue
        recs = recognizer.recognize(logo_images)
        for logo in logo_images:
            info = recs[logo.id]
            info['percent_area'] = logo.metadata['percent_area']
            info['bbox']         = logo.metadata['bbox']
            if info['brand'] == 'UNKNOWN':
                fname = f"frame{frame_no}_logo{logo.metadata['logo_index']}.png"
                logo.image.save(os.path.join(unknown_dir, fname))
            else:
                fname = f"{logo.id}.png"
                logo.image.save(os.path.join(known_dir, fname))
                detected_rows.append({
                    'frame_number':  logo.metadata['frame_number'],
                    'timestamp':     logo.metadata['timestamp'],
                    'brand':         info['brand'],
                    'percent_area':  info['percent_area'],
                    'bbox':          info['bbox'],
                    'source':        info.get('source', 'unknown'),
                    'filename':      fname,
                })
        frame_results.append({
            'frame_number': frame_no,
            'timestamp':    timestamp,
            'results':      recs
        })
    if use_qwen_filter:
        qwen_model = Qwen2_5VLModel(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
        qwen_top2k_strategy = QwenCorrectnessStrategy(
            model=qwen_model,
            sources_to_filter={"Top_2K_Brands", "audio"},
            prompt_template=ocr_check_prompt,
            image_size=(240, 240),
            batch_size=16,
            model_kwargs={"temperature": 0.1, "do_sample": True},
        )
        filtered_rows, annotated_rows, frame_results, filtered_frame_results = apply_post_filters(
            detected_rows,
            image_dir=known_dir,
            frame_results=frame_results,
            strategies=[qwen_top2k_strategy],
        )
        if annotated_rows:
            df_all = pd.DataFrame(annotated_rows)
            all_csv = os.path.join(known_dir, f"{sanitized}_all_detections_with_verdicts.csv")
            df_all.to_csv(all_csv, index=False)
            print(f"▶ Exported (all detections w/ verdicts) to {all_csv}")
        if filtered_rows:
            df = pd.DataFrame(filtered_rows)
            csv_path = os.path.join(known_dir, f"{sanitized}_detected_logos.csv")
            df.to_csv(csv_path, index=False)
            print(f"▶ Exported {len(df)} detections to {csv_path}")
        frame_results = filtered_frame_results
    else:
        if detected_rows:
            df = pd.DataFrame(detected_rows)
            csv_path = os.path.join(known_dir, f"{sanitized}_detected_logos.csv")
            df.to_csv(csv_path, index=False)
            print(f"▶ Exported {len(df)} detections to {csv_path}")
    annotate_video(
        video_path,
        frame_results,
        output_path=os.path.join(output_root, f"output_{sanitized}.mp4"),
    )
    print(frame_results)
    return frame_results
def process_frame_results_merged(frame_results, default_end_offset: float = 2.0) -> pd.DataFrame:
    brand_records = {}
    for frame_result in frame_results:
        frame_timestamp = frame_result['timestamp']
        frame_items = frame_result['results']
        for logo_id, data in frame_items.items():
            brand = data['brand']
            if brand == 'UNKNOWN':
                continue
            source = data.get('source', 'unknown')
            if brand not in brand_records:
                brand_records[brand] = {
                    'logo': data['image'],
                    'start_timestamp': frame_timestamp,
                    'end_timestamp': frame_timestamp,
                    'source': source
                }
            else:
                brand_records[brand]['end_timestamp'] = frame_timestamp
                if brand_records[brand]['source'] != source:
                    brand_records[brand]['source'] += f", {source}"
    records = []
    for brand, info in brand_records.items():
        records.append({
            'logo': info['logo'],
            'brand_name': brand,
            'start_timestamp': info['start_timestamp'],
            'end_timestamp': info['end_timestamp'],
            'source': info['source'],
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df['brand_name'] = df['brand_name'].str.replace('\n', ' or ')
        df['info'] = (
            "Brand Name(s): " + df['brand_name'] +
            ", Start Timestamp: " + df['start_timestamp'].astype(str) +
            ", End Timestamp: " + df['end_timestamp'].astype(str)
        )
    return df
def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    video_path = "../videos/1.mp4"
    process_video(video_path, 0.98)
if __name__ == "__main__":
    main()
```

## File: utils/fuzzy_matching.py
```python
from rapidfuzz import process, fuzz
import easyocr
import numpy as np
from typing import List, Tuple, Dict
import textdistance
import re
reader = easyocr.Reader(['en'])
def combine_easyocr_text_ordered(results, sep=" "):
    sorted_results = sorted(results, key=lambda x: (x[0][1][1], x[0][0][0]))
    return sep.join([text for (_, text, _) in sorted_results])
def get_ocr_image(image):
    ocr_result = reader.readtext(np.array(image))
    ocr_result = combine_easyocr_text_ordered(ocr_result).strip().lower()
    return ocr_result
def smart_fuzzy_brand_match(ocr_text, brand_list, base_threshold=70):
    clean_ocr = ocr_text.strip()
    is_short = len(clean_ocr) < 4
    threshold = base_threshold + 15 if is_short else base_threshold
    match, score, _ = process.extractOne(clean_ocr, brand_list, scorer=fuzz.WRatio)
    return (match, score) if score >= threshold else ("UNKNOWN", score)
def robust_fuzzy_match(ocr_text: str, db_text: str, base_threshold=75, min_text_len=3) -> str:
    ocr_clean = ocr_text.strip()
    if len(ocr_clean) < min_text_len:
        return "UNKNOWN"
    score = fuzz.WRatio(ocr_clean, db_text)
    partial_score = fuzz.partial_ratio(ocr_clean, db_text)
    if len(ocr_clean) <= 3 and score < base_threshold + 10:
        return "UNKNOWN"
    if score >= base_threshold and partial_score >= 80:
        return db_text
    return "UNKNOWN"
def smart_fuzzy_brand_match_batch(
    texts: List[str],
    brand_list: List[str],
    base_threshold: int = 70
) -> List[Tuple[str, int]]:
    queries = [t.strip().lower() for t in texts]
    thresholds = [base_threshold + 15 if len(q) < 4 else base_threshold for q in queries]
    # Bulk compute pairwise scores >= base_threshold
    raw_matches = process.cdist(
        queries,
        brand_list,
        scorer=fuzz.WRatio,
        processor=lambda s: s,
        score_cutoff=base_threshold
    )
    best_match_score = np.max(raw_matches, axis=1)
    best_match_index = np.argmax(raw_matches, axis=1)
    is_matched = best_match_score > thresholds
    results = []
    for i, is_match in enumerate(is_matched):
        if is_match:
            score = best_match_score[i]
            brand = brand_list[best_match_index[i]]
            results.append((brand, score))
        else:
            results.append(("UNKNOWN", best_match_score[i]))
    return results
# --- Batched fuzzy matcher ---
def robust_fuzzy_match_batch(
    ocr_texts: List[str],
    db_texts: List[str],
    base_threshold:    int = 75,
    min_text_len:      int = 3,
    partial_threshold: int = 80,
    workers:           int = -1
) -> List[str]:
    # Clean and length‐filter
    ocr_clean = [t.strip() for t in ocr_texts]
    lengths   = np.array([len(t) for t in ocr_clean])
    results   = ["UNKNOWN"] * len(ocr_clean)
    # Only match those long enough
    mask = lengths >= min_text_len
    if not mask.any():
        return results
    queries = [ocr_clean[i] for i in np.nonzero(mask)[0]]
    # Compute WRatio & partial_ratio in parallel
    full_scores    = process.cdist(queries, db_texts,    scorer=fuzz.WRatio,        workers=workers)
    partial_scores = process.cdist(queries, db_texts,    scorer=fuzz.partial_ratio, workers=workers)
    # For each query, pick best db index
    best_idx     = full_scores.argmax(axis=1)
    best_full    = full_scores[np.arange(full_scores.shape[0]), best_idx]
    best_partial = partial_scores[np.arange(full_scores.shape[0]), best_idx]
    # Apply thresholds
    accept = (best_full >= base_threshold) & (best_partial >= partial_threshold)
    short  = np.array([len(q) <= 3 for q in queries])
    accept |= (short & (best_full >= base_threshold + 10))
    valid_indices = np.nonzero(mask)[0]
    for qi, orig_i in enumerate(valid_indices):
        if accept[qi]:
            results[orig_i] = db_texts[best_idx[qi]]
    return results
def normalize_digits_letters(s: str) -> str:
    # (?<=\d)(?=[A-Za-z])  = position between digit and letter
    # (?<=[A-Za-z])(?=\d)  = position between letter and digit
    return re.sub(r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)', ' ', s)
def smart_score(a: str, b: str) -> float:
    # print(a, b)
    a = normalize_digits_letters(a.lower().strip())
    b = normalize_digits_letters(b.lower().strip())
    # if b in ['rai', 'omega', 'yakult']:
    #     print("a, b: ", (a, b))
    if len(a.split()) > 1 and len(b.split()) > 1:
        return fuzz.WRatio(a, b)
    if max(len(a), len(b)) <= 6:
        return textdistance.jaro_winkler.normalized_similarity(a, b) * 100
    return fuzz.partial_ratio(a, b)
```

## File: utils/prompts.py
```python
QWEN_TRANSCRIPT_LOGO_PROMPT = lambda transcript: f"""
You are an expert assistant tasked with extracting brand and product names from advertisement transcripts.
Your goal is to extract a **newline-separated list** of all proper brand or product names mentioned in the text below.
✅ Include:
- Pharmaceutical brands
- Digital platforms
- Food and fashion brands
- TV channels or media networks
- Any other **proper named brand entity**
🚫 Exclude:
- Personal names
- Countries or cities
- Sports teams or events (unless branded)
- General nouns or slogans
🧠 Notes:
- Do not include duplicates
- Some brand names may have minor spelling errors or be embedded in longer sentences
Transcript:
\"\"\"{transcript.strip()}\"\"\"
List the brand names below, one per line:
"""
QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT = "You are an expert in analyzing advertisements. You will be given a transcript transcribed from an advertisement video in a non-English language. Your job is to carefully analyze this transcript and give me a thorough and complete list of ALL brand names mentioned in this transcript using both your knowledge, and contextual information present in the transcript."
```

## File: utils/shot_detection.py
```python
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
from scipy.signal import find_peaks
import numpy as np
import cv2
from scenedetect import detect, AdaptiveDetector
class ShotDetector:
    def __init__(self, video_path, stats_path='./video_stats.csv'):
        self.video_path = video_path
        self.stats_path = stats_path
    def detect_scenes(self):
        scenes = detect(self.video_path,
                        AdaptiveDetector(),
                        stats_file_path=self.stats_path,
                        show_progress=True)
        df = pd.read_csv(self.stats_path)
        return df
    def extract_keyframes(self, df, metric='adaptive_ratio (w=2)', threshold='quantile', quantile=0.98):
        x = df[metric].values
        thresh=threshold
        if threshold=='quantile':
            thresh=df[metric].quantile(quantile)
        peaks, _ = find_peaks(x, threshold=thresh)
        frames, frame_numbers = self._read_keyframes(peaks)
        timecodes = df.loc[df['Frame Number'].isin(frame_numbers), 'Timecode'].values
        return frames, frame_numbers, timecodes
    def process_video(self, **kwargs):
        df = self.detect_scenes()
        return self.extract_keyframes(df, **kwargs)
    def _read_keyframes(self, peaks):
        peaks = np.insert(peaks, 0, 0)
        frames = []
        frame_numbers = []
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video")
        amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Total Frames: ", amount_of_frames)
        for i in range(len(peaks)-1):
            frame_no = int((peaks[i] + peaks[i+1])/2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
                frame_numbers.append(frame_no)
        return frames, frame_numbers
def main():
    video_path = "C:/Users/Admin/Desktop/milestone 2/videos/2.mp4"
    stats_path= "C:/Users/Admin/Desktop/milestone 2/videos/video_2_stats.csv"
    detector = ShotDetector(video_path=video_path, stats_path=stats_path)
    frames, frame_numbers, timestamps = detector.process_video()
    print(len(frames), len(frame_numbers), len(timestamps))
if __name__ == "__main__":
    main()
```

## File: utils/video_annotations_to_crops.py
```python
import json
import cv2
import numpy as np
import pandas as pd
import os
import uuid
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from PIL import Image
LS_FPS = 30.0
def process_video_annotations(video_path: str,
                              annotations_json):
    annotations = annotations_json[0]
    metadata = {}
    boxes = annotations['box']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")
    frame_box_map = {}
    for box in boxes:
        label = box['labels'][0]
        for seq in box['sequence']:
            original_frame = seq['frame']
            frame_num = int((original_frame / LS_FPS) * video_fps)
            x = int((seq['x'] / 100) * frame_width)
            y = int((seq['y'] / 100) * frame_height)
            w = int((seq['width'] / 100) * frame_width)
            h = int((seq['height'] / 100) * frame_height)
            frame_box_map.setdefault(frame_num, []).append({
                'label': label,
                'coords': (x, y, w, h),
                'time': seq['time'],
            })
    for frame_num in sorted(frame_box_map.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_num}")
            continue
        for box in frame_box_map[frame_num]:
            x, y, w, h = box['coords']
            label = box['label']
            crop = frame[y:y+h, x:x+w]
            unique_name = f"{uuid.uuid4().hex}.jpg"
            metadata[unique_name] ={
                'frame_number': frame_num,
                'label': label,
                'timestamp': box['time'],
                'crop': Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            }
    cap.release()
    return metadata
```

## File: .gitignore
```
.venv
.gradio
__pycache__
app_video1.py 
pipeline1.py 
video_stats.csv
ARY_Digital_Logo.png
venv_new
logo_audio.wav
logo_index.faiss
metadata.json
```

## File: gradio_app.py
```python
import gradio as gr
from utils.shot_detection import ShotDetector
from pipelines.pipeline import process_video, process_frame_results_merged
import os
def run_pipeline(hf_key, video, threshold, detector_threshold, detector_nms_threshold, use_db: bool, use_translation: bool, use_llm_db: bool, use_qwen_filter: bool):
  if hf_key:
      os.environ["HF_TOKEN"] = hf_key
  else:
      return "Please set a valid Huggingface API Key"
  frame_results = process_video(video, threshold, detector_threshold, detector_nms_threshold, use_db, use_translation, use_llm_db, use_qwen_filter)
  results_df = process_frame_results_merged(frame_results)
  logos = results_df['logo'].values.tolist()
  infos = results_df['info'].values.tolist()
  return list(zip(logos, infos))
DESCRIPTION = "# Automated Logo Detection - Phase 3"
css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Logo Detection"):
        hf_key = gr.Textbox(label="Huggingface_API_KEY", placeholder="Huggingface API KEY", type="password")
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Input Video")
                submit_btn = gr.Button(value="Run Pipeline")
                input_threshold = gr.Slider(
                  label="Video Threshold (Quantile)",
                  info="Larger value will detect fewer keyframes from video and vice versa",
                  minimum=0.80,
                  maximum=0.99,
                  value=0.98,
                  step=0.01)
                detector_threshold = gr.Slider(
                  label="Detector Threshold",
                  info="Larger value will detect fewer logos within a keyframe vice versa",
                  minimum=0.005,
                  maximum=0.1,
                  value=0.03,
                  step=0.005)
                detector_nms_threshold = gr.Slider(
                  label="Detector NMS Threshold",
                  info="Larger value will remove more overlapping logos/regions and vice versa",
                  minimum=0.05,
                  maximum=0.4,
                  value=0.3,
                  step=0.05)
                use_db_checkbox = gr.Checkbox(label="Use Database?", value=False)
                use_llm_db = gr.Checkbox(label="Use Qwen for Database OCR?", value=False)
                use_translation_checkbox = gr.Checkbox(label="Use Engilish Translation?", value=False)
                use_qwen_filter = gr.Checkbox(label="Use Qwen for False Positive Filtering?", value=False)
            with gr.Column():
                output_frames = gr.Gallery(columns=1, label="Keyframes with timestamps", preview=True, show_label=True)
        with gr.Accordion("Instructions"):
          gr.Markdown(
)
        submit_btn.click(run_pipeline, [hf_key, input_video, input_threshold, detector_threshold, detector_nms_threshold, use_db_checkbox, use_translation_checkbox, use_llm_db, use_qwen_filter], [output_frames])
if __name__ == '__main__':
   demo.launch(debug=True)
```

## File: LICENSE
```
MIT License

Copyright (c) 2025 Syed Mustafa Ali Abbasi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## File: README.md
```markdown
# 🎯 Semi-Automated Logo Detection in Brand Advertisement Videos

An application for semi-automated logo detection in brand advertisement videos using multimodal machine learning.

## Overview
![Flowchart](assets/logo_detection_pipeline.png)

This is an application that allow detection of logos in non-English brand advertisment videos using multimodal ML techniques. The overall pipeline is:
- Run the `Whisper` model on audio of the advertisement to transcribe it from the source language (Italian for e.g.) to English.
- Apply an LLM like `Qwen 2.5` to obtain all brand names mentioned in the audio transcript.
- Run _shot detection_ on the video to obtain the most distinct, relevant keyframes.
- On each keyframe, run a _zero-shot object detection_ model such as `OWLv2` with prompt to extract as many _logo-like_ regions as possible i.e. crops that may contain an actual logo.
- Some of the regions that are not logos i.e. false positives are removed using CLIP-based filtering.
- All of the crops/regions then run through the following brand assignment techniques:
  - Use `CLIP` model to assign each region a brand from a list of top ~2000 brands (Netflix, Apple, etc.) obtained publicly from Kaggle.
  - Use `Optical Character Recognition (OCR)` along with `fuzzy string matching` to assign leftover regions a brand from the brand names extracted from the audio.
  - Use `FAISS` vector store to assign leftover regions a brand from the nearest-matching logo in the vector store. The vector store is pre-populated with the [LogoDet-3K](https://github.com/Wangjing1551/LogoDet-3K-Dataset) dataset for now.
- The `FAISS` vector store enables continual learning of new logos over time via human labelling, logo scraping, etc.
- A `Gradio` application allows the user to upload a video, run the pipeline, and view the matched regions/logos with corresponding brand names and timestamps.

## Advantages
- No need for manual training/fine-tuning of object detection models on custom logos.
- Local, indigenous brands detected using the audio transcript with OCR & fuzzy matching.
- CLIP model works quite well for detecting global, popular brands.
- Vector stores like FAISS enable continual learning and detection of new logos over time.
- Overall pipeline is agnostic to the domain (Ads, sports, etc.) and the source language.

 ## Challenges & Limitations
 - Some false positives obtained from OWLv2 pass unfiltered through the remaining pipeline.
 - If a brand is not popular, not present in the vector store, and is not mentioned in the audio explicitly, it's logos will not be detected.
 - Currently, the pipeline runs slower than would be ideal. We are trying to implement optimizations such as replacing CLIP-based filtering with heuristic-based filtering, replacing EasyOCR with PaddleOCR, and using batched version of RapidFuzz for fuzzy matching.

## Usage
1. Clone the Repository.
2. Set up and activate a python virtual environment (optional).
3. Run `pip install requirements.txt` to install the necessary dependencies.
4. Run `gradio gradio_app.py` to launch the gradio UI.
6. Upload a short brand advertisement video of your choice.
7. Type or copy-paste your `HuggingFace Hub` API key. You can obtain this for free.
8. Click on `Run Pipeline` to run the pipeline.
9. View the logos detected with corresponding brand names and timestamps on the right after pipeline finishes.
```
