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
# import albumentations as A





def resize_images(images):
    """
    Resize each PIL image in the list if its width or height is less than 28 pixels.
    The image is scaled up uniformly so that both dimensions are at least 28 pixels,
    preserving the original aspect ratio.

    Parameters:
        images (list): A list of PIL.Image objects.
    
    Returns:
        list: A new list of PIL.Image objects with images resized if necessary.
    """
    resized = []
    for img in images:
        width, height = img.size
        if width < 28 or height < 28:
            # Calculate the scale factor to ensure both dimensions are at least 28.
            scale = max(28 / width, 28 / height)
            new_size = (math.ceil(width * scale), math.ceil(height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        resized.append(img)
    return resized

def extract_audio_brands(video_path, 
                         out_dir=None,
                         mode: str = "translate",   # "transcribe" OR "translate"
                         ):
    qwen_text_model = Qwen2_5TextModel()
    whisper = WhisperModel()

    if mode not in ("transcribe", "translate"):
        raise ValueError(f"Invalid mode '{mode}', expected 'transcribe' or 'translate'")

    out = Path(out_dir) if out_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)

    #convert video to audio file
    convert_video_to_audio(video_path)

    #transcript extraction
    # transcript, _ = whisper.run_example('logo_audio.wav')
    # if transcript:
    #     (out / "transcript.en.txt").write_text(transcript, encoding="utf-8")

    # 2) Transcript (either original language OR English)
    if mode == "transcribe":
        transcript, chunks_df, lang = whisper.run_example(
            'logo_audio.wav',
            mode="transcribe",
            language=None,          # auto-detect
            return_language=True,
        )
        lang = lang or "orig"
        (out / f"transcript.{lang}.txt").write_text(transcript, encoding="utf-8")

    else:  # mode == "translate"
        transcript, chunks_df, _ = whisper.run_example(
            'logo_audio.wav',
            mode="translate",
            language="english",
            return_language=True,
        )
        lang='english'
        (out / "transcript.en.txt").write_text(transcript, encoding="utf-8")

    # print(transcript)
    brand_string = qwen_text_model.run_example(
        QWEN_TRANSCRIPT_LOGO_PROMPT(transcript),
        QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT
    )
    audio_brands = brand_string.split('\n')
    audio_brands = [name.lower().strip() for name in audio_brands]

    (out / f"brands_audio_{lang}.txt").write_text("\n".join(sorted(set(audio_brands))), encoding="utf-8")

    return audio_brands

def filter_logos_with_clip(logo_images: List[LogoImage], clip_model, positive_prompts: List[str], negative_prompts: List[str]) -> List[LogoImage]:
    """
    Filters out noisy logos using CLIP model and clustering.
    
    Parameters:
        logo_images: List of LogoImage objects.
        clip_model: CLIP model instance to run.
        positive_prompts: List of positive prompts for CLIP filtering.
        negative_prompts: List of negative prompts for CLIP filtering.
    
    Returns:
        List of LogoImage objects after filtering.
    """
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
    
    # if not sorted_clusters:
    #     # No clusters found, return all logos
    #     return logo_images

    exclude_cluster = sorted_clusters[0]

    # Identify logos to keep (not in the excluded cluster)
    include_logo_indices = clip_filter_results_df.loc[
        clip_filter_results_df['cluster'] != exclude_cluster, 'image_path'
    ].values
    include_logo_indices = [int(idx) for idx in include_logo_indices]

    filtered_logo_images = [logo_images[i] for i in include_logo_indices]

    return filtered_logo_images


# def annotate_video(video_path, frame_results, output_path):
#     """
#     Draws bounding boxes, brand names, percent areas, and timestamps
#     onto each frame where logos were detected, and writes _all_ frames
#     (annotated or not) into a new video.
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video {video_path}")

#     fps    = cap.get(cv2.CAP_PROP_FPS)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Build a lookup: frame_number → detection info
#     det_by_frame = { fr['frame_number']: fr for fr in frame_results }

#     frame_no = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # If we have detections for this frame, draw them
#         if frame_no in det_by_frame:
#             fr = det_by_frame[frame_no]
#             timestamp = fr['timestamp']
#             for logo_id, info in fr['results'].items():
#                 x1, y1, x2, y2 = map(int, info['bbox'])
#                 brand         = info['brand']
#                 pct           = info['percent_area']

#                 if brand == "UNKNOWN":
#                     continue

#                 # draw box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#                 # label above box
#                 label = f"{brand} {pct:.1f}%"
#                 cv2.putText(
#                     frame, label,
#                     (x1, y1-8),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0,255,0), 1
#                 )

#             # optional: timestamp on every annotated frame
#             cv2.putText(
#                 frame, f"Time: {timestamp}",
#                 (10, height-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5, (255,255,255), 1
#             )

#         # --- Write _every_ frame ---
#         out.write(frame)
#         frame_no += 1

#     cap.release()
#     out.release()
#     print(f"Annotated video with all {frame_no} frames saved to {output_path}")

# def annotate_video(video_path, frame_results, output_path, window=3):
#     """
#     Draws bounding boxes, brand names, percent areas, and timestamps
#     onto each frame where logos were detected—and also on the +/- `window`
#     frames around each detection. Writes all frames into a new video.
#     """
#     # --- Open and read all frames ---
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video {video_path}")

#     fps         = cap.get(cv2.CAP_PROP_FPS)
#     width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()

#     # --- Build extended detection mapping ---
#     # maps frame_idx -> list of detections to draw
#     # each detection: dict with keys 'bbox', 'brand', 'percent_area', 'timestamp'
#     extended = {}
#     for fr in frame_results:
#         F = fr['frame_number']
#         # for each offset frame in [F-window .. F+window]
#         for f in range(max(0, F - window), min(len(frames), F + window + 1)):
#             for info in fr['results'].values():
#                 # skip unknowns if you like
#                 if info.get('brand', 'UNKNOWN') == 'UNKNOWN':
#                     continue
#                 extended.setdefault(f, []).append({
#                     'bbox':         info['bbox'],
#                     'brand':        info['brand'],
#                     'percent_area': info['percent_area'],
#                     'timestamp':    fr['timestamp']
#                 })

#     # --- Prepare VideoWriter ---
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # --- Annotate & write frames ---
#     for idx, frame in enumerate(frames):
#         if idx in extended:
#             for det in extended[idx]:
#                 x1, y1, x2, y2 = map(int, det['bbox'])
#                 brand, pct     = det['brand'], det['percent_area']

#                 # draw box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                 # draw label background
#                 label = f"{brand} {pct:.1f}%"
#                 (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#                 cv2.rectangle(
#                     frame,
#                     (x1, y1 - th - baseline - 4),
#                     (x1 + tw + 4, y1),
#                     (0, 0, 255),
#                     thickness=-1
#                 )
#                 # draw label text
#                 cv2.putText(
#                     frame, label,
#                     (x1 + 2, y1 - baseline - 2),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0, 0, 0), 1
#                 )

#             # draw a timestamp once per frame (if desired)
#             ts = extended[idx][0]['timestamp']
#             cv2.putText(
#                 frame, f"Time: {ts}",
#                 (10, height - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5, (255, 255, 255), 1
#             )

#         out.write(frame)

    # out.release()
    # print(f"Annotated video with all {len(frames)} frames saved to {output_path}")


# def annotate_video(video_path, frame_results, output_path, window=3):
#     """
#     Draws bounding boxes, brand names, percent areas, and timestamps
#     onto each frame where logos were detected—and also on the +/- `window`
#     frames around each detection.  Writes ONLY those annotated frames
#     (not the whole video) into a new video.
#     """
#     # --- Read all frames into memory ---
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video {video_path}")

#     fps          = cap.get(cv2.CAP_PROP_FPS)
#     width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()

#     # --- Build extended detection mapping (±window frames) ---
#     extended = {}
#     for fr in frame_results:
#         F = fr['frame_number']
#         for f in range(max(0, F - window), min(len(frames), F + window + 1)):
#             for info in fr['results'].values():
#                 if info.get('brand', 'UNKNOWN') == 'UNKNOWN':
#                     continue
#                 extended.setdefault(f, []).append({
#                     'bbox':         info['bbox'],
#                     'brand':        info['brand'],
#                     'percent_area': info['percent_area'],
#                     'timestamp':    fr['timestamp']
#                 })

#     # If there are no detections, bail out early
#     if not extended:
#         print("No detections found; no output video created.")
#         return

#     # --- Prepare per-brand color assignment ---
#     palette = [
#         (255, 0, 0),    # Blue
#         (0, 255, 0),    # Green
#         (0, 0, 255),    # Red
#         (255, 255, 0),  # Cyan
#         (255, 0, 255),  # Magenta
#         (0, 255, 255),  # Yellow
#         (255, 127, 0),  # Orange
#         (127, 0, 255),  # Purple
#     ]
#     brand_colors = {}
#     next_color = 0

#     # --- VideoWriter for only annotated frames ---
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # --- Annotate & write only selected frames ---
#     annotated_count = 0
#     for idx, frame in enumerate(frames):
#         if idx not in extended:
#             continue

#         for det in extended[idx]:
#             x1, y1, x2, y2 = map(int, det['bbox'])
#             brand, pct     = det['brand'], det['percent_area']

#             # get or assign color
#             if brand not in brand_colors:
#                 brand_colors[brand] = palette[next_color % len(palette)]
#                 next_color += 1
#             color = brand_colors[brand]

#             # draw box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             # label background
#             label = f"{brand} {pct:.1f}%"
#             (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             cv2.rectangle(
#                 frame,
#                 (x1, y1 - th - baseline - 4),
#                 (x1 + tw + 4, y1),
#                 color,
#                 thickness=-1
#             )

#             # choose text color
#             brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
#             text_color = (0,0,0) if brightness > 128 else (255,255,255)
#             cv2.putText(
#                 frame, label,
#                 (x1 + 2, y1 - baseline - 2),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5, text_color, 1
#             )

#         # draw timestamp
#         ts = extended[idx][0]['timestamp']
#         cv2.putText(
#             frame, f"Time: {ts}",
#             (10, height - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5, (255, 255, 255), 1
#         )

#         # write this one annotated frame
#         out.write(frame)
#         annotated_count += 1

#     out.release()
#     print(f"Saved {annotated_count} annotated frames to {output_path}")

def annotate_video(video_path, frame_results, output_path, window=3):
    """
    Draws bounding boxes, brand names, percent areas, and timestamps
    onto each frame where logos were detected—and also on the +/- `window`
    frames around each detection.  Writes ONLY those annotated frames
    (not the whole video) into a new video.
    """
    # --- Read all frames into memory ---
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

    # --- Build extended detection mapping (±window frames) ---
    extended = {}
    for fr in frame_results:
        F = fr['frame_number']
        for f in range(max(0, F - window), min(len(frames), F + window + 1)):
            for info in fr['results'].values():
                # REMOVED: No longer skip unknown brands
                extended.setdefault(f, []).append({
                    'bbox':         info['bbox'],
                    'brand':        info['brand'],
                    'percent_area': info['percent_area'],
                    'timestamp':    fr['timestamp']
                })

    # If there are no detections, bail out early
    if not extended:
        print("No detections found; no output video created.")
        return

    # --- Prepare per-brand color assignment ---
    palette = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 127, 0),  # Orange
        (127, 0, 255),  # Purple
    ]
    brand_colors = {}
    next_color = 0

    # --- VideoWriter for only annotated frames ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Annotate & write only selected frames ---
    annotated_count = 0
    for idx, frame in enumerate(frames):
        if idx not in extended:
            continue

        for det in extended[idx]:
            x1, y1, x2, y2 = map(int, det['bbox'])
            brand, pct     = det['brand'], det['percent_area']

            # Use grey color for unknown brands, assign color for known brands
            if brand == 'UNKNOWN':
                color = (128, 128, 128)  # Grey color for unknown brands
            else:
                if brand not in brand_colors:
                    brand_colors[brand] = palette[next_color % len(palette)]
                    next_color += 1
                color = brand_colors[brand]

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # label background (only for known brands)
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

                # choose text color
                brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
                text_color = (0,0,0) if brightness > 128 else (255,255,255)
                cv2.putText(
                    frame, label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, text_color, 1
                )
            else:
                # Optional: Add a simple label for unknown brands without background
                label = "unknown"
                cv2.putText(
                    frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (128, 128, 128), 1  # Grey text
                )

        # draw timestamp
        ts = extended[idx][0]['timestamp']
        cv2.putText(
            frame, f"Time: {ts}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        # write this one annotated frame
        out.write(frame)
        annotated_count += 1

    out.release()
    print(f"Saved {annotated_count} annotated frames to {output_path}")

# # def process_video(video_path, threshold, use_db=False):
# #     print('use_db', use_db)
# #     shot_detector = ShotDetector(video_path)
# #     detector  = LogoDetector()
# #     clip_model = CLIPModel()
# #     filter_pipeline = FilterPipeline([
# #         area_aspect_filter,
# #         texture_filter,
# #         edge_density_filter,
# #         color_variance_filter,
# #     ])

# #     # qwen_model = Qwen2_5VLModel()

# #     db_index_path =  "D:\\milestone 2\\faiss_db\\logo_index.faiss"
# #     metadata_path = "D:\\milestone 2\\faiss_db\\metadata.json"
# #     db = LogoDatabaseNew(db_index_path, metadata_path, batch_size=64)

#     # audio_brands = extract_audio_brands(video_path)
#     audio_brands = ['sky', 'neoborocillina', 'storeroom', 'merluzzo', 'nurofen', 'monifarma', 'moneypharm', 'verishure', 'verisure', 'barbie', 'chanted evening', 'orogel', 'rai sport', 'rai 2', 'guinness world records']
#     # print(audio_brands)
    
#     #Video Shot Detection
#     frames, frame_numbers, timestamps = shot_detector.process_video(quantile=threshold)
#     print('Total Frames:', len(frames))

#     #Object detection on all frames
#     detector_results = detector.process_images(frames, threshold=0.03, nms_threshold=0.3, batch_size=16)
#     # print(len(detector_results))
#     # for result in detector_results:
#     #     print(len(result['logos']))
    
#     frame_results = []
#     for result, timestamp, frame_no in tqdm(zip(detector_results, timestamps, frame_numbers)):
        
#         logos, scores = result['logos'], result['scores']
#         print("Total logos in frame: ", len(logos))
#         if len(logos) <= 0:
#             continue

#         #Create LogoImage objects
#         logo_images = []
#         for idx, (logo, score) in enumerate(zip(logos, scores)):
#             metadata = {
#                 'frame_number': frame_no,
#                 'timestamp': timestamp,
#                 'logo_index': idx,
#                 'score': score,
#             }
#             logo_image = LogoImage.create(logo, metadata)
#             logo_images.append(logo_image)
        
#         #Filter out easy examples using CLIP
#         if len(logo_images) >= 10:
#             # logo_images = filter_logos_with_clip(logo_images, clip_model, positive_prompts, negative_prompts)
#             logo_images = filter_pipeline.apply(logo_images)
#             print(f"Filtered logos kept for frame {frame_no}: {len(logo_images)}")
            
#         if len(logo_images) == 0:
#             continue
    
#         # Then run recognizer (same as earlier)
#         techniques = [
#             ClipTechnique(clip_model, clip_threshold=0.8),
#             OcrFuzzyBatchTechnique(get_ocr_image, smart_fuzzy_brand_match_batch, audio_brands, fuzzy_threshold=0.8),
#             # OcrFuzzyTechnique(get_ocr_image, smart_fuzzy_brand_match, audio_brands, fuzzy_threshold=0.8),
#             # DatabaseFaissTechnique(db, get_ocr_image),
#         ]

#         if use_db:
#             techniques.append(DatabaseFaissTechniqueBatch(db, get_ocr_image))

#         recognizer = BrandRecognizer(techniques)
#         frame_recognition_results = recognizer.recognize(logo_images)

#         frame_results.append({
#             'frame_number': frame_no,
#             'timestamp': timestamp,
#             'results': frame_recognition_results
#         })

#     print(frame_results)
#     return frame_results

# def process_video(video_path, threshold, use_db=False):
#     print('use_db', use_db)
#     # --- Prepare output dirs & CSV accumulator ---
#     base = os.path.splitext(os.path.basename(video_path))[0]
#     sanitized = re.sub(r'[^A-Za-z0-9_\-]', '_', base)
#     unknown_dir = os.path.join(sanitized, "unknowns")
#     known_dir   = os.path.join(sanitized, "knowns")       
#     os.makedirs(unknown_dir, exist_ok=True)
#     os.makedirs(known_dir, exist_ok=True)         

#     detected_rows = []  # will collect rows for CSV
#     frame_results = []

#     # --- Initialize your components ---
#     shot_detector   = ShotDetector(video_path)
#     detector        = LogoDetector()
#     clip_model      = CLIPModel()

#     #Heuristic-based filtering of false positives
#     filter_pipeline = FilterPipeline([
#         area_aspect_filter,
#         texture_filter,
#         edge_density_filter,
#         color_variance_filter,
#     ])

#     db = LogoDatabaseNew(
#         index_path="D:\\milestone 2\\faiss_database_with_italian_logos\\logo_index.faiss",
#         metadata_path="D:\\milestone 2\\faiss_database_with_italian_logos\\metadata.json",
#         batch_size=64
#     )

#     audio_brands = extract_audio_brands(video_path)
#     # audio_brands = [
#     #     'sky','neoborocillina','storeroom','merluzzo','nurofen',
#     #     'monifarma','moneypharm','verishure','verisure','barbie',
#     #     'chanted evening','orogel','rai sport','rai 2','guinness world records'
#     # ]
#     print(audio_brands)

#     # 1) Shot detection → get frames, numbers, timestamps
#     frames, frame_numbers, timestamps = shot_detector.process_video(quantile=threshold)
#     print('Total Frames:', len(frames))

#     # 2) Logo detection on all frames (batched)
#     detector_results = detector.process_images(
#         frames,
#         threshold=0.03,
#         nms_threshold=0.3,
#         batch_size=16
#     )

#     # 3) Per-frame loop
#     for frame, result, timestamp, frame_no in tqdm(
#         zip(frames, detector_results, timestamps, frame_numbers),
#         total=len(detector_results),
#         desc="Processing frames"
#     ):
#         logos, scores = result['logos'], result['scores']
#         if not logos:
#             continue

#         # Compute frame area once
#         if isinstance(frame, Image.Image):
#             fw, fh = frame.size
#         else:  # assume numpy array HxWxC
#             fh, fw = frame.shape[:2]
#         frame_area = fw * fh

#         # 3a) Build LogoImage list *with* percent_area in metadata
#         logo_images = []
#         for idx, (crop_img, det_score) in enumerate(zip(logos, scores)):
#             w, h = (crop_img.size if isinstance(crop_img, Image.Image)
#                     else (crop_img.shape[1], crop_img.shape[0]))
#             percent_area = (w * h) / frame_area * 100.0

#             metadata = {
#                 'frame_number':  frame_no,
#                 'timestamp':     timestamp,
#                 'logo_index':    idx,
#                 'detection_score': det_score,
#                 'percent_area':  percent_area,
#             }
#             logo_images.append(LogoImage.create(crop_img, metadata))

#         # 3b) Optional filtering
#         if len(logo_images) >= 10:
#             logo_images = filter_pipeline.apply(logo_images)

#         if not logo_images or len(logo_images)==0:
#             continue

#         # 3c) Recognition via your stacked techniques
#         techniques = [
#             ClipTechnique(clip_model, clip_threshold=0.8),
#             OcrFuzzyBatchTechnique(get_ocr_image, smart_fuzzy_brand_match_batch, audio_brands, fuzzy_threshold=0.8),
#         ]
#         if use_db:
#             techniques.append(DatabaseFaissTechniqueBatch(db, get_ocr_image))

#         recognizer = BrandRecognizer(techniques)
#         recs = recognizer.recognize(logo_images)

#         # 3d) Enrich recs with percent_area, save unknowns & collect CSV rows
#         for logo in logo_images:
#             info = recs[logo.id]
#             # add percent_area to the result
#             info['percent_area'] = logo.metadata['percent_area']

#             if info['brand'] == 'UNKNOWN':
#                 # save unknown crop
#                 fname = f"frame{frame_no}_logo{logo.metadata['logo_index']}.png"
#                 logo.image.save(os.path.join(unknown_dir, fname))
#             else:
#                 # save known crop
#                 fname = f"{logo.id}.png"                             # ← new
#                 logo.image.save(os.path.join(known_dir, fname))      # ← new
#                 # queue for CSV
#                 detected_rows.append({
#                     'frame_number':  logo.metadata['frame_number'],
#                     'timestamp':     logo.metadata['timestamp'],
#                     'brand':         info['brand'],
#                     'percent_area':  logo.metadata['percent_area'],
#                     'filename':     fname,
#                 })

#         # 3e) Append to overall frame_results
#         frame_results.append({
#             'frame_number': frame_no,
#             'timestamp':    timestamp,
#             'results':      recs
#         })

#     # 4) Export known detections to CSV
#     if detected_rows:
#         df = pd.DataFrame(detected_rows)
#         # csv_path = os.path.join(sanitized, f"{sanitized}_detected_logos.csv")
#         csv_path = os.path.join(known_dir, f"{sanitized}_detected_logos.csv")
#         df.to_csv(csv_path, index=False)
#         print(f"▶ Exported {len(df)} detections to {csv_path}")
    
#     print(frame_results)
#     return frame_results

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

    # --- Prepare output dirs & CSV accumulator ---
    base = os.path.splitext(os.path.basename(video_path))[0]
    sanitized = re.sub(r'[^A-Za-z0-9_\-]', '_', base)

    #top level results folder for output
    output_root = os.path.join("results", sanitized)
    os.makedirs(output_root, exist_ok=True)

    unknown_dir = os.path.join(output_root, "unknowns")
    known_dir   = os.path.join(output_root, "knowns")
    os.makedirs(unknown_dir, exist_ok=True)
    os.makedirs(known_dir, exist_ok=True)

    # NEW: audio folder
    audio_dir = os.path.join(output_root, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    detected_rows = []  # will collect rows for CSV
    frame_results = []

    # --- Initialize your components ---
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

    # 1) Shot detection → get frames, numbers, timestamps
    frames, frame_numbers, timestamps = shot_detector.process_video(quantile=threshold)
    print('Total Frames:', len(frames))

    # 2) Logo detection on all frames (batched)
    detector_results = detector.process_images(
        frames,
        threshold=detector_threshold,
        nms_threshold=nms_threshold,
        batch_size=16
    )

    # 3c) Recognition via your stacked techniques
    techniques = [
        ClipTechnique(clip_model, clip_threshold=0.80),
        OcrFuzzyBatchTechnique(get_ocr_image, smart_fuzzy_brand_match_batch, audio_brands, fuzzy_threshold=0.8),
    ]
    if use_db:
        techniques.insert(0, DatabaseFaissTechniqueBatch(db, 
                                                         get_ocr_image,
                                                         use_qwen_ocr=use_llm_db, #Use Qwen for OCR if needed
                                                         ))

    recognizer = BrandRecognizer(techniques)

    # 3) Per-frame loop
    for frame, result, timestamp, frame_no in tqdm(
        zip(frames, detector_results, timestamps, frame_numbers),
        total=len(detector_results),
        desc="Processing frames"
    ):
        logos, scores, boxes = result['logos'], result['scores'], result['boxes']
        if not logos:
            continue

        # Compute frame area once
        if isinstance(frame, Image.Image):
            fw, fh = frame.size
        else:
            fh, fw = frame.shape[:2]
        frame_area = fw * fh

        # 3a) Build LogoImage list *with* bbox + percent_area in metadata
        logo_images = []
        for idx, (crop_img, det_score, box) in enumerate(zip(logos, scores, boxes)):
            # round & unpack
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]

            # compute percent_area
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

        # 3b) Optional filtering
        if len(logo_images) >= 10:
            logo_images = filter_pipeline.apply(logo_images)

        if not logo_images:
            continue

        # Run recognizer to assign brand names
        recs = recognizer.recognize(logo_images)

        # 3d) Enrich recs with bbox + percent_area, save unknowns & collect CSV rows
        for logo in logo_images:
            info = recs[logo.id]
            # propagate metadata fields
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

        # 3e) Append to overall frame_results
        # Now each info dict inside recs[logo_id] has a 'bbox' key
        frame_results.append({
            'frame_number': frame_no,
            'timestamp':    timestamp,
            'results':      recs
        })

    
    # --- after the big for-loop where you fill `detected_rows` & `frame_results` ---

    if use_qwen_filter:
        # NEW: Qwen post-filter (ONLY for Top_2K_Brands by default)
        # Build/Reuse the Qwen model (you already have this class)
        qwen_model = Qwen2_5VLModel(model_id="Qwen/Qwen2.5-VL-7B-Instruct")

        qwen_top2k_strategy = QwenCorrectnessStrategy(
            model=qwen_model,
            sources_to_filter={"Top_2K_Brands", "audio"},      # <- extend later e.g., {"Top_2K_Brands", "audio", "database"}
            prompt_template=ocr_check_prompt,
            image_size=(240, 240),
            batch_size=16,
            model_kwargs={"temperature": 0.1, "do_sample": True},
            # model_kwargs={}, #no temperature enables greedy decoding
        )

        # Apply post-filters (list allows you to stack more strategies later)
        filtered_rows, annotated_rows, frame_results, filtered_frame_results = apply_post_filters(
            detected_rows,
            image_dir=known_dir,
            frame_results=frame_results,
            strategies=[qwen_top2k_strategy],
        )

        # OPTIONAL: Save a diagnostics "keep-all" CSV with verdicts for analysis/debugging
        if annotated_rows:
            df_all = pd.DataFrame(annotated_rows)
            all_csv = os.path.join(known_dir, f"{sanitized}_all_detections_with_verdicts.csv")
            df_all.to_csv(all_csv, index=False)
            print(f"▶ Exported (all detections w/ verdicts) to {all_csv}")

        # 4) Export filtered known detections to CSV (only 'Correct' for Top_2K_Brands; others unchanged)
        if filtered_rows:
            df = pd.DataFrame(filtered_rows)
            csv_path = os.path.join(known_dir, f"{sanitized}_detected_logos.csv")
            df.to_csv(csv_path, index=False)
            print(f"▶ Exported {len(df)} detections to {csv_path}")
        
        frame_results = filtered_frame_results #use filtered detections only

    else:

        # 4) Export known detections to CSV
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
    """
    Processes frame_results by merging multiple detections of the same brand across frames.

    Parameters:
        frame_results: List of frame-wise recognition results.
        default_end_offset: Default duration if no end detected.

    Returns:
        A pandas DataFrame with columns:
        ['logo', 'brand_name', 'start_timestamp', 'end_timestamp', 'source']
    """
    brand_records = {}

    for frame_result in frame_results:
        frame_timestamp = frame_result['timestamp']
        frame_items = frame_result['results']

        for logo_id, data in frame_items.items():
            brand = data['brand']
            if brand == 'UNKNOWN':
                continue  # Skip unknowns

            source = data.get('source', 'unknown')  # NEW: capture source if present

            if brand not in brand_records:
                # First time seeing this brand
                brand_records[brand] = {
                    'logo': data['image'],
                    'start_timestamp': frame_timestamp,
                    'end_timestamp': frame_timestamp,
                    'source': source
                }
            else:
                # Update end timestamp if brand seen again
                brand_records[brand]['end_timestamp'] = frame_timestamp
                # If multiple sources appear, you could overwrite or join
                if brand_records[brand]['source'] != source:
                    brand_records[brand]['source'] += f", {source}"

    # Convert to DataFrame
    records = []
    for brand, info in brand_records.items():
        records.append({
            'logo': info['logo'],
            'brand_name': brand,
            'start_timestamp': info['start_timestamp'],
            'end_timestamp': info['end_timestamp'],
            'source': info['source'],  # NEW
        })

    df = pd.DataFrame(records)

    if not df.empty:
        df['brand_name'] = df['brand_name'].str.replace('\n', ' or ')
        df['info'] = (
            "Brand Name(s): " + df['brand_name'] +
            # ", Source: " + df['source'] +
            ", Start Timestamp: " + df['start_timestamp'].astype(str) +
            ", End Timestamp: " + df['end_timestamp'].astype(str)
        )

    return df



def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    video_path = "../videos/1.mp4"
    # stats_path= "C:/Users/Admin/Desktop/milestone 2/videos/video_2_stats.csv"
    process_video(video_path, 0.98)
    
if __name__ == "__main__":
    main()
        