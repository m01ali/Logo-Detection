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

# --- Label Studio assumed FPS ---
LS_FPS = 30.0
    
def process_video_annotations(video_path: str,
                              annotations_json):

    # --- Load annotations ---
    # with open(annotations_path, 'r') as f:
    #     annotations = json.load(f)[0]
    
    annotations = annotations_json[0]

    # --- Setup output directory ---
    # output_dir = 'cropped_frames'
    # os.makedirs(output_dir, exist_ok=True)

    # --- Metadata dictionary ---
    metadata = {}

    # --- Process each annotation task (video) ---
    boxes = annotations['box']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")

    # --- Map frame numbers to box info ---
    frame_box_map = {}

    for box in boxes:
        label = box['labels'][0]
        for seq in box['sequence']:
            # if not seq['enabled']:
            #     continue
            # frame_num = seq['frame']
            
            # Correct frame number from LS-FPS to actual FPS
            original_frame = seq['frame']
            frame_num = int((original_frame / LS_FPS) * video_fps)

            # Convert 0–100 scale (in Label Studio) to pixel values
            x = int((seq['x'] / 100) * frame_width)
            y = int((seq['y'] / 100) * frame_height)
            w = int((seq['width'] / 100) * frame_width)
            h = int((seq['height'] / 100) * frame_height)

            frame_box_map.setdefault(frame_num, []).append({
                'label': label,
                'coords': (x, y, w, h),
                'time': seq['time'],
            })
            
    # --- Process only annotated frames ---
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

            # Generate unique filename
            unique_name = f"{uuid.uuid4().hex}.jpg"
            # filepath = os.path.join(output_dir, unique_name)

            # Save cropped image
            # cv2.imwrite(filepath, crop)

            # Store metadata
            metadata[unique_name] ={
                'frame_number': frame_num,
                'label': label,
                'timestamp': box['time'],
                'crop': Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            }

    cap.release()
    
    return metadata