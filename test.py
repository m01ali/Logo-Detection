import torch
import numpy as np
import supervision as sv
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import cv2
import os
import colorsys
from IPython.display import Video

class LogoDetector:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.queries = [
            # --- Broadcasting & Digital Overlays ---
            "a television network watermark",
            "a channel logo in the corner of the screen",
            "a digital graphic logo",
            "a sports broadcast logo",

            # --- Sports & Apparel (Tennis, Volleyball, Billiards) ---
            "a brand logo on a sports shirt",
            "a sponsor logo on athletic shorts",
            "a logo on a tennis racket",
            "a sportswear brand logo",
            "a sponsor logo on a sports uniform",

            # --- Advertising Boards & Venues ---
            "an advertising board sponsor logo",
            "a logo printed on a sports court surface",
            "a sponsor logo on a billiard table border",
            "a tournament logo on a wall",
            "a brand name printed on a banner",

            # --- Products & Commercial Packaging (Food, Meds, Toys) ---
            "a brand name on medicine packaging",
            "a company logo on a food box",
            "a brand logo on a frozen food bag",
            "a toy brand logo",
            "a corporate logo on a product label",
            "a logo on a security camera device",

            # --- General / Fallback (EXPANDED) ---
            "a corporate trademark",
            "a small brand icon",
            "a company symbol",
            "a text-based brand logo",
            "a commercial brand mark",
            "an enterprise emblem",
            "a distinctive brand insignia",
            "a recognizable company badge",
            "a generic logo",
            "an organization's crest",
            "a commercial watermark",
            "a manufacturer's mark",
            "an official logo graphic",
            "a product",
            "a potential logo",
            "a company",
            "a brand"
        ]

    def run_example(self, image: Image.Image, threshold=0.1, nms_threshold=0.3):
        texts = [self.queries]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)

        results = self.processor.image_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )

        scores = results[0]['scores'].cpu().numpy().reshape(-1, 1)
        boxes = results[0]['boxes'].cpu().numpy()
        labels = results[0]['labels'].cpu().numpy()

        if len(boxes) == 0:
            return {'boxes': np.empty((0, 4), dtype=np.float32), 'scores': np.empty((0,), dtype=np.float32), 'labels': np.empty((0,), dtype=np.int64)}

        detections = np.hstack((boxes, scores))
        keep_idxs = sv.box_non_max_suppression(detections, iou_threshold=nms_threshold)

        return {
            'boxes': boxes[keep_idxs],
            'scores': scores.flatten()[keep_idxs],
            'labels': labels[keep_idxs]
        }

    def draw_boxes(self, image: Image.Image, result: dict):
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        logos = []

        img_w, img_h = image.size
        for box in boxes:
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
            
            # --- Integrated Clamping Logic from test.py ---
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            if x2 > x1 and y2 > y1:
                logos.append(image.crop((x1, y1, x2, y2)))

        return logos, scores, boxes, labels

    def process_image(self, image: Image.Image, threshold: float = 0.1, nms_threshold: float = 0.3):
        result = self.run_example(image, threshold=threshold, nms_threshold=nms_threshold)
        logos, scores, boxes, labels = self.draw_boxes(image, result)
        return logos, scores, boxes, labels

# --- Helper Functions ---
def generate_distinct_bgr_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

def pick_crop_dir(preferred_dir="/content/cropped-logos/owlv2"):
    os.makedirs(preferred_dir, exist_ok=True)
    return preferred_dir


# 1. Initialize detector
print("Initializing detector...")
detector = LogoDetector()
COLORS = generate_distinct_bgr_colors(len(detector.queries))

# 2. Define video paths and directories
input_video_path = "/content/video_3 (online-video-cutter.com).mp4"
output_video_path = "owlv2_output_raw.mp4"
final_video_path = "owlv2_output_web.mp4"
crop_output_dir = pick_crop_dir() # Set to /content/cropped-logos/owlv2
file_name = os.path.splitext(os.path.basename(input_video_path))[0]

# 3. Setup OpenCV VideoCapture and VideoWriter
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps is None or fps <= 0:
    fps = 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing {total_frames} frames...")
print(f"Saving crops in: {crop_output_dir}")

# 4. Process the video
frame_count = 0
saved_crops = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processing frame {frame_count}/{total_frames}...")

    # Convert OpenCV BGR frame to PIL RGB Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Process frame through the detector
    logos, scores, boxes, labels = detector.process_image(pil_image, threshold=0.15, nms_threshold=0.3)

    # --- Integrated Saving Logic from test.py ---
    for crop_idx, logo_crop in enumerate(logos, start=1):
        crop_filename = f"{frame_count}_{crop_idx}_{file_name}.png"
        crop_path = os.path.join(crop_output_dir, crop_filename)
        logo_crop.save(crop_path)
        saved_crops += 1

    # Draw boxes and labels
    for box, score, label_idx in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]

        box_color = COLORS[label_idx]
        matched_prompt = detector.queries[label_idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        label_text = f"{matched_prompt}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + text_width, y1), box_color, -1)

        brightness = (box_color[0]*114 + box_color[1]*587 + box_color[2]*299) / 1000
        text_color = (0, 0, 0) if brightness > 125 else (255, 255, 255)

        cv2.putText(frame, label_text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    out.write(frame)

cap.release()
out.release()
print(f"Video processing complete! Total crops saved: {saved_crops}")

# 5. Convert to H.264 for playback
print("Converting video for web playback...")
os.system(f"ffmpeg -y -hide_banner -loglevel error -i {output_video_path} -vcodec libx264 {final_video_path}")

# 6. Display the video inline
print("Done!")
Video(final_video_path, embed=True, width=640)