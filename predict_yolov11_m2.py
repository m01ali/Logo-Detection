# =============================================================================
# M2 — Custom YOLOv11 Logo Detector — Predict on a video (local)
# =============================================================================
# Converted 1:1 from Cell 9 ("Predict on new images / a folder") of the
# "M2 - Yolov11" section of Notebooks/Yolov11-M2.ipynb — same
# best.predict(source=..., save=True, imgsz=IMG_SIZE, conf=0.25) call, just
# pointed at a local video instead of a Colab image folder, and given a
# project/name so the output lands in a predictable local folder.
#
# Requires the trained weights (best.pt) downloaded locally — see
# docs/yolov11-m2-validation/README.md for where Cell 10 of
# train_yolov11_m2.py saves them on Drive.
#
# Usage:
#   python predict_yolov11_m2.py --weights weights\best.pt --video video_3-trimmed.mp4
#
# Output:
#   <output-dir>/<video_stem>/<video_stem>.avi   — annotated video (boxes + class + conf)
#   <output-dir>/<video_stem>/labels/*.txt       — per-frame YOLO-format detections
# =============================================================================


# ── Cell 1 — Config (from the command line instead of hardcoded paths) ──────
# %%
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Run the M2 custom YOLOv11 logo detector on a video.")
parser.add_argument("--weights", default="weights/best.pt", help="Path to trained YOLO weights (best.pt)")
parser.add_argument("--video", default="video_3-trimmed.mp4", help="Path to the input video")
parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--output-dir", default="yolo results", help="Root folder for outputs")
args = parser.parse_args()

WEIGHTS_PATH = Path(args.weights)
VIDEO_PATH   = Path(args.video)
IMG_SIZE     = args.imgsz
CONF         = args.conf
OUTPUT_DIR   = args.output_dir

assert WEIGHTS_PATH.exists(), f"Missing: {WEIGHTS_PATH}"
assert VIDEO_PATH.exists(),   f"Missing: {VIDEO_PATH}"
print("Weights OK:", WEIGHTS_PATH, "| Video OK:", VIDEO_PATH)


# ── Cell 2 — Load the trained model (same as Cell 7 of train_yolov11_m2.py) ──
# %%
from ultralytics import YOLO

best = YOLO(str(WEIGHTS_PATH))


# ── Cell 3 — Predict on the video (same call as Cell 9, source = video) ─────
# %%
best.predict(
    source=str(VIDEO_PATH),
    save=True,
    save_txt=True,
    save_conf=True,
    imgsz=IMG_SIZE,
    conf=CONF,
    project=OUTPUT_DIR,
    name=VIDEO_PATH.stem,
    exist_ok=True,
)
print(f"Results saved under: {Path(OUTPUT_DIR) / VIDEO_PATH.stem}")
