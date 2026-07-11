# =============================================================================
# M2 — Custom YOLOv11 Logo Detector (training + validation)
# =============================================================================
# Converted 1:1 from the "M2 - Yolov11" section of Notebooks/Yolov11-M2.ipynb.
# This is a Google Colab based workflow — run the cells top-to-bottom on a GPU
# runtime. The code is kept exactly as in the notebook.
#
# Pipeline:
#   Cell 1  — download + extract the labelled YOLO dataset from Drive
#   Cell 2  — config (paths, model, epochs, split ratio)
#   Cell 3  — 70/30 train/val split
#   Cell 4  — write data.yaml for the new split
#   Cell 5  — label sanity check
#   Cell 6  — train YOLO11-L
#   Cell 7  — validate on the held-out val split
#   Cell 8  — training curves + sample predictions
#   Cell 9  — (optional) predict on new images
#   Cell 10 — (optional) save the weights to Drive (WEIGHTS_DIR/<run>/best.pt)
#
# Reference results from the recorded run (YOLO11-L, 50 epochs, imgsz 640,
# 4081 train / 1749 val images, 3658 val instances):
#   mAP50-95: 0.6767   |   mAP50: 0.8498   |   mAP75: 0.7524
# Full per-class metrics: docs/yolov11-m2-validation/
# =============================================================================


# ── Cell 1 — Setup + download dataset from shared Drive link ─────────────────
# %%
# !pip install -q ultralytics gdown

import torch, gdown, zipfile, os
from pathlib import Path

print("CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# --- Download the shared file ---
FILE_ID  = "17P5o1JPULX6H-cO_vxubFjYviPBTPp_z"
ZIP_PATH = "/content/logo2.zip"
EXTRACT  = Path("/content/logo2_raw")

if not Path(ZIP_PATH).exists():
    gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)

# --- Extract ---
EXTRACT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT)

# Peek at what got extracted so we can locate data.yaml
for root, dirs, files in os.walk(EXTRACT):
    depth = root.replace(str(EXTRACT), "").count(os.sep)
    if depth <= 2:
        print("  " * depth + os.path.basename(root) + "/")


# ── Cell 2 — Config (point SRC_DIR at wherever data.yaml landed) ─────────────
# %%
from pathlib import Path

# After Cell 1's tree print, set this to the folder containing data.yaml.
# Common case: extraction creates one top-level folder, e.g. /content/logo2_raw/logo2.yolov11
SRC_DIR     = Path("/content/logo2_raw")   # ← adjust based on the tree printout
SRC_IMAGES  = SRC_DIR / "train" / "images"
SRC_LABELS  = SRC_DIR / "train" / "labels"
SRC_YAML    = SRC_DIR / "data.yaml"

WORK_DIR    = Path("/content/yolo_dataset")

MODEL_NAME  = "yolo11l.pt"
EPOCHS      = 50
IMG_SIZE    = 640
BATCH       = 8
VAL_RATIO   = 0.30
SEED        = 42

# Save runs locally if you don't want to mount Drive:
PROJECT_DIR = Path("/content/yolo_runs")
RUN_NAME    = "logo2_raw"

# Where Cell 10 stores the trained weights on Drive — one subfolder per run,
# so future runs never overwrite earlier ones.
WEIGHTS_DIR = Path("/content/drive/MyDrive/LogoDet/weights")

assert SRC_IMAGES.exists(), f"Missing: {SRC_IMAGES}"
assert SRC_LABELS.exists(), f"Missing: {SRC_LABELS}"
assert SRC_YAML.exists(),   f"Missing: {SRC_YAML}"
print("Source OK:", SRC_DIR)


# ── Cell 3 — 70/30 split → /content/yolo_dataset/{images,labels}/{train,val} ─
# %%
import random, shutil
random.seed(SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

images = [p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS]
pairs, missing = [], 0
for img in images:
    lbl = SRC_LABELS / (img.stem + ".txt")
    if lbl.exists():
        pairs.append((img, lbl))
    else:
        missing += 1

print(f"Images found: {len(images)} | with labels: {len(pairs)} | missing labels: {missing}")

random.shuffle(pairs)
n_val = int(len(pairs) * VAL_RATIO)
val_set, train_set = pairs[:n_val], pairs[n_val:]
print(f"Train: {len(train_set)} | Val: {len(val_set)}")

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
for split in ("train", "val"):
    (WORK_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_split(items, split):
    for img, lbl in items:
        shutil.copy2(img, WORK_DIR / "images" / split / img.name)
        shutil.copy2(lbl, WORK_DIR / "labels" / split / lbl.name)

copy_split(train_set, "train")
copy_split(val_set,   "val")
print("Copied to:", WORK_DIR)


# ── Cell 4 — Write new data.yaml (reuse class names from yours) ──────────────
# %%
import yaml

with open(SRC_YAML, "r") as f:
    src_cfg = yaml.safe_load(f)

names = src_cfg.get("names")
nc    = src_cfg.get("nc", len(names) if names is not None else None)
assert names is not None and nc is not None, "Could not read names/nc from your data.yaml"

new_cfg = {
    "path":  str(WORK_DIR),
    "train": "images/train",
    "val":   "images/val",
    "nc":    nc,
    "names": names,
}
NEW_YAML = WORK_DIR / "data.yaml"
with open(NEW_YAML, "w") as f:
    yaml.safe_dump(new_cfg, f, sort_keys=False)

print("Wrote:", NEW_YAML, "| nc:", nc)


# ── Cell 5 — Label sanity check (class IDs must be in [0, nc-1]) ─────────────
# %%
bad_files, bad_lines, max_cls = set(), 0, -1
for split in ("train", "val"):
    for lbl in (WORK_DIR / "labels" / split).glob("*.txt"):
        for ln in lbl.read_text().splitlines():
            parts = ln.strip().split()
            if not parts: continue
            try:
                cls = int(parts[0])
            except ValueError:
                bad_lines += 1; bad_files.add(str(lbl)); continue
            max_cls = max(max_cls, cls)
            if cls < 0 or cls >= nc:
                bad_lines += 1; bad_files.add(str(lbl))

print(f"Max class id seen: {max_cls} (nc={nc}) | bad lines: {bad_lines} in {len(bad_files)} files")
if bad_lines:
    print("Some class IDs are outside [0, nc-1]. Fix nc/names in data.yaml or the labels.")


# ── Cell 6 — Train YOLO11l ───────────────────────────────────────────────────
# %%
from ultralytics import YOLO

model = YOLO(MODEL_NAME)   # downloads pretrained weights on first run

results = model.train(
    data=str(NEW_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    seed=SEED,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    exist_ok=True,
    patience=20,
    save=True,
    plots=True,
    device=0 if torch.cuda.is_available() else "cpu",
    # workers=8,
    # cache="ram",     # ~5.8k imgs at 640 fits comfortably on A100/L4 RAM → big speedup
    # amp=True,        # mixed precision is on by default; leave it
)
print("Best weights:", PROJECT_DIR / RUN_NAME / "weights" / "best.pt")


# ── Cell 7 — Validate ────────────────────────────────────────────────────────
# %%
best_pt = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
best = YOLO(str(best_pt))
metrics = best.val(data=str(NEW_YAML), imgsz=IMG_SIZE, split="val")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50   : {metrics.box.map50:.4f}")
print(f"mAP75   : {metrics.box.map75:.4f}")


# ── Cell 8 — Curves & sample predictions ─────────────────────────────────────
# %%
from IPython.display import Image, display
run_dir = PROJECT_DIR / RUN_NAME
for fname in ["results.png", "confusion_matrix.png",
              "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
    p = run_dir / fname
    if p.exists():
        print(fname); display(Image(filename=str(p)))


# ── Cell 9 — (Optional) Predict on new images / a folder ─────────────────────
# %%
best.predict(source="/content/test_imgs", save=True, imgsz=IMG_SIZE, conf=0.25)


# ── Cell 10 — (Optional) Save weights to Drive at the end ────────────────────
# %%
from google.colab import drive
import shutil
drive.mount('/content/drive')

dest = WEIGHTS_DIR / RUN_NAME
dest.mkdir(parents=True, exist_ok=True)
for w in ("best.pt", "last.pt"):
    src = PROJECT_DIR / RUN_NAME / "weights" / w
    if src.exists():
        shutil.copy2(src, dest / w)
        print("Saved:", dest / w)

# Also keep the full run folder (curves, logs, plots), uncomment if needed:
# shutil.copytree(PROJECT_DIR / RUN_NAME, dest / "run", dirs_exist_ok=True)
