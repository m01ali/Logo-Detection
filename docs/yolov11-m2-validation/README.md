# Custom YOLOv11 Logo Detector — M2 Validation Results

Custom, domain-specific logo detector fine-tuned on the labelled M2 logo dataset.
Training/validation code: [`train_yolov11_m2.py`](../../train_yolov11_m2.py)
(1:1 conversion of the "M2 - Yolov11" section of `Notebooks/Yolov11-M2.ipynb`).

## Setup

| | |
|---|---|
| Base model | YOLO11-L (`yolo11l.pt`, pretrained) |
| Epochs | 50 (patience 20) |
| Image size | 640 |
| Batch | 8 |
| Split | 70% train / 30% val (seed 42) |
| Train images | 4,081 |
| Val images | 1,749 (3,658 logo instances) |
| Classes | 267 brand classes present in the val split |

## Headline metrics (held-out validation split)

| Metric | Value |
|---|---|
| mAP50-95 | **0.6767** |
| mAP50 | **0.8498** |
| mAP75 | 0.7524 |
| Precision (all) | 0.836 |
| Recall (all) | 0.816 |
| Inference speed | ~24.8 ms/image (Colab GPU) |

## Files in this folder

- `val_results_full.txt` — full per-class precision / recall / mAP50 / mAP50-95 table (267 classes)
- `results.png` — training curves (losses + metrics per epoch)
- `confusion_matrix.png` — per-class confusion matrix
- `val_batch0_labels.jpg` / `val_batch0_pred.jpg` — ground truth vs model predictions on a validation batch
- `train_log_tail.txt` — final training epochs log

## Model weights

`best.pt` is produced by Cell 6 of `train_yolov11_m2.py`. Cell 10 saves weights
to Google Drive under a dedicated weights folder, one subfolder per run:
`MyDrive/LogoDet/weights/<run_name>/best.pt` (e.g. `weights/logo2_raw/best.pt`).
The originally recorded run predates this layout and was saved to
`MyDrive/LogoDet/yolo_runs/logo2_raw/weights/best.pt`.
(Weights are not committed to this repo.)

## Notes / caveats

- 18 of 267 classes score 0 mAP — nearly all have only 1–2 validation instances
  (e.g. `abbott`, `alpine`, `canyon`, `frecciarossa`, `hp`). These brands need
  more labelled examples before per-class numbers are meaningful.
- Metrics are on a random 70/30 split of the same dataset, not on the fixed
  gold evaluation suite discussed in the planning doc — treat them as an
  optimistic upper bound for in-domain performance.
