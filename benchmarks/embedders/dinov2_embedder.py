"""
DINOv2 ViT-L/14 embedder.

Purely visual features — no text alignment. Strong spatial and structural
understanding, which can help distinguish logos that share similar colour
palettes but differ in shape.
"""

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from benchmarks.embedders.base import BaseLogoEmbedder, EmbedderInfo


def _select_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class DINOv2Embedder(BaseLogoEmbedder):
    """
    facebook/dinov2-large  →  1024-dim L2-normalised embeddings.

    Extraction: AutoModel(pixel_values).pooler_output  (CLS token after linear head).
    Note: DINOv2 was not trained with text; output is purely visual.
    """

    MODEL_ID = "facebook/dinov2-large"

    def __init__(self, device: str = "auto", batch_size: int = 16):
        self._device = _select_device(device)
        self._batch_size = batch_size

        self._processor = AutoImageProcessor.from_pretrained(self.MODEL_ID, use_fast=True)
        self._model = AutoModel.from_pretrained(self.MODEL_ID).to(self._device)
        self._model.eval()

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name="dinov2",
            model_id=self.MODEL_ID,
            dim=1024,
            description="DINOv2 ViT-L/14 — purely visual features",
        )

    def embed(self, images: List[Image.Image]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(images), self._batch_size):
            batch = images[start : start + self._batch_size]
            rgb_batch = [img.convert("RGB") for img in batch]

            inputs = self._processor(images=rgb_batch, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                emb = outputs.pooler_output  # (B, 1024)

            all_embeddings.append(emb.cpu().numpy())

        combined = np.concatenate(all_embeddings, axis=0)
        return self.l2_normalise(combined)
