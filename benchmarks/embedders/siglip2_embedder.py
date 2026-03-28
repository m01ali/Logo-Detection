"""
SigLIP2 embedder.

SigLIP2 uses a sigmoid contrastive loss (instead of softmax like CLIP), which
gives better-calibrated similarity scores for image-image retrieval. The
so400m-patch14-384 checkpoint is the strongest in the SigLIP2 family for
retrieval tasks.

Extraction path: vision_model(pixel_values).pooler_output
SigLIP2 does not expose get_image_features() the same way CLIP does.
"""

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from benchmarks.embedders.base import BaseLogoEmbedder, EmbedderInfo


def _select_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class SigLIP2Embedder(BaseLogoEmbedder):
    """
    google/siglip2-so400m-patch14-384  →  1152-dim L2-normalised embeddings.

    Input resolution: 384×384 (higher than CLIP's 224×224 → heavier per image,
    hence smaller default batch_size).

    Extraction:
        AutoModel.vision_model(pixel_values).pooler_output  → (B, 1152)
    If visual_projection exists, it is applied (same defensive pattern as
    the production _extract_image_embeddings fallback in faiss_db.py).
    """

    MODEL_ID = "google/siglip2-so400m-patch14-384"

    def __init__(self, device: str = "auto", batch_size: int = 8):
        self._device = _select_device(device)
        self._batch_size = batch_size

        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModel.from_pretrained(self.MODEL_ID).to(self._device)
        self._model.eval()

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name="siglip2",
            model_id=self.MODEL_ID,
            dim=1152,
            description="SigLIP2 so400m-patch14-384 — sigmoid contrastive, 384px",
        )

    def embed(self, images: List[Image.Image]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(images), self._batch_size):
            batch = images[start : start + self._batch_size]
            rgb_batch = [img.convert("RGB") for img in batch]

            inputs = self._processor(images=rgb_batch, return_tensors="pt").to(self._device)

            with torch.no_grad():
                vision_out = self._model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                if (
                    hasattr(vision_out, "pooler_output")
                    and vision_out.pooler_output is not None
                ):
                    pooled = vision_out.pooler_output
                else:
                    # Fallback: CLS token from last hidden state
                    pooled = vision_out.last_hidden_state[:, 0, :]

                if (
                    hasattr(self._model, "visual_projection")
                    and self._model.visual_projection is not None
                ):
                    emb = self._model.visual_projection(pooled)
                else:
                    emb = pooled

            all_embeddings.append(emb.cpu().numpy())

        combined = np.concatenate(all_embeddings, axis=0)
        return self.l2_normalise(combined)
