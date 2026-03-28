"""
CLIP ViT-L/14 embedder — baseline model.

Mirrors the production embedding logic in models/faiss_db.py (_embed_batch /
_extract_image_embeddings) exactly, so benchmark results are directly comparable
to what the live system produces.
"""

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from benchmarks.embedders.base import BaseLogoEmbedder, EmbedderInfo


def _select_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _tensor_from_output(emb) -> torch.Tensor:
    """Unpack a tensor or ModelOutput object into a plain tensor."""
    if isinstance(emb, torch.Tensor):
        return emb
    if hasattr(emb, "image_embeds") and isinstance(emb.image_embeds, torch.Tensor):
        return emb.image_embeds
    if hasattr(emb, "pooler_output") and isinstance(emb.pooler_output, torch.Tensor):
        return emb.pooler_output
    raise RuntimeError(f"Cannot extract tensor from type {type(emb)}")


class CLIPEmbedder(BaseLogoEmbedder):
    """
    openai/clip-vit-large-patch14  →  768-dim L2-normalised embeddings.

    Extraction path (matches production faiss_db._extract_image_embeddings):
        CLIPProcessor → pixel_values → CLIPModel.get_image_features()
    Fallback (same as production):
        vision_model().pooler_output  → visual_projection (if present)
    """

    MODEL_ID = "openai/clip-vit-large-patch14"

    def __init__(self, device: str = "auto", batch_size: int = 16):
        self._device = _select_device(device)
        self._batch_size = batch_size

        self._processor = CLIPProcessor.from_pretrained(self.MODEL_ID, use_fast=True)
        self._model = CLIPModel.from_pretrained(self.MODEL_ID).to(self._device)
        self._model.eval()

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name="clip",
            model_id=self.MODEL_ID,
            dim=768,
            description="CLIP ViT-L/14 (production baseline)",
        )

    def embed(self, images: List[Image.Image]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(images), self._batch_size):
            batch = images[start : start + self._batch_size]
            rgb_batch = [img.convert("RGB") for img in batch]

            inputs = self._processor(images=rgb_batch, return_tensors="pt").to(self._device)

            with torch.no_grad():
                try:
                    emb = self._model.get_image_features(
                        pixel_values=inputs["pixel_values"]
                    )
                except Exception:
                    vision_out = self._model.vision_model(
                        pixel_values=inputs["pixel_values"]
                    )
                    if (
                        hasattr(vision_out, "pooler_output")
                        and vision_out.pooler_output is not None
                    ):
                        pooled = vision_out.pooler_output
                    else:
                        pooled = vision_out.last_hidden_state[:, 0, :]

                    if (
                        hasattr(self._model, "visual_projection")
                        and self._model.visual_projection is not None
                    ):
                        emb = self._model.visual_projection(pooled)
                    else:
                        emb = pooled

            all_embeddings.append(_tensor_from_output(emb).cpu().numpy())

        combined = np.concatenate(all_embeddings, axis=0)
        return self.l2_normalise(combined)
