"""
Hybrid embedder: CLIP + DINOv2 concatenation.

Combines text-aligned (CLIP) and purely visual (DINOv2) feature spaces.
Both sub-embedders independently produce unit vectors (768 and 1024 dims).
Concatenation yields a 1792-dim vector which is then re-L2-normalised,
giving equal implicit weighting to both sub-spaces (since ||clip||=||dino||=1
before concat, each contributes equally to the final unit vector).
"""

from typing import List

import numpy as np
from PIL import Image

from benchmarks.embedders.base import BaseLogoEmbedder, EmbedderInfo
from benchmarks.embedders.clip_embedder import CLIPEmbedder
from benchmarks.embedders.dinov2_embedder import DINOv2Embedder


class HybridEmbedder(BaseLogoEmbedder):
    """
    CLIP (768) + DINOv2 (1024)  →  concatenate  →  re-L2-normalise  →  1792-dim.

    Both sub-embedders share the same device and batch_size.
    Images are embedded once per sub-model (two forward passes total per batch).
    """

    def __init__(self, device: str = "auto", batch_size: int = 8):
        self._clip  = CLIPEmbedder(device=device, batch_size=batch_size)
        self._dinov2 = DINOv2Embedder(device=device, batch_size=batch_size)

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name="hybrid",
            model_id="clip+dinov2",
            dim=1792,
            description="Hybrid: CLIP ViT-L/14 (768) ⊕ DINOv2 ViT-L/14 (1024), concat + L2-norm",
        )

    def embed(self, images: List[Image.Image]) -> np.ndarray:
        clip_emb  = self._clip.embed(images)   # (N, 768)  unit vectors
        dino_emb  = self._dinov2.embed(images) # (N, 1024) unit vectors
        concat    = np.concatenate([clip_emb, dino_emb], axis=1)  # (N, 1792)
        return self.l2_normalise(concat)        # re-normalise → unit vectors
