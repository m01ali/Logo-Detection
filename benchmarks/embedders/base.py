"""Abstract base class for all logo embedders used in the benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class EmbedderInfo:
    name: str         # short key: "clip", "siglip2", "dinov2", "hybrid"
    model_id: str     # HuggingFace ID or description
    dim: int          # output embedding dimension (post-normalisation)
    description: str  # human-readable description for reports


class BaseLogoEmbedder(ABC):
    """
    Contract for all benchmark embedders.

    All embedders MUST return L2-normalised float32 vectors of shape (N, dim).

    Similarity notes
    ----------------
    The production system uses IndexFlatL2 and converts distances via:
        production_score = 1 - dist / 4
    For L2-normalised unit vectors, the mathematically correct cosine similarity is:
        cosine = 1 - dist / 2
    So production_score = (1 + cosine) / 2  — a linear rescaling, NOT true cosine.
    The benchmark reports both values for transparency.
    """

    @property
    @abstractmethod
    def info(self) -> EmbedderInfo:
        ...

    @abstractmethod
    def embed(self, images: List[Image.Image]) -> np.ndarray:
        """
        Embed a list of PIL Images.

        Args:
            images: List of RGB PIL Images (any resolution).
        Returns:
            np.ndarray of shape (N, dim), dtype float32, L2-normalised rows.
        """
        ...

    def embed_single(self, image: Image.Image) -> np.ndarray:
        """Convenience wrapper. Returns shape (dim,)."""
        return self.embed([image])[0]

    @staticmethod
    def l2_normalise(vectors: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation with epsilon guard. Returns float32."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return (vectors / norms).astype(np.float32)
