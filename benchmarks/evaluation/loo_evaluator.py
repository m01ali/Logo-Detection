"""
Leave-One-Out (LOO) evaluator for the logo retrieval benchmark.

Protocol
--------
For each image I in the database:
  1. Build an IndexFlatL2 from all OTHER images (112 out of 113).
  2. Query with I's embedding.
  3. Record the ranked list of brand names returned.
  4. A hit at rank k means the true brand appears in the top-k results.

All 113 images are embedded once up-front; the FAISS index is rebuilt per query
(113 rebuilds of a 112-vector index — negligible cost, avoids deletion complexity
since IndexFlatL2 has no native deletion API).

Edge cases
----------
- Brand with exactly 1 image: skipped and logged in LOOResult.n_skipped.
  (Not expected given the actual database, but handled defensively.)
- FAISS returning -1 indices (can occur if k > index size): filtered out.
"""

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from benchmarks.benchmark_config import (
    DATABASE_PATH,
    DEFAULT_K_VALUES,
    IMAGE_EXTENSIONS,
    RESULTS_DIR,
    THRESHOLD_SWEEP,
)
from benchmarks.embedders.base import BaseLogoEmbedder
from benchmarks.evaluation.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
    recall_at_k,
    threshold_sensitivity,
)


@dataclass
class LOOResult:
    model_name: str
    recall: Dict[int, float]            # {1: 0.85, 3: 0.92, 5: 0.95}
    mrr: float
    threshold_curve: Dict[float, Dict]  # threshold -> {recall_at_k, mrr, coverage}
    n_queries: int
    n_skipped: int
    per_brand_recall: Dict[str, float]  # brand -> Recall@1
    raw_query_results: List[dict] = field(default_factory=list)


class LOOEvaluator:

    def __init__(
        self,
        embedder: BaseLogoEmbedder,
        database_path: str = DATABASE_PATH,
        k_values: List[int] = None,
        threshold_sweep: List[float] = None,
    ):
        self.embedder = embedder
        self.database_path = Path(database_path)
        self.k_values = k_values or DEFAULT_K_VALUES
        self.threshold_sweep = threshold_sweep or THRESHOLD_SWEEP
        self.max_k = max(self.k_values)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_database(self) -> Dict[str, List[Path]]:
        """Return {brand_name: [sorted image paths]}."""
        db: Dict[str, List[Path]] = {}
        for brand_dir in sorted(self.database_path.iterdir()):
            if not brand_dir.is_dir():
                continue
            imgs = sorted(
                p for p in brand_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if imgs:
                db[brand_dir.name] = imgs
        return db

    def _build_index(
        self, all_embeddings: np.ndarray, exclude_idx: int
    ):
        """Build IndexFlatL2 excluding one row; return (index, brand_list)."""
        mask = np.ones(len(all_embeddings), dtype=bool)
        mask[exclude_idx] = False
        db_vectors = all_embeddings[mask].astype(np.float32)
        index = faiss.IndexFlatL2(self.embedder.info.dim)
        index.add(db_vectors)
        return index, mask  # caller reconstructs brand list using mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> LOOResult:
        brand_image_map = self._load_database()

        # Flatten to parallel lists
        all_paths: List[Path] = []
        all_brands_flat: List[str] = []
        for brand, paths in brand_image_map.items():
            all_paths.extend(paths)
            all_brands_flat.extend([brand] * len(paths))

        n_total = len(all_paths)
        print(
            f"\n[{self.embedder.info.name}] Embedding {n_total} images…"
        )
        all_images = [Image.open(p).convert("RGB") for p in tqdm(all_paths, desc="loading")]
        all_embeddings = self.embedder.embed(all_images)  # (N, dim) float32

        query_results: List[dict] = []
        n_skipped = 0

        for query_idx in tqdm(range(n_total), desc="LOO queries"):
            true_brand = all_brands_flat[query_idx]
            brand_count = sum(1 for b in all_brands_flat if b == true_brand)

            if brand_count < 2:
                print(
                    f"  WARNING: skipping brand '{true_brand}' — only 1 image, "
                    "cannot LOO without removing it from DB."
                )
                n_skipped += 1
                continue

            # Build index without this image
            mask = np.ones(n_total, dtype=bool)
            mask[query_idx] = False
            db_vectors = all_embeddings[mask].astype(np.float32)
            db_brands  = [b for i, b in enumerate(all_brands_flat) if mask[i]]

            index = faiss.IndexFlatL2(self.embedder.info.dim)
            index.add(db_vectors)

            k = min(self.max_k, index.ntotal)
            query_vec = all_embeddings[query_idx : query_idx + 1].astype(np.float32)
            distances, indices = index.search(query_vec, k)

            # Filter invalid FAISS indices (-1 can appear if k > ntotal)
            valid = [(d, i) for d, i in zip(distances[0], indices[0]) if i >= 0]
            retrieved_brands = [db_brands[i] for _, i in valid]

            # Similarity scores — both production formula and true cosine
            prod_scores   = [1.0 - float(d) / 4.0 for d, _ in valid]
            cosine_scores = [1.0 - float(d) / 2.0 for d, _ in valid]

            query_results.append(
                {
                    "query_path":       str(all_paths[query_idx]),
                    "true_brand":       true_brand,
                    "retrieved_brands": retrieved_brands,
                    "production_scores": prod_scores,
                    "cosine_scores":     cosine_scores,
                }
            )

        # ------------------------------------------------------------------
        # Aggregate metrics
        # ------------------------------------------------------------------
        all_retrieved = [r["retrieved_brands"] for r in query_results]
        all_true      = [r["true_brand"]       for r in query_results]
        all_scores    = [r["production_scores"] for r in query_results]

        recall = {k: mean_recall_at_k(all_retrieved, all_true, k) for k in self.k_values}
        mrr    = mean_reciprocal_rank(all_retrieved, all_true)

        threshold_curve = threshold_sensitivity(
            all_scores, all_retrieved, all_true, self.threshold_sweep, k=1
        )

        per_brand: Dict[str, List[float]] = {}
        for r in query_results:
            brand = r["true_brand"]
            hit = recall_at_k(r["retrieved_brands"], brand, k=1)
            per_brand.setdefault(brand, []).append(hit)
        per_brand_recall = {b: sum(v) / len(v) for b, v in per_brand.items()}

        return LOOResult(
            model_name=self.embedder.info.name,
            recall=recall,
            mrr=mrr,
            threshold_curve=threshold_curve,
            n_queries=len(query_results),
            n_skipped=n_skipped,
            per_brand_recall=per_brand_recall,
            raw_query_results=query_results,
        )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_per_query_csv(self, result: LOOResult, output_dir: str = RESULTS_DIR) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"loo_per_query_{result.model_name}.csv")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_image", "true_brand",
                "rank1_brand", "rank1_prod_score", "rank1_cosine_score",
                "rank3_brands", "rank5_brands",
                "recall_at_1", "recall_at_3", "recall_at_5", "reciprocal_rank",
            ])
            for r in result.raw_query_results:
                rb = r["retrieved_brands"]
                ps = r["production_scores"]
                cs = r["cosine_scores"]
                writer.writerow([
                    os.path.basename(r["query_path"]),
                    r["true_brand"],
                    rb[0] if rb else "",
                    f"{ps[0]:.4f}" if ps else "",
                    f"{cs[0]:.4f}" if cs else "",
                    ";".join(rb[:3]),
                    ";".join(rb[:5]),
                    int(recall_at_k(rb, r["true_brand"], 1)),
                    int(recall_at_k(rb, r["true_brand"], 3)),
                    int(recall_at_k(rb, r["true_brand"], 5)),
                    f"{reciprocal_rank(rb, r['true_brand']):.4f}",
                ])

        return path
