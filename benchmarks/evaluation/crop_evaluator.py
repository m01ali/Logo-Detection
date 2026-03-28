"""
Real video crop evaluator.

Unlike LOO, this evaluator searches each video crop against the FULL database
(no images held out). The crops are expected in a labelled folder with the same
structure as the logo database:

    benchmarks/labelled_crops/
    ├── ING/
    │   ├── frame_042_crop.jpg
    │   └── frame_107_crop.jpg
    ├── verisure/
    │   └── video2_t12s.png
    └── ...

The folder name = ground-truth brand label.

How to build the labelled_crops folder
---------------------------------------
1. Run the Streamlit app on videos where you know which brands appear.
2. Save the detected logo crops (the bounding-box crops, not the full frame).
3. Place them into benchmarks/labelled_crops/BRAND_NAME/ folders.
4. Aim for 2–3 crops per brand, preferring hard cases (blur, occlusion, low-res).
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
    LABELLED_CROPS_DIR,
    RESULTS_DIR,
    THRESHOLD_SWEEP,
)
from benchmarks.embedders.base import BaseLogoEmbedder
from benchmarks.evaluation.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    threshold_sensitivity,
)


@dataclass
class CropResult:
    model_name: str
    scenario: str = "crops"
    recall: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    threshold_curve: Dict[float, Dict] = field(default_factory=dict)
    n_queries: int = 0
    per_brand_recall: Dict[str, float] = field(default_factory=dict)
    raw_query_results: List[dict] = field(default_factory=list)


class CropEvaluator:

    def __init__(
        self,
        embedder: BaseLogoEmbedder,
        database_path: str = DATABASE_PATH,
        crops_path: str = LABELLED_CROPS_DIR,
        k_values: List[int] = None,
        threshold_sweep: List[float] = None,
    ):
        self.embedder = embedder
        self.database_path = Path(database_path)
        self.crops_path = Path(crops_path)
        self.k_values = k_values or DEFAULT_K_VALUES
        self.threshold_sweep = threshold_sweep or THRESHOLD_SWEEP
        self.max_k = max(self.k_values)

    def _load_folder(self, folder: Path) -> Dict[str, List[Path]]:
        db: Dict[str, List[Path]] = {}
        for brand_dir in sorted(folder.iterdir()):
            if not brand_dir.is_dir():
                continue
            imgs = sorted(
                p for p in brand_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if imgs:
                db[brand_dir.name] = imgs
        return db

    def run(self) -> CropResult:
        if not self.crops_path.exists():
            raise FileNotFoundError(
                f"Labelled crops folder not found: {self.crops_path}\n"
                "Create it by placing video crop images in sub-folders named by brand.\n"
                "Example: benchmarks/labelled_crops/ING/frame_001.jpg"
            )

        # Load DB images and embed → build fixed FAISS index
        db_map = self._load_folder(self.database_path)
        db_paths: List[Path] = []
        db_brands: List[str] = []
        for brand, paths in db_map.items():
            db_paths.extend(paths)
            db_brands.extend([brand] * len(paths))

        print(f"\n[{self.embedder.info.name}] Embedding {len(db_paths)} DB images…")
        db_images = [Image.open(p).convert("RGB") for p in tqdm(db_paths, desc="db")]
        db_embeddings = self.embedder.embed(db_images)

        index = faiss.IndexFlatL2(self.embedder.info.dim)
        index.add(db_embeddings.astype(np.float32))

        # Load crop queries
        crop_map = self._load_folder(self.crops_path)
        crop_paths: List[Path] = []
        crop_true_brands: List[str] = []
        for brand, paths in crop_map.items():
            crop_paths.extend(paths)
            crop_true_brands.extend([brand] * len(paths))

        print(f"Embedding {len(crop_paths)} video crop queries…")
        crop_images = [Image.open(p).convert("RGB") for p in tqdm(crop_paths, desc="crops")]
        crop_embeddings = self.embedder.embed(crop_images)

        # Query
        query_results: List[dict] = []
        k = min(self.max_k, index.ntotal)

        for query_idx, (true_brand, query_path) in enumerate(
            tqdm(zip(crop_true_brands, crop_paths), total=len(crop_paths), desc="querying")
        ):
            query_vec = crop_embeddings[query_idx : query_idx + 1].astype(np.float32)
            distances, indices = index.search(query_vec, k)

            valid = [(d, i) for d, i in zip(distances[0], indices[0]) if i >= 0]
            retrieved_brands = [db_brands[i] for _, i in valid]
            prod_scores   = [1.0 - float(d) / 4.0 for d, _ in valid]
            cosine_scores = [1.0 - float(d) / 2.0 for d, _ in valid]

            query_results.append({
                "query_path":       str(query_path),
                "true_brand":       true_brand,
                "retrieved_brands": retrieved_brands,
                "production_scores": prod_scores,
                "cosine_scores":     cosine_scores,
            })

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

        return CropResult(
            model_name=self.embedder.info.name,
            recall=recall,
            mrr=mrr,
            threshold_curve=threshold_curve,
            n_queries=len(query_results),
            per_brand_recall=per_brand_recall,
            raw_query_results=query_results,
        )

    def export_per_query_csv(self, result: CropResult, output_dir: str = RESULTS_DIR) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"crop_per_query_{result.model_name}.csv")

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
