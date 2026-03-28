"""
Video crop benchmark CLI.

Usage (from project root, using .venv311):
    python -m benchmarks.run_crop_benchmark
    python -m benchmarks.run_crop_benchmark --models clip dinov2
    python -m benchmarks.run_crop_benchmark --crops-path /path/to/labelled_crops

Requirements:
    Populate benchmarks/labelled_crops/BRAND_NAME/*.jpg with real video crops
    before running. The folder name is the ground-truth brand label.

Options:
    --models      Space-separated model keys (default: all four)
    --crops-path  Path to labelled crops folder (default: benchmarks/labelled_crops/)
    --db-path     Path to logo database (default: from benchmark_config.py)
    --output-dir  Results output folder (default: benchmarks/results/)
    --device      'auto', 'cpu', or 'cuda'
    --batch-size  Images per forward pass
    --no-plots    Skip plots
"""

import argparse
import os

# M1 Mac: suppress duplicate OpenMP runtime warning from PyTorch + FAISS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from benchmarks.benchmark_config import DATABASE_PATH, LABELLED_CROPS_DIR, RESULTS_DIR
from benchmarks.embedders.clip_embedder import CLIPEmbedder
from benchmarks.embedders.dinov2_embedder import DINOv2Embedder
from benchmarks.embedders.hybrid_embedder import HybridEmbedder
from benchmarks.embedders.siglip2_embedder import SigLIP2Embedder
from benchmarks.evaluation.crop_evaluator import CropEvaluator

EMBEDDER_REGISTRY = {
    "clip":    CLIPEmbedder,
    "siglip2": SigLIP2Embedder,
    "dinov2":  DINOv2Embedder,
    "hybrid":  HybridEmbedder,
}
ALL_MODELS = list(EMBEDDER_REGISTRY.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Video crop retrieval benchmark")
    parser.add_argument("--models",     nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, metavar="MODEL")
    parser.add_argument("--crops-path", default=LABELLED_CROPS_DIR)
    parser.add_argument("--db-path",    default=DATABASE_PATH)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-plots",   action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running crop benchmark for models: {args.models}")
    print(f"Crops:    {args.crops_path}")
    print(f"Database: {args.db_path}")
    print(f"Output:   {args.output_dir}\n")

    results = []

    for model_name in args.models:
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        embedder = EMBEDDER_REGISTRY[model_name](
            device=args.device,
            batch_size=args.batch_size,
        )
        evaluator = CropEvaluator(
            embedder,
            database_path=args.db_path,
            crops_path=args.crops_path,
        )

        try:
            result = evaluator.run()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return

        results.append(result)
        csv_path = evaluator.export_per_query_csv(result, args.output_dir)
        print(f"Per-query CSV → {csv_path}")
        print(
            f"  R@1={result.recall.get(1,0):.3f}  "
            f"R@3={result.recall.get(3,0):.3f}  "
            f"R@5={result.recall.get(5,0):.3f}  "
            f"MRR={result.mrr:.3f}  "
            f"Queries={result.n_queries}"
        )

    # Summary CSV
    import csv, os
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "crop_summary.csv")
    from benchmarks.benchmark_config import EMBEDDING_DIMS
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dim", "recall_at_1", "recall_at_3", "recall_at_5", "mrr", "n_queries"])
        for r in sorted(results, key=lambda x: x.recall.get(1, 0), reverse=True):
            writer.writerow([
                r.model_name,
                EMBEDDING_DIMS.get(r.model_name, "?"),
                f"{r.recall.get(1,0):.4f}",
                f"{r.recall.get(3,0):.4f}",
                f"{r.recall.get(5,0):.4f}",
                f"{r.mrr:.4f}",
                r.n_queries,
            ])
    print(f"Crop summary CSV → {summary_path}")

    # Plots
    if not args.no_plots and results:
        try:
            from benchmarks.reporting.plot_reporter import (
                plot_per_brand_heatmap,
                plot_threshold_sensitivity,
            )
            # Adapt CropResult to match LOOResult interface for plotting
            plot_threshold_sensitivity(results, args.output_dir)  # type: ignore[arg-type]
            plot_per_brand_heatmap(results, args.output_dir)       # type: ignore[arg-type]
        except ImportError as e:
            print(f"Plotting skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
