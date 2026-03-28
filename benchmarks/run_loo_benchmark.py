"""
Leave-One-Out benchmark CLI.

Usage (from project root, using .venv311):
    python -m benchmarks.run_loo_benchmark
    python -m benchmarks.run_loo_benchmark --models clip dinov2
    python -m benchmarks.run_loo_benchmark --models clip --db-path /path/to/logos
    python -m benchmarks.run_loo_benchmark --device cpu --batch-size 8

Options:
    --models      Space-separated list of models to run.
                  Choices: clip  siglip2  dinov2  hybrid  (default: all four)
    --db-path     Path to logo database folder (default: from benchmark_config.py)
    --output-dir  Where to write CSV and PNG results (default: benchmarks/results/)
    --device      'auto', 'cpu', or 'cuda'  (default: auto)
    --batch-size  Images per forward pass (default: 16; use 8 for SigLIP2/Hybrid)
    --no-plots    Skip matplotlib plots (useful on headless servers)
"""

import argparse
import os
import sys

# M1 Mac: suppress duplicate OpenMP runtime warning from PyTorch + FAISS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from benchmarks.benchmark_config import DATABASE_PATH, RESULTS_DIR
from benchmarks.embedders.clip_embedder import CLIPEmbedder
from benchmarks.embedders.dinov2_embedder import DINOv2Embedder
from benchmarks.embedders.hybrid_embedder import HybridEmbedder
from benchmarks.embedders.siglip2_embedder import SigLIP2Embedder
from benchmarks.evaluation.loo_evaluator import LOOEvaluator
from benchmarks.reporting.table_reporter import export_summary_csv, print_comparison_table

EMBEDDER_REGISTRY = {
    "clip":    CLIPEmbedder,
    "siglip2": SigLIP2Embedder,
    "dinov2":  DINOv2Embedder,
    "hybrid":  HybridEmbedder,
}
ALL_MODELS = list(EMBEDDER_REGISTRY.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="LOO retrieval benchmark")
    parser.add_argument("--models",     nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, metavar="MODEL")
    parser.add_argument("--db-path",    default=DATABASE_PATH)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-plots",   action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running LOO benchmark for models: {args.models}")
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
        evaluator = LOOEvaluator(embedder, database_path=args.db_path)
        result = evaluator.run()
        results.append(result)

        # Per-model per-query CSV
        csv_path = evaluator.export_per_query_csv(result, args.output_dir)
        print(f"Per-query CSV → {csv_path}")

        # Quick summary line
        print(
            f"  R@1={result.recall.get(1,0):.3f}  "
            f"R@3={result.recall.get(3,0):.3f}  "
            f"R@5={result.recall.get(5,0):.3f}  "
            f"MRR={result.mrr:.3f}  "
            f"Queries={result.n_queries}"
        )

    # Comparison table + summary CSV
    print_comparison_table(results)
    export_summary_csv(results, args.output_dir)

    # Plots
    if not args.no_plots:
        try:
            from benchmarks.reporting.plot_reporter import (
                plot_per_brand_heatmap,
                plot_threshold_sensitivity,
            )
            plot_threshold_sensitivity(results, args.output_dir)
            plot_per_brand_heatmap(results, args.output_dir)
        except ImportError as e:
            print(f"Plotting skipped (matplotlib not available): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
