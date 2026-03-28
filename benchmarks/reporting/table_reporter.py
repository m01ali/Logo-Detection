"""Terminal table and CSV export for benchmark results."""

import csv
import os
from typing import List

from benchmarks.benchmark_config import RESULTS_DIR
from benchmarks.evaluation.loo_evaluator import LOOResult


def print_comparison_table(results: List[LOOResult]) -> None:
    """Print a formatted comparison table sorted by R@1 descending."""
    try:
        from tabulate import tabulate
        _tabulate_available = True
    except ImportError:
        _tabulate_available = False

    sorted_results = sorted(results, key=lambda r: r.recall.get(1, 0), reverse=True)

    headers = ["Model", "Dim", "R@1", "R@3", "R@5", "MRR", "Queries"]
    rows = []
    for r in sorted_results:
        from benchmarks.benchmark_config import EMBEDDING_DIMS
        dim = EMBEDDING_DIMS.get(r.model_name, "?")
        rows.append([
            r.model_name,
            dim,
            f"{r.recall.get(1, 0):.3f}",
            f"{r.recall.get(3, 0):.3f}",
            f"{r.recall.get(5, 0):.3f}",
            f"{r.mrr:.3f}",
            r.n_queries,
        ])

    print("\n" + "=" * 60)
    print("LOGO RETRIEVAL BENCHMARK — LOO RESULTS")
    print("=" * 60)

    if _tabulate_available:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        # Fallback plain-text table
        col_w = [max(len(str(h)), max(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*row))

    print()
    if results and results[0].n_skipped > 0:
        print(f"  NOTE: {results[0].n_skipped} brands skipped (only 1 image each).")


def export_summary_csv(results: List[LOOResult], output_dir: str = RESULTS_DIR) -> str:
    """Export one-row-per-model summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "loo_summary.csv")

    from benchmarks.benchmark_config import EMBEDDING_DIMS
    sorted_results = sorted(results, key=lambda r: r.recall.get(1, 0), reverse=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dim", "recall_at_1", "recall_at_3", "recall_at_5",
                         "mrr", "n_queries", "n_skipped"])
        for r in sorted_results:
            writer.writerow([
                r.model_name,
                EMBEDDING_DIMS.get(r.model_name, "?"),
                f"{r.recall.get(1, 0):.4f}",
                f"{r.recall.get(3, 0):.4f}",
                f"{r.recall.get(5, 0):.4f}",
                f"{r.mrr:.4f}",
                r.n_queries,
                r.n_skipped,
            ])

    print(f"Summary CSV → {path}")
    return path
