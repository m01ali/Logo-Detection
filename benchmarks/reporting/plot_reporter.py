"""Matplotlib plots for benchmark results."""

import os
from typing import List

from benchmarks.benchmark_config import PRODUCTION_THRESHOLD, RESULTS_DIR
from benchmarks.evaluation.loo_evaluator import LOOResult

# Colour palette — one per model
_MODEL_COLOURS = {
    "clip":    "#2196F3",  # blue
    "siglip2": "#4CAF50",  # green
    "dinov2":  "#FF9800",  # orange
    "hybrid":  "#9C27B0",  # purple
}
_DEFAULT_COLOUR = "#607D8B"


def plot_threshold_sensitivity(
    results: List[LOOResult],
    output_dir: str = RESULTS_DIR,
    metric: str = "recall_at_k",
    k: int = 1,
) -> str:
    """
    X-axis: similarity threshold (production score scale).
    Y-axis: Recall@k (default), MRR, or Coverage.
    One line per model. Vertical dashed line at production default threshold.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "threshold_sensitivity.png")

    fig, ax = plt.subplots(figsize=(9, 5))

    for result in results:
        thresholds = sorted(result.threshold_curve.keys())
        values = [result.threshold_curve[t][metric] for t in thresholds]
        colour = _MODEL_COLOURS.get(result.model_name, _DEFAULT_COLOUR)
        ax.plot(
            thresholds, values,
            marker="o", linewidth=2, markersize=5,
            color=colour, label=result.model_name,
        )

    ax.axvline(
        PRODUCTION_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
        label=f"Production threshold ({PRODUCTION_THRESHOLD})",
    )

    metric_label = {
        "recall_at_k": f"Recall@{k}",
        "mrr":         "MRR",
        "coverage":    "Coverage",
    }.get(metric, metric)

    ax.set_xlabel("Similarity threshold (production score: 1 − dist/4)", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Threshold Sensitivity — {metric_label}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Threshold plot → {output_path}")
    return output_path


def plot_per_brand_heatmap(
    results: List[LOOResult],
    output_dir: str = RESULTS_DIR,
) -> str:
    """
    Heatmap: rows = brands, cols = models, values = per-brand Recall@1.
    0 (red) → 1 (green).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "per_brand_heatmap.png")

    # Collect all brands across all results
    all_brands = sorted({b for r in results for b in r.per_brand_recall})
    model_names = [r.model_name for r in results]

    matrix = np.zeros((len(all_brands), len(model_names)))
    for j, result in enumerate(results):
        for i, brand in enumerate(all_brands):
            matrix[i, j] = result.per_brand_recall.get(brand, 0.0)

    fig, ax = plt.subplots(
        figsize=(max(6, len(model_names) * 2), max(5, len(all_brands) * 0.55))
    )
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_yticks(range(len(all_brands)))
    ax.set_yticklabels(all_brands, fontsize=9)
    ax.set_title("Per-Brand Recall@1 by Model", fontsize=13)

    # Annotate cells
    for i in range(len(all_brands)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            text_colour = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_colour)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Recall@1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap → {output_path}")
    return output_path
