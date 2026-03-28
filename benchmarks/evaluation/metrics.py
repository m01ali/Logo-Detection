"""
Pure metric functions for retrieval benchmarking.
No I/O, no model loading — all inputs are plain Python lists / numpy arrays.

Metric definitions
------------------
Recall@k   : 1 if true_brand appears in the top-k retrieved list, else 0.
             Macro-averaged across all queries → mean_recall_at_k().

MRR        : Mean Reciprocal Rank. For each query, RR = 1/rank if true_brand
             is found within the retrieved list, else 0. rank is 1-indexed.

Coverage(θ): Fraction of queries that returned at least one result with
             score >= threshold θ. Useful for understanding how aggressive
             a threshold is.

threshold_sensitivity(): sweeps θ and returns Recall@k, MRR, Coverage at each.
"""

from typing import List, Dict


def recall_at_k(retrieved_brands: List[str], true_brand: str, k: int) -> float:
    """Recall@k for a single query. retrieved_brands sorted descending by score."""
    return 1.0 if true_brand in retrieved_brands[:k] else 0.0


def mean_recall_at_k(
    all_retrieved: List[List[str]],
    all_true_brands: List[str],
    k: int,
) -> float:
    """Macro-average Recall@k across all queries."""
    if not all_retrieved:
        return 0.0
    scores = [recall_at_k(r, t, k) for r, t in zip(all_retrieved, all_true_brands)]
    return sum(scores) / len(scores)


def reciprocal_rank(retrieved_brands: List[str], true_brand: str) -> float:
    """
    Reciprocal rank for a single query.
    RR = 1/rank (1-indexed) if found, else 0.0.
    """
    try:
        rank = retrieved_brands.index(true_brand) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_true_brands: List[str],
) -> float:
    """MRR across all queries."""
    if not all_retrieved:
        return 0.0
    rrs = [reciprocal_rank(r, t) for r, t in zip(all_retrieved, all_true_brands)]
    return sum(rrs) / len(rrs)


def threshold_sensitivity(
    all_scores: List[List[float]],
    all_brands: List[List[str]],
    all_true_brands: List[str],
    thresholds: List[float],
    k: int = 1,
) -> Dict[float, Dict[str, float]]:
    """
    For each threshold, filter retrieved results to those >= threshold,
    then compute Recall@k, MRR, and Coverage.

    Args:
        all_scores:      Per-query list of similarity scores (descending order).
        all_brands:      Per-query list of brand names parallel to all_scores.
        all_true_brands: Ground-truth brand per query.
        thresholds:      List of threshold values to evaluate.
        k:               k for Recall@k inside each threshold bucket.

    Returns:
        {threshold: {"recall_at_k": float, "mrr": float, "coverage": float}}
    """
    results: Dict[float, Dict[str, float]] = {}
    n = len(all_true_brands)

    for thresh in thresholds:
        filtered_brands: List[List[str]] = []
        has_result: List[bool] = []

        for scores_q, brands_q in zip(all_scores, all_brands):
            above = [b for s, b in zip(scores_q, brands_q) if s >= thresh]
            filtered_brands.append(above)
            has_result.append(len(above) > 0)

        results[thresh] = {
            "recall_at_k": mean_recall_at_k(filtered_brands, all_true_brands, k),
            "mrr":         mean_reciprocal_rank(filtered_brands, all_true_brands),
            "coverage":    sum(has_result) / n if n > 0 else 0.0,
        }

    return results
