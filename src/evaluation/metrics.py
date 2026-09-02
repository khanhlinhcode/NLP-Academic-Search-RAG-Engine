"""
Evaluation metrics for Information Retrieval.

Implements standard IR metrics to compare different search methods:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Latency measurement
"""

import time
from typing import Callable, Dict, List, Set

import numpy as np

from src.search.bm25_search import SearchResult


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Precision@K — Fraction of retrieved documents that are relevant.

    P@K = |{relevant ∩ retrieved[:K]}| / K

    Args:
        retrieved_ids: List of document IDs returned by search (in order).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cutoff position.

    Returns:
        Precision score in [0, 1].
    """
    retrieved_at_k = retrieved_ids[:k]
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_ids)
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@K — Fraction of relevant documents that were retrieved.

    R@K = |{relevant ∩ retrieved[:K]}| / |relevant|

    Args:
        retrieved_ids: List of document IDs returned by search (in order).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cutoff position.

    Returns:
        Recall score in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_ids)
    return relevant_count / len(relevant_ids)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Mean Reciprocal Rank @ K — Reciprocal of the rank of the first relevant result.

    MRR@K = 1 / rank(first relevant document in top K)

    Args:
        retrieved_ids: List of document IDs returned by search (in order).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cutoff position.

    Returns:
        MRR score in [0, 1]. Returns 0 if no relevant document in top K.
    """
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Measures ranking quality considering the position of relevant results.
    DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K
    nDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: List of document IDs returned by search (in order).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cutoff position.

    Returns:
        nDCG score in [0, 1].
    """
    # DCG: Discounted Cumulative Gain
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 1)

    # IDCG: Ideal DCG (all relevant docs ranked first)
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_count + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_search_method(
    search_fn: Callable,
    queries: List[dict],
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate a search method across multiple queries.

    Args:
        search_fn: Search function that takes (query: str, top_k: int) and
                   returns List[SearchResult].
        queries: List of dicts with 'query' (str) and 'relevant_ids' (List[str]).
        k: Cutoff position for metrics.

    Returns:
        Dictionary with averaged metric scores and latency.
    """
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []
    latencies = []

    for i, q_data in enumerate(queries, start=1):
        query = q_data["query"]
        relevant_ids = set(q_data["relevant_ids"])

        # Measure latency
        start_time = time.time()
        results = search_fn(query, top_k=k)
        latency = (time.time() - start_time) * 1000  # ms

        retrieved_ids = [r.paper.id for r in results]

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_ids, k))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))
        latencies.append(latency)

    return {
        f"Precision@{k}": round(np.mean(precisions), 4),
        f"Recall@{k}": round(np.mean(recalls), 4),
        f"MRR@{k}": round(np.mean(mrrs), 4),
        f"nDCG@{k}": round(np.mean(ndcgs), 4),
        "Latency_ms": round(np.mean(latencies), 2),
    }


def format_evaluation_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format evaluation results as a markdown table.

    Args:
        results: Dict mapping method name to metric scores.

    Returns:
        Formatted markdown table string.
    """
    if not results:
        return "No results to display."

    # Get all metric names from the first result
    metrics = list(next(iter(results.values())).keys())

    # Header
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|--------|" + "|".join("-" * (len(m) + 2) for m in metrics) + "|"

    # Rows
    rows = []
    for method_name, scores in results.items():
        values = " | ".join(str(scores.get(m, "N/A")) for m in metrics)
        rows.append(f"| {method_name} | {values} |")

    return "\n".join([header, separator] + rows)
