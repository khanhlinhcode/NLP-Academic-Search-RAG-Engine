"""Deterministic information-retrieval metrics for binary or graded qrels."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import numpy as np


def _positive_ids(relevance: set[str] | dict[str, int]) -> set[str]:
    if isinstance(relevance, set):
        return relevance
    return {document_id for document_id, grade in relevance.items() if grade > 0}


MetricValue: TypeAlias = float | int | str | None
MetricSummary: TypeAlias = dict[str, MetricValue]


def _ranked_prefix(retrieved_ids: list[str], k: int) -> list[str]:
    """Extract top-k unique document IDs in rank order (First-Occurrence Deduplication Policy).

    Duplicate retrieved IDs are deduplicated preserving the rank of their first appearance.
    If k <= 0 or retrieved_ids is empty, returns an empty list.
    """
    if k <= 0 or not retrieved_ids:
        return []
    seen = set()
    unique_ids = []
    for doc_id in retrieved_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_ids.append(doc_id)
            if len(unique_ids) == k:
                break
    return unique_ids


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@K clamped to [0.0, 1.0]."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    ranked = _ranked_prefix(retrieved_ids, k)
    hits = sum(item in relevant_ids for item in ranked)
    return min(1.0, max(0.0, hits / k))


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K clamped to [0.0, 1.0]."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    ranked = _ranked_prefix(retrieved_ids, k)
    hits = sum(item in relevant_ids for item in ranked)
    return min(1.0, max(0.0, hits / len(relevant_ids)))


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Mean Reciprocal Rank @ K clamped to [0.0, 1.0]."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    ranked = _ranked_prefix(retrieved_ids, k)
    score = next(
        (1.0 / rank for rank, item in enumerate(ranked, start=1) if item in relevant_ids),
        0.0,
    )
    return min(1.0, max(0.0, score))


def average_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Average Precision @ K clamped to [0.0, 1.0]."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    ranked = _ranked_prefix(retrieved_ids, k)
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    denom = min(len(relevant_ids), k)
    if denom <= 0:
        return 0.0
    return min(1.0, max(0.0, precision_sum / denom))


def ndcg_at_k(retrieved_ids: list[str], relevance: set[str] | dict[str, int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K clamped to [0.0, 1.0]."""
    if k <= 0 or not retrieved_ids or not relevance:
        return 0.0
    ranked = _ranked_prefix(retrieved_ids, k)
    grades = {item: 1 for item in relevance} if isinstance(relevance, set) else relevance
    gains = [grades.get(item, 0) for item in ranked]
    dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(gains, 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    if idcg <= 0:
        return 0.0
    return min(1.0, max(0.0, dcg / idcg))


def evaluate_search_method(
    search_fn: Callable[..., list[Any]], queries: list[dict[str, Any]], k: int = 10
) -> MetricSummary:
    if k <= 0:
        raise ValueError("k must be positive")
    if not queries:
        return {
            f"Precision@{k}": 0.0,
            f"Recall@{k}": 0.0,
            f"MRR@{k}": 0.0,
            f"MAP@{k}": 0.0,
            f"nDCG@{k}": 0.0,
            f"HitRate@{k}": 0.0,
            "nDCG_95pct_CI_lower": None,
            "nDCG_95pct_CI_upper": None,
            "nDCG_CI_method": "unavailable for n<2",
            "query_count": 0,
            "Latency_p50_ms": None,
            "Latency_p95_ms": None,
            "Latency_p99_ms": None,
        }

    values: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "mrr": [],
        "map": [],
        "ndcg": [],
        "hit": [],
    }
    latencies = []
    for query in queries:
        started = time.perf_counter()
        results = search_fn(query["query"], top_k=k)
        latencies.append((time.perf_counter() - started) * 1000)
        retrieved = [result.paper.id for result in results]
        if "qrels" in query and query["qrels"]:
            relevance = query["qrels"]
        elif "relevant_ids" in query and query["relevant_ids"]:
            relevance = {item: 1 for item in query["relevant_ids"]}
        else:
            relevance = {}
        relevant = _positive_ids(relevance)
        values["precision"].append(precision_at_k(retrieved, relevant, k))
        values["recall"].append(recall_at_k(retrieved, relevant, k))
        values["mrr"].append(mrr_at_k(retrieved, relevant, k))
        values["map"].append(average_precision_at_k(retrieved, relevant, k))
        values["ndcg"].append(ndcg_at_k(retrieved, relevance, k))
        values["hit"].append(float(bool(set(_ranked_prefix(retrieved, k)) & relevant)))

    def mean(name: str) -> float:
        return float(np.mean(values[name])) if values[name] else 0.0

    ndcg_list = values["ndcg"]
    n_queries = len(ndcg_list)
    if n_queries >= 2:
        stdev = statistics.stdev(ndcg_list)
        margin = 1.96 * stdev / math.sqrt(n_queries)
        ndcg_mean = mean("ndcg")
        ci_lower = round(max(0.0, ndcg_mean - margin), 4)
        ci_upper = round(min(1.0, ndcg_mean + margin), 4)
        ci_method = "normal-approximation (n>=2)"
    else:
        ci_lower = None
        ci_upper = None
        ci_method = "unavailable for n<2"

    return {
        f"Precision@{k}": round(mean("precision"), 4),
        f"Recall@{k}": round(mean("recall"), 4),
        f"MRR@{k}": round(mean("mrr"), 4),
        f"MAP@{k}": round(mean("map"), 4),
        f"nDCG@{k}": round(mean("ndcg"), 4),
        f"HitRate@{k}": round(mean("hit"), 4),
        "nDCG_95pct_CI_lower": ci_lower,
        "nDCG_95pct_CI_upper": ci_upper,
        "nDCG_CI_method": ci_method,
        "query_count": n_queries,
        "Latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "Latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "Latency_p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else None,
    }


def format_evaluation_table(results: Mapping[str, Mapping[str, MetricValue]]) -> str:
    if not results:
        return "No results to display."
    metrics = list(next(iter(results.values())))
    rows = ["| Method | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for method, scores in results.items():
        rows.append(
            "| " + method + " | " + " | ".join(str(scores[name]) for name in metrics) + " |"
        )
    return "\n".join(rows)
