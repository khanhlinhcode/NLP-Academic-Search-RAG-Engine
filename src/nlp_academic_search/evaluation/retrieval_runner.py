"""Reproducible retrieval benchmark orchestration."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper
from nlp_academic_search.data.manifest import sha256_file
from nlp_academic_search.evaluation.metrics import MetricSummary, evaluate_search_method
from nlp_academic_search.search.bm25_search import BM25Searcher
from nlp_academic_search.search.hybrid_search import FusionMethod, HybridSearcher
from nlp_academic_search.search.models import SearchResult
from nlp_academic_search.search.semantic_search import SemanticSearcher


class RetrievalReport(TypedDict):
    benchmark: dict[str, Any]
    run: dict[str, Any]
    results: dict[str, MetricSummary]
    limitations: list[str]


def run_retrieval_evaluation(
    benchmark_path: Path,
    k: int,
    *,
    include_reranker: bool = False,
    bm25_weight: float | None = None,
    candidate_pool: int | None = None,
    rrf_k: int = 60,
    reranker_model: str | None = None,
) -> RetrievalReport:
    """Evaluate sparse, dense, weighted fusion, RRF and optional reranking on a versioned benchmark."""
    if k < 1:
        raise ValueError("k must be positive")
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    papers = [Paper.from_dict(item) for item in payload["documents"]]
    queries = payload["queries"]
    if payload.get("kind") == "human-authored-small-fixture" and any(
        query["query"].strip().casefold() in paper.abstract.casefold()
        for query in queries
        for paper in papers
    ):
        raise ValueError("Benchmark rejected: a query is copied verbatim from a document abstract")
    bm25 = BM25Searcher(papers)
    semantic = SemanticSearcher(papers, load_existing=False, corpus_path=benchmark_path)
    effective_weight = settings.search.bm25_weight if bm25_weight is None else bm25_weight
    effective_pool = candidate_pool or settings.search.candidate_pool
    if effective_pool < k:
        raise ValueError("candidate_pool must be greater than or equal to k")
    hybrid = HybridSearcher(bm25, semantic, alpha=effective_weight)

    results = {
        "BM25": evaluate_search_method(bm25.search, queries, k=k),
        "Dense": evaluate_search_method(semantic.search, queries, k=k),
        "RRF": evaluate_search_method(
            lambda q, top_k: hybrid.search(
                q,
                top_k=top_k,
                method=FusionMethod.RRF,
                candidate_pool=effective_pool,
                rrf_k=rrf_k,
            ),
            queries,
            k=k,
        ),
        "Weighted": evaluate_search_method(
            lambda q, top_k: hybrid.search(
                q,
                top_k=top_k,
                method=FusionMethod.WEIGHTED,
                candidate_pool=effective_pool,
            ),
            queries,
            k=k,
        ),
    }

    reranker_info = "disabled"
    if include_reranker:
        from nlp_academic_search.search.reranker import Reranker

        reranker = Reranker(model_name=reranker_model)
        reranker_info = reranker.model_name

        def hybrid_reranked_search(query: str, top_k: int) -> list[SearchResult]:
            candidates = hybrid.search(
                query,
                top_k=effective_pool,
                method=FusionMethod.RRF,
                candidate_pool=effective_pool,
                rrf_k=rrf_k,
            )
            return reranker.rerank(query, candidates, top_k=top_k)

        results["Hybrid+Reranker"] = evaluate_search_method(hybrid_reranked_search, queries, k=k)

    return {
        "benchmark": {
            "name": payload.get("name", "unknown"),
            "kind": payload.get("kind", "standard-benchmark"),
            "provenance": payload.get("provenance", "unknown"),
            "sha256": sha256_file(benchmark_path),
            "query_count": len(queries),
            "document_count": len(papers),
            "standard_benchmark": payload.get("kind", "evaluated"),
        },
        "run": {
            "timestamp": datetime.now(UTC).isoformat(),
            "hardware": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "embedding_model": settings.embedding.model_name,
            "embedding_revision": settings.embedding.model_revision or "latest",
            "reranker_model": reranker_info,
            "bm25_weight": effective_weight,
            "semantic_weight": 1.0 - effective_weight,
            "rrf_k": rrf_k,
            "candidate_pool": effective_pool,
            "k": k,
        },
        "results": results,
        "limitations": [
            "Repository fixture benchmark measures internal sanity; external BEIR/SciFact measures generalization.",
            "Concurrent throughput is not evaluated by this fixture.",
        ],
    }
