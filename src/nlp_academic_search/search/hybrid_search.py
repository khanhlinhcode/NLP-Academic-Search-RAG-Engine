"""Hybrid sparse/dense retrieval with explicit fusion scores."""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from nlp_academic_search.config import settings
from nlp_academic_search.search.bm25_search import BM25Searcher
from nlp_academic_search.search.models import SearchFilters, SearchResult
from nlp_academic_search.search.semantic_search import SemanticSearcher


class FusionMethod(StrEnum):
    WEIGHTED = "weighted"
    RRF = "rrf"


class HybridSearcher:
    def __init__(
        self,
        bm25: BM25Searcher,
        semantic: SemanticSearcher,
        alpha: float | None = None,
    ) -> None:
        if len(bm25.papers) != len(semantic.papers):
            raise ValueError("BM25 and semantic search must use the same corpus")
        self.bm25 = bm25
        self.semantic = semantic
        self.alpha = settings.search.bm25_weight if alpha is None else alpha
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

    def search(
        self,
        query: str,
        top_k: int = 10,
        method: FusionMethod = FusionMethod.RRF,
        candidate_pool: int | None = None,
        *,
        filters: SearchFilters | None = None,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        pool = candidate_pool or settings.search.candidate_pool
        if pool < top_k:
            raise ValueError("candidate_pool must be greater than or equal to top_k")
        if method == FusionMethod.WEIGHTED:
            return self._weighted_fusion(query, top_k, filters)
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        return self._rrf_fusion(query, top_k, pool, filters, rrf_k)

    def _weighted_fusion(
        self, query: str, top_k: int, filters: SearchFilters | None
    ) -> list[SearchResult]:
        bm25_scores = self.bm25.get_scores(query)
        semantic_scores = self.semantic.get_scores(query)
        bm25_normalized = self._min_max(bm25_scores)
        semantic_normalized = self._min_max(semantic_scores)
        fused = self.alpha * bm25_normalized + (1 - self.alpha) * semantic_normalized
        results = []
        active_filters = filters or SearchFilters()
        for index in np.argsort(fused)[::-1]:
            if fused[index] <= 0:
                break
            paper = self.bm25.papers[int(index)]
            if active_filters.matches(paper):
                results.append(
                    SearchResult(
                        paper=paper,
                        bm25_score=float(bm25_scores[index]),
                        semantic_score=float(semantic_scores[index]),
                        weighted_score=float(fused[index]),
                    )
                )
            if len(results) == top_k:
                break
        return results

    @staticmethod
    def _min_max(scores: np.ndarray) -> np.ndarray:
        minimum, maximum = scores.min(), scores.max()
        if maximum <= minimum:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - minimum) / (maximum - minimum)

    def _rrf_fusion(
        self,
        query: str,
        top_k: int,
        candidate_pool: int,
        filters: SearchFilters | None,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=candidate_pool, filters=filters)
        semantic_results = self.semantic.search(query, top_k=candidate_pool, filters=filters)
        combined: dict[str, SearchResult] = {}
        for rank, result in enumerate(bm25_results, start=1):
            combined[result.paper.id] = SearchResult(
                paper=result.paper,
                bm25_score=result.bm25_score,
                rrf_score=1.0 / (rrf_k + rank),
            )
        for rank, result in enumerate(semantic_results, start=1):
            increment = 1.0 / (rrf_k + rank)
            existing = combined.get(result.paper.id)
            if existing:
                existing.semantic_score = result.semantic_score
                existing.rrf_score = (existing.rrf_score or 0.0) + increment
            else:
                combined[result.paper.id] = SearchResult(
                    paper=result.paper,
                    semantic_score=result.semantic_score,
                    rrf_score=increment,
                )
        return sorted(combined.values(), key=lambda item: item.score, reverse=True)[:top_k]
