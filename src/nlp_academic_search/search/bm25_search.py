"""Sparse BM25 retrieval."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.data.preprocessor import tokenize_for_bm25
from nlp_academic_search.search.models import SearchFilters, SearchResult

__all__ = ["BM25Searcher", "SearchResult"]


class BM25Searcher:
    def __init__(self, papers: list[Paper]):
        if not papers:
            raise ValueError("BM25 requires at least one paper")
        self.papers = papers
        self.bm25 = BM25Okapi([tokenize_for_bm25(paper.text) for paper in papers])

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if top_k < 1 or not query.strip():
            return []
        scores = self.get_scores(query)
        results = []
        active_filters = filters or SearchFilters()
        for index in np.argsort(scores)[::-1]:
            if scores[index] <= 0:
                break
            paper = self.papers[int(index)]
            if active_filters.matches(paper):
                results.append(SearchResult(paper=paper, bm25_score=float(scores[index])))
            if len(results) == top_k:
                break
        return results

    def get_scores(self, query: str) -> np.ndarray:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return np.zeros(len(self.papers), dtype=np.float32)
        return self.bm25.get_scores(tokens)
