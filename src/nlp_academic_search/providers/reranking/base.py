"""Provider-neutral reranker protocol."""

from __future__ import annotations

from typing import Protocol

from nlp_academic_search.search.models import SearchResult


class RerankerProvider(Protocol):
    provider_name: str
    model_name: str

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
