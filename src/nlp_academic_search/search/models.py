"""Shared retrieval result and filter contracts."""

from __future__ import annotations

from dataclasses import dataclass

from nlp_academic_search.data.loader import Paper


@dataclass(frozen=True)
class SearchFilters:
    category: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    author: str | None = None

    def matches(self, paper: Paper) -> bool:
        if self.category and self.category not in paper.categories:
            return False
        if self.year_from and (paper.year is None or paper.year < self.year_from):
            return False
        if self.year_to and (paper.year is None or paper.year > self.year_to):
            return False
        if self.author:
            needle = self.author.casefold()
            if not any(needle in author.casefold() for author in paper.authors):
                return False
        return True


@dataclass
class SearchResult:
    paper: Paper
    bm25_score: float | None = None
    semantic_score: float | None = None
    rrf_score: float | None = None
    weighted_score: float | None = None
    reranker_score: float | None = None

    @property
    def score(self) -> float:
        for value in (
            self.reranker_score,
            self.rrf_score,
            self.weighted_score,
            self.semantic_score,
            self.bm25_score,
        ):
            if value is not None:
                return value
        return 0.0

    @property
    def score_type(self) -> str:
        for name in (
            "reranker_score",
            "rrf_score",
            "weighted_score",
            "semantic_score",
            "bm25_score",
        ):
            if getattr(self, name) is not None:
                return name
        return "unscored"

    def to_dict(self) -> dict[str, object]:
        result = self.paper.to_dict()
        result.update(
            {
                "year": self.paper.year,
                "score": round(self.score, 6),
                "score_type": self.score_type,
                "bm25_score": self.bm25_score,
                "semantic_score": self.semantic_score,
                "rrf_score": self.rrf_score,
                "weighted_score": self.weighted_score,
                "reranker_score": self.reranker_score,
            }
        )
        return result
