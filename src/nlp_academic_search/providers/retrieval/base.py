"""Lightweight retrieval contracts shared by local and cloud deployments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from nlp_academic_search.search.models import FusionMethod, SearchFilters, SearchResult


class RetrievalProviderError(RuntimeError):
    """Base retrieval-provider failure."""


class RetrievalUnavailableError(RetrievalProviderError):
    """The configured retrieval backend is unavailable or incompatible."""


@dataclass(frozen=True)
class RetrievalBatch:
    results: list[SearchResult]
    retrieval_mode: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalStatus:
    ready: bool
    total_papers: int = 0
    provenance: str | None = None
    embedding_model: str | None = None
    embedding_revision: str | None = None
    embedding_dimension: int | None = None
    reason: str | None = None


class RetrievalProvider(Protocol):
    provider_name: str

    def search(
        self,
        query: str,
        method: str,
        top_k: int,
        *,
        filters: SearchFilters | None = None,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> RetrievalBatch: ...

    def status(self) -> RetrievalStatus: ...

    def close(self) -> None: ...
