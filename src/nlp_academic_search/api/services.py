"""Application service container and bounded execution policies."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastapi import Request

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper
from nlp_academic_search.providers.generation.base import GenerationProvider, ModelUnavailableError
from nlp_academic_search.providers.reranking.base import RerankerProvider
from nlp_academic_search.providers.retrieval.base import (
    RetrievalBatch,
    RetrievalProvider,
    RetrievalStatus,
)
from nlp_academic_search.providers.verification.base import SemanticVerificationProvider
from nlp_academic_search.search.models import FusionMethod, SearchFilters, SearchResult

T = TypeVar("T")


class ServiceTimeoutError(RuntimeError):
    pass


class ServiceBusyError(RuntimeError):
    pass


@dataclass
class ServiceContainer:
    papers: list[Paper] = field(default_factory=list)
    bm25: Any | None = None
    semantic: Any | None = None
    hybrid: Any | None = None
    rag_generator: GenerationProvider | Any | None = None
    reranker: RerankerProvider | Any | None = None
    retrieval_provider: RetrievalProvider | None = None
    semantic_verifier: SemanticVerificationProvider | None = None
    started_at: float = field(default_factory=time.monotonic)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=settings.api.concurrency_limit, thread_name_prefix="retrieval"
        )
    )
    dense_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="dense")
    )
    generation_slots: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(settings.ollama.concurrency_limit)
    )
    _status_cache: tuple[float, RetrievalStatus] | None = field(default=None, init=False)

    def run(
        self,
        function: Callable[[], T],
        timeout: float,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> T:
        future = (executor or self.executor).submit(function)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise ServiceTimeoutError("The pipeline stage exceeded its timeout") from exc

    def search(
        self,
        query: str,
        method: str,
        top_k: int,
        *,
        filters: SearchFilters | None = None,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> list[SearchResult]:
        return self.search_batch(query, method, top_k, filters=filters, fusion=fusion).results

    def search_batch(
        self,
        query: str,
        method: str,
        top_k: int,
        *,
        filters: SearchFilters | None = None,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> RetrievalBatch:
        def execute() -> RetrievalBatch | list[SearchResult]:
            if self.retrieval_provider is not None:
                return self.retrieval_provider.search(
                    query, method, top_k, filters=filters, fusion=fusion
                )
            if method == "bm25":
                if self.bm25 is None:
                    raise RuntimeError("BM25 retrieval is not configured")
                return self.bm25.search(query, top_k=top_k, filters=filters)
            if method == "semantic":
                if self.semantic is None:
                    raise RuntimeError("Semantic retrieval is not configured")
                return self.semantic.search(query, top_k=top_k, filters=filters)
            if self.hybrid is None:
                raise RuntimeError("Hybrid retrieval is not configured")
            return self.hybrid.search(
                query,
                top_k=top_k,
                method=fusion,
                candidate_pool=max(settings.search.candidate_pool, top_k),
                filters=filters,
            )

        executor = self.executor if method == "bm25" else self.dense_executor
        value = self.run(execute, settings.search.timeout_seconds, executor=executor)
        if isinstance(value, RetrievalBatch):
            return value
        mode = fusion.value if method == "hybrid" else method
        return RetrievalBatch(results=value, retrieval_mode=mode)

    def retrieve_for_rag(
        self, question: str, top_k: int, use_reranker: bool
    ) -> tuple[list[SearchResult], list[str], str]:
        if not settings.rag_enabled:
            raise ModelUnavailableError("RAG is disabled by configuration")
        warnings = []
        candidate_k = max(top_k, min(settings.search.candidate_pool, top_k * 4))
        batch = self.search_batch(question, "hybrid", candidate_k)
        results = batch.results
        warnings = list(batch.warnings)
        method = batch.retrieval_mode
        if use_reranker:
            if self.reranker is None:
                warnings.append("Reranker requested but unavailable; using RRF order.")
            else:
                reranker = self.reranker
                results = self.run(
                    lambda: reranker.rerank(question, results, top_k=top_k),
                    settings.reranker.timeout_seconds,
                )
                method = "rrf+reranker"
        return results[:top_k], warnings, method

    def acquire_generation(self) -> None:
        if not self.generation_slots.acquire(blocking=False):
            raise ServiceBusyError("Generation capacity is busy; retry shortly")

    def release_generation(self) -> None:
        self.generation_slots.release()

    def ollama_available(self) -> bool:
        if not settings.rag_enabled or self.rag_generator is None:
            return False
        checker = getattr(self.rag_generator, "is_available", None)
        return bool(checker()) if callable(checker) else False

    def generation_available(self) -> bool:
        if not settings.rag_enabled or self.rag_generator is None:
            return False
        if settings.generation_provider == "ollama":
            return self.ollama_available()
        checker = getattr(self.rag_generator, "is_available", None)
        return bool(checker()) if callable(checker) else False

    def semantic_verification_available(self) -> bool:
        if not settings.verification.enabled or self.semantic_verifier is None:
            return False
        checker = getattr(self.semantic_verifier, "is_available", None)
        return bool(checker()) if callable(checker) else False

    def retrieval_status(self, *, use_cache: bool = True) -> RetrievalStatus:
        now = time.monotonic()
        if (
            use_cache
            and self._status_cache is not None
            and now - self._status_cache[0] < settings.health_cache_seconds
        ):
            return self._status_cache[1]
        if self.retrieval_provider is not None:
            status = self.retrieval_provider.status()
        elif self.semantic is not None:
            try:
                manifest = self.semantic.manifest
                manifest.validate(
                    corpus_path=self.semantic.corpus_path,
                    papers=self.papers,
                    model_name=self.semantic.model_name,
                    model_revision=self.semantic.model_revision,
                    embeddings=self.semantic.embeddings,
                    index=self.semantic.index,
                )
                status = RetrievalStatus(
                    ready=True,
                    total_papers=len(self.papers),
                    provenance=manifest.provenance,
                    embedding_model=manifest.embedding_model,
                    embedding_revision=manifest.embedding_revision,
                    embedding_dimension=manifest.embedding_dimension,
                )
            except (OSError, ValueError) as exc:
                status = RetrievalStatus(False, len(self.papers), reason=str(exc))
        else:
            status = RetrievalStatus(False, reason="retrieval provider is not configured")
        self._status_cache = (now, status)
        return status

    def index_compatible(self) -> bool:
        return self.retrieval_status().ready

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.dense_executor.shutdown(wait=False, cancel_futures=True)
        if self.retrieval_provider is not None:
            self.retrieval_provider.close()
        if self.rag_generator is not None:
            close = getattr(self.rag_generator, "close", None)
            if callable(close):
                close()
        if self.semantic_verifier is not None:
            close = getattr(self.semantic_verifier, "close", None)
            if callable(close):
                close()


def build_services() -> ServiceContainer:
    from nlp_academic_search.providers.factory import build_provider_bundle

    providers = build_provider_bundle()
    local = providers.retrieval if providers.retrieval.provider_name == "local" else None
    return ServiceContainer(
        papers=getattr(local, "papers", []),
        bm25=getattr(local, "bm25", None),
        semantic=getattr(local, "semantic", None),
        hybrid=getattr(local, "hybrid", None),
        reranker=providers.reranker,
        retrieval_provider=providers.retrieval,
        rag_generator=providers.generation,
        semantic_verifier=providers.verifier,
    )


def get_services(request: Request) -> ServiceContainer:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise RuntimeError("Application services are not initialized")
    return services
