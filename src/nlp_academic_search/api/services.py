"""Application service container and bounded execution policies."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from fastapi import Request

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper, load_papers
from nlp_academic_search.rag.generator import ModelUnavailableError, RAGGenerator
from nlp_academic_search.search.bm25_search import BM25Searcher
from nlp_academic_search.search.hybrid_search import FusionMethod, HybridSearcher
from nlp_academic_search.search.index_manifest import IndexCompatibilityError
from nlp_academic_search.search.models import SearchFilters, SearchResult
from nlp_academic_search.search.reranker import Reranker
from nlp_academic_search.search.semantic_search import SemanticSearcher

T = TypeVar("T")


class ServiceTimeoutError(RuntimeError):
    pass


class ServiceBusyError(RuntimeError):
    pass


@dataclass
class ServiceContainer:
    papers: list[Paper]
    bm25: BM25Searcher
    semantic: SemanticSearcher
    hybrid: HybridSearcher
    rag_generator: RAGGenerator
    reranker: Reranker | None = None
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
        def execute() -> list[SearchResult]:
            if method == "bm25":
                return self.bm25.search(query, top_k=top_k, filters=filters)
            if method == "semantic":
                return self.semantic.search(query, top_k=top_k, filters=filters)
            return self.hybrid.search(
                query,
                top_k=top_k,
                method=fusion,
                candidate_pool=max(settings.search.candidate_pool, top_k),
                filters=filters,
            )

        executor = self.executor if method == "bm25" else self.dense_executor
        return self.run(execute, settings.search.timeout_seconds, executor=executor)

    def retrieve_for_rag(
        self, question: str, top_k: int, use_reranker: bool
    ) -> tuple[list[SearchResult], list[str], str]:
        if not settings.rag_enabled:
            raise ModelUnavailableError("RAG is disabled by configuration")
        warnings = []
        candidate_k = max(top_k, min(settings.search.candidate_pool, top_k * 4))
        results = self.search(question, "hybrid", candidate_k)
        method = "rrf"
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
        if not settings.rag_enabled:
            return False
        try:
            response = httpx.get(f"{settings.ollama.base_url}/api/tags", timeout=2.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            names = [model.get("name", "") for model in models]
            return any(
                name == settings.ollama.model_name
                or name.startswith(f"{settings.ollama.model_name}:")
                for name in names
            )
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    def index_compatible(self) -> bool:
        try:
            self.semantic.manifest.validate(
                corpus_path=self.semantic.corpus_path,
                papers=self.papers,
                model_name=self.semantic.model_name,
                model_revision=self.semantic.model_revision,
                embeddings=self.semantic.embeddings,
                index=self.semantic.index,
            )
            return True
        except (IndexCompatibilityError, OSError, ValueError):
            return False

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.dense_executor.shutdown(wait=False, cancel_futures=True)


def build_services() -> ServiceContainer:
    papers = load_papers()
    bm25 = BM25Searcher(papers)
    semantic = SemanticSearcher(papers)
    hybrid = HybridSearcher(bm25, semantic)
    reranker = Reranker() if settings.reranker.enabled else None
    return ServiceContainer(
        papers=papers,
        bm25=bm25,
        semantic=semantic,
        hybrid=hybrid,
        reranker=reranker,
        rag_generator=RAGGenerator(),
    )


def get_services(request: Request) -> ServiceContainer:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise RuntimeError("Application services are not initialized")
    return services
