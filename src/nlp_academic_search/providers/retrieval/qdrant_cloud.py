"""Qdrant Cloud dense, BM25, and server-side RRF retrieval provider."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nlp_academic_search.config import QdrantConfig
from nlp_academic_search.data.loader import Paper
from nlp_academic_search.providers.retrieval.base import (
    RetrievalBatch,
    RetrievalStatus,
    RetrievalUnavailableError,
)
from nlp_academic_search.search.models import FusionMethod, SearchFilters, SearchResult


class QdrantCloudRetrievalProvider:
    provider_name = "qdrant"

    def __init__(
        self,
        config: QdrantConfig,
        *,
        client: Any | None = None,
        allow_degraded: bool = False,
        candidate_pool: int = 50,
    ) -> None:
        if not config.url or not config.api_key or not config.dense_model:
            raise ValueError("Qdrant URL, API key, and dense model are required")
        self.config = config
        self.collection_name = config.collection_alias
        self.dense_model = config.dense_model
        self.sparse_model = config.sparse_model
        self.allow_degraded = allow_degraded
        self.candidate_pool = candidate_pool
        self._owns_client = client is None
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
                cloud_inference=True,
                timeout=max(1, int(config.timeout_seconds)),
                check_compatibility=False,
            )
        self.client = client
        self._status: RetrievalStatus | None = None

    @staticmethod
    def _models() -> Any:
        from qdrant_client import models

        return models

    def _paper_filter(self, filters: SearchFilters | None = None) -> Any:
        models = self._models()
        must: list[Any] = [
            models.FieldCondition(key="record_type", match=models.MatchValue(value="paper"))
        ]
        active = filters or SearchFilters()
        if active.category:
            must.append(
                models.FieldCondition(
                    key="categories", match=models.MatchValue(value=active.category)
                )
            )
        if active.year_from is not None or active.year_to is not None:
            must.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=active.year_from, lte=active.year_to),
                )
            )
        if active.author:
            must.append(
                models.FieldCondition(key="authors", match=models.MatchText(text=active.author))
            )
        if active.source:
            must.append(
                models.FieldCondition(key="source", match=models.MatchValue(value=active.source))
            )
        return models.Filter(must=must)

    def _query_points(
        self, query: str, method: str, top_k: int, filters: SearchFilters | None
    ) -> list[Any]:
        models = self._models()
        query_filter = self._paper_filter(filters)
        if method == "bm25":
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=models.Document(text=query, model=self.sparse_model),
                using="sparse",
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        elif method == "semantic":
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=models.Document(text=query, model=self.dense_model),
                using="dense",
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            pool = max(top_k, self.candidate_pool)
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query, model=self.dense_model),
                        using="dense",
                        filter=query_filter,
                        limit=pool,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model=self.sparse_model),
                        using="sparse",
                        filter=query_filter,
                        limit=pool,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        return list(getattr(response, "points", response))

    @staticmethod
    def _paper_from_payload(payload: dict[str, Any]) -> Paper:
        published_at = payload.get("published_at") or payload.get("published_date")
        if isinstance(published_at, datetime):
            published_at = published_at.isoformat()
        return Paper.model_validate(
            {
                "id": payload.get("paper_id") or payload.get("id"),
                "arxiv_id": payload.get("arxiv_id"),
                "doi": payload.get("doi"),
                "title": payload.get("title"),
                "abstract": payload.get("abstract") or payload.get("text"),
                "authors": payload.get("authors") or [],
                "categories": payload.get("categories") or [],
                "published_at": published_at,
                "updated_at": payload.get("updated_at"),
                "source_url": payload.get("source_url") or payload.get("url"),
                "pdf_url": payload.get("pdf_url"),
                "source": payload.get("source") or "qdrant",
                "license": payload.get("license"),
                "content_hash": payload.get("content_hash") or "",
                "schema_version": str(payload.get("paper_schema_version") or "1.0"),
            }
        )

    def _results(self, points: list[Any], method: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            paper = self._paper_from_payload(payload)
            score = float(getattr(point, "score", 0.0) or 0.0)
            result = SearchResult(paper=paper)
            if method == "bm25":
                result.bm25_score = score
            elif method == "semantic":
                result.semantic_score = score
            else:
                result.rrf_score = score
            results.append(result)
        return results

    def search(
        self,
        query: str,
        method: str,
        top_k: int,
        *,
        filters: SearchFilters | None = None,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> RetrievalBatch:
        if not query.strip() or top_k < 1:
            return RetrievalBatch(results=[], retrieval_mode=method)
        if method == "hybrid" and fusion != FusionMethod.RRF:
            raise RetrievalUnavailableError("Qdrant cloud profile supports RRF fusion only")
        try:
            points = self._query_points(query, method, top_k, filters)
            mode = fusion.value if method == "hybrid" else method
            return RetrievalBatch(self._results(points, method), mode)
        except Exception as exc:
            if method == "hybrid" and self.allow_degraded:
                try:
                    points = self._query_points(query, "bm25", top_k, filters)
                    return RetrievalBatch(
                        self._results(points, "bm25"),
                        "bm25_degraded",
                        ["Dense retrieval unavailable; serving BM25-only results."],
                    )
                except Exception as fallback_exc:
                    raise RetrievalUnavailableError(
                        "Qdrant retrieval is unavailable"
                    ) from fallback_exc
            raise RetrievalUnavailableError("Qdrant retrieval is unavailable") from exc

    def status(self) -> RetrievalStatus:
        try:
            self.client.get_collection(self.collection_name)
            count = self.client.count(
                collection_name=self.collection_name,
                count_filter=self._paper_filter(),
                exact=True,
            )
            total = int(getattr(count, "count", 0))
            if total < 1:
                raise ValueError("Qdrant collection contains no paper records")
            models = self._models()
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="record_type", match=models.MatchValue(value="manifest")
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            payload = dict(records[0].payload or {}) if records else {}
            if not payload:
                raise ValueError("Qdrant collection manifest is missing")
            if int(payload.get("paper_count", -1)) != total:
                raise ValueError("Qdrant paper count does not match its manifest")
            if int(payload.get("schema_version", 0)) != self.config.expected_schema_version:
                raise ValueError("Qdrant schema version does not match configuration")
            if payload.get("dense_model") != self.dense_model:
                raise ValueError("Qdrant dense model does not match configuration")
            if payload.get("sparse_model") != self.sparse_model:
                raise ValueError("Qdrant sparse model does not match configuration")
            if (
                self.config.expected_corpus_sha256
                and payload.get("corpus_sha256") != self.config.expected_corpus_sha256
            ):
                raise ValueError("Qdrant corpus checksum does not match configuration")
            self._status = RetrievalStatus(
                ready=True,
                total_papers=total,
                provenance=f"qdrant-cloud:{payload.get('corpus_version', 'unknown')}",
                embedding_model=self.dense_model,
            )
        except Exception as exc:
            self._status = RetrievalStatus(ready=False, reason=type(exc).__name__)
        return self._status

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
