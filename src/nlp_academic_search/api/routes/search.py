"""Synchronous search routes executed in FastAPI's worker pool."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from nlp_academic_search.api.schemas import PaperResponse, SearchResponse
from nlp_academic_search.api.services import ServiceContainer, get_services
from nlp_academic_search.search.models import FusionMethod, SearchFilters

router = APIRouter(prefix="/search", tags=["Search"])


def _search(
    request: Request,
    services: ServiceContainer,
    *,
    query: str,
    method: str,
    top_k: int,
    offset: int,
    category: str | None,
    year_from: int | None,
    year_to: int | None,
    author: str | None,
    source: str | None,
    fusion: FusionMethod = FusionMethod.RRF,
) -> SearchResponse:
    started = time.perf_counter()
    filters = SearchFilters(
        category=category,
        year_from=year_from,
        year_to=year_to,
        author=author,
        source=source,
    )
    batch = services.search_batch(query, method, top_k + offset + 1, filters=filters, fusion=fusion)
    results = batch.results[offset : offset + top_k]
    return SearchResponse(
        query=query,
        method=batch.retrieval_mode,
        total_results=len(results),
        results=[PaperResponse.model_validate(result.to_dict()) for result in results],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        offset=offset,
        page_size=top_k,
        has_more=len(batch.results) > offset + top_k,
        request_id=getattr(request.state, "request_id", None),
        retrieval_mode=batch.retrieval_mode,
        warnings=batch.warnings,
    )


def common_parameters(
    q: Annotated[
        str, Query(min_length=1, max_length=500, examples=["retrieval augmented generation"])
    ],
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
    offset: Annotated[int, Query(ge=0, le=450)] = 0,
    category: str | None = None,
    year_from: Annotated[int | None, Query(ge=1900, le=2100)] = None,
    year_to: Annotated[int | None, Query(ge=1900, le=2100)] = None,
    author: Annotated[str | None, Query(max_length=200)] = None,
    source: Annotated[str | None, Query(max_length=100)] = None,
) -> dict[str, object]:
    return {
        "query": q,
        "top_k": top_k,
        "offset": offset,
        "category": category,
        "year_from": year_from,
        "year_to": year_to,
        "author": author,
        "source": source,
    }


CommonParams = Annotated[dict[str, object], Depends(common_parameters)]
Services = Annotated[ServiceContainer, Depends(get_services)]


@router.get("", response_model=SearchResponse, summary="Hybrid search")
def hybrid_search(
    request: Request,
    params: CommonParams,
    services: Services,
    fusion: FusionMethod = FusionMethod.RRF,
) -> SearchResponse:
    return _search(request, services, method="hybrid", fusion=fusion, **params)  # type: ignore[arg-type]


@router.get("/bm25", response_model=SearchResponse, summary="BM25 keyword search")
def bm25_search(request: Request, params: CommonParams, services: Services) -> SearchResponse:
    return _search(request, services, method="bm25", **params)  # type: ignore[arg-type]


@router.get("/semantic", response_model=SearchResponse, summary="Dense semantic search")
def semantic_search(request: Request, params: CommonParams, services: Services) -> SearchResponse:
    return _search(request, services, method="semantic", **params)  # type: ignore[arg-type]
