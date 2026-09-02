"""
Search API routes.

Provides endpoints for BM25, semantic, and hybrid search.
"""

import time
from typing import List

from fastapi import APIRouter, Depends, Query

from src.api.schemas import PaperResponse, SearchResponse

router = APIRouter(prefix="/search", tags=["Search"])


def _format_results(results, query: str, method: str, start_time: float) -> SearchResponse:
    """Format search results into API response."""
    latency_ms = (time.time() - start_time) * 1000

    paper_responses = [
        PaperResponse(
            id=r.paper.id,
            title=r.paper.title,
            abstract=r.paper.abstract,
            authors=r.paper.authors,
            category=r.paper.category,
            year=r.paper.year,
            score=round(r.score, 4),
        )
        for r in results
    ]

    return SearchResponse(
        query=query,
        method=method,
        total_results=len(paper_responses),
        results=paper_responses,
        latency_ms=round(latency_ms, 2),
    )


@router.get("", response_model=SearchResponse, summary="Hybrid Search (default)")
async def hybrid_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    """
    Perform hybrid search combining BM25 keyword matching and SBERT semantic similarity.

    This is the recommended search method as it captures both exact keyword matches
    and meaning-based similarity.
    """
    from src.api.main import app_state

    start_time = time.time()
    results = app_state.hybrid_searcher.search(q, top_k=top_k)
    return _format_results(results, q, "hybrid", start_time)


@router.get("/bm25", response_model=SearchResponse, summary="BM25 Keyword Search")
async def bm25_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    """
    Perform keyword-based search using BM25 (Okapi) algorithm.

    Best for queries with specific technical terms or exact phrases.
    """
    from src.api.main import app_state

    start_time = time.time()
    results = app_state.bm25_searcher.search(q, top_k=top_k)
    return _format_results(results, q, "bm25", start_time)


@router.get("/semantic", response_model=SearchResponse, summary="Semantic Search")
async def semantic_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    """
    Perform semantic search using Sentence-BERT embeddings + FAISS.

    Best for natural language queries where meaning matters more than exact keywords.
    """
    from src.api.main import app_state

    start_time = time.time()
    results = app_state.semantic_searcher.search(q, top_k=top_k)
    return _format_results(results, q, "semantic", start_time)
