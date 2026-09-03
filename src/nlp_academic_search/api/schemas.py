"""Versioned API request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nlp_academic_search.rag.citations import CitationValidation


class AskRequest(BaseModel):
    question: str = Field(
        min_length=5, max_length=1000, examples=["How does RRF combine rankings?"]
    )
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = Field(default=False)


class PaperResponse(BaseModel):
    id: str
    arxiv_id: str | None
    doi: str | None
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    year: int | None
    source_url: str | None
    pdf_url: str | None
    source: str
    score: float
    score_type: str
    bm25_score: float | None = None
    semantic_score: float | None = None
    rrf_score: float | None = None
    weighted_score: float | None = None
    reranker_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    method: str
    total_results: int
    results: list[PaperResponse]
    latency_ms: float
    offset: int = 0
    page_size: int = 10
    has_more: bool = False
    request_id: str | None = None


class SourceReference(BaseModel):
    index: int
    id: str
    arxiv_id: str | None
    doi: str | None
    title: str
    authors: list[str]
    categories: list[str]
    year: int | None
    source_url: str | None
    pdf_url: str | None
    source: str


class StageLatencies(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class AnswerMetadata(BaseModel):
    model: str
    retrieval_method: str
    source_ids: list[str]
    prompt_version: str
    estimated_context_tokens: int
    context_truncated: bool
    latencies: StageLatencies
    warnings: list[str]
    citation_validation: CitationValidation


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    metadata: AnswerMetadata


class LiveResponse(BaseModel):
    status: Literal["alive"]
    version: str


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    version: str
    total_papers: int
    search_ready: bool
    rag_enabled: bool
    ollama_available: bool
    index_provenance: str | None
    models: dict[str, str]
    checks: dict[str, bool]


class StatsResponse(BaseModel):
    total_papers: int
    embedding_model: str
    embedding_revision: str | None
    embedding_dim: int
    llm_model: str
    reranker_model: str
    reranker_enabled: bool
    bm25_weight: float
    semantic_weight: float
    index_provenance: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    retryable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail
