"""Versioned API request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nlp_academic_search.rag.citations import CitationValidation
from nlp_academic_search.rag.verification import SemanticValidation


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
    retrieval_mode: str | None = None
    warnings: list[str] = Field(default_factory=list)


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
    verification_ms: float = 0.0


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
    citation_repair_attempted: bool = False
    citation_repair_succeeded: bool | None = None
    answer_status: Literal[
        "verified",
        "structurally_valid",
        "refused_insufficient_context",
        "refused_unverified",
        "verification_unavailable",
    ] = "structurally_valid"
    semantic_validation: SemanticValidation | None = None
    semantic_verification_attempted: bool = False
    semantic_verification_succeeded: bool | None = None
    final_answer_replaced: bool = False
    verification_latency_ms: float = 0.0


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
    generation_available: bool = False
    semantic_verification_available: bool = False
    index_provenance: str | None
    models: dict[str, str]
    checks: dict[str, bool]
    providers: dict[str, str] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    total_papers: int
    embedding_model: str
    embedding_revision: str | None
    embedding_dim: int | None
    llm_model: str
    reranker_model: str
    reranker_enabled: bool
    bm25_weight: float
    semantic_weight: float
    index_provenance: str | None
    retrieval_provider: str = "local"
    generation_provider: str = "ollama"


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    retryable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail
