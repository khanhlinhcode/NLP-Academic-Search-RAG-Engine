"""
Pydantic schemas for API request/response models.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ─── Request Models ───────────────────────────────────────────────

class SearchQuery(BaseModel):
    """Search request parameters."""

    q: str = Field(..., description="Search query string", min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    method: Literal["bm25", "semantic", "hybrid"] = Field(
        default="hybrid",
        description="Search method: bm25 (keyword), semantic (embedding), or hybrid (combined)",
    )


class AskRequest(BaseModel):
    """RAG question-answering request."""

    question: str = Field(..., description="Question to answer using RAG", min_length=5, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of papers to use as context")
    use_reranker: bool = Field(default=True, description="Whether to use Cross-Encoder reranking")


# ─── Response Models ──────────────────────────────────────────────

class PaperResponse(BaseModel):
    """A single paper in search results."""

    id: str
    title: str
    abstract: str
    authors: List[str]
    category: str
    year: Optional[int]
    score: float


class SearchResponse(BaseModel):
    """Search API response."""

    query: str
    method: str
    total_results: int
    results: List[PaperResponse]
    latency_ms: float


class SourceReference(BaseModel):
    """A source paper referenced in the RAG answer."""

    index: int = Field(description="Citation index [1], [2], etc.")
    id: str
    title: str
    authors: List[str]
    category: str
    year: Optional[int]


class AskResponse(BaseModel):
    """RAG question-answering response."""

    question: str
    answer: str
    sources: List[SourceReference]
    retrieval_method: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    total_papers: int
    ollama_available: bool
    models: dict


class StatsResponse(BaseModel):
    """Dataset and model statistics."""

    total_papers: int
    embedding_model: str
    embedding_dim: int
    llm_model: str
    reranker_model: str
    reranker_enabled: bool
    bm25_weight: float
    semantic_weight: float
