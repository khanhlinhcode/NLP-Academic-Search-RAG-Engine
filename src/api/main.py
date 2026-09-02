"""
FastAPI Application — NLP Academic Search & RAG Engine.

Main entry point for the REST API.
Initializes all search engines and models on startup.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes import rag, search
from src.api.schemas import HealthResponse, StatsResponse
from src.config import settings
from src.data.loader import load_papers
from src.rag.generator import RAGGenerator
from src.search.bm25_search import BM25Searcher
from src.search.hybrid_search import HybridSearcher
from src.search.reranker import Reranker
from src.search.semantic_search import SemanticSearcher


@dataclass
class AppState:
    """Application state holding all initialized components."""

    bm25_searcher: Optional[BM25Searcher] = None
    semantic_searcher: Optional[SemanticSearcher] = None
    hybrid_searcher: Optional[HybridSearcher] = None
    reranker: Optional[Reranker] = None
    rag_generator: Optional[RAGGenerator] = None
    total_papers: int = 0


# Global app state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — initialize all components on startup.

    This loads the dataset, builds search indices, and initializes
    the LLM generator once when the server starts.
    """
    global app_state

    print("\n" + "=" * 60)
    print("🚀 NLP Academic Search & RAG Engine — Starting up...")
    print("=" * 60 + "\n")

    start_time = time.time()

    # 1. Load papers
    papers = load_papers()
    app_state.total_papers = len(papers)

    # 2. Initialize BM25 search
    app_state.bm25_searcher = BM25Searcher(papers)

    # 3. Initialize semantic search (loads embeddings from disk if available)
    app_state.semantic_searcher = SemanticSearcher(papers)

    # 4. Initialize hybrid search
    app_state.hybrid_searcher = HybridSearcher(
        bm25=app_state.bm25_searcher,
        semantic=app_state.semantic_searcher,
    )

    # 5. Initialize reranker (optional)
    if settings.reranker.enabled:
        try:
            app_state.reranker = Reranker()
        except Exception as e:
            print(f"⚠️  Reranker initialization failed: {e}")
            print("   Continuing without reranker.")

    # 6. Initialize RAG generator
    app_state.rag_generator = RAGGenerator()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"✅ Server ready in {elapsed:.1f}s")
    print(f"   📚 Papers: {app_state.total_papers}")
    print(f"   🔍 Search: BM25 + Semantic + Hybrid")
    print(f"   🔄 Reranker: {'enabled' if app_state.reranker else 'disabled'}")
    print(f"   🤖 LLM: {settings.ollama.model_name}")
    print(f"   📖 Docs: http://{settings.api.host}:{settings.api.port}/docs")
    print("=" * 60 + "\n")

    yield

    # Cleanup on shutdown
    print("\n🛑 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="NLP Academic Search & RAG Engine",
    description=(
        "A semantic search system for scientific papers with LLM-powered Q&A.\n\n"
        "**Features:**\n"
        "- 🔍 BM25 keyword search\n"
        "- 🧠 SBERT semantic search with FAISS\n"
        "- ⚡ Hybrid retrieval (BM25 + Semantic)\n"
        "- 🔄 Cross-Encoder reranking\n"
        "- 🤖 RAG with Ollama LLM\n"
    ),
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(search.router)
app.include_router(rag.router)


# ─── Root & Health Endpoints ──────────────────────────────────────


@app.get("/", tags=["Root"])
async def root():
    """API root — redirect to documentation."""
    return {
        "name": "NLP Academic Search & RAG Engine",
        "version": __version__,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    ollama_available = False
    if app_state.rag_generator:
        ollama_available = app_state.rag_generator.is_available()

    return HealthResponse(
        status="healthy",
        version=__version__,
        total_papers=app_state.total_papers,
        ollama_available=ollama_available,
        models={
            "embedding": settings.embedding.model_name,
            "llm": settings.ollama.model_name,
            "reranker": settings.reranker.model_name if settings.reranker.enabled else "disabled",
        },
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats():
    """Get dataset and model statistics."""
    embedding_dim = 0
    if app_state.semantic_searcher and app_state.semantic_searcher.embeddings is not None:
        embedding_dim = app_state.semantic_searcher.embeddings.shape[1]

    return StatsResponse(
        total_papers=app_state.total_papers,
        embedding_model=settings.embedding.model_name,
        embedding_dim=embedding_dim,
        llm_model=settings.ollama.model_name,
        reranker_model=settings.reranker.model_name,
        reranker_enabled=settings.reranker.enabled,
        bm25_weight=settings.search.bm25_weight,
        semantic_weight=settings.search.semantic_weight,
    )
