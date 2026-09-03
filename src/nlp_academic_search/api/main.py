"""FastAPI application factory with bounded services and truthful readiness."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nlp_academic_search import __version__
from nlp_academic_search.api.routes import rag, search
from nlp_academic_search.api.schemas import HealthResponse, LiveResponse, StatsResponse
from nlp_academic_search.api.services import (
    ServiceBusyError,
    ServiceContainer,
    ServiceTimeoutError,
    build_services,
)
from nlp_academic_search.config import settings
from nlp_academic_search.rag.generator import (
    GenerationTimeoutError,
    ModelUnavailableError,
    RAGGenerationError,
)

logger = logging.getLogger("academic_search.api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _error(request: Request, status: int, code: str, message: str, retryable: bool) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    response = JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "retryable": retryable,
            }
        },
    )
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


def create_app(service_container: ServiceContainer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.services = service_container or await asyncio.to_thread(build_services)
        try:
            yield
        finally:
            application.state.services.close()

    application = FastAPI(
        title="NLP Academic Search & RAG Engine",
        description="Versioned hybrid retrieval and evidence-constrained local RAG.",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.request_slots = asyncio.Semaphore(settings.api.concurrency_limit)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            await asyncio.wait_for(application.state.request_slots.acquire(), timeout=0.1)
        except TimeoutError:
            return _error(request, 429, "capacity_busy", "Request capacity is busy", True)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                json.dumps(
                    {
                        "event": "request_complete",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
            )
            return response
        finally:
            application.state.request_slots.release()

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(part) for part in item["loc"]) for item in exc.errors()]
        return _error(
            request, 422, "invalid_request", f"Invalid fields: {', '.join(fields)}", False
        )

    @application.exception_handler(ServiceTimeoutError)
    async def service_timeout(request: Request, _: ServiceTimeoutError) -> JSONResponse:
        return _error(request, 504, "pipeline_timeout", "A pipeline stage timed out", True)

    @application.exception_handler(ServiceBusyError)
    async def service_busy(request: Request, _: ServiceBusyError) -> JSONResponse:
        return _error(
            request, 429, "capacity_busy", "Inference capacity is busy; retry shortly", True
        )

    @application.exception_handler(ModelUnavailableError)
    async def model_unavailable(request: Request, _: ModelUnavailableError) -> JSONResponse:
        return _error(
            request, 503, "model_unavailable", "The configured Ollama model is unavailable", True
        )

    @application.exception_handler(GenerationTimeoutError)
    async def generation_timeout(request: Request, _: GenerationTimeoutError) -> JSONResponse:
        return _error(request, 504, "generation_timeout", "Ollama generation timed out", True)

    @application.exception_handler(RAGGenerationError)
    async def generation_error(request: Request, _: RAGGenerationError) -> JSONResponse:
        return _error(request, 500, "generation_error", "Answer generation failed", True)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled request error request_id=%s type=%s",
            getattr(request.state, "request_id", None),
            type(exc).__name__,
        )
        return _error(request, 500, "internal_error", "An internal error occurred", False)

    application.include_router(search.router)
    application.include_router(rag.router)
    application.include_router(search.router, prefix="/api/v1")
    application.include_router(rag.router, prefix="/api/v1")

    @application.get("/", tags=["System"])
    def root() -> dict[str, str]:
        return {"name": application.title, "version": __version__, "docs": "/docs"}

    @application.get("/health/live", response_model=LiveResponse, tags=["System"])
    def live() -> LiveResponse:
        return LiveResponse(status="alive", version=__version__)

    def health_payload(check_ollama: bool = True) -> HealthResponse:
        services: ServiceContainer | None = getattr(application.state, "services", None)
        corpus_ready = services is not None and bool(services.papers)
        index_ready = services.index_compatible() if services else False
        search_ready = corpus_ready and index_ready
        ollama_available = services.ollama_available() if services and check_ollama else False
        rag_ready = not settings.rag_enabled or ollama_available
        status = (
            "ready" if search_ready and rag_ready else "degraded" if search_ready else "not_ready"
        )
        manifest = services.semantic.manifest if services else None
        return HealthResponse(
            status=status,
            version=__version__,
            total_papers=len(services.papers) if services else 0,
            search_ready=search_ready,
            rag_enabled=settings.rag_enabled,
            ollama_available=ollama_available,
            index_provenance=manifest.provenance if manifest else None,
            models={
                "embedding": settings.embedding.model_name,
                "llm": settings.ollama.model_name,
                "reranker": settings.reranker.model_name
                if settings.reranker.enabled
                else "disabled",
            },
            checks={"corpus": corpus_ready, "index": index_ready, "ollama": rag_ready},
        )

    @application.get("/health", response_model=HealthResponse, tags=["System"])
    def health() -> HealthResponse:
        return health_payload()

    @application.get("/health/ready", response_model=HealthResponse, tags=["System"])
    def ready() -> HealthResponse | JSONResponse:
        payload = health_payload()
        if payload.status != "ready":
            return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))
        return payload

    @application.get("/stats", response_model=StatsResponse, tags=["System"])
    def stats() -> StatsResponse:
        services: ServiceContainer = application.state.services
        manifest = services.semantic.manifest
        return StatsResponse(
            total_papers=len(services.papers),
            embedding_model=manifest.embedding_model,
            embedding_revision=manifest.embedding_revision,
            embedding_dim=manifest.embedding_dimension,
            llm_model=settings.ollama.model_name,
            reranker_model=settings.reranker.model_name,
            reranker_enabled=services.reranker is not None,
            bm25_weight=settings.search.bm25_weight,
            semantic_weight=settings.search.semantic_weight,
            index_provenance=manifest.provenance,
        )

    return application


app = create_app()
