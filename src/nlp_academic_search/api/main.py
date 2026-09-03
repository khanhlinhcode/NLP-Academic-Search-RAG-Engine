"""FastAPI application factory with bounded services and truthful readiness."""

from __future__ import annotations

import asyncio
import hmac
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
from nlp_academic_search.api.security import InMemoryRateLimiter
from nlp_academic_search.api.services import (
    ServiceBusyError,
    ServiceContainer,
    ServiceTimeoutError,
    build_services,
)
from nlp_academic_search.config import settings
from nlp_academic_search.providers.generation.base import (
    GenerationInvalidResponseError,
    GenerationRateLimitedError,
    GenerationTimeoutError,
    ModelUnavailableError,
    RAGGenerationError,
)
from nlp_academic_search.providers.retrieval.base import RetrievalUnavailableError

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
    application.state.health_cache = None
    application.state.rate_limiter = InMemoryRateLimiter()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        token = ""
        public_paths = {"/", "/health", "/health/live", "/health/ready", "/docs", "/openapi.json"}
        if settings.backend_api_token and request.url.path not in public_paths:
            authorization = request.headers.get("Authorization", "")
            scheme, _, token = authorization.partition(" ")
            valid = scheme.casefold() == "bearer" and hmac.compare_digest(
                token, settings.backend_api_token
            )
            if not valid:
                return _error(request, 401, "unauthorized", "Authentication required", False)
        token_subject = token if settings.backend_api_token else ""
        client_host = request.client.host if request.client else "unknown"
        subject = application.state.rate_limiter.subject(token_subject, client_host)
        if "/ask" in request.url.path:
            allowed = application.state.rate_limiter.allow(
                subject, "ask", settings.ask_rate_limit_per_minute
            )
        elif "/search" in request.url.path:
            allowed = application.state.rate_limiter.allow(
                subject, "search", settings.search_rate_limit_per_minute
            )
        else:
            allowed = True
        if not allowed:
            return _error(request, 429, "rate_limited", "Request rate limit exceeded", True)
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
            request, 503, "model_unavailable", "The generation provider is unavailable", True
        )

    @application.exception_handler(GenerationTimeoutError)
    async def generation_timeout(request: Request, _: GenerationTimeoutError) -> JSONResponse:
        return _error(request, 504, "generation_timeout", "Generation provider timed out", True)

    @application.exception_handler(GenerationRateLimitedError)
    async def generation_rate_limited(
        request: Request, _: GenerationRateLimitedError
    ) -> JSONResponse:
        return _error(
            request,
            429,
            "generation_rate_limited",
            "Generation quota is temporarily exhausted",
            True,
        )

    @application.exception_handler(GenerationInvalidResponseError)
    async def invalid_generation(
        request: Request, _: GenerationInvalidResponseError
    ) -> JSONResponse:
        return _error(
            request,
            502,
            "generation_invalid_response",
            "Generation provider returned invalid data",
            True,
        )

    @application.exception_handler(RetrievalUnavailableError)
    async def retrieval_unavailable(request: Request, _: RetrievalUnavailableError) -> JSONResponse:
        return _error(
            request, 503, "retrieval_unavailable", "Retrieval provider is unavailable", True
        )

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
        retrieval = services.retrieval_status() if services else None
        corpus_ready = bool(retrieval and retrieval.total_papers)
        index_ready = bool(retrieval and retrieval.ready)
        search_ready = corpus_ready and index_ready
        generation_available = (
            services.generation_available() if services and check_ollama else False
        )
        ollama_available = settings.generation_provider == "ollama" and generation_available
        verification_available = (
            services.semantic_verification_available() if services and check_ollama else False
        )
        verification_required = settings.verification.enabled and settings.verification.fail_closed
        rag_ready = not settings.rag_enabled or (
            generation_available and (not verification_required or verification_available)
        )
        status = (
            "ready" if search_ready and rag_ready else "degraded" if search_ready else "not_ready"
        )
        return HealthResponse(
            status=status,
            version=__version__,
            total_papers=retrieval.total_papers if retrieval else 0,
            search_ready=search_ready,
            rag_enabled=settings.rag_enabled,
            ollama_available=ollama_available,
            generation_available=generation_available,
            semantic_verification_available=verification_available,
            index_provenance=retrieval.provenance if retrieval else None,
            models={
                "embedding": (retrieval.embedding_model if retrieval else None)
                or settings.embedding.model_name,
                "llm": settings.active_generation_model,
                "reranker": settings.reranker.model_name
                if settings.reranker.enabled
                else "disabled",
                "verifier": settings.verification.model_name
                if settings.verification.enabled
                else "disabled",
            },
            checks={
                "corpus": corpus_ready,
                "index": index_ready,
                "generation": generation_available,
                "verification": not verification_required or verification_available,
            },
            providers={
                "retrieval": settings.retrieval_provider,
                "generation": settings.generation_provider,
                "reranker": settings.reranker_provider,
                "verification": settings.verification.provider,
            },
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
        retrieval = services.retrieval_status()
        return StatsResponse(
            total_papers=retrieval.total_papers,
            embedding_model=retrieval.embedding_model or settings.embedding.model_name,
            embedding_revision=retrieval.embedding_revision,
            embedding_dim=retrieval.embedding_dimension,
            llm_model=settings.active_generation_model,
            reranker_model=settings.reranker.model_name,
            reranker_enabled=services.reranker is not None,
            bm25_weight=settings.search.bm25_weight,
            semantic_weight=settings.search.semantic_weight,
            index_provenance=retrieval.provenance,
            retrieval_provider=settings.retrieval_provider,
            generation_provider=settings.generation_provider,
        )

    return application


app = create_app()
