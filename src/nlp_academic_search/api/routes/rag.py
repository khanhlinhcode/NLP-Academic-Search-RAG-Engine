"""Grounded RAG routes with typed HTTP errors and structured SSE."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Generator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from nlp_academic_search.api.schemas import (
    AnswerMetadata,
    AskRequest,
    AskResponse,
    SourceReference,
    StageLatencies,
)
from nlp_academic_search.api.services import ServiceBusyError, ServiceContainer, get_services
from nlp_academic_search.config import settings
from nlp_academic_search.providers.generation.base import (
    GenerationInvalidResponseError,
    GenerationRateLimitedError,
    GenerationTimeoutError,
    ModelUnavailableError,
    RAGGenerationError,
)
from nlp_academic_search.rag.citations import validate_citations
from nlp_academic_search.rag.prompt_builder import (
    PROMPT_VERSION,
    InsufficientContextError,
    PromptPackage,
    build_rag_messages,
    build_source_list,
)

router = APIRouter(prefix="/ask", tags=["RAG"])
Services = Annotated[ServiceContainer, Depends(get_services)]
REFUSAL = "Not enough evidence in the retrieved sources."


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _prepare(
    request: AskRequest, services: ServiceContainer
) -> tuple[PromptPackage, list[SourceReference], list[str], str, float]:
    started = time.perf_counter()
    results, warnings, retrieval_method = services.retrieve_for_rag(
        request.question, request.top_k, request.use_reranker
    )
    thresholded = [
        result
        for result in results
        if (result.bm25_score or 0) > 0
        or (result.semantic_score or -1) >= settings.rag_min_relevance_score
        or (result.rrf_score or 0) > 0
    ]
    papers = [result.paper for result in thresholded]
    retrieval_ms = (time.perf_counter() - started) * 1000
    package = build_rag_messages(request.question, papers)
    sources = [SourceReference.model_validate(item) for item in build_source_list(package.papers)]
    if package.truncated:
        warnings.append("Context was truncated to fit the configured token budget.")
    return package, sources, warnings, retrieval_method, retrieval_ms


def _metadata(
    *,
    services: ServiceContainer,
    sources: list[SourceReference],
    package: PromptPackage,
    retrieval_method: str,
    retrieval_ms: float,
    generation_ms: float,
    warnings: list[str],
    answer: str,
) -> AnswerMetadata:
    if services.rag_generator is None:
        raise ModelUnavailableError("Generation provider is not configured")
    citation_validation = validate_citations(answer, len(sources))
    warnings.extend(citation_validation.warnings)
    return AnswerMetadata(
        model=services.rag_generator.model_name,
        retrieval_method=retrieval_method,
        source_ids=[source.id for source in sources],
        prompt_version=PROMPT_VERSION,
        estimated_context_tokens=package.estimated_context_tokens,
        context_truncated=package.truncated,
        latencies=StageLatencies(
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(retrieval_ms + generation_ms, 2),
        ),
        warnings=list(dict.fromkeys(warnings)),
        citation_validation=citation_validation,
    )


@router.post("", response_model=AskResponse, summary="Grounded question answering")
def ask_question(payload: AskRequest, services: Services) -> AskResponse:
    if services.rag_generator is None:
        raise ModelUnavailableError("Generation provider is not configured")
    try:
        package, sources, warnings, method, retrieval_ms = _prepare(payload, services)
    except InsufficientContextError:
        validation = validate_citations(REFUSAL, 0)
        return AskResponse(
            question=payload.question,
            answer=REFUSAL,
            sources=[],
            metadata=AnswerMetadata(
                model=services.rag_generator.model_name,
                retrieval_method="rrf",
                source_ids=[],
                prompt_version=PROMPT_VERSION,
                estimated_context_tokens=0,
                context_truncated=False,
                latencies=StageLatencies(retrieval_ms=0, generation_ms=0, total_ms=0),
                warnings=["No relevant context was retrieved; the model was not called."],
                citation_validation=validation,
            ),
        )
    services.acquire_generation()
    generated_at = time.perf_counter()
    try:
        answer = services.rag_generator.generate(package.messages)  # type: ignore[attr-defined]
    finally:
        services.release_generation()
    generation_ms = (time.perf_counter() - generated_at) * 1000
    metadata = _metadata(
        services=services,
        sources=sources,
        package=package,
        retrieval_method=method,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        warnings=warnings,
        answer=answer,
    )
    return AskResponse(question=payload.question, answer=answer, sources=sources, metadata=metadata)


@router.post("/stream", summary="Streaming grounded question answering")
def ask_question_stream(
    payload: AskRequest, services: Services, http_request: Request
) -> StreamingResponse:
    try:
        package, sources, warnings, method, retrieval_ms = _prepare(payload, services)
    except InsufficientContextError:

        def refuse() -> Generator[str, None, None]:
            yield _sse_event("sources", {"sources": [], "retrieval_method": "rrf"})
            yield _sse_event(
                "warning",
                {"message": "No relevant context was retrieved; the model was not called."},
            )
            yield _sse_event("token", {"token": REFUSAL})
            yield _sse_event(
                "done",
                {"metadata": {"citation_validation": validate_citations(REFUSAL, 0).model_dump()}},
            )

        return StreamingResponse(refuse(), media_type="text/event-stream")

    async def generate() -> AsyncGenerator[str, None]:
        answer_parts: list[str] = []
        acquired = False
        try:
            yield _sse_event("stage", {"name": "retrieval", "status": "complete"})
            yield _sse_event(
                "sources",
                {
                    "sources": [source.model_dump(mode="json") for source in sources],
                    "retrieval_method": method,
                    "retrieval_ms": round(retrieval_ms, 2),
                },
            )
            services.acquire_generation()
            acquired = True
            if services.rag_generator is None:
                raise ModelUnavailableError("Generation provider is not configured")
            yield _sse_event("stage", {"name": "generation", "status": "running"})
            generated_at = time.perf_counter()
            async for token in services.rag_generator.generate_stream_async(package.messages):
                if await http_request.is_disconnected():
                    break
                answer_parts.append(token)
                yield _sse_event("token", {"token": token})
            generation_ms = (time.perf_counter() - generated_at) * 1000
            metadata = _metadata(
                services=services,
                sources=sources,
                package=package,
                retrieval_method=method,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                warnings=warnings,
                answer="".join(answer_parts),
            )
            for warning in metadata.warnings:
                yield _sse_event("warning", {"message": warning})
            yield _sse_event("stage", {"name": "generation", "status": "complete"})
            yield _sse_event("done", {"metadata": metadata.model_dump(mode="json")})
        except (
            ServiceBusyError,
            ModelUnavailableError,
            GenerationTimeoutError,
            GenerationRateLimitedError,
            GenerationInvalidResponseError,
            RAGGenerationError,
        ) as exc:
            code = (
                "generation_timeout"
                if isinstance(exc, GenerationTimeoutError)
                else "generation_error"
            )
            if isinstance(exc, ModelUnavailableError):
                code = "model_unavailable"
            if isinstance(exc, ServiceBusyError):
                code = "capacity_busy"
            if isinstance(exc, GenerationRateLimitedError):
                code = "generation_rate_limited"
            if isinstance(exc, GenerationInvalidResponseError):
                code = "generation_invalid_response"
            yield _sse_event("error", {"code": code, "message": str(exc), "retryable": True})
        finally:
            if acquired:
                services.release_generation()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
