"""Grounded RAG routes with typed HTTP errors and structured SSE."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
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
from nlp_academic_search.providers.verification.base import SemanticVerificationError
from nlp_academic_search.rag.citations import CitationValidation, validate_citations
from nlp_academic_search.rag.prompt_builder import (
    PROMPT_VERSION,
    InsufficientContextError,
    PromptPackage,
    build_citation_repair_messages,
    build_rag_messages,
    build_source_list,
)
from nlp_academic_search.rag.verification import SemanticValidation, validate_semantic_assessment

router = APIRouter(prefix="/ask", tags=["RAG"])
Services = Annotated[ServiceContainer, Depends(get_services)]
REFUSAL = "Not enough evidence in the retrieved sources."


@dataclass(frozen=True)
class CitationOutcome:
    answer: str
    validation: CitationValidation
    semantic_validation: SemanticValidation | None = None
    repair_attempted: bool = False
    repair_succeeded: bool | None = None
    semantic_attempted: bool = False
    semantic_succeeded: bool | None = None
    answer_status: str = "structurally_valid"
    verification_latency_ms: float = 0.0
    warning: str | None = None


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _validation_quality(validation: CitationValidation) -> tuple[int, int, int, float, float]:
    return (
        int(validation.valid),
        -len(validation.invalid_indices),
        -validation.uncited_claim_count,
        validation.claim_citation_coverage,
        validation.citation_precision,
    )


def _verify(
    services: ServiceContainer, package: PromptPackage, answer: str
) -> tuple[SemanticValidation | None, float, str | None]:
    """Run the optional judge and retain only locally checked evidence spans."""
    if not settings.verification.enabled:
        return None, 0.0, None
    verifier = services.semantic_verifier
    if verifier is None:
        return None, 0.0, "Semantic verification is unavailable."
    started = time.perf_counter()
    try:
        result = verifier.assess(package.messages[-1]["content"], answer, package.papers)
        claims = result["claims"]
        validation = validate_semantic_assessment(
            answer,
            package.papers,
            claims,
            provider=verifier.provider_name,
            model=verifier.model_name,
            independent=verifier.verifier_independent,
        )
        return validation, (time.perf_counter() - started) * 1000, None
    except (SemanticVerificationError, RuntimeError, KeyError, TypeError) as exc:
        return None, (time.perf_counter() - started) * 1000, str(exc)


def _status_for(
    structural: CitationValidation, semantic: SemanticValidation | None, verifier_error: str | None
) -> str:
    if structural.valid and semantic is not None and semantic.valid:
        return "verified"
    if structural.valid and not settings.verification.enabled:
        return "structurally_valid"
    if structural.valid and verifier_error:
        return "verification_unavailable"
    if structural.valid:
        return "verification_unavailable"
    return "refused_unverified" if settings.verification.fail_closed else "verification_unavailable"


def _validate_and_repair_sync(
    services: ServiceContainer,
    package: PromptPackage,
    answer: str,
) -> CitationOutcome:
    initial = validate_citations(answer, len(package.papers))
    semantic, latency, verifier_error = (
        _verify(services, package, answer) if initial.valid else (None, 0.0, None)
    )
    status = _status_for(initial, semantic, verifier_error)
    needs_repair = not initial.valid or (
        initial.valid and semantic is not None and not semantic.valid
    )
    if not needs_repair or settings.verification.max_repair_attempts == 0:
        return CitationOutcome(
            answer,
            initial,
            semantic,
            False,
            None,
            settings.verification.enabled,
            semantic.valid if semantic else None,
            status,
            latency,
            verifier_error,
        )
    if services.rag_generator is None:
        raise ModelUnavailableError("Generation provider is not configured")
    try:
        repaired = services.rag_generator.generate(
            build_citation_repair_messages(package, answer), temperature=0.0
        ).strip()
        if not repaired:
            raise GenerationInvalidResponseError("Citation repair returned an empty answer")
    except RAGGenerationError:
        repaired = answer
    final_structural = validate_citations(repaired, len(package.papers))
    final_semantic, repair_latency, final_error = (
        _verify(services, package, repaired) if final_structural.valid else (None, 0.0, None)
    )
    final_status = _status_for(final_structural, final_semantic, final_error)
    final_valid = (
        final_structural.valid
        and final_status in {"verified", "structurally_valid"}
        and not (
            settings.verification.enabled
            and settings.verification.fail_closed
            and final_status != "verified"
        )
    )
    if not final_valid and settings.verification.fail_closed:
        return CitationOutcome(
            REFUSAL,
            validate_citations(REFUSAL, 0),
            final_semantic,
            True,
            False,
            settings.verification.enabled,
            bool(final_semantic and final_semantic.valid),
            "refused_unverified",
            latency + repair_latency,
            "Answer withheld because evidence could not be verified.",
        )
    final_warning = final_error or (
        "Please review the answer for accuracy because some claims lack citations."
        if not final_structural.valid
        else None
    )
    return CitationOutcome(
        repaired,
        final_structural,
        final_semantic,
        True,
        final_valid,
        settings.verification.enabled,
        bool(final_semantic and final_semantic.valid),
        final_status,
        latency + repair_latency,
        final_warning,
    )


async def _validate_and_repair_async(
    services: ServiceContainer,
    package: PromptPackage,
    answer: str,
) -> CitationOutcome:
    initial = validate_citations(answer, len(package.papers))
    semantic, latency, verifier_error = (
        await asyncio.to_thread(_verify, services, package, answer)
        if initial.valid
        else (None, 0.0, None)
    )
    status = _status_for(initial, semantic, verifier_error)
    needs_repair = not initial.valid or (
        initial.valid and semantic is not None and not semantic.valid
    )
    if not needs_repair or settings.verification.max_repair_attempts == 0:
        return CitationOutcome(
            answer,
            initial,
            semantic,
            False,
            None,
            settings.verification.enabled,
            semantic.valid if semantic else None,
            status,
            latency,
            verifier_error,
        )
    if services.rag_generator is None:
        raise ModelUnavailableError("Generation provider is not configured")
    try:
        parts = [
            token
            async for token in services.rag_generator.generate_stream_async(
                build_citation_repair_messages(package, answer), temperature=0.0
            )
        ]
        repaired = "".join(parts).strip()
        if not repaired:
            raise GenerationInvalidResponseError("Citation repair returned an empty answer")
    except RAGGenerationError:
        repaired = answer
    final_structural = validate_citations(repaired, len(package.papers))
    final_semantic, repair_latency, final_error = (
        await asyncio.to_thread(_verify, services, package, repaired)
        if final_structural.valid
        else (None, 0.0, None)
    )
    final_status = _status_for(final_structural, final_semantic, final_error)
    final_valid = (
        final_structural.valid
        and final_status in {"verified", "structurally_valid"}
        and not (
            settings.verification.enabled
            and settings.verification.fail_closed
            and final_status != "verified"
        )
    )
    if not final_valid and settings.verification.fail_closed:
        return CitationOutcome(
            REFUSAL,
            validate_citations(REFUSAL, 0),
            final_semantic,
            True,
            False,
            settings.verification.enabled,
            bool(final_semantic and final_semantic.valid),
            "refused_unverified",
            latency + repair_latency,
            "Answer withheld because evidence could not be verified.",
        )
    final_warning = final_error or (
        "Please review the answer for accuracy because some claims lack citations."
        if not final_structural.valid
        else None
    )
    return CitationOutcome(
        repaired,
        final_structural,
        final_semantic,
        True,
        final_valid,
        settings.verification.enabled,
        bool(final_semantic and final_semantic.valid),
        final_status,
        latency + repair_latency,
        final_warning,
    )


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
    outcome: CitationOutcome,
) -> AnswerMetadata:
    if services.rag_generator is None:
        raise ModelUnavailableError("Generation provider is not configured")
    if outcome.warning:
        warnings.append(outcome.warning)
    warnings.extend(outcome.validation.warnings)
    if outcome.semantic_validation:
        warnings.extend(outcome.semantic_validation.warnings)
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
            verification_ms=round(outcome.verification_latency_ms, 2),
        ),
        warnings=list(dict.fromkeys(warnings)),
        citation_validation=outcome.validation,
        citation_repair_attempted=outcome.repair_attempted,
        citation_repair_succeeded=outcome.repair_succeeded,
        answer_status=outcome.answer_status,  # type: ignore[arg-type]
        semantic_validation=outcome.semantic_validation,
        semantic_verification_attempted=outcome.semantic_attempted,
        semantic_verification_succeeded=outcome.semantic_succeeded,
        final_answer_replaced=outcome.repair_attempted,
        verification_latency_ms=round(outcome.verification_latency_ms, 2),
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
                answer_status="refused_insufficient_context",
            ),
        )
    services.acquire_generation()
    generated_at = time.perf_counter()
    try:
        answer = services.rag_generator.generate(package.messages)  # type: ignore[attr-defined]
        outcome = _validate_and_repair_sync(services, package, answer)
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
        outcome=outcome,
    )
    return AskResponse(
        question=payload.question,
        answer=outcome.answer,
        sources=sources,
        metadata=metadata,
    )


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
                {
                    "metadata": {
                        "citation_validation": validate_citations(REFUSAL, 0).model_dump(),
                        "answer_status": "refused_insufficient_context",
                    }
                },
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
            answer = "".join(answer_parts).strip()
            if not answer:
                raise GenerationInvalidResponseError("Generation provider returned an empty answer")
            yield _sse_event("stage", {"name": "structural_validation", "status": "running"})
            # Compatibility alias retained for existing v1 SSE consumers.
            yield _sse_event("stage", {"name": "citation_validation", "status": "running"})
            initial_validation = validate_citations(answer, len(sources))
            yield _sse_event(
                "stage",
                {
                    "name": "structural_validation",
                    "status": "complete" if initial_validation.valid else "needs_repair",
                },
            )
            yield _sse_event(
                "stage",
                {
                    "name": "citation_validation",
                    "status": "complete" if initial_validation.valid else "needs_repair",
                },
            )
            if initial_validation.valid and settings.verification.enabled:
                yield _sse_event("stage", {"name": "semantic_validation", "status": "running"})
            outcome = await _validate_and_repair_async(services, package, answer)
            if settings.verification.enabled:
                yield _sse_event(
                    "stage",
                    {
                        "name": "semantic_validation",
                        "status": "complete"
                        if outcome.semantic_validation and outcome.semantic_validation.valid
                        else "unavailable"
                        if outcome.answer_status == "verification_unavailable"
                        else "needs_repair",
                    },
                )
            if outcome.repair_attempted:
                yield _sse_event(
                    "stage",
                    {
                        "name": "citation_repair",
                        "status": "complete" if outcome.repair_succeeded else "failed",
                    },
                )
                yield _sse_event(
                    "stage",
                    {
                        "name": "answer_repair",
                        "status": "complete" if outcome.repair_succeeded else "failed",
                    },
                )
            if outcome.answer != answer:
                yield _sse_event("answer_replacement", {"answer": outcome.answer})
            yield _sse_event(
                "stage",
                {
                    "name": "final_validation",
                    "status": "complete"
                    if outcome.answer_status in {"verified", "structurally_valid"}
                    else "failed",
                },
            )
            generation_ms = (time.perf_counter() - generated_at) * 1000
            metadata = _metadata(
                services=services,
                sources=sources,
                package=package,
                retrieval_method=method,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                warnings=warnings,
                outcome=outcome,
            )
            for warning in metadata.warnings:
                yield _sse_event("warning", {"message": warning})
            yield _sse_event(
                "stage",
                {
                    "name": "generation",
                    "status": "complete"
                    if outcome.answer_status in {"verified", "structurally_valid"}
                    else "needs_review",
                },
            )
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
