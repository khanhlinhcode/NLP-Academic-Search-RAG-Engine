"""Grounded RAG routes with typed HTTP errors and structured SSE."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Annotated, Literal

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
from nlp_academic_search.rag.citations import CitationValidation, is_refusal, validate_citations
from nlp_academic_search.rag.prompt_builder import (
    PROMPT_VERSION,
    InsufficientContextError,
    PromptPackage,
    build_citation_repair_messages,
    build_rag_messages,
    build_source_list,
)
from nlp_academic_search.rag.verification import SemanticValidation

router = APIRouter(prefix="/ask", tags=["RAG"])
logger = logging.getLogger("academic_search.rag")
Services = Annotated[ServiceContainer, Depends(get_services)]
REFUSAL = "Not enough evidence in the retrieved sources."
UNVERIFIED_REFUSAL = "Not enough verified evidence in the retrieved sources."
AnswerStatus = Literal[
    "verified",
    "structurally_valid",
    "refused_insufficient_context",
    "refused_unverified",
    "verification_unavailable",
]


@dataclass(frozen=True)
class SemanticCheck:
    validation: SemanticValidation | None
    attempted: bool
    latency_ms: float = 0.0
    error_category: str | None = None
    provider_http_status: int | None = None
    provider_request_id: str | None = None


@dataclass(frozen=True)
class CitationOutcome:
    answer: str
    validation: CitationValidation
    initial_validation: CitationValidation
    semantic_validation: SemanticValidation | None = None
    initial_semantic_validation: SemanticValidation | None = None
    repair_attempted: bool = False
    repair_succeeded: bool | None = None
    semantic_attempted: bool = False
    semantic_succeeded: bool | None = None
    answer_status: AnswerStatus = "structurally_valid"
    verification_latency_ms: float = 0.0
    final_answer_replaced: bool = False
    failure_reason: str | None = None
    verification_provider_http_status: int | None = None
    verification_provider_request_id: str | None = None
    warning: str | None = None


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _log_outcome(request: Request, outcome: CitationOutcome, source_count: int) -> None:
    logger.info(
        json.dumps(
            {
                "event": "rag_validation_complete",
                "request_id": getattr(request.state, "request_id", None),
                "provider": settings.verification.provider,
                "model": (
                    settings.verification.model_name
                    if settings.verification.enabled
                    else "disabled"
                ),
                "answer_status": outcome.answer_status,
                "source_count": source_count,
                "structural_valid": outcome.validation.valid,
                "semantic_valid": bool(
                    outcome.semantic_validation and outcome.semantic_validation.valid
                ),
                "repair_attempted": outcome.repair_attempted,
                "verification_latency_ms": round(outcome.verification_latency_ms, 2),
                "provider_http_status": outcome.verification_provider_http_status,
                "provider_error_category": outcome.failure_reason,
                "provider_request_id": outcome.verification_provider_request_id,
            }
        )
    )


def _verify(services: ServiceContainer, package: PromptPackage, answer: str) -> SemanticCheck:
    """Run the optional judge and retain only locally checked evidence spans."""
    if not settings.verification.enabled:
        return SemanticCheck(None, False)
    verifier = services.semantic_verifier
    if verifier is None:
        return SemanticCheck(None, False, error_category="not_configured")
    started = time.perf_counter()
    try:
        validation = verifier.verify(answer, package.papers, package.question)
        return SemanticCheck(
            validation,
            True,
            (time.perf_counter() - started) * 1000,
        )
    except SemanticVerificationError as exc:
        return SemanticCheck(
            None,
            True,
            (time.perf_counter() - started) * 1000,
            type(exc).__name__,
            getattr(exc, "provider_http_status", None),
            getattr(exc, "provider_request_id", None),
        )


def _verification_warning(status: AnswerStatus, reason: str | None) -> str | None:
    if status == "verification_unavailable":
        return "Semantic verification is unavailable; the answer has structural checks only."
    if status != "refused_unverified":
        return None
    messages = {
        "SemanticVerificationInvalidRequest": (
            "Answer withheld because the semantic verifier rejected its request contract."
        ),
        "SemanticVerificationInvalidResponse": (
            "Answer withheld because the semantic verifier returned an invalid response."
        ),
        "SemanticVerificationAuthenticationError": (
            "Answer withheld because semantic-verifier authentication failed."
        ),
        "SemanticVerificationRateLimited": (
            "Answer withheld because the semantic verifier is rate-limited."
        ),
        "SemanticVerificationTimeout": ("Answer withheld because the semantic verifier timed out."),
        "SemanticVerificationUnavailable": (
            "Answer withheld because the semantic verifier is unavailable."
        ),
        "semantic_assessment_failed": (
            "Answer withheld because one or more claims lacked verified evidence."
        ),
    }
    if reason is not None and reason in messages:
        return messages[reason]
    return "Answer withheld because retrieved evidence could not verify every claim."


def _status_for(
    answer: str, structural: CitationValidation, semantic: SemanticCheck
) -> AnswerStatus:
    if is_refusal(answer):
        return "refused_insufficient_context"
    validation = semantic.validation
    if structural.valid and validation is not None and validation.valid:
        return "verified"
    if structural.valid and not settings.verification.enabled:
        return "structurally_valid"
    if structural.valid and validation is None and not settings.verification.fail_closed:
        return "verification_unavailable"
    return "refused_unverified"


def _accepted(status: AnswerStatus) -> bool:
    return status in {
        "verified",
        "structurally_valid",
        "refused_insufficient_context",
        "verification_unavailable",
    }


def _outcome(
    *,
    draft: str,
    answer: str,
    initial_structural: CitationValidation,
    final_structural: CitationValidation,
    initial_semantic: SemanticCheck,
    final_semantic: SemanticCheck,
    status: AnswerStatus,
    repair_attempted: bool,
    repair_succeeded: bool | None,
    failure_reason: str | None = None,
) -> CitationOutcome:
    semantic_attempted = initial_semantic.attempted or final_semantic.attempted
    provider_error = final_semantic if final_semantic.error_category else initial_semantic
    return CitationOutcome(
        answer=answer,
        validation=final_structural,
        initial_validation=initial_structural,
        semantic_validation=final_semantic.validation,
        initial_semantic_validation=initial_semantic.validation,
        repair_attempted=repair_attempted,
        repair_succeeded=repair_succeeded,
        semantic_attempted=semantic_attempted,
        semantic_succeeded=(
            final_semantic.validation.valid
            if final_semantic.validation is not None
            else False
            if final_semantic.attempted
            else False
            if semantic_attempted
            else None
        ),
        answer_status=status,
        verification_latency_ms=(
            initial_semantic.latency_ms
            if initial_semantic is final_semantic
            else initial_semantic.latency_ms + final_semantic.latency_ms
        ),
        final_answer_replaced=answer != draft,
        failure_reason=failure_reason,
        verification_provider_http_status=provider_error.provider_http_status,
        verification_provider_request_id=provider_error.provider_request_id,
        warning=_verification_warning(status, failure_reason),
    )


def _refused_outcome(
    *,
    draft: str,
    package: PromptPackage,
    initial_structural: CitationValidation,
    initial_semantic: SemanticCheck,
    final_semantic: SemanticCheck,
    repair_attempted: bool,
    reason: str,
) -> CitationOutcome:
    return _outcome(
        draft=draft,
        answer=UNVERIFIED_REFUSAL,
        initial_structural=initial_structural,
        final_structural=validate_citations(UNVERIFIED_REFUSAL, len(package.papers)),
        initial_semantic=initial_semantic,
        final_semantic=final_semantic,
        status="refused_unverified",
        repair_attempted=repair_attempted,
        repair_succeeded=False if repair_attempted else None,
        failure_reason=reason,
    )


def _validate_and_repair_sync(
    services: ServiceContainer,
    package: PromptPackage,
    answer: str,
) -> CitationOutcome:
    initial = validate_citations(answer, len(package.papers))
    initial_semantic = (
        _verify(services, package, answer)
        if initial.valid and not is_refusal(answer)
        else SemanticCheck(None, False)
    )
    status = _status_for(answer, initial, initial_semantic)
    if _accepted(status):
        return _outcome(
            draft=answer,
            answer=answer,
            initial_structural=initial,
            final_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=initial_semantic,
            status=status,
            repair_attempted=False,
            repair_succeeded=None,
        )
    if initial_semantic.validation is None and initial.valid:
        return _refused_outcome(
            draft=answer,
            package=package,
            initial_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=initial_semantic,
            repair_attempted=False,
            reason=initial_semantic.error_category or "verification_unavailable",
        )
    if settings.verification.max_repair_attempts == 0:
        return _refused_outcome(
            draft=answer,
            package=package,
            initial_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=initial_semantic,
            repair_attempted=False,
            reason=(
                "semantic_assessment_failed"
                if initial_semantic.validation is not None
                else "repair_disabled"
            ),
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
        return _refused_outcome(
            draft=answer,
            package=package,
            initial_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=SemanticCheck(None, False),
            repair_attempted=True,
            reason="repair_provider_error",
        )
    final_structural = validate_citations(repaired, len(package.papers))
    final_semantic = (
        _verify(services, package, repaired)
        if final_structural.valid
        else SemanticCheck(None, False)
    )
    final_status = _status_for(repaired, final_structural, final_semantic)
    if _accepted(final_status) and not (
        initial_semantic.validation is not None
        and not initial_semantic.validation.valid
        and final_status == "verification_unavailable"
    ):
        return _outcome(
            draft=answer,
            answer=repaired,
            initial_structural=initial,
            final_structural=final_structural,
            initial_semantic=initial_semantic,
            final_semantic=final_semantic,
            status=final_status,
            repair_attempted=True,
            repair_succeeded=True,
        )
    return _refused_outcome(
        draft=answer,
        package=package,
        initial_structural=initial,
        initial_semantic=initial_semantic,
        final_semantic=final_semantic,
        repair_attempted=True,
        reason=(
            final_semantic.error_category
            or (
                "semantic_assessment_failed"
                if final_semantic.validation is not None
                else "final_validation_failed"
            )
        ),
    )


async def _validate_and_repair_async(
    services: ServiceContainer,
    package: PromptPackage,
    answer: str,
    initial: CitationValidation,
    initial_semantic: SemanticCheck,
) -> CitationOutcome:
    status = _status_for(answer, initial, initial_semantic)
    if _accepted(status):
        return _outcome(
            draft=answer,
            answer=answer,
            initial_structural=initial,
            final_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=initial_semantic,
            status=status,
            repair_attempted=False,
            repair_succeeded=None,
        )
    if initial_semantic.validation is None and initial.valid:
        return _refused_outcome(
            draft=answer,
            package=package,
            initial_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=initial_semantic,
            repair_attempted=False,
            reason=initial_semantic.error_category or "verification_unavailable",
        )
    if settings.verification.max_repair_attempts == 0:
        return _refused_outcome(
            draft=answer,
            package=package,
            initial_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=initial_semantic,
            repair_attempted=False,
            reason=(
                "semantic_assessment_failed"
                if initial_semantic.validation is not None
                else "repair_disabled"
            ),
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
        return _refused_outcome(
            draft=answer,
            package=package,
            initial_structural=initial,
            initial_semantic=initial_semantic,
            final_semantic=SemanticCheck(None, False),
            repair_attempted=True,
            reason="repair_provider_error",
        )
    final_structural = validate_citations(repaired, len(package.papers))
    final_semantic = (
        await asyncio.to_thread(_verify, services, package, repaired)
        if final_structural.valid
        else SemanticCheck(None, False)
    )
    final_status = _status_for(repaired, final_structural, final_semantic)
    if _accepted(final_status) and not (
        initial_semantic.validation is not None
        and not initial_semantic.validation.valid
        and final_status == "verification_unavailable"
    ):
        return _outcome(
            draft=answer,
            answer=repaired,
            initial_structural=initial,
            final_structural=final_structural,
            initial_semantic=initial_semantic,
            final_semantic=final_semantic,
            status=final_status,
            repair_attempted=True,
            repair_succeeded=True,
        )
    return _refused_outcome(
        draft=answer,
        package=package,
        initial_structural=initial,
        initial_semantic=initial_semantic,
        final_semantic=final_semantic,
        repair_attempted=True,
        reason=(
            final_semantic.error_category
            or (
                "semantic_assessment_failed"
                if final_semantic.validation is not None
                else "final_validation_failed"
            )
        ),
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
            total_ms=round(retrieval_ms + generation_ms + outcome.verification_latency_ms, 2),
            verification_ms=round(outcome.verification_latency_ms, 2),
        ),
        warnings=list(dict.fromkeys(warnings)),
        citation_validation=outcome.validation,
        initial_citation_validation=outcome.initial_validation,
        citation_repair_attempted=outcome.repair_attempted,
        citation_repair_succeeded=outcome.repair_succeeded,
        answer_status=outcome.answer_status,  # type: ignore[arg-type]
        semantic_validation=outcome.semantic_validation,
        initial_semantic_validation=outcome.initial_semantic_validation,
        semantic_verification_attempted=outcome.semantic_attempted,
        semantic_verification_succeeded=outcome.semantic_succeeded,
        final_answer_replaced=outcome.final_answer_replaced,
        verification_latency_ms=round(outcome.verification_latency_ms, 2),
        failure_reason=outcome.failure_reason,
        verification_provider_http_status=outcome.verification_provider_http_status,
        verification_provider_request_id=outcome.verification_provider_request_id,
    )


@router.post("", response_model=AskResponse, summary="Grounded question answering")
def ask_question(payload: AskRequest, services: Services, http_request: Request) -> AskResponse:
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
    pipeline_ms = (time.perf_counter() - generated_at) * 1000
    generation_ms = max(0.0, pipeline_ms - outcome.verification_latency_ms)
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
    _log_outcome(http_request, outcome, len(sources))
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
                    "answer": REFUSAL,
                    "metadata": {
                        "citation_validation": validate_citations(REFUSAL, 0).model_dump(),
                        "answer_status": "refused_insufficient_context",
                    },
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
                    return
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
            initial_semantic = SemanticCheck(None, False)
            if (
                initial_validation.valid
                and not is_refusal(answer)
                and settings.verification.enabled
            ):
                yield _sse_event("stage", {"name": "semantic_validation", "status": "running"})
                if await http_request.is_disconnected():
                    return
                initial_semantic = await asyncio.to_thread(_verify, services, package, answer)
                yield _sse_event(
                    "stage",
                    {
                        "name": "semantic_validation",
                        "status": "complete"
                        if initial_semantic.validation and initial_semantic.validation.valid
                        else "unavailable"
                        if initial_semantic.validation is None
                        else "failed",
                    },
                )
            elif not is_refusal(answer):
                yield _sse_event(
                    "stage",
                    {
                        "name": "semantic_validation",
                        "status": "skipped" if settings.verification.enabled else "disabled",
                    },
                )
            initial_status = _status_for(answer, initial_validation, initial_semantic)
            will_repair = (
                not _accepted(initial_status)
                and not (initial_validation.valid and initial_semantic.validation is None)
                and settings.verification.max_repair_attempts > 0
            )
            if will_repair:
                yield _sse_event("stage", {"name": "answer_repair", "status": "running"})
            outcome = await _validate_and_repair_async(
                services, package, answer, initial_validation, initial_semantic
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
            pipeline_ms = (time.perf_counter() - generated_at) * 1000
            generation_ms = max(0.0, pipeline_ms - outcome.verification_latency_ms)
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
            _log_outcome(http_request, outcome, len(sources))
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
            yield _sse_event(
                "done",
                {
                    "answer": outcome.answer,
                    "metadata": metadata.model_dump(mode="json"),
                },
            )
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
