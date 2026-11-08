"""Generation provider contract and provider-neutral failure taxonomy."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol


class RAGGenerationError(RuntimeError):
    """Base generation failure."""

    def __init__(
        self,
        message: str,
        *,
        provider_http_status: int | None = None,
        provider_request_id: str | None = None,
        retry_after: float | None = None,
        finish_reason: str | None = None,
        output_token_usage: dict[str, Any] | None = None,
        tokens_emitted: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider_http_status = provider_http_status
        self.provider_request_id = provider_request_id
        self.retry_after = retry_after
        self.finish_reason = finish_reason
        self.output_token_usage = output_token_usage
        self.tokens_emitted = tokens_emitted


class ModelUnavailableError(RAGGenerationError):
    """The configured generation provider or model is unavailable."""


class GenerationTimeoutError(RAGGenerationError):
    """Generation exceeded its configured deadline."""


class GenerationRateLimitedError(RAGGenerationError):
    """The provider rejected generation because its quota was exhausted."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        *,
        provider_http_status: int | None = 429,
        provider_request_id: str | None = None,
        finish_reason: str | None = None,
        output_token_usage: dict[str, Any] | None = None,
        tokens_emitted: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider_http_status=provider_http_status,
            provider_request_id=provider_request_id,
            retry_after=retry_after,
            finish_reason=finish_reason,
            output_token_usage=output_token_usage,
            tokens_emitted=tokens_emitted,
        )


class GenerationInvalidResponseError(RAGGenerationError):
    """The provider returned a response that violates its documented schema."""


class GenerationProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str: ...

    def generate_stream_async(
        self, messages: list[dict[str, str]], temperature: float = 0.2
    ) -> AsyncGenerator[str, None]: ...

    def is_available(self) -> bool: ...

    def close(self) -> None: ...
