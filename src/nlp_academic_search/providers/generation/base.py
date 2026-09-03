"""Generation provider contract and provider-neutral failure taxonomy."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol


class RAGGenerationError(RuntimeError):
    """Base generation failure."""


class ModelUnavailableError(RAGGenerationError):
    """The configured generation provider or model is unavailable."""


class GenerationTimeoutError(RAGGenerationError):
    """Generation exceeded its configured deadline."""


class GenerationRateLimitedError(RAGGenerationError):
    """The provider rejected generation because its quota was exhausted."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


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
