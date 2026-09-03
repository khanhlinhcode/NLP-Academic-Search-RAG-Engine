"""Contracts and typed failures for semantic evidence verification."""

from __future__ import annotations

from typing import Protocol

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.rag.verification import SemanticValidation


class SemanticVerificationError(RuntimeError):
    """A verifier could not return a trustworthy structured assessment."""


class SemanticVerificationUnavailable(SemanticVerificationError):
    """The configured verifier or model is unavailable."""


class SemanticVerificationInvalidResponse(SemanticVerificationError):
    """The verifier returned output that does not match its strict contract."""


class SemanticVerificationTimeout(SemanticVerificationUnavailable):
    """The verifier exceeded its bounded request deadline."""


class SemanticVerificationRateLimited(SemanticVerificationUnavailable):
    """The verifier rejected the request due to quota or rate limits."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SemanticVerificationProvider(Protocol):
    provider_name: str
    model_name: str
    verifier_independent: bool

    def verify(self, answer: str, sources: list[Paper], question: str) -> SemanticValidation: ...

    def is_available(self) -> bool: ...

    def close(self) -> None: ...
