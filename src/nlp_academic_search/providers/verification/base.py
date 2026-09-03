"""Contracts and typed failures for semantic evidence verification."""

from __future__ import annotations

from typing import Protocol

from nlp_academic_search.data.loader import Paper


class SemanticVerificationError(RuntimeError):
    """A verifier could not return a trustworthy structured assessment."""


class SemanticVerificationUnavailable(SemanticVerificationError):
    """The configured verifier or model is unavailable."""


class SemanticVerificationInvalidResponse(SemanticVerificationError):
    """The verifier returned output that does not match its strict contract."""


class SemanticVerificationProvider(Protocol):
    provider_name: str
    model_name: str
    verifier_independent: bool

    def assess(self, question: str, answer: str, papers: list[Paper]) -> dict: ...

    def is_available(self) -> bool: ...

    def close(self) -> None: ...
