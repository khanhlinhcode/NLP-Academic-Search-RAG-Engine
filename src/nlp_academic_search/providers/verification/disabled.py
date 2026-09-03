"""Explicit disabled verifier used for local/offline deployments."""

from __future__ import annotations

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.providers.verification.base import SemanticVerificationUnavailable
from nlp_academic_search.rag.verification import SemanticValidation


class DisabledSemanticVerificationProvider:
    provider_name = "disabled"
    model_name = "disabled"
    verifier_independent = False

    def verify(self, answer: str, sources: list[Paper], question: str) -> SemanticValidation:
        del answer, sources, question
        raise SemanticVerificationUnavailable("Semantic verification is disabled")

    def is_available(self) -> bool:
        return False

    def close(self) -> None:
        return None
