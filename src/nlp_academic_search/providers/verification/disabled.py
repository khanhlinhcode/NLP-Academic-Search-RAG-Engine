"""Explicit disabled verifier used for local/offline deployments."""

from __future__ import annotations


class DisabledSemanticVerificationProvider:
    provider_name = "disabled"
    model_name = "disabled"
    verifier_independent = False

    def assess(self, question: str, answer: str, papers: list) -> dict:
        del question, answer, papers
        raise RuntimeError("Semantic verification is disabled")

    def is_available(self) -> bool:
        return False

    def close(self) -> None:
        return None
