"""Select provider implementations without importing local ML code in cloud mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlp_academic_search.config import Settings, settings
from nlp_academic_search.providers.generation.base import GenerationProvider
from nlp_academic_search.providers.reranking.base import RerankerProvider
from nlp_academic_search.providers.retrieval.base import RetrievalProvider
from nlp_academic_search.providers.verification.base import SemanticVerificationProvider


@dataclass(frozen=True)
class ProviderBundle:
    retrieval: RetrievalProvider
    generation: GenerationProvider
    reranker: RerankerProvider | None
    verifier: SemanticVerificationProvider | None


def build_provider_bundle(config: Settings = settings) -> ProviderBundle:
    if config.deployment_profile == "cloud":
        from nlp_academic_search.providers.generation.groq import GroqGenerationProvider
        from nlp_academic_search.providers.retrieval.qdrant_cloud import (
            QdrantCloudRetrievalProvider,
        )

        retrieval = QdrantCloudRetrievalProvider(
            config.qdrant,
            allow_degraded=config.allow_degraded_retrieval,
            candidate_pool=config.search.candidate_pool,
        )
        generation = GroqGenerationProvider(config.groq)
        verifier: SemanticVerificationProvider | None = None
        if config.verification.enabled and config.verification.provider == "groq":
            from nlp_academic_search.providers.verification.groq import (
                GroqSemanticVerificationProvider,
            )

            verifier = GroqSemanticVerificationProvider(config.groq, config.verification)
        return ProviderBundle(
            retrieval=retrieval, generation=generation, reranker=None, verifier=verifier
        )

    from nlp_academic_search.providers.generation.ollama import build_ollama_provider
    from nlp_academic_search.providers.retrieval.local import LocalRetrievalProvider

    retrieval = LocalRetrievalProvider()
    generation = build_ollama_provider()
    reranker: Any = None
    if config.reranker.enabled and config.reranker_provider == "local":
        from nlp_academic_search.providers.reranking.local import LocalRerankerProvider

        reranker = LocalRerankerProvider()
    return ProviderBundle(
        retrieval=retrieval, generation=generation, reranker=reranker, verifier=None
    )
