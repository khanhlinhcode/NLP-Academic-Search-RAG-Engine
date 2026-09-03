"""Credential-gated smoke checks for externally managed cloud providers."""

from __future__ import annotations

import os

import pytest

from nlp_academic_search.config import GroqConfig, QdrantConfig
from nlp_academic_search.providers.generation.groq import GroqGenerationProvider
from nlp_academic_search.providers.retrieval.qdrant_cloud import (
    QdrantCloudRetrievalProvider,
)


def _required(*names: str) -> dict[str, str]:
    values = {name: os.getenv(name, "") for name in names}
    if missing := [name for name, value in values.items() if not value]:
        pytest.skip(f"Missing cloud integration credentials: {', '.join(missing)}")
    return values


@pytest.mark.integration
def test_qdrant_cloud_collection_is_ready():
    values = _required("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_DENSE_MODEL")
    provider = QdrantCloudRetrievalProvider(
        QdrantConfig(
            url=values["QDRANT_URL"],
            api_key=values["QDRANT_API_KEY"],
            collection_alias=os.getenv("QDRANT_COLLECTION_ALIAS", "academic-papers-current"),
            dense_model=values["QDRANT_DENSE_MODEL"],
            sparse_model=os.getenv("QDRANT_SPARSE_MODEL", "qdrant/bm25"),
            timeout_seconds=15,
            expected_corpus_sha256=os.getenv("QDRANT_EXPECTED_CORPUS_SHA256") or None,
            expected_schema_version=1,
        )
    )
    try:
        assert provider.status().ready
        assert provider.search("information retrieval", "hybrid", 3).results
    finally:
        provider.close()


@pytest.mark.integration
def test_groq_configured_model_is_available():
    values = _required("GROQ_API_KEY")
    provider = GroqGenerationProvider(
        GroqConfig(
            base_url=os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=values["GROQ_API_KEY"],
            model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            timeout_seconds=15,
            max_output_tokens=32,
        )
    )
    try:
        assert provider.is_available()
    finally:
        provider.close()
