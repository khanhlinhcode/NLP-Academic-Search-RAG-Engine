"""Deterministic tests for lightweight cloud retrieval and generation providers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from nlp_academic_search.config import GroqConfig, QdrantConfig
from nlp_academic_search.providers.generation.base import (
    GenerationInvalidResponseError,
    GenerationRateLimitedError,
)
from nlp_academic_search.providers.generation.groq import GroqGenerationProvider
from nlp_academic_search.providers.retrieval.qdrant_cloud import (
    QdrantCloudRetrievalProvider,
)
from nlp_academic_search.search.models import SearchFilters


def qdrant_config() -> QdrantConfig:
    return QdrantConfig(
        url="https://fixture.qdrant.test",
        api_key="secret",
        collection_alias="papers-current",
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        sparse_model="qdrant/bm25",
        timeout_seconds=5,
        expected_corpus_sha256="abc123",
        expected_schema_version=1,
    )


def point(score: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        payload={
            "record_type": "paper",
            "paper_id": "arxiv:1706.03762",
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "A Transformer uses attention.",
            "authors": ["Ashish Vaswani"],
            "categories": ["cs.CL"],
            "year": 2017,
            "published_at": "2017-06-12T00:00:00Z",
            "source": "arxiv",
        },
    )


class FakeQdrantClient:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = list(responses or [SimpleNamespace(points=[point()])])
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get_collection(self, _name):
        return SimpleNamespace(status="green")

    def count(self, **_kwargs):
        return SimpleNamespace(count=1)

    def scroll(self, **_kwargs):
        return (
            [
                SimpleNamespace(
                    payload={
                        "paper_count": 1,
                        "schema_version": 1,
                        "corpus_version": "fixture-v1",
                        "corpus_sha256": "abc123",
                        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "sparse_model": "qdrant/bm25",
                    }
                )
            ],
            None,
        )


@pytest.mark.parametrize(
    ("method", "score_name"),
    [("bm25", "bm25_score"), ("semantic", "semantic_score"), ("hybrid", "rrf_score")],
)
def test_qdrant_maps_each_retrieval_method(method, score_name):
    client = FakeQdrantClient()
    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=client)

    batch = provider.search(
        "attention",
        method,
        3,
        filters=SearchFilters(category="cs.CL", year_from=2010, author="Vaswani"),
    )

    assert batch.retrieval_mode == ("rrf" if method == "hybrid" else method)
    assert batch.results[0].paper.id == "arxiv:1706.03762"
    assert getattr(batch.results[0], score_name) == 0.5
    assert client.calls[0]["collection_name"] == "papers-current"


def test_qdrant_hybrid_can_report_explicit_bm25_degradation():
    client = FakeQdrantClient(
        [RuntimeError("dense unavailable"), SimpleNamespace(points=[point(2.1)])]
    )
    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=client, allow_degraded=True)

    batch = provider.search("attention", "hybrid", 3)

    assert batch.retrieval_mode == "bm25_degraded"
    assert batch.results[0].bm25_score == 2.1
    assert "BM25-only" in batch.warnings[0]


def test_qdrant_status_validates_manifest():
    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=FakeQdrantClient())
    status = provider.status()
    assert status.ready is True
    assert status.total_papers == 1
    assert status.provenance == "qdrant-cloud:fixture-v1"


def groq_config() -> GroqConfig:
    return GroqConfig(
        base_url="https://api.groq.test/openai/v1",
        api_key="groq-secret",
        model_name="openai/gpt-oss-20b",
        reasoning_effort="low",
        timeout_seconds=5,
        max_output_tokens=128,
    )


def test_groq_non_streaming_and_authorization_header():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer groq-secret"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == 128
        assert body["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer [1]."}}]})

    client = httpx.Client(
        base_url="https://api.groq.test/openai/v1", transport=httpx.MockTransport(handler)
    )
    provider = GroqGenerationProvider(groq_config(), client=client)
    assert provider.generate([{"role": "user", "content": "Question"}]) == "Answer [1]."


@pytest.mark.asyncio
async def test_groq_streaming_parses_deltas():
    def handler(_request: httpx.Request) -> httpx.Response:
        content = (
            'data: {"choices":[{"delta":{"content":"Grounded "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"[1]."}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    provider = GroqGenerationProvider(groq_config(), async_transport=httpx.MockTransport(handler))
    tokens = [
        token
        async for token in provider.generate_stream_async([{"role": "user", "content": "Question"}])
    ]
    assert tokens == ["Grounded ", "[1]."]


def test_groq_maps_rate_limit_and_invalid_payload():
    def limited(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "2"})

    rate_client = httpx.Client(
        base_url="https://api.groq.test/openai/v1", transport=httpx.MockTransport(limited)
    )
    with pytest.raises(GenerationRateLimitedError) as raised:
        GroqGenerationProvider(groq_config(), client=rate_client).generate([])
    assert raised.value.retry_after == 2

    invalid_client = httpx.Client(
        base_url="https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    )
    with pytest.raises(GenerationInvalidResponseError):
        GroqGenerationProvider(groq_config(), client=invalid_client).generate([])
