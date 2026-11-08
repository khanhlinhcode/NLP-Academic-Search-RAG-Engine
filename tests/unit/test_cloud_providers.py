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
        if not self.responses:
            return SimpleNamespace(points=[])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def query_batch_points(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            return [SimpleNamespace(points=[]), SimpleNamespace(points=[])]
        if isinstance(self.responses[0], BaseException):
            exc = self.responses.pop(0)
            if isinstance(exc, BaseException):
                raise exc
        dense_resp = self.responses.pop(0)
        sparse_resp = self.responses.pop(0) if self.responses else SimpleNamespace(points=[])
        return [dense_resp, sparse_resp]

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


def test_qdrant_hybrid_batch_populates_per_channel_scores():
    client = FakeQdrantClient(
        [
            SimpleNamespace(points=[point(0.72)]),
            SimpleNamespace(points=[point(8.5)]),
        ]
    )
    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=client)

    batch = provider.search("attention mechanisms", "hybrid", 3)

    assert batch.retrieval_mode == "rrf"
    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.paper.id == "arxiv:1706.03762"
    assert result.semantic_score == 0.72
    assert result.bm25_score == 8.5
    # Rank 0 in both dense and sparse: 1/(2+0) + 1/(2+0) = 0.5 + 0.5 = 1.00
    assert result.rrf_score == 1.00


def test_qdrant_real_local_engine_batch_query_parity():
    """Verify Qdrant batch querying and RRF parity directly against real local Qdrant Rust engine."""
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection(
        "papers-current",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    client.upsert(
        "papers-current",
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "dense": [1.0, 0.0, 0.0, 0.0],
                    "sparse": models.SparseVector(indices=[1], values=[10.0]),
                },
                payload={
                    "record_type": "paper",
                    "paper_id": "arxiv:1706.03762",
                    "title": "Attention Is All You Need",
                    "abstract": "The Transformer uses attention mechanisms.",
                    "authors": ["Ashish Vaswani"],
                    "categories": ["cs.CL"],
                    "year": 2017,
                    "published_at": "2017-06-12T00:00:00Z",
                    "source": "arxiv",
                },
            ),
            models.PointStruct(
                id=2,
                vector={
                    "dense": [0.0, 1.0, 0.0, 0.0],
                    "sparse": models.SparseVector(indices=[2], values=[5.0]),
                },
                payload={
                    "record_type": "paper",
                    "paper_id": "arxiv:1810.04805",
                    "title": "BERT",
                    "abstract": "Deep Bidirectional Transformers.",
                    "authors": ["Jacob Devlin"],
                    "categories": ["cs.CL"],
                    "year": 2018,
                    "published_at": "2018-10-11T00:00:00Z",
                    "source": "arxiv",
                },
            ),
        ],
    )

    # 1. Run actual server-side FusionQuery RRF
    fusion_resp = client.query_points(
        collection_name="papers-current",
        prefetch=[
            models.Prefetch(query=[1.0, 0.0, 0.0, 0.0], using="dense", limit=10),
            models.Prefetch(
                query=models.SparseVector(indices=[1], values=[1.0]), using="sparse", limit=10
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=2,
    )
    server_points = fusion_resp.points

    # 2. Run actual batch query on local engine
    batch_resp = client.query_batch_points(
        collection_name="papers-current",
        requests=[
            models.QueryRequest(
                query=[1.0, 0.0, 0.0, 0.0], using="dense", limit=10, with_payload=True
            ),
            models.QueryRequest(
                query=models.SparseVector(indices=[1], values=[1.0]),
                using="sparse",
                limit=10,
                with_payload=True,
            ),
        ],
    )
    dense_pts = batch_resp[0].points
    sparse_pts = batch_resp[1].points

    # Compute manual RRF with k=2 and 0-based rank
    rrf_k = 2.0
    scores = {}
    for r, p in enumerate(dense_pts):
        scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (rrf_k + r)
    for r, p in enumerate(sparse_pts):
        scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (rrf_k + r)

    manual_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Assert exact parity between server-side RRF and batched calculation
    assert len(server_points) == len(manual_sorted) == 2
    for s_pt, (m_id, m_score) in zip(server_points, manual_sorted, strict=True):
        assert s_pt.id == m_id
        assert abs(s_pt.score - m_score) < 1e-5


def make_custom_point(paper_id: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        payload={
            "record_type": "paper",
            "paper_id": paper_id,
            "title": f"Title for {paper_id}",
            "abstract": f"Abstract for {paper_id}",
            "authors": ["Author"],
            "categories": ["cs.CL"],
            "year": 2020,
            "published_at": "2020-01-01T00:00:00Z",
            "source": "arxiv",
        },
    )


def test_qdrant_tie_breaking_dense_zero_strictly_precedes_none():
    """Verify that when RRF scores tie, measured dense score 0.0 strictly precedes missing score None."""

    class Client:
        def query_batch_points(self, **_):
            return [
                SimpleNamespace(
                    points=[make_custom_point("dummy", 0.9), make_custom_point("p_zero", 0.0)]
                ),
                SimpleNamespace(
                    points=[make_custom_point("dummy2", 10.0), make_custom_point("p_none", 5.0)]
                ),
            ]

    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=Client())
    batch = provider.search("test", "hybrid", 4)
    res = {r.paper.id: r for r in batch.results}
    assert res["p_zero"].semantic_score == 0.0
    assert res["p_none"].semantic_score is None

    order = [r.paper.id for r in batch.results]
    assert order.index("p_zero") < order.index("p_none")


def test_qdrant_tie_breaking_sparse_zero_handled_and_ordered():
    """Verify that when RRF and dense scores tie, higher BM25 score precedes 0.0, which is preserved."""

    class Client:
        def query_batch_points(self, **_):
            return [
                SimpleNamespace(
                    points=[make_custom_point("doc_1", 0.5), make_custom_point("doc_2", 0.5)]
                ),
                SimpleNamespace(
                    points=[make_custom_point("doc_2", 10.0), make_custom_point("doc_1", 0.0)]
                ),
            ]

    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=Client())
    batch = provider.search("test", "hybrid", 2)
    order = [r.paper.id for r in batch.results]
    assert order == ["doc_2", "doc_1"]
    assert batch.results[1].bm25_score == 0.0


def test_qdrant_missing_scores_preserved_as_none():
    """Verify that missing modality scores remain strictly None without artificial defaults."""

    class Client:
        def query_batch_points(self, **_):
            return [
                SimpleNamespace(points=[make_custom_point("dense_only", 0.8)]),
                SimpleNamespace(points=[make_custom_point("sparse_only", 12.0)]),
            ]

    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=Client())
    batch = provider.search("test", "hybrid", 2)
    res = {r.paper.id: r for r in batch.results}
    assert res["dense_only"].bm25_score is None
    assert res["sparse_only"].semantic_score is None


def test_qdrant_unrounded_rrf_preserves_sub_micro_differences():
    """Verify that real unrounded RRF scores sort strictly without 6-decimal truncation collision."""

    # Difference: 1/(2+198) + 1/(2+200) vs 2/(2+199) is ~0.000000246 (< 1e-6)
    class Client:
        def query_batch_points(self, **_):
            dense = [make_custom_point(f"d_{i}", 0.5) for i in range(198)] + [
                make_custom_point("cand_A", 0.1),
                make_custom_point("cand_B", 0.9),
            ]
            sparse = [make_custom_point(f"s_{i}", 5.0) for i in range(199)] + [
                make_custom_point("cand_B", 5.0),
                make_custom_point("cand_A", 5.0),
            ]
            return [SimpleNamespace(points=dense), SimpleNamespace(points=sparse)]

    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=Client(), candidate_pool=250)
    batch = provider.search("test", "hybrid", 250)
    order = [r.paper.id for r in batch.results]
    assert order.index("cand_A") < order.index("cand_B")


def test_qdrant_truly_equal_rrf_breaks_tie_alphabetically_by_paper_id():
    """Verify that truly equal RRF and channel scores break ties alphabetically by paper.id."""

    class Client:
        def query_batch_points(self, **_):
            return [
                SimpleNamespace(
                    points=[
                        make_custom_point("arxiv:2005.11401", 0.6),
                        make_custom_point("arxiv:1706.03762", 0.6),
                    ]
                ),
                SimpleNamespace(
                    points=[
                        make_custom_point("arxiv:1706.03762", 8.0),
                        make_custom_point("arxiv:2005.11401", 8.0),
                    ]
                ),
            ]

    provider = QdrantCloudRetrievalProvider(qdrant_config(), client=Client())
    batch = provider.search("test", "hybrid", 2)
    order = [r.paper.id for r in batch.results]
    assert order == ["arxiv:1706.03762", "arxiv:2005.11401"]


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


@pytest.mark.asyncio
async def test_groq_streaming_empty_choices_chunk_does_not_raise():
    def handler(_request: httpx.Request) -> httpx.Response:
        content = (
            'data: {"choices":[{"delta":{"content":"WordNet synsets "}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":10}}\n\n'
            'data: {"choices":[{"delta":{"content":"improve retrieval [1]."}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    provider = GroqGenerationProvider(groq_config(), async_transport=httpx.MockTransport(handler))
    tokens = [
        token
        async for token in provider.generate_stream_async([{"role": "user", "content": "Question"}])
    ]
    assert tokens == ["WordNet synsets ", "improve retrieval [1]."]


def test_groq_http_error_attaches_status_and_request_id():
    def limited(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={
                "retry-after": "3.5",
                "x-groq-request-id": "req-groq-429",
            },
        )

    client = httpx.Client(
        base_url="https://api.groq.test/openai/v1", transport=httpx.MockTransport(limited)
    )
    with pytest.raises(GenerationRateLimitedError) as raised:
        GroqGenerationProvider(groq_config(), client=client).generate([])
    assert raised.value.provider_http_status == 429
    assert raised.value.provider_request_id == "req-groq-429"
    assert raised.value.retry_after == 3.5


def test_groq_generate_empty_answer_raises_with_status_and_usage():
    def empty_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-groq-request-id": "req-empty-1"},
            json={
                "choices": [{"message": {"content": "   "}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 1024},
            },
        )

    client = httpx.Client(
        base_url="https://api.groq.test/openai/v1", transport=httpx.MockTransport(empty_handler)
    )
    with pytest.raises(GenerationInvalidResponseError) as raised:
        GroqGenerationProvider(groq_config(), client=client).generate([])
    assert raised.value.provider_http_status == 200
    assert raised.value.provider_request_id == "req-empty-1"
    assert raised.value.finish_reason == "length"
    assert raised.value.output_token_usage == {"prompt_tokens": 120, "completion_tokens": 1024}


@pytest.mark.asyncio
async def test_groq_streaming_interruption_records_tokens_emitted():
    def handler(_request: httpx.Request) -> httpx.Response:
        content = (
            'data: {"choices":[{"delta":{"content":"Partial token"}}]}\n\ndata: {invalid json\n\n'
        )
        return httpx.Response(
            200,
            text=content,
            headers={"content-type": "text/event-stream", "x-groq-request-id": "req-stream-err"},
        )

    provider = GroqGenerationProvider(groq_config(), async_transport=httpx.MockTransport(handler))
    generator = provider.generate_stream_async([{"role": "user", "content": "Question"}])
    token = await anext(generator)
    assert token == "Partial token"
    with pytest.raises(GenerationInvalidResponseError) as raised:
        await anext(generator)
    assert raised.value.tokens_emitted is True
    assert raised.value.provider_request_id == "req-stream-err"


@pytest.mark.asyncio
async def test_groq_streaming_missing_done_raises_invalid_stream():
    def handler(_request: httpx.Request) -> httpx.Response:
        content = 'data: {"choices":[{"delta":{"content":"Truncated stream"}}]}\n\n'
        return httpx.Response(
            200,
            text=content,
            headers={"content-type": "text/event-stream", "x-groq-request-id": "req-missing-done"},
        )

    provider = GroqGenerationProvider(groq_config(), async_transport=httpx.MockTransport(handler))
    generator = provider.generate_stream_async([{"role": "user", "content": "Question"}])
    token = await anext(generator)
    assert token == "Truncated stream"
    with pytest.raises(GenerationInvalidResponseError) as raised:
        await anext(generator)
    assert "without [DONE]" in str(raised.value)
    assert raised.value.tokens_emitted is True
    assert raised.value.provider_request_id == "req-missing-done"


@pytest.mark.asyncio
async def test_groq_streaming_finish_reason_length_raises():
    def handler(_request: httpx.Request) -> httpx.Response:
        content = (
            'data: {"choices":[{"delta":{"content":"Partial start "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":""},"finish_reason":"length"}]}\n\n'
        )
        return httpx.Response(
            200,
            text=content,
            headers={"content-type": "text/event-stream", "x-groq-request-id": "req-length-stream"},
        )

    provider = GroqGenerationProvider(groq_config(), async_transport=httpx.MockTransport(handler))
    generator = provider.generate_stream_async([{"role": "user", "content": "Question"}])
    token = await anext(generator)
    assert token == "Partial start "
    with pytest.raises(GenerationInvalidResponseError) as raised:
        await anext(generator)
    assert raised.value.finish_reason == "length"
    assert raised.value.tokens_emitted is True
    assert raised.value.provider_request_id == "req-length-stream"


def test_groq_generate_finish_reason_length_raises():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-groq-request-id": "req-length-gen"},
            json={
                "choices": [
                    {"message": {"content": "Incomplete answer due to"}, "finish_reason": "length"}
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 128},
            },
        )

    client = httpx.Client(
        base_url="https://api.groq.test/openai/v1", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(GenerationInvalidResponseError) as raised:
        GroqGenerationProvider(groq_config(), client=client).generate([])
    assert raised.value.finish_reason == "length"
    assert raised.value.provider_request_id == "req-length-gen"
    assert raised.value.output_token_usage == {"prompt_tokens": 80, "completion_tokens": 128}
