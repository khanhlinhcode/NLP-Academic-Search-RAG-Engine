"""Tests for the Streamlit API client and SSE parser."""

import json

import httpx

from nlp_academic_search.ui.api_client import AcademicSearchClient, _iter_sse


def test_iter_sse_decodes_structured_events():
    lines = [
        "event: sources",
        'data: {"sources": [{"index": 1, "title": "Paper"}]}',
        "",
        "event: token",
        'data: {"token": "Hello"}',
        "",
        "event: answer_replacement",
        'data: {"answer": "Hello [1]."}',
        "",
        "event: done",
        'data: {"latency_ms": 12.4}',
        "",
    ]

    events = list(_iter_sse(lines))

    assert [event["event"] for event in events] == [
        "sources",
        "token",
        "answer_replacement",
        "done",
    ]
    assert events[0]["data"]["sources"][0]["index"] == 1
    assert events[1]["data"]["token"] == "Hello"
    assert events[2]["data"]["answer"] == "Hello [1]."
    assert events[3]["data"]["latency_ms"] == 12.4


def test_client_uses_method_specific_search_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "query": "rag",
                "method": "semantic",
                "total_results": 0,
                "results": [],
                "latency_ms": 1,
            },
        )

    client = AcademicSearchClient("http://test", transport=httpx.MockTransport(handler))
    response = client.search("rag", method="semantic", top_k=5)

    assert captured == {"path": "/search/semantic", "query": {"q": "rag", "top_k": "5"}}
    assert response["method"] == "semantic"


def test_client_sends_backend_bearer_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer backend-secret"
        return httpx.Response(200, json={"status": "ready"})

    client = AcademicSearchClient(
        "https://backend.test",
        api_token="backend-secret",
        transport=httpx.MockTransport(handler),
    )
    assert client.health()["status"] == "ready"


def test_client_stream_answer_parses_api_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {"question": "What is RAG?", "top_k": 3, "use_reranker": False}
        content = (
            'event: sources\ndata: {"sources": [], "retrieval_method": "hybrid"}\n\n'
            'event: token\ndata: {"token": "Grounded"}\n\n'
            'event: done\ndata: {"latency_ms": 9.8}\n\n'
        )
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    client = AcademicSearchClient("http://test", transport=httpx.MockTransport(handler))
    events = list(client.stream_answer("What is RAG?", top_k=3, use_reranker=False))

    assert [event["event"] for event in events] == ["sources", "token", "done"]
    assert events[1]["data"]["token"] == "Grounded"
