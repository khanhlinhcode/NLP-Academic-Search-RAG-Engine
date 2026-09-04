"""Tests for the Streamlit API client and SSE parser."""

import json

import httpx
import pytest

from nlp_academic_search.ui.api_client import (
    DEFAULT_API_REQUEST_TIMEOUT_SECONDS,
    AcademicSearchClient,
    APIError,
    _iter_sse,
    parse_request_timeout,
)


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


def test_iter_sse_supports_crlf_heartbeat_multiline_data_and_final_eof():
    lines = [
        ": keep-alive\r",
        "event: done\r",
        'data: {"answer":\r',
        'data: "Final answer [1].", "metadata": {"answer_status": "verified"}}\r',
    ]

    events = list(_iter_sse(lines))

    assert events == [
        {
            "event": "done",
            "data": {
                "answer": "Final answer [1].",
                "metadata": {"answer_status": "verified"},
            },
        }
    ]


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


def test_authentication_error_does_not_expose_bearer_token():
    secret = "backend-secret-that-must-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        return httpx.Response(401, json={"detail": "Authentication required"})

    client = AcademicSearchClient(
        "https://backend.test",
        api_token=secret,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(APIError) as caught:
        client.health()

    assert secret not in str(caught.value)
    assert "BACKEND_API_TOKEN" in str(caught.value)


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


def test_client_rejects_stream_that_ends_before_terminal_event():
    def handler(_: httpx.Request) -> httpx.Response:
        content = 'event: token\ndata: {"token": "Unverified draft"}\n\n'
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    client = AcademicSearchClient("http://test", transport=httpx.MockTransport(handler))

    with pytest.raises(APIError, match="before final validation completed"):
        list(client.stream_answer("What is RAG?"))


def test_client_accepts_error_as_terminal_event():
    def handler(_: httpx.Request) -> httpx.Response:
        content = 'event: error\ndata: {"message": "Generation failed."}\n\n'
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    client = AcademicSearchClient("http://test", transport=httpx.MockTransport(handler))

    events = list(client.stream_answer("What is RAG?"))

    assert events == [{"event": "error", "data": {"message": "Generation failed."}}]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("75", 75.0),
        (None, DEFAULT_API_REQUEST_TIMEOUT_SECONDS),
        ("not-a-number", DEFAULT_API_REQUEST_TIMEOUT_SECONDS),
        ("nan", DEFAULT_API_REQUEST_TIMEOUT_SECONDS),
        ("0", DEFAULT_API_REQUEST_TIMEOUT_SECONDS),
        ("601", DEFAULT_API_REQUEST_TIMEOUT_SECONDS),
    ],
)
def test_parse_request_timeout_is_bounded(configured, expected):
    assert parse_request_timeout(configured) == expected


def test_client_applies_configured_timeout():
    client = AcademicSearchClient("http://test", timeout=75)

    assert client._client.timeout.read == 75
    client.close()


def test_client_rejects_events_after_terminal_event():
    def handler(_: httpx.Request) -> httpx.Response:
        content = (
            'event: done\ndata: {"answer": "Final", "metadata": {}}\n\n'
            'event: done\ndata: {"answer": "Duplicate", "metadata": {}}\n\n'
        )
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    client = AcademicSearchClient("http://test", transport=httpx.MockTransport(handler))

    with pytest.raises(APIError, match="events after the answer stream completed"):
        list(client.stream_answer("What is RAG?"))
