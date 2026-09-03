"""Unit tests for the bounded load-test engine."""

import httpx
import pytest

from nlp_academic_search.evaluation.load_test import run_load_test


@pytest.mark.asyncio
async def test_load_test_search_endpoint_success():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"results": []}))
    report = await run_load_test(
        api_url="http://test-load",
        endpoint_type="search",
        concurrency=2,
        request_count=5,
        warmup_requests=1,
        transport=transport,
    )
    assert report["mode"] == "fixed-count"
    assert report["request_count"] == 5
    assert report["warmup_requests"] == 1
    metrics = report["metrics"]
    assert metrics["successful_requests"] == 5
    assert metrics["failed_requests"] == 0
    assert metrics["status_code_distribution"]["200"] == 5
    assert metrics["latency_population"] == "successful HTTP 2xx requests only"


@pytest.mark.asyncio
async def test_load_test_ask_endpoint_requires_guard():
    with pytest.raises(ValueError, match="requires explicit"):
        await run_load_test(api_url="http://test-load", endpoint_type="ask")


@pytest.mark.asyncio
async def test_load_test_ask_endpoint_with_flag():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"answer": "ok"}))
    report = await run_load_test(
        api_url="http://test-load",
        endpoint_type="ask",
        concurrency=1,
        request_count=2,
        transport=transport,
        allow_ask_load=True,
    )
    assert report["metrics"]["successful_requests"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"concurrency": 0}, "concurrency"),
        ({"request_count": 0}, "request_count"),
        ({"timeout_sec": 0}, "timeout_sec"),
        ({"request_count": None, "duration_sec": 0}, "duration_sec"),
        ({"warmup_requests": -1}, "warmup_requests"),
        ({"endpoint_type": "unknown"}, "endpoint_type"),
    ],
)
async def test_load_test_rejects_invalid_parameters(kwargs: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        await run_load_test(api_url="http://test-load", **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_timeout_and_http_status_distributions():
    calls = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(503)

    report = await run_load_test(
        api_url="http://test-load",
        request_count=2,
        concurrency=1,
        transport=httpx.MockTransport(handle_request),
    )
    metrics = report["metrics"]
    assert metrics["timeout_rate"] == 0.5
    assert metrics["error_rate"] == 1.0
    assert metrics["error_type_distribution"] == {"timeout": 1, "http_status": 1}
    assert metrics["latency_p50_ms"] is None


@pytest.mark.asyncio
async def test_fixed_duration_mode_runs_requests():
    transport = httpx.MockTransport(lambda _: httpx.Response(200))
    report = await run_load_test(
        api_url="http://test-load",
        request_count=None,
        duration_sec=0.002,
        concurrency=1,
        transport=transport,
    )
    assert report["mode"] == "fixed-duration"
    assert report["request_count"] > 0
