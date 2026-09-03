"""Bounded asynchronous HTTP load testing with explicit measurement policy."""

from __future__ import annotations

import asyncio
import platform
import statistics
import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import httpx
import numpy as np

EndpointType = Literal["search", "ask"]


class LoadMetricsReport(TypedDict):
    total_requests: int
    successful_requests: int
    failed_requests: int
    timed_out_requests: int
    success_rate: float
    error_rate: float
    timeout_rate: float
    throughput_req_per_sec: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    latency_mean_ms: float | None
    latency_population: str
    status_code_distribution: dict[str, int]
    error_type_distribution: dict[str, int]


class LoadTestReport(TypedDict):
    endpoint_type: EndpointType
    mode: str
    concurrency: int
    request_count: int
    requested_duration_sec: float | None
    warmup_requests: int
    timeout_sec: float
    total_elapsed_sec: float
    environment: dict[str, str]
    metrics: LoadMetricsReport


@dataclass
class LoadTestMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timed_out_requests: int = 0
    successful_durations_ms: list[float] = field(default_factory=list)
    status_codes: dict[str, int] = field(default_factory=dict)
    error_types: dict[str, int] = field(default_factory=dict)

    def add_result(self, duration_ms: float, status_code: int | None, error: str | None) -> None:
        self.total_requests += 1
        if status_code is not None:
            key = str(status_code)
            self.status_codes[key] = self.status_codes.get(key, 0) + 1
        if error is not None:
            self.error_types[error] = self.error_types.get(error, 0) + 1
            self.failed_requests += 1
            if error == "timeout":
                self.timed_out_requests += 1
        elif status_code is not None and 200 <= status_code < 300:
            self.successful_requests += 1
            self.successful_durations_ms.append(duration_ms)
        else:
            self.failed_requests += 1
            error_key = "http_status"
            self.error_types[error_key] = self.error_types.get(error_key, 0) + 1

    def summary(self, total_elapsed_sec: float) -> LoadMetricsReport:
        durations = self.successful_durations_ms
        total = self.total_requests
        return {
            "total_requests": total,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "timed_out_requests": self.timed_out_requests,
            "success_rate": round(self.successful_requests / total, 4) if total else 0.0,
            "error_rate": round(self.failed_requests / total, 4) if total else 0.0,
            "timeout_rate": round(self.timed_out_requests / total, 4) if total else 0.0,
            "throughput_req_per_sec": round(total / total_elapsed_sec, 2)
            if total_elapsed_sec > 0
            else 0.0,
            "latency_p50_ms": round(float(np.percentile(durations, 50)), 2) if durations else None,
            "latency_p95_ms": round(float(np.percentile(durations, 95)), 2) if durations else None,
            "latency_p99_ms": round(float(np.percentile(durations, 99)), 2) if durations else None,
            "latency_mean_ms": round(statistics.mean(durations), 2) if durations else None,
            "latency_population": "successful HTTP 2xx requests only",
            "status_code_distribution": self.status_codes,
            "error_type_distribution": self.error_types,
        }


async def _request(
    client: httpx.AsyncClient,
    endpoint: str,
    payload_or_params: dict[str, str | int],
    method: str,
    timeout_sec: float,
) -> tuple[float, int | None, str | None]:
    started = time.perf_counter()
    status_code: int | None = None
    error_name: str | None = None
    try:
        if method == "POST":
            response = await client.post(endpoint, json=payload_or_params, timeout=timeout_sec)
        else:
            response = await client.get(endpoint, params=payload_or_params, timeout=timeout_sec)
        status_code = response.status_code
    except httpx.TimeoutException:
        error_name = "timeout"
    except httpx.ConnectError:
        error_name = "connection_error"
    except httpx.HTTPError:
        error_name = "http_error"
    return (time.perf_counter() - started) * 1000, status_code, error_name


async def run_load_test(
    api_url: str,
    endpoint_type: EndpointType = "search",
    concurrency: int = 2,
    request_count: int | None = 10,
    timeout_sec: float = 10.0,
    *,
    duration_sec: float | None = None,
    warmup_requests: int = 0,
    transport: httpx.AsyncBaseTransport | None = None,
    allow_ask_load: bool = False,
) -> LoadTestReport:
    """Run either a fixed-count or fixed-duration test after excluded warm-up calls."""
    if endpoint_type not in ("search", "ask"):
        raise ValueError("endpoint_type must be 'search' or 'ask'")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if warmup_requests < 0:
        raise ValueError("warmup_requests must not be negative")
    if duration_sec is not None and duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if duration_sec is not None and request_count is not None:
        raise ValueError("choose fixed request_count or duration_sec, not both")
    if duration_sec is None and (request_count is None or request_count <= 0):
        raise ValueError("request_count must be positive in fixed-count mode")
    if endpoint_type == "ask" and not allow_ask_load:
        raise ValueError("Load testing the Ask endpoint requires explicit --allow-ask-load flag")

    path = "/api/v1/search" if endpoint_type == "search" else "/api/v1/ask"
    method = "GET" if endpoint_type == "search" else "POST"
    payload: dict[str, str | int] = (
        {"q": "transformer attention mechanism", "top_k": 5}
        if endpoint_type == "search"
        else {"question": "How does attention work?", "top_k": 3}
    )
    metrics = LoadTestMetrics()
    semaphore = asyncio.Semaphore(concurrency)

    async def measured_request(client: httpx.AsyncClient) -> None:
        async with semaphore:
            duration, status, error = await _request(client, path, payload, method, timeout_sec)
            metrics.add_result(duration, status, error)

    async with httpx.AsyncClient(
        base_url=api_url, transport=transport, timeout=timeout_sec
    ) as client:
        for _ in range(warmup_requests):
            await _request(client, path, payload, method, timeout_sec)

        started_total = time.perf_counter()
        if duration_sec is None:
            assert request_count is not None
            await asyncio.gather(*(measured_request(client) for _ in range(request_count)))
        else:
            deadline = started_total + duration_sec

            async def duration_worker() -> None:
                while time.perf_counter() < deadline:
                    await measured_request(client)

            await asyncio.gather(*(duration_worker() for _ in range(concurrency)))
        total_elapsed = time.perf_counter() - started_total

    return {
        "endpoint_type": endpoint_type,
        "mode": "fixed-duration" if duration_sec is not None else "fixed-count",
        "concurrency": concurrency,
        "request_count": metrics.total_requests,
        "requested_duration_sec": duration_sec,
        "warmup_requests": warmup_requests,
        "timeout_sec": timeout_sec,
        "total_elapsed_sec": round(total_elapsed, 4),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "metrics": metrics.summary(total_elapsed),
    }
