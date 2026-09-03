"""Unit tests for RAG runner and evaluation orchestration using mock HTTP transport."""

import json
from pathlib import Path

import httpx
import pytest

from nlp_academic_search.evaluation.rag_runner import run_rag_evaluation


@pytest.fixture
def sample_benchmark_path(tmp_path: Path) -> Path:
    bench_data = {
        "name": "unit-test-rag-benchmark",
        "cases": [
            {
                "id": "c1",
                "question": "What is attention mechanism?",
                "relevant_source_ids": ["p1", "p2"],
                "expected_keywords": ["attention", "transformer"],
                "should_refuse": False,
            },
            {
                "id": "c2",
                "question": "What is quantum computing in deep learning?",
                "relevant_source_ids": ["p99"],
                "expected_keywords": ["quantum"],
                "should_refuse": True,
            },
        ],
    }
    path = tmp_path / "rag_benchmark.json"
    path.write_text(json.dumps(bench_data), encoding="utf-8")
    return path


def test_rag_runner_successful_response(sample_benchmark_path: Path):
    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        question = body["question"]
        if "attention" in question:
            content = {
                "answer": "Attention mechanism replaces recurrence [1][2].",
                "sources": [
                    {"id": "p1", "title": "Attention Paper 1"},
                    {"id": "p2", "title": "Attention Paper 2"},
                ],
            }
        else:
            content = {
                "answer": "Not enough evidence in the retrieved context.",
                "sources": [],
            }
        return httpx.Response(200, json=content)

    transport = httpx.MockTransport(handle_request)
    result = run_rag_evaluation("http://test-api", sample_benchmark_path, transport=transport)

    assert result["status"] == "evaluated"
    assert result["benchmark"] == "unit-test-rag-benchmark"
    assert result["generator_model"]
    assert "cases" in result
    cases = result["cases"]
    assert len(cases) == 2
    assert cases[0]["case_id"] == "c1"
    assert cases[1]["case_id"] == "c2"

    agg = result["aggregate"]
    assert agg["error_rate"] == 0.0
    assert agg["context_precision"] == 1.0
    assert agg["context_recall"] == 1.0
    assert agg["refusal_correct"] == 1.0
    assert agg["answerable_case_count"] == 1
    assert agg["refusal_case_count"] == 1
    assert agg["successful_case_count"] == 2
    assert agg["latency_p50_ms"] is not None
    assert agg["latency_p95_ms"] is not None


def test_rag_runner_multiple_cases_order_and_percentiles(tmp_path: Path):
    bench_data = {
        "name": "multi-case-benchmark",
        "cases": [
            {
                "id": f"case_{i}",
                "question": f"Question {i}",
                "relevant_source_ids": [f"p{i}"],
                "expected_keywords": ["word"],
            }
            for i in range(5)
        ],
    }
    path = tmp_path / "multi_bench.json"
    path.write_text(json.dumps(bench_data), encoding="utf-8")

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        q_idx = body["question"].split()[-1]
        return httpx.Response(
            200,
            json={
                "answer": f"Word answer {q_idx} [1].",
                "sources": [{"id": f"p{q_idx}", "title": f"Paper {q_idx}"}],
            },
        )

    transport = httpx.MockTransport(handle_request)
    result = run_rag_evaluation("http://test-api", path, transport=transport)

    assert [c["case_id"] for c in result["cases"]] == [f"case_{i}" for i in range(5)]
    assert result["aggregate"]["context_precision"] == 1.0
    assert result["aggregate"]["context_recall"] == 1.0


def test_rag_runner_http_error_handling(sample_benchmark_path: Path):
    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "quantum" in body["question"]:
            return httpx.Response(500, json={"error": "Internal Server Error"})
        return httpx.Response(
            200,
            json={
                "answer": "Attention paper [1].",
                "sources": [{"id": "p1", "title": "Paper 1"}],
            },
        )

    transport = httpx.MockTransport(handle_request)
    result = run_rag_evaluation("http://test-api", sample_benchmark_path, transport=transport)

    assert result["aggregate"]["error_rate"] == 0.5
    assert result["aggregate"]["http_error_rate"] == 0.5
    assert result["cases"][0]["case_id"] == "c1"
    assert "metrics" in result["cases"][0]
    assert result["cases"][1]["case_id"] == "c2"
    assert result["cases"][1]["error"] == "HTTPStatusError(500)"


def test_rag_runner_timeout_and_connection_errors(sample_benchmark_path: Path):
    count = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            raise httpx.TimeoutException("Connection timed out", request=request)
        raise httpx.ConnectError("Connection refused", request=request)

    transport = httpx.MockTransport(handle_request)
    result = run_rag_evaluation("http://test-api", sample_benchmark_path, transport=transport)

    assert result["aggregate"]["error_rate"] == 1.0
    assert result["aggregate"]["timeout_rate"] == 0.5
    assert result["aggregate"]["connection_error_rate"] == 0.5
    assert result["cases"][0]["error"] == "TimeoutError"
    assert result["cases"][1]["error"] == "ConnectError"
    # Ensure raw exception object is not serialized
    assert isinstance(result["cases"][0]["error"], str)


def test_rag_runner_invalid_json(sample_benchmark_path: Path):
    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Not a valid JSON response {{{")

    transport = httpx.MockTransport(handle_request)
    result = run_rag_evaluation("http://test-api", sample_benchmark_path, transport=transport)

    assert result["aggregate"]["error_rate"] == 1.0
    assert result["cases"][0]["error"] == "InvalidResponse"


def test_rag_runner_empty_benchmark(tmp_path: Path):
    bench_data = {"name": "empty-benchmark", "cases": []}
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(bench_data), encoding="utf-8")

    result = run_rag_evaluation("http://test-api", path)

    assert result["status"] == "empty_benchmark"
    assert result["aggregate"]["error_rate"] == 0.0
    assert result["aggregate"]["latency_p50_ms"] is None
    assert result["aggregate"]["successful_case_count"] == 0


def test_rag_runner_uses_injected_clock(sample_benchmark_path: Path):
    ticks = iter([0.0, 0.010, 0.010, 0.030])

    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"answer": "Supported attention claim [1].", "sources": [{"id": "p1"}]},
        )

    result = run_rag_evaluation(
        "http://test-api",
        sample_benchmark_path,
        transport=httpx.MockTransport(handle_request),
        clock=lambda: next(ticks),
    )
    assert [row["latency_ms"] for row in result["cases"]] == [10.0, 20.0]


def test_rag_runner_aggregates_semantic_pilot_metrics(tmp_path: Path):
    path = tmp_path / "semantic.json"
    path.write_text(
        json.dumps(
            {
                "name": "semantic-pilot",
                "cases": [
                    {"id": "verified", "question": "Question one", "should_refuse": False},
                    {"id": "withheld", "question": "Question two", "should_refuse": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        question = json.loads(request.content)["question"]
        verified = question.endswith("one")
        return httpx.Response(
            200,
            json={
                "answer": (
                    "A supported answer [1]."
                    if verified
                    else "Not enough verified evidence in the retrieved sources."
                ),
                "sources": [{"id": "p1"}] if verified else [],
                "metadata": {
                    "answer_status": "verified" if verified else "refused_unverified",
                    "semantic_verification_attempted": True,
                    "semantic_verification_succeeded": verified,
                    "verification_latency_ms": 10 if verified else 30,
                    "citation_repair_attempted": not verified,
                    "citation_repair_succeeded": False,
                    "semantic_validation": {
                        "semantic_claim_coverage": 1.0 if verified else 0.0,
                        "supported_claim_count": 1 if verified else 0,
                        "unsupported_claim_count": 0 if verified else 1,
                        "insufficient_claim_count": 0,
                        "evidence_quote_validity": 1.0 if verified else 0.0,
                    },
                },
            },
        )

    report = run_rag_evaluation("http://test-api", path, transport=httpx.MockTransport(handler))
    aggregate = report["aggregate"]
    assert aggregate["verified_answer_rate"] == 0.5
    assert aggregate["refusal_due_to_verification_rate"] == 0.5
    assert aggregate["semantic_verification_latency_p50_ms"] == 20.0
    assert aggregate["semantic_verification_latency_p95_ms"] == 29.0
    assert aggregate["repair_rate"] == 0.5
    assert aggregate["repair_success_rate"] == 0.0


def test_rag_runner_refusal_cases(tmp_path: Path):
    bench_data = {
        "name": "refusal-test",
        "cases": [
            {
                "id": "should_refuse_correct",
                "question": "Unanswerable Q",
                "relevant_source_ids": [],
                "should_refuse": True,
            },
            {
                "id": "should_refuse_fabricated",
                "question": "Unanswerable Q2",
                "relevant_source_ids": [],
                "should_refuse": True,
            },
        ],
    }
    path = tmp_path / "refusal.json"
    path.write_text(json.dumps(bench_data), encoding="utf-8")

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["question"] == "Unanswerable Q":
            return httpx.Response(
                200,
                json={"answer": "Not enough evidence to answer this.", "sources": []},
            )
        # Fabricated answer
        return httpx.Response(
            200,
            json={"answer": "The answer is 42 without evidence.", "sources": []},
        )

    transport = httpx.MockTransport(handle_request)
    result = run_rag_evaluation("http://test-api", path, transport=transport)

    c1_metrics = result["cases"][0]["metrics"]
    c2_metrics = result["cases"][1]["metrics"]
    assert c1_metrics["refusal_correct"] is True
    assert c2_metrics["refusal_correct"] is False


def test_rag_runner_citation_cases(tmp_path: Path):
    bench_data = {
        "name": "citation-test",
        "cases": [
            {
                "id": "valid_citation",
                "question": "Q1",
                "relevant_source_ids": ["p1"],
            },
            {
                "id": "invalid_citation_index",
                "question": "Q2",
                "relevant_source_ids": ["p1"],
            },
            {
                "id": "no_source",
                "question": "Q3",
                "relevant_source_ids": [],
            },
        ],
    }
    path = tmp_path / "citations.json"
    path.write_text(json.dumps(bench_data), encoding="utf-8")

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        q = body["question"]
        if q == "Q1":
            return httpx.Response(
                200,
                json={
                    "answer": "Claim supported by source [1].",
                    "sources": [{"id": "p1", "title": "P1"}],
                },
            )
        if q == "Q2":
            return httpx.Response(
                200,
                json={
                    "answer": "Claim supported by missing source [99].",
                    "sources": [{"id": "p1", "title": "P1"}],
                },
            )
        return httpx.Response(
            200,
            json={
                "answer": "Claim without any source list.",
                "sources": [],
            },
        )

    transport = httpx.MockTransport(handle_request)
    result = run_rag_evaluation("http://test-api", path, transport=transport)

    m1 = result["cases"][0]["metrics"]
    m2 = result["cases"][1]["metrics"]
    m3 = result["cases"][2]["metrics"]

    assert m1["citation_precision"] == 1.0
    assert m1["invalid_citation_rate"] == 0.0
    assert m2["invalid_citation_rate"] > 0.0
    assert m3["citation_coverage"] == 0.0
