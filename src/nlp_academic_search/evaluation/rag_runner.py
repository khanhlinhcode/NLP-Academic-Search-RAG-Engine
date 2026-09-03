"""Deterministic evaluation orchestration for a running RAG API."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import httpx
import numpy as np

from nlp_academic_search.config import settings
from nlp_academic_search.data.manifest import sha256_file
from nlp_academic_search.evaluation.rag_metrics import evaluate_rag_case
from nlp_academic_search.rag.prompt_builder import PROMPT_VERSION


class RAGAggregate(TypedDict):
    context_precision: float | None
    context_recall: float | None
    answer_relevance: float | None
    faithfulness_proxy: float | None
    citation_precision: float | None
    citation_coverage: float | None
    source_utilization: float | None
    claim_citation_coverage: float | None
    invalid_citation_rate: float | None
    refusal_correct: float | None
    semantic_claim_coverage: float | None
    unsupported_claim_count: float | None
    insufficient_claim_count: float | None
    evidence_quote_validity: float | None
    verified_answer_rate: float | None
    refused_verification_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    error_rate: float
    timeout_rate: float
    connection_error_rate: float
    http_error_rate: float
    answerable_case_count: int
    refusal_case_count: int
    successful_case_count: int
    failed_case_count: int


class RAGReport(TypedDict):
    status: str
    timestamp: str
    benchmark: str
    benchmark_version: str
    benchmark_sha256: str
    corpus_version: str | None
    index_version: str | None
    generator_model: str
    generator_revision: str
    judge_model: str
    retrieval_method: str
    top_k: int
    prompt_version: str
    aggregate: RAGAggregate
    cases: list[dict[str, Any]]
    limitations: list[str]


def _active_version(pointer: Path) -> str | None:
    try:
        return pointer.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(row["metrics"][name]) for row in rows]
    return round(statistics.mean(values), 4) if values else None


def run_rag_evaluation(
    api_url: str,
    benchmark_path: Path,
    *,
    transport: httpx.BaseTransport | None = None,
    client: httpx.Client | None = None,
    clock: Callable[[], float] = time.perf_counter,
    retrieval_method: str = "rrf",
    top_k: int = 5,
    api_token: str | None = None,
) -> RAGReport:
    """Evaluate RAG with explicit metric populations and sanitized errors.

    Context metrics cover successful answerable cases only. Refusal correctness
    covers successful cases marked ``should_refuse``. Citation and answer metrics
    cover all successful cases.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = benchmark.get("cases")
    if not isinstance(cases, list):
        raise ValueError("RAG benchmark must contain a cases list")

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    error_counts = {"timeout": 0, "connection": 0, "http": 0, "response": 0}
    if cases:
        created_client = client is None
        request_headers = {"Authorization": f"Bearer {api_token}"} if api_token else None
        provider_timeout = (
            settings.groq.timeout_seconds
            if settings.generation_provider == "groq"
            else settings.ollama.timeout_seconds
        )
        http_client = client or httpx.Client(
            base_url=api_url,
            transport=transport,
            timeout=provider_timeout,
            headers=request_headers,
        )
        manager = http_client if created_client else nullcontext(http_client)
        with manager as active_client:
            for case in cases:
                started = clock()
                try:
                    response = active_client.post(
                        "/api/v1/ask",
                        headers=request_headers,
                        json={
                            "question": case["question"],
                            "top_k": top_k,
                            "use_reranker": retrieval_method.endswith("reranker"),
                        },
                    )
                    response.raise_for_status()
                    row: dict[str, Any] = {
                        "case_id": case["id"],
                        "should_refuse": bool(case.get("should_refuse", False)),
                        "metrics": evaluate_rag_case(case, response.json()),
                    }
                except httpx.TimeoutException:
                    error_counts["timeout"] += 1
                    row = {"case_id": case["id"], "error": "TimeoutError"}
                except httpx.ConnectError:
                    error_counts["connection"] += 1
                    row = {"case_id": case["id"], "error": "ConnectError"}
                except httpx.HTTPStatusError as exc:
                    error_counts["http"] += 1
                    row = {
                        "case_id": case["id"],
                        "error": f"HTTPStatusError({exc.response.status_code})",
                    }
                except (httpx.HTTPError, ValueError, TypeError, KeyError):
                    error_counts["response"] += 1
                    row = {"case_id": case.get("id", "unknown"), "error": "InvalidResponse"}
                latency = max(0.0, (clock() - started) * 1000)
                latencies.append(latency)
                row["latency_ms"] = round(latency, 2)
                rows.append(row)

    successful = [row for row in rows if "metrics" in row]
    answerable = [row for row in successful if not row["should_refuse"]]
    refusal = [row for row in successful if row["should_refuse"]]
    total = len(cases)
    failed = total - len(successful)
    aggregate: RAGAggregate = {
        "context_precision": _mean(answerable, "context_precision"),
        "context_recall": _mean(answerable, "context_recall"),
        "answer_relevance": _mean(successful, "answer_relevance"),
        "faithfulness_proxy": _mean(successful, "faithfulness_proxy"),
        "citation_precision": _mean(successful, "citation_precision"),
        "citation_coverage": _mean(successful, "citation_coverage"),
        "source_utilization": _mean(successful, "source_utilization"),
        "claim_citation_coverage": _mean(successful, "claim_citation_coverage"),
        "invalid_citation_rate": _mean(successful, "invalid_citation_rate"),
        "refusal_correct": _mean(refusal, "refusal_correct"),
        "semantic_claim_coverage": _mean(successful, "semantic_claim_coverage"),
        "unsupported_claim_count": _mean(successful, "unsupported_claim_count"),
        "insufficient_claim_count": _mean(successful, "insufficient_claim_count"),
        "evidence_quote_validity": _mean(successful, "evidence_quote_validity"),
        "verified_answer_rate": _mean(successful, "verified_answer"),
        "refused_verification_rate": _mean(successful, "refused_verification"),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "error_rate": round(failed / total, 4) if total else 0.0,
        "timeout_rate": round(error_counts["timeout"] / total, 4) if total else 0.0,
        "connection_error_rate": round(error_counts["connection"] / total, 4) if total else 0.0,
        "http_error_rate": round(error_counts["http"] / total, 4) if total else 0.0,
        "answerable_case_count": len(answerable),
        "refusal_case_count": len(refusal),
        "successful_case_count": len(successful),
        "failed_case_count": failed,
    }
    return {
        "status": "evaluated" if cases else "empty_benchmark",
        "timestamp": datetime.now(UTC).isoformat(),
        "benchmark": str(benchmark.get("name", "unknown")),
        "benchmark_version": str(benchmark.get("version", "unversioned")),
        "benchmark_sha256": sha256_file(benchmark_path),
        "corpus_version": _active_version(settings.data.raw_dir / "CURRENT"),
        "index_version": _active_version(settings.data.embeddings_dir / "CURRENT"),
        "generator_model": settings.active_generation_model,
        "generator_revision": (
            "provider-managed" if settings.generation_provider == "groq" else "local-tag-unpinned"
        ),
        "judge_model": "none (deterministic metrics only)",
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "prompt_version": PROMPT_VERSION,
        "aggregate": aggregate,
        "cases": rows,
        "limitations": [
            "faithfulness_proxy is deprecated and checks citation structure, not semantic entailment",
            "No external LLM judge was run; generator tag revision is not content-addressed",
        ],
    }
