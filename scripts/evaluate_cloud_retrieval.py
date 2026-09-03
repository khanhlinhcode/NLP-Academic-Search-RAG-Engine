"""Evaluate the configured Qdrant collection against labelled qrels."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from nlp_academic_search.config import settings
from nlp_academic_search.evaluation.metrics import evaluate_search_method, format_evaluation_table
from nlp_academic_search.providers.retrieval.qdrant_cloud import (
    QdrantCloudRetrievalProvider,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("benchmarks/retrieval/in_domain_golden.json")
    )
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    if args.k < 1:
        raise SystemExit("-k must be positive")
    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise SystemExit("Benchmark must contain a queries list")

    provider = QdrantCloudRetrievalProvider(
        settings.qdrant,
        allow_degraded=False,
        candidate_pool=settings.search.candidate_pool,
    )
    try:
        status = provider.status()
        if not status.ready:
            raise SystemExit(f"Qdrant collection is not ready: {status.reason}")
        results = {
            "Qdrant-BM25": evaluate_search_method(
                lambda query, top_k: provider.search(query, "bm25", top_k).results,
                queries,
                k=args.k,
            ),
            "Qdrant-Dense": evaluate_search_method(
                lambda query, top_k: provider.search(query, "semantic", top_k).results,
                queries,
                k=args.k,
            ),
            "Qdrant-RRF": evaluate_search_method(
                lambda query, top_k: provider.search(query, "hybrid", top_k).results,
                queries,
                k=args.k,
            ),
        }
    finally:
        provider.close()

    run_id = args.run_id or f"cloud_{int(time.time())}"
    output_dir = Path("reports/experiments") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "benchmark": str(args.benchmark),
        "benchmark_name": payload.get("name", "unknown"),
        "provider": "qdrant",
        "collection": settings.qdrant.collection_alias,
        "corpus_sha256": settings.qdrant.expected_corpus_sha256,
        "dense_model": settings.qdrant.dense_model,
        "sparse_model": settings.qdrant.sparse_model,
        "candidate_pool": settings.search.candidate_pool,
        "k": args.k,
        "results": results,
        "limitations": [
            "Qdrant BM25 is not assumed to be identical to the local rank-bm25 implementation.",
            "Results are valid only for the recorded collection, corpus checksum, and model IDs.",
        ],
    }
    destination = output_dir / "cloud_retrieval.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_evaluation_table(results))
    print(f"\nReport JSON: {destination}")


if __name__ == "__main__":
    main()
