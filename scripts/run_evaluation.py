"""Thin CLI for reproducible retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from nlp_academic_search.evaluation.experiment_config import ExperimentConfig
from nlp_academic_search.evaluation.metrics import MetricSummary, format_evaluation_table
from nlp_academic_search.evaluation.retrieval_runner import run_retrieval_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free retrieval evaluation benchmark.")
    parser.add_argument("--config", type=Path, help="Optional experiment TOML configuration")
    parser.add_argument("--benchmark", type=Path, help="Benchmark JSON; overrides config")
    parser.add_argument("-k", type=int, help="Evaluation top-K; overrides config")
    parser.add_argument(
        "--include-reranker",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include Cross-Encoder reranker; overrides config",
    )
    parser.add_argument("--run-id", type=str, default="", help="Experiment run identifier")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config) if args.config else None
    benchmark = args.benchmark or (config.corpus_path if config else None)
    benchmark = benchmark or Path("benchmarks/retrieval/in_domain_golden.json")
    k = args.k if args.k is not None else config.top_k if config else 3
    configured_reranker = bool(config and config.retrieval_method == "hybrid_reranked")
    include_reranker = (
        args.include_reranker if args.include_reranker is not None else configured_reranker
    )

    report = run_retrieval_evaluation(
        benchmark,
        k,
        include_reranker=include_reranker,
        bm25_weight=config.bm25_weight if config else None,
        candidate_pool=config.candidate_pool if config else None,
        rrf_k=config.rrf_k if config else 60,
        reranker_model=config.reranker_model if config else None,
    )
    report_payload: dict[str, Any] = dict(report)
    report_payload["effective_config"] = (
        config.effective()
        if config
        else {
            "benchmark": str(benchmark.resolve()),
            "top_k": k,
            "include_reranker": include_reranker,
            "config_path": None,
            "config_sha256": None,
        }
    )

    run_id = args.run_id or f"run_{int(time.time())}"
    output_dir = (
        Path("reports/experiments") / run_id if args.run_id or config is None else config.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval.json"
    json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    results: dict[str, MetricSummary] = report["results"]
    table_md = format_evaluation_table(results)
    bench_info = report["benchmark"]
    run_info = report["run"]
    md_content = (
        f"""# Retrieval Evaluation Report ({run_id})

- **Benchmark Name**: {bench_info["name"]}
- **Benchmark SHA-256**: `{bench_info["sha256"]}`
- **Provenance**: {bench_info["provenance"]}
- **Document Count**: {bench_info["document_count"]}
- **Query Count**: {bench_info["query_count"]}
- **Embedding Model**: {run_info["embedding_model"]} ({run_info["embedding_revision"]})
- **Reranker Model**: {run_info["reranker_model"]}
- **Evaluation Top-K**: {run_info["k"]}
- **Timestamp**: {run_info["timestamp"]}

## Results Table

{table_md}

## Limitations
"""
        + "\n".join(f"- {lim}" for lim in report["limitations"])
        + "\n"
    )
    md_path = output_dir / "retrieval.md"
    md_path.write_text(md_content, encoding="utf-8")

    print(table_md)
    print(f"\nReport JSON: {json_path}")
    print(f"Report MD:   {md_path}")


if __name__ == "__main__":
    main()
