"""Thin CLI for deterministic RAG API evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from nlp_academic_search.config import settings
from nlp_academic_search.evaluation.experiment_config import ExperimentConfig
from nlp_academic_search.evaluation.rag_runner import run_rag_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation against FastAPI.")
    parser.add_argument("--config", type=Path, help="Optional experiment TOML configuration")
    parser.add_argument("--api-url", help="API base URL; overrides application settings")
    parser.add_argument("--benchmark", type=Path, help="RAG benchmark JSON; overrides config")
    parser.add_argument("--top-k", type=int, help="Retrieved source count; overrides config")
    parser.add_argument("--run-id", type=str, default="", help="Experiment run identifier")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config) if args.config else None
    api_url = args.api_url or settings.api.base_url
    benchmark = args.benchmark or (config.corpus_path if config else None)
    benchmark = benchmark or Path("benchmarks/rag/rag_golden.json")
    top_k = args.top_k if args.top_k is not None else config.top_k if config else 5
    retrieval_method = config.retrieval_method if config else "rrf"

    report = run_rag_evaluation(
        api_url,
        benchmark,
        retrieval_method=retrieval_method,
        top_k=top_k,
    )
    if report["aggregate"]["connection_error_rate"] == 1.0:
        report["status"] = "not_evaluated"
        report["limitations"].append(f"API unavailable at {api_url}")
    report_payload: dict[str, Any] = dict(report)
    report_payload["effective_config"] = (
        config.effective()
        if config
        else {
            "api_url": api_url,
            "benchmark": str(benchmark.resolve()),
            "top_k": top_k,
            "retrieval_method": retrieval_method,
            "config_path": None,
            "config_sha256": None,
        }
    )

    run_id = args.run_id or f"run_{int(time.time())}"
    output_dir = (
        Path("reports/experiments") / run_id if args.run_id or config is None else config.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rag.json"
    json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_content = (
        f"""# RAG Evaluation Report ({run_id})

- **Status**: {report["status"]}
- **Benchmark**: {report["benchmark"]}
- **Benchmark SHA-256**: `{report["benchmark_sha256"]}`
- **Generator Model**: {report["generator_model"]} ({report["generator_revision"]})
- **Judge Model**: {report["judge_model"]}
- **Prompt Version**: {report["prompt_version"]}
- **Timestamp**: {report["timestamp"]}

## Aggregate Metrics

```json
{json.dumps(report["aggregate"], indent=2)}
```

## Case Summary

Total cases: {len(report["cases"])}

## Limitations
"""
        + "\n".join(f"- {lim}" for lim in report["limitations"])
        + "\n"
    )
    md_path = output_dir / "rag.md"
    md_path.write_text(md_content, encoding="utf-8")

    print(json.dumps(report["aggregate"], indent=2))
    print(f"\nReport JSON: {json_path}")
    print(f"Report MD:   {md_path}")


if __name__ == "__main__":
    main()
