"""Thin CLI for bounded Search/Ask load tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

from nlp_academic_search.config import settings
from nlp_academic_search.evaluation.experiment_config import ExperimentConfig
from nlp_academic_search.evaluation.load_test import EndpointType, run_load_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded HTTP load test.")
    parser.add_argument("--config", type=Path, help="Optional experiment TOML configuration")
    parser.add_argument("--api-url", help="API base URL")
    parser.add_argument("--endpoint", choices=["search", "ask"], default="search")
    parser.add_argument("--concurrency", type=int, help="Concurrent workers; overrides config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--requests", type=int, help="Fixed number of measured requests")
    mode.add_argument("--duration", type=float, help="Fixed measurement duration in seconds")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--timeout", type=float, help="Per-request timeout; overrides config")
    parser.add_argument("--allow-ask-load", action="store_true")
    parser.add_argument("--run-id", type=str, default="")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config) if args.config else None
    api_url = args.api_url or settings.api.base_url
    concurrency = (
        args.concurrency if args.concurrency is not None else config.concurrency if config else 2
    )
    timeout = (
        args.timeout if args.timeout is not None else config.timeout_seconds if config else 10.0
    )
    request_count = None if args.duration is not None else args.requests or 10
    endpoint = cast(EndpointType, args.endpoint)

    report = asyncio.run(
        run_load_test(
            api_url=api_url,
            endpoint_type=endpoint,
            concurrency=concurrency,
            request_count=request_count,
            duration_sec=args.duration,
            warmup_requests=args.warmup_requests,
            timeout_sec=timeout,
            allow_ask_load=args.allow_ask_load,
        )
    )
    report_payload: dict[str, Any] = dict(report)
    report_payload["effective_config"] = (
        config.effective()
        if config
        else {
            "api_url": api_url,
            "endpoint": endpoint,
            "concurrency": concurrency,
            "request_count": request_count,
            "duration_sec": args.duration,
            "warmup_requests": args.warmup_requests,
            "timeout_seconds": timeout,
            "config_path": None,
            "config_sha256": None,
        }
    )

    run_id = args.run_id or f"load_{int(time.time())}"
    out_dir = Path("reports/load") / run_id if args.run_id or config is None else config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "load_test.json"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    md_content = f"""# Load Test Report ({run_id})

- **Endpoint**: `{report["endpoint_type"]}`
- **Mode**: {report["mode"]}
- **Concurrency**: {report["concurrency"]}
- **Measured Requests**: {report["request_count"]}
- **Warm-up Requests**: {report["warmup_requests"]}
- **Elapsed**: {report["total_elapsed_sec"]} s
- **Throughput**: {metrics["throughput_req_per_sec"]} req/s

## Latency (successful HTTP 2xx requests only)

- **P50**: {metrics["latency_p50_ms"]} ms
- **P95**: {metrics["latency_p95_ms"]} ms
- **P99**: {metrics["latency_p99_ms"]} ms
- **Mean**: {metrics["latency_mean_ms"]} ms

## Reliability

- **Success rate**: {metrics["success_rate"]}
- **Error rate**: {metrics["error_rate"]}
- **Timeout rate**: {metrics["timeout_rate"]}
- **Status distribution**: `{json.dumps(metrics["status_code_distribution"], sort_keys=True)}`
- **Error distribution**: `{json.dumps(metrics["error_type_distribution"], sort_keys=True)}`
"""
    md_path = out_dir / "load_test.md"
    md_path.write_text(md_content, encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nReport JSON: {json_path}")
    print(f"Report MD:   {md_path}")


if __name__ == "__main__":
    main()
