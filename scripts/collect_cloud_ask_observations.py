"""Collect final synchronous cloud Ask responses into a private review CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx

from nlp_academic_search.evaluation.cloud_observations import collect_cloud_observations

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=ROOT / "benchmarks/rag/cloud_candidates_v1.json"
    )
    parser.add_argument(
        "--template", type=Path, default=ROOT / "benchmarks/rag/cloud_review_sheet_v1.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports/experiments/cloud_holdout_v1"
    )
    parser.add_argument("--api-base-url", default="https://nlp-academic-search-api.onrender.com")
    parser.add_argument("--app-commit", required=True)
    parser.add_argument("--limit", type=int, required=True, help="Maximum new requests, 1..100")
    parser.add_argument("--delay-seconds", type=float, default=6.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    token = os.environ.get("BACKEND_API_TOKEN")
    if not token:
        parser.error("BACKEND_API_TOKEN must be set in the environment")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to((ROOT / "reports/experiments").resolve()):
        parser.error("output directory must be under Git-ignored reports/experiments")
    parsed = urlparse(args.api_base_url)
    production = (
        parsed.scheme == "https" and parsed.hostname == "nlp-academic-search-api.onrender.com"
    )
    local = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
    if (
        not (production or local)
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        parser.error("api-base-url must be the known Render API or local loopback")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be finite and positive")
    candidate_bytes = args.candidates.read_bytes()
    candidates = json.loads(candidate_bytes)
    with httpx.Client(
        base_url=args.api_base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=args.timeout_seconds,
        follow_redirects=False,
    ) as client:
        try:
            summary = collect_cloud_observations(
                candidates=candidates,
                candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
                template_path=args.template,
                sheet_path=output_dir / "review_sheet.csv",
                manifest_path=output_dir / "ask_run_manifest.json",
                api_base_url=args.api_base_url,
                app_commit=args.app_commit,
                limit=args.limit,
                delay_seconds=args.delay_seconds,
                client=client,
            )
        except (ValueError, httpx.HTTPError) as exc:
            parser.error(str(exc))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
