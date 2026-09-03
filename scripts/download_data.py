"""CLI entry point for versioned arXiv ingestion."""

from __future__ import annotations

import argparse

from nlp_academic_search.data.ingestion import ingest_arxiv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=15000)
    parser.add_argument("--set", dest="set_spec", default="cs")
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    count, version = ingest_arxiv(args.max_records, args.set_spec)
    print(f"Activated {count} verified arXiv metadata records: {version}")


if __name__ == "__main__":
    main()
