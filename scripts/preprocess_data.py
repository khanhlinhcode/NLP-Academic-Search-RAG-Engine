"""CLI entry point for validated corpus preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_academic_search.config import settings
from nlp_academic_search.data.processing import preprocess_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=settings.data.raw_dir / "papers.jsonl")
    args = parser.parse_args()
    output = preprocess_corpus(args.input)
    print(f"Activated validated corpus: {output}")


if __name__ == "__main__":
    main()
