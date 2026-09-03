"""CLI entry point for converting a local BEIR dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_academic_search.data.sources.beir import convert_beir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("qrels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="scifact")
    parser.add_argument(
        "--provenance-url",
        default="https://github.com/beir-cellar/beir",
    )
    parser.add_argument(
        "--license",
        default="See upstream dataset card and source dataset license",
    )
    args = parser.parse_args()
    documents, queries = convert_beir(
        args.dataset_dir,
        args.qrels,
        args.output,
        args.name,
        provenance_url=args.provenance_url,
        license_name=args.license,
    )
    print(f"Imported {documents} documents and {queries} judged queries into {args.output}")
    print(f"Manifest: {args.output.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
