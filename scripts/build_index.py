"""CLI entry point for building or adopting a semantic index."""

from __future__ import annotations

import argparse

from nlp_academic_search.search.indexing import adopt_existing_index, build_semantic_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args()
    if args.adopt_existing:
        path = adopt_existing_index()
        print(f"Created compatibility manifest at {path}; model weights remain unverified.")
    else:
        target = build_semantic_index()
        print(f"Built and atomically activated semantic index: {target}")


if __name__ == "__main__":
    main()
