"""Conversion from a local BEIR dataset to the repository benchmark schema."""

from __future__ import annotations

import csv
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.data.manifest import sha256_file


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_beir(
    dataset_dir: Path,
    qrels_path: Path,
    output: Path,
    name: str,
    *,
    provenance_url: str = "https://github.com/beir-cellar/beir",
    license_name: str = "See upstream dataset card and source dataset license",
) -> tuple[int, int]:
    """Convert corpus, queries and qrels without changing relevance judgments."""
    corpus_rows = _read_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = _read_jsonl(dataset_dir / "queries.jsonl")
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = row.get("query-id") or row.get("query_id")
            corpus_id = row.get("corpus-id") or row.get("corpus_id")
            if query_id and corpus_id:
                qrels.setdefault(query_id, {})[f"beir:{corpus_id}"] = int(row["score"])
    papers = [
        Paper(
            id=f"beir:{row['_id']}",
            title=row["title"],
            abstract=row["text"],
            source=f"BEIR/{name}",
        ).to_dict()
        for row in corpus_rows
        if row.get("text") and row.get("title")
    ]
    queries = [
        {"id": row["_id"], "query": row["text"], "qrels": qrels.get(row["_id"], {})}
        for row in query_rows
        if qrels.get(row["_id"])
    ]
    payload = {
        "name": f"BEIR/{name}",
        "version": "BEIR public release; exact files identified by manifest hashes",
        "kind": "standard-benchmark-import",
        "provenance": (
            f"Locally imported BEIR files from {dataset_dir.name}; acquisition and license "
            "must be recorded separately."
        ),
        "documents": papers,
        "queries": queries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    qrel_count = sum(len(items) for items in qrels.values())
    manifest = {
        "dataset": name,
        "benchmark_family": "BEIR",
        "split": qrels_path.stem,
        "provenance_url": provenance_url,
        "license": license_name,
        "imported_at": datetime.now(UTC).isoformat(),
        "import_command": " ".join(
            shlex.quote(part)
            for part in (
                "uv run python -m scripts.import_beir",
                str(dataset_dir),
                str(qrels_path),
                str(output),
                "--name",
                name,
            )
        ),
        "files": {
            "corpus.jsonl": sha256_file(dataset_dir / "corpus.jsonl"),
            "queries.jsonl": sha256_file(dataset_dir / "queries.jsonl"),
            qrels_path.name: sha256_file(qrels_path),
            output.name: sha256_file(output),
        },
        "counts": {
            "documents": len(papers),
            "queries": len(queries),
            "qrels": qrel_count,
        },
        "judgments_transformed": False,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(papers), len(queries)
