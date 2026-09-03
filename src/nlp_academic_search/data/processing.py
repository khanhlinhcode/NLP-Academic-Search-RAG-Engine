"""Validated, deduplicated corpus preprocessing workflow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper
from nlp_academic_search.data.manifest import (
    activate_version,
    build_manifest,
    write_jsonl,
    write_manifest,
)


def preprocess_corpus(source_path: Path) -> Path:
    """Validate JSONL rows, deduplicate them and activate the resulting corpus."""
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    version_dir = settings.data.raw_dir / "versions" / version
    output_path = version_dir / "papers.jsonl"
    quarantine_path = version_dir / "quarantine.jsonl"
    papers: list[Paper] = []
    rejected: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                paper = Paper.from_dict(json.loads(line))
                keys = {paper.id, paper.content_hash}
                if paper.arxiv_id:
                    keys.add(f"arxiv:{paper.arxiv_id}")
                if paper.doi:
                    keys.add(f"doi:{paper.doi}")
                if keys & seen_keys:
                    continue
                seen_keys.update(keys)
                papers.append(paper)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                rejected.append({"line": line_number, "error": str(exc)})

    if not papers:
        raise RuntimeError("No valid records remain after preprocessing")
    write_jsonl(output_path, papers)
    if rejected:
        quarantine_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rejected),
            encoding="utf-8",
        )
    manifest = build_manifest(
        output_path,
        papers,
        source="validated-jsonl",
        filtering_rules=["schema validation", "deduplicate id/arxiv_id/doi/content_hash"],
        quarantined_count=len(rejected),
    )
    write_manifest(version_dir / "corpus_manifest.json", manifest)
    activate_version(settings.data.raw_dir, version)
    return output_path
