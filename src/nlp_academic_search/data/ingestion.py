"""Versioned arXiv ingestion workflow used by the command-line entry point."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import iter_papers
from nlp_academic_search.data.manifest import (
    activate_version,
    atomic_write_text,
    build_manifest,
    write_jsonl,
    write_manifest,
)
from nlp_academic_search.data.sources import ArxivOAIAdapter


def ingest_arxiv(max_records: int, set_spec: str = "cs") -> tuple[int, str]:
    """Ingest, validate and atomically activate an arXiv corpus version."""
    if max_records < 1:
        raise ValueError("max_records must be positive")

    staging = settings.data.raw_dir / "staging" / f"arxiv-{set_spec.replace(':', '-')}"
    staging.mkdir(parents=True, exist_ok=True)
    checkpoint = staging / "resumption_token.txt"
    partial = staging / "papers.partial.jsonl"
    adapter = ArxivOAIAdapter(set_spec=set_spec, checkpoint_path=checkpoint)
    papers = list(iter_papers(partial)) if partial.exists() else []
    seen: set[str] = set()
    quarantine: list[dict[str, str]] = []
    for paper in papers:
        seen.update({paper.id, paper.content_hash})
        if paper.arxiv_id:
            seen.add(f"arxiv:{paper.arxiv_id}")
        if paper.doi:
            seen.add(f"doi:{paper.doi}")

    if len(papers) < max_records:
        with partial.open("a", encoding="utf-8") as handle:
            for paper in adapter.iter_papers():
                keys = {paper.id, paper.content_hash}
                if paper.arxiv_id:
                    keys.add(f"arxiv:{paper.arxiv_id}")
                if paper.doi:
                    keys.add(f"doi:{paper.doi}")
                if keys & seen:
                    quarantine.append({"id": paper.id, "reason": "duplicate"})
                    continue
                seen.update(keys)
                papers.append(paper)
                handle.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                if len(papers) >= max_records:
                    break

    if not papers:
        raise RuntimeError(
            "arXiv ingestion returned no valid records; previous corpus remains active"
        )

    quarantine.extend(adapter.quarantined)
    version = datetime.now(UTC).strftime("arxiv-%Y%m%dT%H%M%S%fZ")
    version_dir = settings.data.raw_dir / "versions" / version
    corpus_path = version_dir / "papers.jsonl"
    write_jsonl(corpus_path, papers)
    if quarantine:
        atomic_write_text(
            version_dir / "quarantine.jsonl",
            "".join(json.dumps(item) + "\n" for item in quarantine),
        )
    manifest = build_manifest(
        corpus_path,
        papers,
        source=adapter.name,
        filtering_rules=[f"arXiv OAI set={set_spec}", "exclude deleted records", "deduplicate"],
        quarantined_count=len(quarantine),
        license_notes="Per-record license is retained when arXiv supplies it.",
    )
    write_manifest(version_dir / "corpus_manifest.json", manifest)
    activate_version(settings.data.raw_dir, version)
    atomic_write_text(checkpoint, "")
    return len(papers), version
