"""Corpus provenance manifests and atomic activation helpers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from nlp_academic_search.data.loader import SCHEMA_VERSION, Paper


class CorpusManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: str
    retrieved_at: datetime
    document_count: int = Field(ge=0)
    corpus_sha256: str
    ordered_id_sha256: str
    filtering_rules: list[str]
    license_notes: str | None = None
    quarantined_count: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_hash(papers: Iterable[Paper]) -> str:
    digest = hashlib.sha256()
    for paper in papers:
        digest.update(paper.id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    corpus_path: Path,
    papers: list[Paper],
    *,
    source: str,
    filtering_rules: list[str],
    quarantined_count: int = 0,
    license_notes: str | None = None,
) -> CorpusManifest:
    return CorpusManifest(
        source=source,
        retrieved_at=datetime.now(UTC),
        document_count=len(papers),
        corpus_sha256=sha256_file(corpus_path),
        ordered_id_sha256=ordered_id_hash(papers),
        filtering_rules=filtering_rules,
        quarantined_count=quarantined_count,
        license_notes=license_notes,
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_manifest(path: Path, manifest: CorpusManifest) -> None:
    atomic_write_text(path, manifest.model_dump_json(indent=2))


def activate_version(data_root: Path, version: str) -> None:
    version_dir = (data_root / "versions" / version).resolve()
    versions_root = (data_root / "versions").resolve()
    if (
        not version_dir.is_relative_to(versions_root)
        or not (version_dir / "papers.jsonl").is_file()
    ):
        raise ValueError(f"Cannot activate incomplete corpus version: {version}")
    atomic_write_text(data_root / "CURRENT", f"{version}\n")


def write_jsonl(path: Path, papers: Iterable[Paper]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
