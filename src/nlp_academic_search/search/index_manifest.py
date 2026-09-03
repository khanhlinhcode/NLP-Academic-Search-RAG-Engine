"""Versioned semantic-index provenance and compatibility validation."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.data.manifest import ordered_id_hash, sha256_file


class IndexCompatibilityError(RuntimeError):
    """The active index cannot safely be paired with the configured corpus/model."""


class IndexManifest(BaseModel):
    schema_version: str = "1.0"
    corpus_sha256: str
    ordered_document_id_sha256: str
    document_count: int
    embedding_model: str
    embedding_revision: str | None
    embedding_dimension: int
    normalized: bool
    faiss_index_type: str
    build_timestamp: datetime
    library_versions: dict[str, str]
    provenance: str = "built"

    @classmethod
    def create(
        cls,
        *,
        corpus_path: Path,
        papers: list[Paper],
        model_name: str,
        model_revision: str | None,
        dimension: int,
        index: faiss.Index,
        provenance: str = "built",
    ) -> IndexManifest:
        versions = {}
        for package in ("faiss-cpu", "numpy", "sentence-transformers"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        return cls(
            corpus_sha256=sha256_file(corpus_path),
            ordered_document_id_sha256=ordered_id_hash(papers),
            document_count=len(papers),
            embedding_model=model_name,
            embedding_revision=model_revision,
            embedding_dimension=dimension,
            normalized=True,
            faiss_index_type=type(index).__name__,
            build_timestamp=datetime.now(UTC),
            library_versions=versions,
            provenance=provenance,
        )

    @classmethod
    def load(cls, path: Path) -> IndexManifest:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise IndexCompatibilityError(f"Invalid index manifest at {path}: {exc}") from exc

    def validate(
        self,
        *,
        corpus_path: Path,
        papers: list[Paper],
        model_name: str,
        model_revision: str | None,
        embeddings: np.ndarray,
        index: faiss.Index,
    ) -> None:
        failures = []
        if self.corpus_sha256 != sha256_file(corpus_path):
            failures.append("corpus content hash differs")
        if self.ordered_document_id_sha256 != ordered_id_hash(papers):
            failures.append("ordered document-ID hash differs")
        if self.document_count != len(papers) or embeddings.shape[0] != len(papers):
            failures.append("paper/vector count differs")
        if index.ntotal != len(papers):
            failures.append("FAISS ntotal differs")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dimension:
            failures.append("embedding array dimension differs")
        if index.d != self.embedding_dimension:
            failures.append("FAISS dimension differs")
        if self.embedding_model != model_name:
            failures.append("embedding model differs")
        if self.embedding_revision != model_revision:
            failures.append("embedding model revision differs")
        if not self.normalized:
            failures.append("index vectors are not declared normalized")
        if failures:
            raise IndexCompatibilityError("Incompatible semantic index: " + "; ".join(failures))
