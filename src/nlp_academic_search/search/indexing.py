"""Build and adoption workflows for corpus-bound semantic indexes."""

from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper, active_corpus_path, load_papers
from nlp_academic_search.data.manifest import atomic_write_text, build_manifest
from nlp_academic_search.search.index_manifest import IndexManifest
from nlp_academic_search.search.semantic_search import SemanticSearcher


def _restore_text(path: Path, previous: str | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.rollback")
    temporary.write_text(previous, encoding="utf-8")
    os.replace(temporary, path)


def adopt_existing_index(
    *,
    embeddings_dir: Path | None = None,
    raw_dir: Path | None = None,
    papers_list: list[Paper] | None = None,
    corpus_file_path: Path | None = None,
) -> Path:
    """Attach compatibility metadata without claiming model-weight provenance."""
    root = embeddings_dir or settings.data.embeddings_dir
    raw_root = raw_dir or settings.data.raw_dir
    c_path = corpus_file_path or active_corpus_path()
    embeddings_path = root / "paper_embeddings.npy"
    index_path = root / "faiss.index"
    if not embeddings_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Legacy embeddings/faiss.index are missing; run a full index build")
    papers = papers_list if papers_list is not None else load_papers(c_path)
    paper_ids = [paper.id for paper in papers]
    if len(paper_ids) != len(set(paper_ids)):
        raise RuntimeError("Legacy corpus contains duplicate paper IDs and cannot be adopted")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    index = faiss.read_index(str(index_path))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(papers) or index.ntotal != len(papers):
        raise RuntimeError("Legacy index does not match the active corpus and cannot be adopted")
    manifest = IndexManifest.create(
        corpus_path=c_path,
        papers=papers,
        model_name=settings.embedding.model_name,
        model_revision=settings.embedding.model_revision,
        dimension=int(embeddings.shape[1]),
        index=index,
        provenance="legacy-adopted-unverified-model-weights",
    )
    manifest.validate(
        corpus_path=c_path,
        papers=papers,
        model_name=settings.embedding.model_name,
        model_revision=settings.embedding.model_revision,
        embeddings=embeddings,
        index=index,
    )
    manifest_path = root / "index_manifest.json"
    corpus_manifest = build_manifest(
        c_path,
        papers,
        source="legacy-ccdv-arxiv-summarization",
        filtering_rules=["legacy placeholders suppressed at load time"],
        license_notes="Historical corpus provenance/license was not recorded; do not redistribute.",
    )
    corpus_manifest_path = raw_root / "corpus_manifest.json"
    previous = {
        manifest_path: manifest_path.read_text(encoding="utf-8")
        if manifest_path.exists()
        else None,
        corpus_manifest_path: corpus_manifest_path.read_text(encoding="utf-8")
        if corpus_manifest_path.exists()
        else None,
    }
    try:
        atomic_write_text(corpus_manifest_path, corpus_manifest.model_dump_json(indent=2))
        atomic_write_text(manifest_path, manifest.model_dump_json(indent=2))
    except Exception:
        for path, content in previous.items():
            _restore_text(path, content)
        raise
    return manifest_path


def build_semantic_index() -> Path:
    """Build and atomically activate an index for the current corpus."""
    papers = load_papers()
    searcher = SemanticSearcher(papers, load_existing=False)
    return searcher.save_versioned()
