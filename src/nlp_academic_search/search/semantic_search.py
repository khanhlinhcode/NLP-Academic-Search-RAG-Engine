"""Dense retrieval with Sentence-Transformers and a validated FAISS index."""

from __future__ import annotations

import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper, active_corpus_path
from nlp_academic_search.data.manifest import atomic_write_text
from nlp_academic_search.search.index_manifest import IndexCompatibilityError, IndexManifest
from nlp_academic_search.search.models import SearchFilters, SearchResult


def active_index_dir(root: Path) -> Path:
    pointer = root / "CURRENT"
    if pointer.is_file():
        version = pointer.read_text(encoding="utf-8").strip()
        candidate = (root / "versions" / version).resolve()
        versions_root = (root / "versions").resolve()
        if candidate.is_relative_to(versions_root) and candidate.is_dir():
            return candidate
    return root


class SemanticSearcher:
    def __init__(
        self,
        papers: list[Paper],
        model_name: str | None = None,
        load_existing: bool = True,
        *,
        model_revision: str | None = None,
        model: Any | None = None,
        index_dir: Path | None = None,
        corpus_path: Path | None = None,
    ) -> None:
        if not papers:
            raise ValueError("Semantic search requires at least one paper")
        self.papers = papers
        self.model_name = model_name or settings.embedding.model_name
        raw_revision = (
            model_revision if model_revision is not None else settings.embedding.model_revision
        )
        self.model_revision = (
            raw_revision.strip() if raw_revision and raw_revision.strip() else None
        )
        # Flat-IP search with one query does not benefit from a large OpenMP pool.
        faiss.omp_set_num_threads(settings.embedding.native_threads)
        self._model = model
        self._inference_lock = threading.RLock()
        self.corpus_path = corpus_path or active_corpus_path()
        self.index_dir = index_dir or active_index_dir(settings.data.embeddings_dir)

        if load_existing:
            self.embeddings, self.index, self.manifest = self._load_existing()
        else:
            self.embeddings = self._compute_embeddings()
            self.index = self._build_faiss_index(self.embeddings)
            self.manifest = IndexManifest.create(
                corpus_path=self.corpus_path,
                papers=self.papers,
                model_name=self.model_name,
                model_revision=self.model_revision,
                dimension=int(self.embeddings.shape[1]),
                index=self.index,
            )

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._inference_lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer

                    torch.set_num_threads(settings.embedding.native_threads)
                    torch.set_grad_enabled(False)
                    self._model = SentenceTransformer(
                        self.model_name,
                        revision=self.model_revision,
                        device=settings.embedding.device,
                    )
                    dimension_getter = getattr(self._model, "get_embedding_dimension", None)
                    model_dimension = (
                        dimension_getter()
                        if dimension_getter is not None
                        else self._model.get_sentence_embedding_dimension()
                    )
                    if (
                        hasattr(self, "manifest")
                        and model_dimension
                        and model_dimension != self.manifest.embedding_dimension
                    ):
                        raise IndexCompatibilityError(
                            "Configured model output dimension differs from index manifest"
                        )
        return self._model

    def _load_existing(self) -> tuple[np.ndarray, faiss.Index, IndexManifest]:
        embeddings_path = self.index_dir / "paper_embeddings.npy"
        index_path = self.index_dir / "faiss.index"
        manifest_path = self.index_dir / "index_manifest.json"
        missing = [
            path.name for path in (embeddings_path, index_path, manifest_path) if not path.exists()
        ]
        if missing:
            raise IndexCompatibilityError(
                f"Active semantic index is incomplete ({', '.join(missing)}). Run 'make index'."
            )
        embeddings = np.load(embeddings_path, mmap_mode="r")
        index = faiss.read_index(str(index_path))
        manifest = IndexManifest.load(manifest_path)
        manifest.validate(
            corpus_path=self.corpus_path,
            papers=self.papers,
            model_name=self.model_name,
            model_revision=self.model_revision,
            embeddings=embeddings,
            index=index,
        )
        return embeddings, index, manifest

    def _compute_embeddings(self) -> np.ndarray:
        with self._inference_lock:
            values = self.model.encode(
                [paper.text for paper in self.papers],
                show_progress_bar=True,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        embeddings = np.asarray(values, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.papers):
            raise IndexCompatibilityError("Embedding model returned an unexpected array shape")
        return embeddings

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.asarray(embeddings, dtype=np.float32))
        return index

    def save_versioned(self, root: Path | None = None) -> Path:
        root = root or settings.data.embeddings_dir
        version = datetime.now(UTC).strftime("index-%Y%m%dT%H%M%S%fZ")
        target = root / "versions" / version
        temporary = root / "versions" / f".{version}.{os.getpid()}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        np.save(temporary / "paper_embeddings.npy", self.embeddings)
        faiss.write_index(self.index, str(temporary / "faiss.index"))
        (temporary / "index_manifest.json").write_text(
            self.manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        os.replace(temporary, target)
        atomic_write_text(root / "CURRENT", f"{version}\n")
        self.index_dir = target
        return target

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if top_k < 1 or not query.strip():
            return []
        candidate_k = min(len(self.papers), max(top_k, settings.search.candidate_pool))
        with self._inference_lock:
            scores, indices = self.index.search(self.get_query_embedding(query), candidate_k)
        results = []
        active_filters = filters or SearchFilters()
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            paper = self.papers[int(index)]
            if active_filters.matches(paper):
                results.append(SearchResult(paper=paper, semantic_score=float(score)))
            if len(results) == top_k:
                break
        return results

    def get_query_embedding(self, query: str) -> np.ndarray:
        with self._inference_lock:
            values = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(values, dtype=np.float32)

    def get_scores(self, query: str) -> np.ndarray:
        return np.dot(self.embeddings, self.get_query_embedding(query).T).flatten()
