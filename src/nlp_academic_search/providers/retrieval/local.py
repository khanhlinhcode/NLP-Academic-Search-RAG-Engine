"""Local BM25, FAISS, and Sentence-Transformer retrieval provider."""

from __future__ import annotations

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper, load_papers
from nlp_academic_search.providers.retrieval.base import RetrievalBatch, RetrievalStatus
from nlp_academic_search.search.bm25_search import BM25Searcher
from nlp_academic_search.search.hybrid_search import HybridSearcher
from nlp_academic_search.search.index_manifest import IndexCompatibilityError
from nlp_academic_search.search.models import FusionMethod, SearchFilters
from nlp_academic_search.search.semantic_search import SemanticSearcher


class LocalRetrievalProvider:
    provider_name = "local"

    def __init__(
        self,
        papers: list[Paper] | None = None,
        bm25: BM25Searcher | None = None,
        semantic: SemanticSearcher | None = None,
        hybrid: HybridSearcher | None = None,
    ) -> None:
        self.papers = papers or load_papers()
        self.bm25 = bm25 or BM25Searcher(self.papers)
        self.semantic = semantic or SemanticSearcher(self.papers)
        self.hybrid = hybrid or HybridSearcher(self.bm25, self.semantic)

    def search(
        self,
        query: str,
        method: str,
        top_k: int,
        *,
        filters: SearchFilters | None = None,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> RetrievalBatch:
        if method == "bm25":
            results = self.bm25.search(query, top_k=top_k, filters=filters)
            mode = "bm25"
        elif method == "semantic":
            results = self.semantic.search(query, top_k=top_k, filters=filters)
            mode = "semantic"
        else:
            results = self.hybrid.search(
                query,
                top_k=top_k,
                method=fusion,
                candidate_pool=max(settings.search.candidate_pool, top_k),
                filters=filters,
            )
            mode = fusion.value
        return RetrievalBatch(results=results, retrieval_mode=mode)

    def status(self) -> RetrievalStatus:
        try:
            self.semantic.manifest.validate(
                corpus_path=self.semantic.corpus_path,
                papers=self.papers,
                model_name=self.semantic.model_name,
                model_revision=self.semantic.model_revision,
                embeddings=self.semantic.embeddings,
                index=self.semantic.index,
            )
        except (IndexCompatibilityError, OSError, ValueError) as exc:
            return RetrievalStatus(ready=False, total_papers=len(self.papers), reason=str(exc))
        manifest = self.semantic.manifest
        return RetrievalStatus(
            ready=True,
            total_papers=len(self.papers),
            provenance=manifest.provenance,
            embedding_model=manifest.embedding_model,
            embedding_revision=manifest.embedding_revision,
            embedding_dimension=manifest.embedding_dimension,
        )

    def close(self) -> None:
        return None
