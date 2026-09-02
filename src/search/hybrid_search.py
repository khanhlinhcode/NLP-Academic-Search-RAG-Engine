"""
Hybrid Search module.

Combines BM25 (sparse) and Semantic (dense) retrieval using score fusion.
Supports both weighted linear combination and Reciprocal Rank Fusion (RRF).
"""

from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from src.config import settings
from src.data.loader import Paper
from src.search.bm25_search import BM25Searcher, SearchResult
from src.search.semantic_search import SemanticSearcher


class FusionMethod(str, Enum):
    """Available fusion methods for combining search scores."""

    WEIGHTED = "weighted"  # Weighted linear combination
    RRF = "rrf"  # Reciprocal Rank Fusion


class HybridSearcher:
    """
    Hybrid search engine combining BM25 keyword search and SBERT semantic search.

    Retrieves candidates from both search methods and fuses the results
    using either weighted linear combination or Reciprocal Rank Fusion (RRF).

    This approach captures both exact keyword matches (BM25) and
    semantic/meaning-based similarity (SBERT), resulting in higher recall
    and more robust retrieval.

    Attributes:
        bm25: BM25 keyword search engine.
        semantic: SBERT semantic search engine.
        alpha: Weight for BM25 scores in weighted fusion (1 - alpha for semantic).
    """

    def __init__(
        self,
        bm25: BM25Searcher,
        semantic: SemanticSearcher,
        alpha: Optional[float] = None,
    ):
        """
        Initialize hybrid searcher with both search engines.

        Args:
            bm25: Initialized BM25Searcher instance.
            semantic: Initialized SemanticSearcher instance.
            alpha: Weight for BM25 scores (0-1). Defaults to config value.
        """
        self.bm25 = bm25
        self.semantic = semantic
        self.alpha = alpha if alpha is not None else settings.search.bm25_weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        method: FusionMethod = FusionMethod.RRF,
        candidate_pool: int = 50,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and semantic retrieval.

        Args:
            query: User search query string.
            top_k: Number of final results to return.
            method: Score fusion method (weighted or RRF).
            candidate_pool: Number of candidates to retrieve from each method
                           before fusion.

        Returns:
            List of SearchResult objects sorted by fused score (descending).
        """
        if method == FusionMethod.RRF:
            return self._rrf_fusion(query, top_k, candidate_pool)
        else:
            return self._weighted_fusion(query, top_k)

    def _weighted_fusion(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Combine scores using weighted linear combination.

        score = alpha * norm(BM25_score) + (1 - alpha) * semantic_score

        Scores are min-max normalized before combination.
        """
        # Get raw scores from both methods
        bm25_scores = self.bm25.get_scores(query)
        semantic_scores = self.semantic.get_scores(query)

        # Min-max normalize BM25 scores to [0, 1]
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_normalized = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_normalized = np.zeros_like(bm25_scores)

        # Semantic scores are already in [-1, 1] (cosine similarity), normalize to [0, 1]
        sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
        if sem_max > sem_min:
            sem_normalized = (semantic_scores - sem_min) / (sem_max - sem_min)
        else:
            sem_normalized = np.zeros_like(semantic_scores)

        # Weighted combination
        fused_scores = self.alpha * bm25_normalized + (1 - self.alpha) * sem_normalized

        # Get top-K
        top_indices = np.argsort(fused_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if fused_scores[idx] > 0:
                results.append(
                    SearchResult(
                        paper=self.bm25.papers[idx],
                        score=float(fused_scores[idx]),
                    )
                )

        return results

    def _rrf_fusion(
        self,
        query: str,
        top_k: int,
        candidate_pool: int,
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF is rank-based fusion that doesn't require score normalization:
            RRF_score(d) = sum(1 / (k + rank_i(d)))

        where k is a constant (typically 60) and rank_i is the rank of
        document d in the i-th result list.

        This method is more robust than weighted fusion as it doesn't
        depend on score distributions.

        Args:
            query: User search query string.
            top_k: Number of final results to return.
            candidate_pool: Number of candidates from each method.
            k: RRF constant (default 60, standard in literature).
        """
        # Get candidates from both methods
        bm25_results = self.bm25.search(query, top_k=candidate_pool)
        semantic_results = self.semantic.search(query, top_k=candidate_pool)

        # Build RRF score map: paper_id -> (rrf_score, Paper)
        rrf_scores: Dict[str, tuple] = {}

        for rank, result in enumerate(bm25_results, start=1):
            paper_id = result.paper.id
            rrf_score = 1.0 / (k + rank)
            if paper_id in rrf_scores:
                existing_score, paper = rrf_scores[paper_id]
                rrf_scores[paper_id] = (existing_score + rrf_score, paper)
            else:
                rrf_scores[paper_id] = (rrf_score, result.paper)

        for rank, result in enumerate(semantic_results, start=1):
            paper_id = result.paper.id
            rrf_score = 1.0 / (k + rank)
            if paper_id in rrf_scores:
                existing_score, paper = rrf_scores[paper_id]
                rrf_scores[paper_id] = (existing_score + rrf_score, paper)
            else:
                rrf_scores[paper_id] = (rrf_score, result.paper)

        # Sort by RRF score and take top-K
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)

        return [
            SearchResult(paper=paper, score=score)
            for score, paper in sorted_results[:top_k]
        ]
