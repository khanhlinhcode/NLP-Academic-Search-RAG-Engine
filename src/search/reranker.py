"""
Cross-Encoder Reranker module.

Uses a Cross-Encoder model to re-score (query, document) pairs
for more accurate ranking of the top candidates retrieved by the
hybrid search pipeline.

Cross-Encoders are more accurate than bi-encoders (like SBERT) because
they jointly encode the query and document, allowing full attention
between them. However, they are slower since each (query, doc) pair
must be processed individually.

Typical pipeline:
    Hybrid Search (fast, top 20) → Cross-Encoder Reranker (accurate, top 5)
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import settings
from src.search.bm25_search import SearchResult


class Reranker:
    """
    Cross-Encoder based reranker for search result refinement.

    Takes the top-N results from the hybrid search pipeline and
    re-scores each (query, document) pair using a Cross-Encoder
    for more accurate final ranking.

    Attributes:
        model: CrossEncoder model instance.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = "cpu"):
        """
        Initialize the Cross-Encoder reranker.

        Args:
            model_name: Name of the Cross-Encoder model.
                       Defaults to config value (cross-encoder/ms-marco-MiniLM-L-6-v2).
            device: Computing device ('cpu', 'cuda', 'mps'). Defaults to 'cpu' for stability.
        """
        model_name = model_name or settings.reranker.model_name

        print(f"🔄 Loading Cross-Encoder reranker: {model_name} (device={device})")
        self.model = CrossEncoder(model_name, device=device)
        print("✅ Reranker loaded.")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Re-rank search results using Cross-Encoder scoring.

        Each (query, document) pair is scored by the Cross-Encoder,
        which provides more accurate relevance estimation than
        bi-encoder similarity.

        Args:
            query: User search query string.
            results: List of SearchResult candidates from hybrid search.
            top_k: Number of top results to return after reranking.

        Returns:
            List of SearchResult objects sorted by Cross-Encoder score (descending).
        """
        if not results:
            return []

        import os
        import torch

        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        # Create (query, document) pairs for Cross-Encoder
        pairs = [(query, result.paper.text) for result in results]

        # Score all pairs with explicit batch size
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)

        # Sort by Cross-Encoder score (descending)
        scored_indices = np.argsort(scores)[::-1][:top_k]

        reranked = []
        for idx in scored_indices:
            reranked.append(
                SearchResult(
                    paper=results[idx].paper,
                    score=float(scores[idx]),
                )
            )

        return reranked
