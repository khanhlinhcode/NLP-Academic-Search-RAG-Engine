"""Optional cross-encoder reranking."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from nlp_academic_search.config import settings
from nlp_academic_search.search.models import SearchResult


class Reranker:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = "cpu",
        *,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker.model_name
        self.model = model or CrossEncoder(self.model_name, device=device)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results or top_k < 1:
            return []
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        scores = np.asarray(
            self.model.predict(
                [(query, result.paper.text) for result in results],
                batch_size=8,
                show_progress_bar=False,
            )
        )
        return [
            SearchResult(
                paper=results[int(index)].paper,
                bm25_score=results[int(index)].bm25_score,
                semantic_score=results[int(index)].semantic_score,
                rrf_score=results[int(index)].rrf_score,
                weighted_score=results[int(index)].weighted_score,
                reranker_score=float(scores[index]),
            )
            for index in np.argsort(scores)[::-1][:top_k]
        ]
