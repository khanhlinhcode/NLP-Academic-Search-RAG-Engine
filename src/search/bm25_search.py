"""
BM25 Keyword Search module.

Implements sparse retrieval using the BM25 (Okapi) algorithm.
BM25 is effective for keyword-based queries where exact term matching matters.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from src.data.loader import Paper
from src.data.preprocessor import tokenize_for_bm25


@dataclass
class SearchResult:
    """A single search result with paper and relevance score."""

    paper: Paper
    score: float

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        result = self.paper.to_dict()
        result["score"] = round(float(self.score), 4)
        return result


class BM25Searcher:
    """
    BM25-based keyword search engine.

    Uses BM25 (Okapi variant) for sparse retrieval — ranking documents
    by term frequency, inverse document frequency, and document length normalization.

    Attributes:
        papers: List of Paper objects in the index.
        bm25: The BM25Okapi index instance.
    """

    def __init__(self, papers: List[Paper]):
        """
        Initialize BM25 index from a list of papers.

        Args:
            papers: List of Paper objects to index.
        """
        self.papers = papers

        # Tokenize all paper texts for BM25 indexing
        print("🔧 Building BM25 index...")
        tokenized_corpus = [tokenize_for_bm25(paper.text) for paper in papers]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 index built with {len(papers)} documents.")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search papers using BM25 keyword matching.

        Args:
            query: User search query string.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult objects sorted by BM25 score (descending).
        """
        # Tokenize the query the same way as the documents
        tokenized_query = tokenize_for_bm25(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-K indices sorted by score (descending)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include papers with positive scores
                results.append(
                    SearchResult(paper=self.papers[idx], score=float(scores[idx]))
                )

        return results

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get raw BM25 scores for all documents (used in hybrid search).

        Args:
            query: User search query string.

        Returns:
            numpy array of BM25 scores for all papers.
        """
        tokenized_query = tokenize_for_bm25(query)
        return self.bm25.get_scores(tokenized_query)
