"""
Unit tests for BM25 search module.
"""

import pytest

from src.data.loader import Paper
from src.search.bm25_search import BM25Searcher


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        Paper(
            id="p001",
            title="Attention Is All You Need",
            abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            authors=["Vaswani"],
            category="cs.CL",
        ),
        Paper(
            id="p002",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="We introduce a new language representation model called BERT, designed to pre-train deep bidirectional representations.",
            authors=["Devlin"],
            category="cs.CL",
        ),
        Paper(
            id="p003",
            title="ImageNet Classification with Deep Convolutional Neural Networks",
            abstract="We trained a large, deep convolutional neural network to classify images into categories.",
            authors=["Krizhevsky"],
            category="cs.CV",
        ),
        Paper(
            id="p004",
            title="Generative Adversarial Networks",
            abstract="We propose a new framework for estimating generative models via an adversarial process.",
            authors=["Goodfellow"],
            category="cs.LG",
        ),
        Paper(
            id="p005",
            title="Playing Atari with Deep Reinforcement Learning",
            abstract="We present the first deep learning model to successfully learn control policies from raw pixels using reinforcement learning.",
            authors=["Mnih"],
            category="cs.AI",
        ),
    ]


@pytest.fixture
def bm25_searcher(sample_papers):
    """Create BM25 searcher with sample papers."""
    return BM25Searcher(sample_papers)


class TestBM25Searcher:
    def test_init(self, bm25_searcher, sample_papers):
        """Test BM25 searcher initialization."""
        assert len(bm25_searcher.papers) == len(sample_papers)

    def test_search_returns_results(self, bm25_searcher):
        """Test that search returns results for a relevant query."""
        results = bm25_searcher.search("transformer attention", top_k=3)
        assert len(results) > 0
        assert results[0].score > 0

    def test_search_relevance(self, bm25_searcher):
        """Test that the most relevant paper is ranked first."""
        results = bm25_searcher.search("attention transformer", top_k=5)
        # "Attention Is All You Need" should rank high for this query
        top_ids = [r.paper.id for r in results]
        assert "p001" in top_ids[:3]

    def test_search_top_k(self, bm25_searcher):
        """Test that top_k limits results."""
        results = bm25_searcher.search("deep learning", top_k=2)
        assert len(results) <= 2

    def test_search_no_results(self, bm25_searcher):
        """Test search with irrelevant query returns empty or low-score results."""
        results = bm25_searcher.search("quantum physics superconductor", top_k=5)
        # Should return empty or very low scores
        if results:
            assert all(r.score >= 0 for r in results)

    def test_get_scores(self, bm25_searcher, sample_papers):
        """Test raw score retrieval."""
        scores = bm25_searcher.get_scores("BERT language model")
        assert len(scores) == len(sample_papers)

    def test_search_result_structure(self, bm25_searcher):
        """Test search result contains expected fields."""
        results = bm25_searcher.search("neural network", top_k=1)
        if results:
            result = results[0]
            assert hasattr(result, "paper")
            assert hasattr(result, "score")
            assert hasattr(result.paper, "id")
            assert hasattr(result.paper, "title")

    def test_search_result_to_dict(self, bm25_searcher):
        """Test search result serialization."""
        results = bm25_searcher.search("deep learning", top_k=1)
        if results:
            d = results[0].to_dict()
            assert "id" in d
            assert "title" in d
            assert "score" in d
