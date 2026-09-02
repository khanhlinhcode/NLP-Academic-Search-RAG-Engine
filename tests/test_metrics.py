"""
Unit tests for evaluation metrics.
"""

import pytest

from src.evaluation.metrics import (
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_none_relevant(self):
        retrieved = ["d4", "d5", "d6"]
        relevant = {"d1", "d2", "d3"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial(self):
        retrieved = ["d1", "d4", "d2"]
        relevant = {"d1", "d2", "d3"}
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


class TestRecallAtK:
    def test_all_found(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_none_found(self):
        retrieved = ["d4", "d5"]
        relevant = {"d1", "d2", "d3"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_partial(self):
        retrieved = ["d1", "d4", "d5"]
        relevant = {"d1", "d2"}
        assert recall_at_k(retrieved, relevant, 3) == 0.5

    def test_empty_relevant(self):
        retrieved = ["d1"]
        relevant = set()
        assert recall_at_k(retrieved, relevant, 1) == 0.0


class TestMRRAtK:
    def test_first_position(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1"}
        assert mrr_at_k(retrieved, relevant, 3) == 1.0

    def test_second_position(self):
        retrieved = ["d4", "d1", "d3"]
        relevant = {"d1"}
        assert mrr_at_k(retrieved, relevant, 3) == 0.5

    def test_not_found(self):
        retrieved = ["d4", "d5", "d6"]
        relevant = {"d1"}
        assert mrr_at_k(retrieved, relevant, 3) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_no_relevant(self):
        retrieved = ["d4", "d5", "d6"]
        relevant = {"d1"}
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_ranking(self):
        retrieved = ["d4", "d1", "d2"]
        relevant = {"d1", "d2"}
        score = ndcg_at_k(retrieved, relevant, 3)
        assert 0 < score < 1  # Should be less than perfect
