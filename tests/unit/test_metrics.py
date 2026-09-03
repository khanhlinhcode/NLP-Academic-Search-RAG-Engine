"""Exhaustive unit tests for information retrieval metrics and deterministic RAG metrics."""

import math

import pytest

from nlp_academic_search.evaluation.metrics import (
    _positive_ids,
    average_precision_at_k,
    evaluate_search_method,
    format_evaluation_table,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from nlp_academic_search.evaluation.rag_metrics import evaluate_rag_case


class TestPositiveIds:
    def test_set_input(self):
        s = {"d1", "d2"}
        assert _positive_ids(s) is s

    def test_dict_input(self):
        d = {"d1": 2, "d2": 0, "d3": 1}
        assert _positive_ids(d) == {"d1", "d3"}


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["d1", "d2", "d3"], {"d1", "d2", "d3"}, 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["d4", "d5", "d6"], {"d1", "d2", "d3"}, 3) == 0.0

    def test_partial(self):
        assert precision_at_k(["d1", "d4", "d2"], {"d1", "d2", "d3"}, 3) == pytest.approx(2 / 3)

    def test_k_zero_or_negative(self):
        assert precision_at_k(["d1"], {"d1"}, 0) == 0.0
        assert precision_at_k(["d1"], {"d1"}, -1) == 0.0

    def test_retrieved_shorter_than_k(self):
        # 1 hit in 2 retrieved items, but k=5 => precision = 1/5
        assert precision_at_k(["d1", "d4"], {"d1", "d2"}, 5) == pytest.approx(1 / 5)

    def test_duplicate_retrieved(self):
        # First-Occurrence Deduplication Policy: ["d1", "d1", "d2"] -> unique ["d1", "d2"], hits=2, k=3 => 2/3
        assert precision_at_k(["d1", "d1", "d2"], {"d1", "d2"}, 3) == pytest.approx(2 / 3)


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, 3) == 1.0

    def test_none_found(self):
        assert recall_at_k(["d4", "d5"], {"d1", "d2", "d3"}, 2) == 0.0

    def test_partial(self):
        assert recall_at_k(["d1", "d4", "d5"], {"d1", "d2"}, 3) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["d1"], set(), 1) == 0.0

    def test_relevant_outside_top_k(self):
        assert recall_at_k(["d4", "d5", "d1"], {"d1", "d2"}, 2) == 0.0

    def test_duplicate_cannot_inflate_recall(self):
        # Duplicate d1 is deduplicated, 2 unique hits found out of 2 relevant
        assert recall_at_k(["d1", "d1", "d2"], {"d1", "d2"}, 3) == 1.0


class TestMRRAtK:
    def test_first_position(self):
        assert mrr_at_k(["d1", "d2", "d3"], {"d1"}, 3) == 1.0

    def test_second_position(self):
        assert mrr_at_k(["d4", "d1", "d3"], {"d1"}, 3) == 0.5

    def test_third_position(self):
        assert mrr_at_k(["d4", "d5", "d1"], {"d1"}, 3) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert mrr_at_k(["d4", "d5", "d6"], {"d1"}, 3) == 0.0

    def test_empty_retrieved_or_relevant(self):
        assert mrr_at_k([], {"d1"}, 5) == 0.0
        assert mrr_at_k(["d1"], set(), 5) == 0.0


class TestAveragePrecisionAtK:
    def test_perfect_map(self):
        assert average_precision_at_k(["d1", "d2"], {"d1", "d2"}, 2) == 1.0

    def test_empty_relevant(self):
        assert average_precision_at_k(["d1"], set(), 1) == 0.0

    def test_interleaved_hits(self):
        # retrieved: [d1 (hit, P@1=1/1), d4 (miss), d2 (hit, P@3=2/3)]
        # min(|rel|, k) = min(2, 3) = 2
        # AP = (1.0 + 2/3) / 2 = 5/6 = 0.83333...
        assert average_precision_at_k(["d1", "d4", "d2"], {"d1", "d2"}, 3) == pytest.approx(5 / 6)

    def test_duplicate_cannot_inflate_ap(self):
        # Duplicate d1 deduplicated to ["d1"], P@1=1/1 => AP = 1.0
        assert average_precision_at_k(["d1", "d1"], {"d1"}, 2) == 1.0


class TestNDCGAtK:
    def test_perfect_ranking_binary(self):
        assert ndcg_at_k(["d1", "d2", "d3"], {"d1", "d2", "d3"}, 3) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k(["d4", "d5", "d6"], {"d1"}, 3) == 0.0

    def test_graded_relevance(self):
        # qrels: d1=3, d2=2, d3=1
        relevance = {"d1": 3, "d2": 2, "d3": 1}
        expected = (3 + 7 / math.log2(3) + 1 / math.log2(4)) / (
            7 + 3 / math.log2(3) + 1 / math.log2(4)
        )
        assert ndcg_at_k(["d2", "d1", "d3"], relevance, 3) == pytest.approx(expected)

    def test_empty_relevance(self):
        assert ndcg_at_k(["d1"], {}, 1) == 0.0

    def test_duplicate_cannot_inflate_ndcg(self):
        assert ndcg_at_k(["d1", "d1"], {"d1": 3}, 2) == 1.0


class DummyResult:
    def __init__(self, paper_id: str):
        self.paper = type("PaperObj", (), {"id": paper_id})()


class TestEvaluateSearchMethod:
    def test_evaluate_search_method_calculation(self):
        def mock_search_fn(query: str, top_k: int = 10):
            if "deep" in query:
                return [DummyResult("d1"), DummyResult("d2")]
            return [DummyResult("d4")]

        queries = [
            {"query": "deep learning", "relevant_ids": ["d1", "d2"]},
            {"query": "quantum", "relevant_ids": ["d99"]},
        ]

        results = evaluate_search_method(mock_search_fn, queries, k=2)

        assert "Precision@2" in results
        assert "Recall@2" in results
        assert "MRR@2" in results
        assert "MAP@2" in results
        assert "nDCG@2" in results
        assert "HitRate@2" in results
        assert "Latency_p50_ms" in results

        # Q1: precision@2 = 1.0, Q2: precision@2 = 0.0 -> mean = 0.5
        assert results["Precision@2"] == 0.5
        assert results["HitRate@2"] == 0.5
        assert results["nDCG_95pct_CI_lower"] is not None
        assert results["nDCG_95pct_CI_upper"] is not None

    def test_evaluate_search_method_with_graded_qrels(self):
        def mock_search_fn(query: str, top_k: int = 10):
            return [DummyResult("d1"), DummyResult("d2")]

        queries = [
            {"query": "search query", "qrels": {"d1": 2, "d2": 1}},
        ]
        results = evaluate_search_method(mock_search_fn, queries, k=2)
        assert results["Precision@2"] == 1.0
        assert results["nDCG@2"] == 1.0
        assert results["nDCG_95pct_CI_lower"] is None
        assert results["nDCG_95pct_CI_upper"] is None

    def test_empty_query_collection(self):
        results = evaluate_search_method(lambda *_args, **_kwargs: [], [], k=2)
        assert results["query_count"] == 0
        assert results["Latency_p50_ms"] is None
        assert results["nDCG_95pct_CI_lower"] is None

    def test_query_without_judgments_rejected(self):
        results = evaluate_search_method(lambda *_args, **_kwargs: [], [{"query": "missing"}], k=2)
        assert results["query_count"] == 1
        assert results["Precision@2"] == 0.0
        assert results["Recall@2"] == 0.0


class TestFormatEvaluationTable:
    def test_empty_results(self):
        assert format_evaluation_table({}) == "No results to display."

    def test_formatting(self):
        data = {
            "BM25": {"Precision@10": 0.5, "Recall@10": 0.8},
            "Dense": {"Precision@10": 0.6, "Recall@10": 0.9},
        }
        table = format_evaluation_table(data)
        assert "| Method | Precision@10 | Recall@10 |" in table
        assert "| BM25 | 0.5 | 0.8 |" in table
        assert "| Dense | 0.6 | 0.9 |" in table


class TestEvaluateRAGCase:
    def test_evaluate_rag_case_valid(self):
        case = {
            "relevant_source_ids": ["p1", "p2"],
            "expected_keywords": ["transformer", "attention"],
            "should_refuse": False,
        }
        response = {
            "answer": "The transformer uses attention mechanism [1][2].",
            "sources": [{"id": "p1"}, {"id": "p2"}],
        }
        metrics = evaluate_rag_case(case, response)
        assert metrics["context_precision"] == 1.0
        assert metrics["context_recall"] == 1.0
        assert metrics["answer_relevance"] == 1.0
        assert metrics["citation_precision"] == 1.0
        assert metrics["citation_coverage"] == 1.0
        assert metrics["source_utilization"] == 1.0
        assert metrics["claim_citation_coverage"] == 1.0
        assert metrics["invalid_citation_rate"] == 0.0
        assert metrics["refusal_correct"] is True
        assert metrics["faithfulness_proxy"] == 1.0

    def test_evaluate_rag_case_empty_sources(self):
        case = {
            "relevant_source_ids": ["p1"],
            "expected_keywords": ["keyword"],
            "should_refuse": True,
        }
        response = {
            "answer": "Not enough evidence to answer.",
            "sources": [],
        }
        metrics = evaluate_rag_case(case, response)
        assert metrics["context_precision"] == 0.0
        assert metrics["context_recall"] == 0.0
        assert metrics["refusal_correct"] is True

    def test_semantic_verification_metrics_are_not_structural_proxies(self):
        case = {"relevant_source_ids": ["p1"], "should_refuse": False}
        response = {
            "answer": "A supported answer [1].",
            "sources": [{"id": "p1"}],
            "metadata": {
                "answer_status": "verified",
                "semantic_verification_attempted": True,
                "semantic_verification_succeeded": True,
                "verification_latency_ms": 12.5,
                "citation_repair_attempted": True,
                "citation_repair_succeeded": True,
                "semantic_validation": {
                    "semantic_claim_coverage": 1.0,
                    "supported_claim_count": 1,
                    "unsupported_claim_count": 0,
                    "insufficient_claim_count": 0,
                    "evidence_quote_validity": 1.0,
                },
            },
        }
        metrics = evaluate_rag_case(case, response)
        assert metrics["verified_answer"] == 1.0
        assert metrics["semantic_claim_coverage"] == 1.0
        assert metrics["supported_claim_count"] == 1.0
        assert metrics["semantic_verification_latency_ms"] == 12.5
        assert metrics["repair_attempted"] == 1.0
        assert metrics["repair_succeeded"] == 1.0
        assert metrics["verifier_error"] == 0.0

    def test_verification_refusal_is_counted_as_refusal(self):
        metrics = evaluate_rag_case(
            {"relevant_source_ids": [], "should_refuse": True},
            {
                "answer": "Not enough verified evidence in the retrieved sources.",
                "sources": [],
                "metadata": {"answer_status": "refused_unverified"},
            },
        )
        assert metrics["refusal_correct"] is True
        assert metrics["refusal_due_to_verification"] == 1.0
