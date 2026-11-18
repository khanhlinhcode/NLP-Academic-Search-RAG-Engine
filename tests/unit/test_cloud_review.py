import copy

import pytest

from nlp_academic_search.evaluation.cloud_review import score_cloud_review


@pytest.fixture
def reviewed_cases():
    return {
        "name": "reviewed-fixture",
        "cases": [
            {
                "id": "answerable-supported",
                "expected_answerability": "answerable",
                "observed": {"refused": False, "total_latency_ms": 100},
                "review": {
                    "reviewer_kind": "human",
                    "answer_correct": True,
                    "factual_claims": [
                        {"claim": "First claim", "supported": True, "citation_correct": True},
                        {"claim": "Second claim", "supported": False, "citation_correct": False},
                    ],
                },
            },
            {
                "id": "answerable-refused",
                "expected_answerability": "answerable",
                "observed": {"refused": True, "total_latency_ms": 200},
                "review": {"reviewer_kind": "human"},
            },
            {
                "id": "unanswerable-refused",
                "expected_answerability": "unanswerable",
                "observed": {"refused": True, "total_latency_ms": 300},
                "review": {"reviewer_kind": "human", "refusal_appropriate": True},
            },
        ],
    }


def test_score_separates_answer_quality_from_refusal_and_claim_support(reviewed_cases):
    report = score_cloud_review(reviewed_cases)
    assert report["grounded_answer_accuracy"] == {"passed": 0, "total": 2, "rate": 0.0}
    assert report["correct_refusal_rate"] == {"passed": 1, "total": 1, "rate": 1.0}
    assert report["false_refusal_rate"] == {"failed": 1, "total": 2, "rate": 0.5}
    assert report["supported_claim_rate"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert report["total_latency_ms"] == {"measured": 3, "p50": None, "p95": None}
    assert report["error_rate"] == {"failed": 0, "total": 3, "rate": 0.0}
    assert report["minimum_recorded_requirements_met"] is False


def test_score_rejects_missing_review_and_duplicate_ids(reviewed_cases):
    incomplete = copy.deepcopy(reviewed_cases)
    del incomplete["cases"][0]["review"]["factual_claims"]
    with pytest.raises(ValueError, match="factual_claims must be non-empty"):
        score_cloud_review(incomplete)

    duplicate = copy.deepcopy(reviewed_cases)
    duplicate["cases"][1]["id"] = duplicate["cases"][0]["id"]
    with pytest.raises(ValueError, match="duplicate case id"):
        score_cloud_review(duplicate)

    blank_claim = copy.deepcopy(reviewed_cases)
    blank_claim["cases"][0]["review"]["factual_claims"][0]["claim"] = "  "
    with pytest.raises(ValueError, match="invalid factual_claims labels"):
        score_cloud_review(blank_claim)


def test_provider_error_counts_as_failed_question_without_diluting_success_latency(reviewed_cases):
    failed = copy.deepcopy(reviewed_cases)
    failed["cases"][0]["observed"] = {
        "refused": False,
        "error": True,
        "total_latency_ms": 9000,
    }
    report = score_cloud_review(failed)
    assert report["grounded_answer_accuracy"] == {"passed": 0, "total": 2, "rate": 0.0}
    assert report["error_rate"] == {"failed": 1, "total": 3, "rate": 0.3333}
    assert report["total_latency_ms"]["measured"] == 2
