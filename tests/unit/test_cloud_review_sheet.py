import copy

import pytest

from nlp_academic_search.evaluation.cloud_review import score_cloud_review
from nlp_academic_search.evaluation.cloud_review_sheet import convert_cloud_review_sheet


@pytest.fixture
def completed_sheet():
    candidates = {
        "name": "candidate-fixture",
        "dataset_role": "candidate_draft_not_scored",
        "corpus": {"paper_count": 1000},
        "cases": [{"id": "a", "anchor_paper_id": "arxiv:1"}, {"id": "u"}],
    }
    base = {
        "reviewer_kind": "human",
        "reviewer_name_or_id": "reviewer-1",
        "reviewed_at_utc": "2026-11-18T00:00:00Z",
        "observed_error": "false",
        "total_latency_ms": "1200",
        "generation_model": "model-a",
        "verification_model": "model-b",
        "verification_status": "verified",
    }
    rows = [
        {
            **base,
            "id": "a",
            "anchor_paper_id": "arxiv:1",
            "question": "What is the finding?",
            "human_expected_answerability": "answerable",
            "reference_source_ids": '["arxiv:1"]',
            "expected_answer": "A finding.",
            "evidence_quote": "A finding.",
            "observed_answer": "A finding [1].",
            "observed_refused": "false",
            "cited_source_ids": '["arxiv:1"]',
            "answer_correct": "true",
            "factual_claims_json": '[{"claim":"A finding","supported":true,"citation_correct":true}]',
        },
        {
            **base,
            "id": "u",
            "anchor_paper_id": "",
            "question": "What is the future result?",
            "human_expected_answerability": "unanswerable",
            "reference_source_ids": "[]",
            "review_notes": "No indexed evidence.",
            "observed_answer": "Not enough evidence.",
            "observed_refused": "true",
            "cited_source_ids": "[]",
            "refusal_appropriate": "true",
        },
    ]
    return candidates, rows


def test_completed_sheet_converts_to_unlocked_scoring_input(completed_sheet):
    candidates, rows = completed_sheet
    result = convert_cloud_review_sheet(candidates, rows, app_commit="a" * 40)
    assert result["dataset_role"] == "reviewed_candidate_unlocked"
    assert result["cases"][0]["reference_source_ids"] == ["arxiv:1"]
    assert result["cases"][1]["observed"]["refused"] is True
    assert score_cloud_review(result)["minimum_recorded_requirements_met"] is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("human_expected_answerability", "", "missing human_expected_answerability"),
        ("reference_source_ids", "[]", "needs reference_source_ids"),
        ("observed_refused", "", "missing observed_refused"),
        ("total_latency_ms", "", "missing total_latency_ms"),
        ("factual_claims_json", "[]", "factual_claims must be non-empty"),
        ("reviewer_kind", "agent_pilot", "only completed human review"),
    ],
)
def test_incomplete_or_unreviewed_sheet_is_rejected(completed_sheet, field, value, error):
    candidates, rows = copy.deepcopy(completed_sheet)
    rows[0][field] = value
    with pytest.raises(ValueError, match=error):
        convert_cloud_review_sheet(candidates, rows, app_commit="a" * 40)


def test_review_sheet_must_match_candidate_ids(completed_sheet):
    candidates, rows = completed_sheet
    with pytest.raises(ValueError, match="IDs must match"):
        convert_cloud_review_sheet(candidates, rows[:1], app_commit="a" * 40)
