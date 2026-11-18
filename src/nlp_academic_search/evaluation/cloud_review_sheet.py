"""Convert a completed human review sheet to an unlocked cloud scoring input."""

from __future__ import annotations

import json
from typing import Any

from nlp_academic_search.evaluation.cloud_review import score_cloud_review


def _required(row: dict[str, str], key: str, case_id: str) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{case_id}: missing {key}")
    return value


def _boolean(row: dict[str, str], key: str, case_id: str) -> bool:
    value = _required(row, key, case_id).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{case_id}: {key} must be true or false")
    return value == "true"


def _json_list(row: dict[str, str], key: str, case_id: str) -> list[Any]:
    try:
        value = json.loads(_required(row, key, case_id))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{case_id}: invalid JSON in {key}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{case_id}: {key} must be a JSON array")
    return value


def _source_ids(row: dict[str, str], key: str, case_id: str) -> list[str]:
    values = _json_list(row, key, case_id)
    if not all(isinstance(value, str) and value.startswith("arxiv:") for value in values):
        raise ValueError(f"{case_id}: {key} must contain arxiv source IDs")
    return values


def convert_cloud_review_sheet(
    candidates: dict[str, Any], rows: list[dict[str, str]], *, app_commit: str
) -> dict[str, Any]:
    """Require every candidate row and human label before creating scoring input."""
    if len(app_commit) != 40 or any(char not in "0123456789abcdef" for char in app_commit):
        raise ValueError("app_commit must be a full lowercase Git SHA")
    cases = candidates.get("cases")
    if candidates.get("dataset_role") != "candidate_draft_not_scored" or not isinstance(
        cases, list
    ):
        raise ValueError("expected a candidate draft JSON with cases")
    candidate_ids = [case["id"] for case in cases]
    row_ids = [row.get("id", "") for row in rows]
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be non-empty and unique")
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(candidate_ids):
        raise ValueError("review sheet IDs must match the candidate IDs exactly")
    candidates_by_id = {case["id"]: case for case in cases}
    converted: list[dict[str, Any]] = []

    for row in rows:
        case_id = row["id"]
        candidate = candidates_by_id[case_id]
        if row.get("anchor_paper_id", "") != (candidate.get("anchor_paper_id") or ""):
            raise ValueError(f"{case_id}: anchor_paper_id differs from candidate")
        question = _required(row, "question", case_id)
        if not 5 <= len(question) <= 1000:
            raise ValueError(f"{case_id}: question length is outside Ask limits")
        answerability = _required(row, "human_expected_answerability", case_id)
        if answerability not in {"answerable", "unanswerable"}:
            raise ValueError(f"{case_id}: invalid human_expected_answerability")
        reference_ids = _source_ids(row, "reference_source_ids", case_id)
        if answerability == "answerable" and not reference_ids:
            raise ValueError(f"{case_id}: answerable question needs reference_source_ids")
        if answerability == "unanswerable" and reference_ids:
            raise ValueError(f"{case_id}: unanswerable question must have no reference_source_ids")
        expected_answer = (row.get("expected_answer") or "").strip()
        evidence_quote = (row.get("evidence_quote") or "").strip()
        if answerability == "answerable" and (not expected_answer or not evidence_quote):
            raise ValueError(f"{case_id}: answerable question needs answer and evidence quote")
        if answerability == "unanswerable":
            _required(row, "review_notes", case_id)

        refused = _boolean(row, "observed_refused", case_id)
        error = _boolean(row, "observed_error", case_id)
        observed_answer = (row.get("observed_answer") or "").strip()
        if not error and not observed_answer:
            raise ValueError(f"{case_id}: missing observed_answer")
        cited_ids = _source_ids(row, "cited_source_ids", case_id)
        latency_text = (row.get("total_latency_ms") or "").strip()
        if not latency_text and not error:
            raise ValueError(f"{case_id}: missing total_latency_ms")
        try:
            latency = float(latency_text) if latency_text else None
        except ValueError as exc:
            raise ValueError(f"{case_id}: invalid total_latency_ms") from exc
        if latency is not None and not 0 <= latency < float("inf"):
            raise ValueError(f"{case_id}: invalid total_latency_ms")

        reviewer_kind = _required(row, "reviewer_kind", case_id)
        if reviewer_kind != "human":
            raise ValueError(f"{case_id}: only completed human review is accepted")
        reviewer_id = _required(row, "reviewer_name_or_id", case_id)
        reviewed_at = _required(row, "reviewed_at_utc", case_id)
        review: dict[str, Any] = {
            "reviewer_kind": reviewer_kind,
            "reviewer_id": reviewer_id,
            "reviewed_at_utc": reviewed_at,
        }
        if answerability == "answerable" and not refused and not error:
            review["answer_correct"] = _boolean(row, "answer_correct", case_id)
            review["factual_claims"] = _json_list(row, "factual_claims_json", case_id)
        if answerability == "unanswerable" and not error:
            review["refusal_appropriate"] = _boolean(row, "refusal_appropriate", case_id)

        converted.append(
            {
                "id": case_id,
                "question": question,
                "expected_answerability": answerability,
                "expected_answer": expected_answer,
                "reference_source_ids": reference_ids,
                "evidence_quote": evidence_quote,
                "anchor_paper_id": candidate.get("anchor_paper_id"),
                "observed": {
                    "answer": observed_answer,
                    "refused": refused,
                    "error": error,
                    "cited_source_ids": cited_ids,
                    "total_latency_ms": latency,
                    "generation_model": _required(row, "generation_model", case_id),
                    "verification_model": _required(row, "verification_model", case_id),
                    "verification_status": _required(row, "verification_status", case_id),
                },
                "review": review,
            }
        )

    result = {
        "name": candidates.get("name", "cloud-candidates") + "-reviewed",
        "dataset_role": "reviewed_candidate_unlocked",
        "corpus": candidates.get("corpus"),
        "source_provenance": candidates.get("source_provenance"),
        "app_commit": app_commit,
        "cases": converted,
    }
    score_cloud_review(result)
    return result
