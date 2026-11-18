"""Score independently reviewed answers from a cloud RAG evaluation set."""

from __future__ import annotations

from statistics import median
from typing import Any


def _percentile(values: list[float], fraction: float) -> float | None:
    if len(values) < 5:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def score_cloud_review(payload: dict[str, Any]) -> dict[str, Any]:
    """Score explicit reviewer labels; never infer correctness from API verifier metadata."""
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    answerable = 0
    grounded = 0
    false_refusals = 0
    unanswerable = 0
    correct_refusals = 0
    factual_claims = 0
    supported_claims = 0
    errors = 0
    latencies: list[float] = []
    reviewer_kinds: set[str] = set()
    seen_ids: set[str] = set()

    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("every case must have a string id")
        if case["id"] in seen_ids:
            raise ValueError(f"duplicate case id: {case['id']}")
        seen_ids.add(case["id"])
        answerability = case.get("expected_answerability")
        if answerability not in {"answerable", "unanswerable"}:
            raise ValueError(f"invalid expected_answerability in {case['id']}")
        observed = case.get("observed")
        review = case.get("review")
        if not isinstance(observed, dict) or not isinstance(review, dict):
            raise ValueError(f"missing observed response or review in {case['id']}")
        if not isinstance(observed.get("refused"), bool):
            raise ValueError(f"refused must be a boolean in {case['id']}")
        if not isinstance(observed.get("error", False), bool):
            raise ValueError(f"error must be a boolean in {case['id']}")
        reviewer_kind = review.get("reviewer_kind")
        if reviewer_kind not in {"human", "agent_pilot"}:
            raise ValueError(f"invalid reviewer_kind in {case['id']}")
        reviewer_kinds.add(reviewer_kind)
        latency = observed.get("total_latency_ms")
        if latency is not None:
            if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0:
                raise ValueError(f"invalid total_latency_ms in {case['id']}")
            if not observed.get("error", False):
                latencies.append(float(latency))

        if observed.get("error", False):
            errors += 1
            if answerability == "answerable":
                answerable += 1
            else:
                unanswerable += 1
            continue

        if answerability == "answerable":
            answerable += 1
            if observed["refused"]:
                false_refusals += 1
            else:
                if not isinstance(review.get("answer_correct"), bool):
                    raise ValueError(f"missing answer_correct label in {case['id']}")
                claims = review.get("factual_claims")
                if not isinstance(claims, list) or not claims:
                    raise ValueError(f"factual_claims must be non-empty in {case['id']}")
                if not all(
                    isinstance(claim, dict)
                    and isinstance(claim.get("claim"), str)
                    and bool(claim["claim"].strip())
                    and isinstance(claim.get("supported"), bool)
                    and isinstance(claim.get("citation_correct"), bool)
                    for claim in claims
                ):
                    raise ValueError(f"invalid factual_claims labels in {case['id']}")
                factual_claims += len(claims)
                supported_claims += sum(
                    claim["supported"] and claim["citation_correct"] for claim in claims
                )
                if review["answer_correct"] and all(
                    claim["supported"] and claim["citation_correct"] for claim in claims
                ):
                    grounded += 1
        else:
            unanswerable += 1
            if not isinstance(review.get("refusal_appropriate"), bool):
                raise ValueError(f"missing refusal_appropriate label in {case['id']}")
            if observed["refused"] and review["refusal_appropriate"]:
                correct_refusals += 1

    return {
        "benchmark": payload.get("name"),
        "dataset_role": payload.get("dataset_role"),
        "corpus": payload.get("corpus"),
        "app_commit": payload.get("app_commit"),
        "reviewer_kinds": sorted(reviewer_kinds),
        "sample_size": len(cases),
        "error_rate": {
            "failed": errors,
            "total": len(cases),
            "rate": round(errors / len(cases), 4),
        },
        "grounded_answer_accuracy": {
            "passed": grounded,
            "total": answerable,
            "rate": round(grounded / answerable, 4) if answerable else None,
        },
        "correct_refusal_rate": {
            "passed": correct_refusals,
            "total": unanswerable,
            "rate": round(correct_refusals / unanswerable, 4) if unanswerable else None,
        },
        "false_refusal_rate": {
            "failed": false_refusals,
            "total": answerable,
            "rate": round(false_refusals / answerable, 4) if answerable else None,
        },
        "supported_claim_rate": {
            "passed": supported_claims,
            "total": factual_claims,
            "rate": round(supported_claims / factual_claims, 4) if factual_claims else None,
        },
        "total_latency_ms": {
            "measured": len(latencies),
            "p50": round(median(latencies), 2) if len(latencies) >= 5 else None,
            "p95": _percentile(latencies, 0.95),
        },
        "minimum_recorded_requirements_met": (
            payload.get("dataset_role") == "locked_holdout"
            and len(cases) >= 100
            and answerable >= 20
            and unanswerable >= 20
            and reviewer_kinds == {"human"}
        ),
        "limitations": [
            "Scoring uses reviewer labels rather than API verification; label independence requires audit.",
            "Rates from a small or agent-reviewed pilot must not be used as CV quality claims.",
            "Minimum recorded requirements do not replace an audit of label quality or sampling.",
        ],
    }
