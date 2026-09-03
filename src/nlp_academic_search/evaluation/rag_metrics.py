"""Deterministic RAG metrics independent of an LLM judge."""

from __future__ import annotations

import re

from nlp_academic_search.rag.citations import validate_citations


def evaluate_rag_case(case: dict, response: dict) -> dict[str, float | bool]:
    sources = response.get("sources", [])
    retrieved = {source.get("id") for source in sources}
    relevant = set(case.get("relevant_source_ids", []))
    overlap = retrieved & relevant
    context_precision = len(overlap) / len(retrieved) if retrieved else 0.0
    context_recall = len(overlap) / len(relevant) if relevant else 1.0
    answer = response.get("answer", "")
    words = set(re.findall(r"[\w-]+", answer.casefold()))
    expected = {word.casefold() for word in case.get("expected_keywords", [])}
    answer_relevance = len(words & expected) / len(expected) if expected else 1.0
    citation = validate_citations(answer, len(sources))
    refused = "not enough evidence" in answer.casefold()
    refusal_correct = refused == bool(case.get("should_refuse", False))
    semantic = (response.get("metadata") or {}).get("semantic_validation") or {}
    status = (response.get("metadata") or {}).get("answer_status")
    groundedness = float(citation.valid and citation.uncited_claim_count == 0)
    return {
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
        "answer_relevance": round(answer_relevance, 4),
        "faithfulness_proxy": groundedness,  # Deprecated structural proxy; not semantic entailment.
        "citation_precision": citation.citation_precision,
        "citation_coverage": citation.citation_coverage,
        "source_utilization": citation.source_utilization,
        "claim_citation_coverage": citation.claim_citation_coverage,
        "invalid_citation_rate": round(
            len(citation.invalid_indices)
            / max(1, len(citation.invalid_indices) + len(citation.cited_indices)),
            4,
        ),
        "refusal_correct": refusal_correct,
        "semantic_claim_coverage": float(semantic.get("semantic_claim_coverage", 0.0)),
        "unsupported_claim_count": float(semantic.get("unsupported_claim_count", 0)),
        "insufficient_claim_count": float(semantic.get("insufficient_claim_count", 0)),
        "evidence_quote_validity": float(semantic.get("evidence_quote_validity", 0.0)),
        "verified_answer": float(status == "verified"),
        "refused_verification": float(status == "refused_unverified"),
    }
