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
    groundedness = float(citation.valid and citation.uncited_claim_count == 0)
    return {
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
        "answer_relevance": round(answer_relevance, 4),
        "faithfulness_proxy": groundedness,
        "citation_precision": citation.citation_precision,
        "citation_coverage": citation.citation_coverage,
        "invalid_citation_rate": round(
            len(citation.invalid_indices)
            / max(1, len(citation.invalid_indices) + len(citation.cited_indices)),
            4,
        ),
        "refusal_correct": refusal_correct,
    }
