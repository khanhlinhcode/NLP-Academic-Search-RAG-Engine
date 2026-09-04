"""Strict semantic claim verification with deterministic evidence checks.

The LLM may classify support, but it never gets to invent evidence: every returned
claim and quote is checked locally against the answer and retrieved title/abstract.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.rag.citations import factual_sentences, segment_sentences

_CITATION = re.compile(r"\[((?:\d+\s*,\s*)*\d+)\]")


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_index: int = Field(ge=0)
    quote: str = Field(min_length=1, max_length=2000)


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_text: str = Field(min_length=1, max_length=6000)
    factual: bool
    cited_indices: list[int]
    verdict: Literal["supported", "unsupported", "insufficient"]
    evidence: list[EvidenceSpan]
    explanation: str = Field(max_length=1000)


class VerifierResponse(BaseModel):
    """The only JSON object accepted from a semantic-verification provider."""

    model_config = ConfigDict(extra="forbid")
    claims: list[ClaimAssessment]


class SemanticValidation(BaseModel):
    valid: bool
    total_factual_claims: int
    supported_claim_count: int
    unsupported_claim_count: int
    insufficient_claim_count: int
    semantic_claim_coverage: float
    evidence_quote_validity: float
    invalid_evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    claims: list[ClaimAssessment] = Field(default_factory=list)
    verifier_provider: str
    verifier_model: str
    verifier_independent: bool
    warnings: list[str] = Field(default_factory=list)


def normalize_evidence_text(value: str) -> str:
    """Lossless-enough comparison normalizer: no fuzzy or embedding matching."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def citation_indices(text: str) -> list[int]:
    return [int(part) for group in _CITATION.findall(text) for part in re.split(r"\s*,\s*", group)]


def _quote_exists(paper: Paper, quote: str) -> bool:
    normalized_quote = normalize_evidence_text(quote)
    return bool(normalized_quote) and any(
        normalized_quote in normalize_evidence_text(value)
        for value in (paper.title, paper.abstract)
    )


def _assessment_claim_is_in_answer(claim: ClaimAssessment, answer: str) -> bool:
    return claim.claim_text in answer


def _containing_sentence(claim_text: str, sentences: list[str]) -> str | None:
    return next((sentence for sentence in sentences if claim_text in sentence), None)


def validate_semantic_assessment(
    answer: str,
    papers: list[Paper],
    assessments: list[ClaimAssessment],
    *,
    provider: str,
    model: str,
    independent: bool,
) -> SemanticValidation:
    """Validate LLM assessments deterministically before accepting semantic support."""
    invalid_spans: list[EvidenceSpan] = []
    checked: list[ClaimAssessment] = []
    answer_claim_sentences = factual_sentences(answer)
    factual_claims = 0
    supported = 0
    unsupported = 0
    insufficient = 0
    valid_quotes = 0
    total_quotes = 0
    covered_sentences: set[str] = set()
    for claim in assessments:
        containing_sentence = _containing_sentence(claim.claim_text, answer_claim_sentences)
        if not _assessment_claim_is_in_answer(claim, answer) or (
            claim.factual and containing_sentence is None
        ):
            invalid_spans.extend(claim.evidence)
            if claim.factual:
                factual_claims += 1
                unsupported += 1
                checked.append(claim.model_copy(update={"verdict": "unsupported"}))
            else:
                checked.append(claim)
            continue
        if not claim.factual:
            checked.append(claim)
            continue
        factual_claims += 1
        if containing_sentence:
            covered_sentences.add(containing_sentence)
        cited = set(claim.cited_indices)
        sentence_citations = set(citation_indices(containing_sentence or claim.claim_text))
        evidence_sources: set[int] = set()
        claim_has_invalid_evidence = False
        for span in claim.evidence:
            total_quotes += 1
            # EvidenceSpan is zero-indexed; user-facing citations are one-indexed.
            within_range = 0 <= span.source_index < len(papers)
            citation_index = span.source_index + 1
            quote_in_source = within_range and _quote_exists(papers[span.source_index], span.quote)
            citation_matches = citation_index in cited and citation_index in sentence_citations
            if within_range and quote_in_source and citation_matches:
                valid_quotes += 1
                evidence_sources.add(citation_index)
            else:
                invalid_spans.append(span)
                claim_has_invalid_evidence = True
        citations_supported = bool(cited) and cited == evidence_sources
        if claim.verdict == "supported" and citations_supported and not claim_has_invalid_evidence:
            supported += 1
        elif claim.verdict == "unsupported":
            unsupported += 1
        elif claim.verdict == "insufficient" and not claim_has_invalid_evidence:
            insufficient += 1
        else:
            unsupported += 1
            claim = claim.model_copy(update={"verdict": "unsupported"})
        checked.append(claim)
    missing_sentences = set(answer_claim_sentences) - covered_sentences
    if missing_sentences:
        unsupported += len(missing_sentences)
    total_factual = factual_claims + len(missing_sentences)
    coverage = supported / total_factual if total_factual else 0.0
    quote_validity = valid_quotes / total_quotes if total_quotes else 0.0
    warnings: list[str] = []
    if invalid_spans:
        warnings.append("Verifier evidence spans did not exactly match the cited retrieved source.")
    if factual_claims == 0:
        warnings.append(
            "Semantic verifier found no factual claim eligible for evidence verification."
        )
    if unsupported or insufficient:
        warnings.append(
            "One or more factual claims were not semantically supported by verified evidence."
        )
    if missing_sentences:
        warnings.append(
            "One or more factual answer sentences were omitted by the semantic verifier."
        )
    return SemanticValidation(
        valid=(
            bool(total_factual)
            and supported == total_factual
            and not unsupported
            and not insufficient
            and not invalid_spans
            and not missing_sentences
            and quote_validity == 1.0
        ),
        total_factual_claims=total_factual,
        supported_claim_count=supported,
        unsupported_claim_count=unsupported,
        insufficient_claim_count=insufficient,
        semantic_claim_coverage=round(coverage, 4),
        evidence_quote_validity=round(quote_validity, 4),
        invalid_evidence_spans=invalid_spans,
        claims=checked,
        verifier_provider=provider,
        verifier_model=model,
        verifier_independent=independent,
        warnings=warnings,
    )


def fallback_claims(answer: str) -> list[ClaimAssessment]:
    """A deterministic parser used only by test/offline verifier implementations."""
    return [
        ClaimAssessment(
            claim_text=sentence,
            factual=not sentence.rstrip().endswith("?"),
            cited_indices=citation_indices(sentence),
            verdict="insufficient",
            evidence=[],
            explanation="Deterministic fallback; semantic support was not assessed.",
        )
        for sentence in segment_sentences(answer)
        if sentence.strip()
    ]
