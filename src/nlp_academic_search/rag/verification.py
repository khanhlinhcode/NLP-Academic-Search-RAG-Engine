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
from nlp_academic_search.rag.citations import segment_sentences

_CITATION = re.compile(r"\[((?:\d+\s*,\s*)*\d+)\]")


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_index: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=2000)


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_text: str = Field(min_length=1, max_length=6000)
    factual: bool
    cited_indices: list[int] = Field(default_factory=list)
    verdict: Literal["supported", "unsupported", "insufficient"]
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    explanation: str = Field(default="", max_length=1000)


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


def _source_text(paper: Paper) -> str:
    return normalize_evidence_text(f"{paper.title}\n{paper.abstract}")


def _assessment_claim_is_in_answer(claim: ClaimAssessment, answer: str) -> bool:
    return normalize_evidence_text(claim.claim_text) in normalize_evidence_text(answer)


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
    answer_citations = set(citation_indices(answer))
    factual_claims = 0
    supported = 0
    unsupported = 0
    insufficient = 0
    valid_quotes = 0
    total_quotes = 0
    for claim in assessments:
        if not _assessment_claim_is_in_answer(claim, answer):
            invalid_spans.extend(claim.evidence)
            checked.append(claim)
            continue
        if not claim.factual:
            checked.append(claim)
            continue
        factual_claims += 1
        cited = set(claim.cited_indices)
        claim_citations = set(citation_indices(claim.claim_text))
        if not cited:
            cited = claim_citations
        evidence_ok = False
        for span in claim.evidence:
            total_quotes += 1
            within_range = 1 <= span.source_index <= len(papers)
            quote_in_source = within_range and normalize_evidence_text(span.quote) in _source_text(
                papers[span.source_index - 1]
            )
            citation_matches = span.source_index in cited and span.source_index in answer_citations
            if within_range and quote_in_source and citation_matches:
                valid_quotes += 1
                evidence_ok = True
            else:
                invalid_spans.append(span)
        if claim.verdict == "supported" and evidence_ok and cited:
            supported += 1
        elif claim.verdict == "unsupported":
            unsupported += 1
        else:
            insufficient += 1
        checked.append(claim)
    coverage = supported / factual_claims if factual_claims else 0.0
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
    return SemanticValidation(
        valid=bool(factual_claims) and supported == factual_claims and not invalid_spans,
        total_factual_claims=factual_claims,
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
        )
        for sentence in segment_sentences(answer)
        if sentence.strip()
    ]
