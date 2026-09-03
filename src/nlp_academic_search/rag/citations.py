"""Deterministic validation of numbered RAG citations."""

from __future__ import annotations

import re

from pydantic import BaseModel

_CITATION = re.compile(r"\[((?:\d+\s*,\s*)*\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class CitationValidation(BaseModel):
    valid: bool
    cited_indices: list[int]
    invalid_indices: list[int]
    citation_precision: float
    citation_coverage: float
    uncited_claim_count: int
    warnings: list[str]


def validate_citations(answer: str, source_count: int) -> CitationValidation:
    mentioned = [
        int(item) for match in _CITATION.findall(answer) for item in re.split(r"\s*,\s*", match)
    ]
    valid_mentions = [index for index in mentioned if 1 <= index <= source_count]
    invalid = sorted(set(mentioned) - set(valid_mentions))
    cited = sorted(set(valid_mentions))
    precision = len(valid_mentions) / len(mentioned) if mentioned else 0.0
    coverage = len(cited) / source_count if source_count else 0.0
    refusal = any(
        phrase in answer.casefold()
        for phrase in ("not enough evidence", "insufficient evidence", "không đủ bằng chứng")
    )
    uncited_claims = 0
    if not refusal:
        for sentence in _SENTENCE.split(answer):
            if len(sentence.split()) >= 8 and not _CITATION.search(sentence):
                uncited_claims += 1
    warnings = []
    if invalid:
        warnings.append(f"Invalid citation indices: {invalid}")
    if source_count and not cited and not refusal:
        warnings.append("The answer contains no valid source citation.")
    if uncited_claims:
        warnings.append(f"Potential factual sentences without citations: {uncited_claims}")
    return CitationValidation(
        valid=not invalid and (bool(cited) or refusal or source_count == 0),
        cited_indices=cited,
        invalid_indices=invalid,
        citation_precision=round(precision, 4),
        citation_coverage=round(coverage, 4),
        uncited_claim_count=uncited_claims,
        warnings=warnings,
    )
