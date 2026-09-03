"""Deterministic, sentence-scoped validation of numbered RAG citations."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_CITATION = re.compile(r"\[((?:\d+\s*,\s*)*\d+)\]")
_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_PRIVATE_DOT = "\ue000"
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "fig.",
    "eq.",
    "sec.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "vs.",
    "etc.",
)
_REFUSAL_SENTENCES = (
    "not enough evidence in the retrieved sources",
    "not enough verified evidence in the retrieved sources",
    "insufficient evidence in the retrieved sources",
    "không đủ bằng chứng trong các nguồn đã truy xuất",
    "không đủ bằng chứng để trả lời",
)


class CitationValidation(BaseModel):
    valid: bool
    cited_indices: list[int]
    invalid_indices: list[int]
    citation_precision: float
    citation_coverage: float = Field(
        description="Deprecated compatibility alias for source_utilization."
    )
    source_utilization: float = Field(
        description="Fraction of retrieved sources referenced at least once."
    )
    claim_citation_coverage: float = Field(
        description="Fraction of detected factual sentences containing a citation."
    )
    uncited_claim_count: int
    warnings: list[str]


def _protect_periods(text: str) -> str:
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = re.sub(
            rf"(?<!\w){re.escape(abbreviation)}",
            lambda match: match.group(0).replace(".", _PRIVATE_DOT),
            protected,
            flags=re.IGNORECASE,
        )
    return re.sub(r"(?<=\d)\.(?=\d)", _PRIVATE_DOT, protected)


def segment_sentences(answer: str) -> list[str]:
    """Split prose and Markdown list items without adding an NLP dependency.

    Newlines are explicit boundaries because generated answers often use
    Markdown bullets without terminal punctuation. Common academic
    abbreviations and decimal points are protected before punctuation splitting.
    """

    sentences: list[str] = []
    normalized = answer.replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        protected = _protect_periods(line)
        for part in re.split(r"(?<=[.!?])\s+", protected):
            restored = part.replace(_PRIVATE_DOT, ".").strip()
            if restored:
                sentences.append(restored)
    return sentences


def is_refusal(answer: str) -> bool:
    content = [sentence for sentence in segment_sentences(answer) if not _HEADING.match(sentence)]
    if len(content) != 1:
        return False
    normalized = " ".join(_LIST_MARKER.sub("", content[0]).casefold().split())
    normalized = normalized.rstrip(".!?")
    return normalized in _REFUSAL_SENTENCES


def _is_factual_sentence(sentence: str) -> bool:
    if _HEADING.match(sentence):
        return False
    content = _LIST_MARKER.sub("", sentence).strip()
    if not content or content.endswith("?"):
        return False
    without_citations = _CITATION.sub("", content)
    without_markup = re.sub(r"[`*_>#]", "", without_citations).strip()
    words = _WORD.findall(without_markup)
    if len(words) < 3:
        return False
    if without_markup.endswith(":") and len(words) <= 6:
        return False
    return True


def factual_sentences(answer: str) -> list[str]:
    """Return sentence-scoped claims that require evidence validation."""
    if is_refusal(answer):
        return []
    return [sentence for sentence in segment_sentences(answer) if _is_factual_sentence(sentence)]


def validate_citations(answer: str, source_count: int) -> CitationValidation:
    if source_count < 0:
        raise ValueError("source_count must not be negative")

    mentioned = [
        int(item) for match in _CITATION.findall(answer) for item in re.split(r"\s*,\s*", match)
    ]
    valid_mentions = [index for index in mentioned if 1 <= index <= source_count]
    invalid = sorted(set(mentioned) - set(valid_mentions))
    cited = sorted(set(valid_mentions))
    precision = len(valid_mentions) / len(mentioned) if mentioned else 0.0
    source_utilization = len(cited) / source_count if source_count else 0.0
    refusal = is_refusal(answer)
    claim_sentences = factual_sentences(answer)
    cited_claims = sum(bool(_CITATION.search(sentence)) for sentence in claim_sentences)
    uncited_claims = 0 if refusal else len(claim_sentences) - cited_claims
    claim_coverage = (
        1.0 if refusal else cited_claims / len(claim_sentences) if claim_sentences else 0.0
    )
    warnings = []
    if invalid:
        warnings.append(f"Invalid citation indices: {invalid}")
    if source_count and not cited and not refusal:
        warnings.append("The answer contains no valid source citation.")
    if uncited_claims:
        warnings.append(f"Potential factual sentences without citations: {uncited_claims}")
    return CitationValidation(
        valid=not invalid and (refusal or (bool(cited) and uncited_claims == 0)),
        cited_indices=cited,
        invalid_indices=invalid,
        citation_precision=round(precision, 4),
        citation_coverage=round(source_utilization, 4),
        source_utilization=round(source_utilization, 4),
        claim_citation_coverage=round(claim_coverage, 4),
        uncited_claim_count=uncited_claims,
        warnings=warnings,
    )
