"""Shared Unicode-safe preprocessing for indexing and queries."""

from __future__ import annotations

import re
import unicodedata

_LATEX_COMMAND = re.compile(r"\\(?:text|mathrm|mathbf|mathit|emph)\{([^{}]*)\}")
_LATEX_OTHER = re.compile(r"\\[A-Za-z]+(?:\[[^\]]*\])?")
_CITATION = re.compile(r"\[(?:\d+\s*,?\s*)+\]")
_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|[A-Za-z]+\d+[A-Za-z0-9-]*", re.UNICODE)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _LATEX_COMMAND.sub(r"\1", text)
    text = _LATEX_OTHER.sub(" ", text)
    text = text.replace("$", " ")
    text = _CITATION.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = clean_text(text).casefold()
    return [token for token in _TOKEN.findall(normalized) if token not in STOP_WORDS]


def prepare_text_for_embedding(title: str, abstract: str) -> str:
    clean_title = clean_text(title)
    clean_abstract = clean_text(abstract)
    return f"{clean_title}. {clean_abstract}".strip()
