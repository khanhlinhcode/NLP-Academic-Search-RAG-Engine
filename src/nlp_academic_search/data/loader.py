"""Validated academic-paper schema and JSONL corpus loader."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nlp_academic_search.config import settings

ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.I)
DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.I)
SCHEMA_VERSION = "1.0"


class CorpusValidationError(ValueError):
    """A corpus record is malformed or violates the paper schema."""


def content_hash(title: str, abstract: str) -> str:
    normalized = " ".join(f"{title}\n{abstract}".casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Paper(BaseModel):
    """A validated academic paper with provenance-safe optional identifiers."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=256)
    arxiv_id: str | None = None
    doi: str | None = None
    title: str = Field(min_length=1, max_length=2000)
    abstract: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    source_url: str | None = None
    pdf_url: str | None = None
    source: str = "unknown"
    license: str | None = None
    content_hash: str = ""
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        category = data.pop("category", None)
        data.pop("year", None)
        if category and "categories" not in data:
            data["categories"] = [category]
        paper_id = str(data.get("id", ""))
        if paper_id.startswith("paper_") and "arxiv_id" not in data:
            data["authors"] = []
            data["categories"] = []
            data["source"] = "legacy-ccdv-arxiv-summarization"
            data["source_url"] = None
            data["pdf_url"] = None
        return data

    @field_validator("arxiv_id")
    @classmethod
    def validate_arxiv_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.removeprefix("arXiv:").strip()
        if not ARXIV_ID_RE.fullmatch(normalized):
            raise ValueError("must be a valid arXiv identifier")
        return normalized

    @field_validator("doi")
    @classmethod
    def validate_doi(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        if not DOI_RE.fullmatch(normalized):
            raise ValueError("must be a valid DOI")
        return normalized.lower()

    @field_validator("authors", "categories")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def complete_and_check_provenance(self) -> Paper:
        if not self.content_hash:
            self.content_hash = content_hash(self.title, self.abstract)
        if self.arxiv_id:
            self.source_url = self.source_url or f"https://arxiv.org/abs/{self.arxiv_id}"
            self.pdf_url = self.pdf_url or f"https://arxiv.org/pdf/{self.arxiv_id}"
        return self

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}"

    @property
    def category(self) -> str:
        return self.categories[0] if self.categories else ""

    @property
    def year(self) -> int | None:
        return self.published_at.year if self.published_at else None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        return cls.model_validate(data)


def active_corpus_path(raw_dir: Path | None = None) -> Path:
    """Resolve a version pointer without allowing it to escape the data directory."""
    root = (raw_dir or settings.data.raw_dir).resolve()
    pointer = root / "CURRENT"
    if pointer.is_file():
        version = pointer.read_text(encoding="utf-8").strip()
        candidate = (root / "versions" / version / "papers.jsonl").resolve()
        versions_root = (root / "versions").resolve()
        if candidate.is_relative_to(versions_root) and candidate.is_file():
            return candidate
    return root / "papers.jsonl"


def iter_papers(path: Path) -> Iterator[Paper]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield Paper.from_dict(payload)
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
                raise CorpusValidationError(f"{path.name}:{line_number}: {exc}") from exc


def load_papers(path: Path | None = None) -> list[Paper]:
    path = path or active_corpus_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run 'make download' or set DATA_RAW_DIR."
        )
    papers = list(iter_papers(path))
    ids = [paper.id for paper in papers]
    if len(ids) != len(set(ids)):
        raise CorpusValidationError(f"{path}: duplicate paper IDs are not allowed")
    return papers
