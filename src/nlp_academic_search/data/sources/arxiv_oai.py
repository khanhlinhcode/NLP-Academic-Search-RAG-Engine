"""Rate-limited arXiv OAI-PMH metadata ingestion."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import httpx

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.data.manifest import atomic_write_text

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
ARXIV_NS = "http://arxiv.org/OAI/arXiv/"
DEFAULT_ARXIV_OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"


class ArxivOAIAdapter:
    """Fetch verifiable bibliographic metadata from the official arXiv endpoint."""

    name = "arxiv-oai-pmh"

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_ARXIV_OAI_ENDPOINT,
        set_spec: str = "cs",
        checkpoint_path: Path | None = None,
        request_interval_seconds: float = 3.0,
        timeout_seconds: float = 45.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.set_spec = set_spec
        self.checkpoint_path = checkpoint_path
        self.request_interval_seconds = request_interval_seconds
        self.max_retries = max_retries
        self.quarantined: list[dict[str, str]] = []
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "nlp-academic-search/1.1 (metadata research tool)"},
        )

    def _request(self, params: dict[str, str]) -> ET.Element:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.get(self.endpoint, params=params)
                response.raise_for_status()
                return ET.fromstring(response.content)
            except (httpx.HTTPError, ET.ParseError) as exc:
                last_error = exc
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"arXiv OAI-PMH request failed after {self.max_retries} attempts"
        ) from last_error

    def _resume_token(self) -> str | None:
        if self.checkpoint_path and self.checkpoint_path.is_file():
            return self.checkpoint_path.read_text(encoding="utf-8").strip() or None
        return None

    def iter_papers(self, *, max_records: int | None = None) -> Iterator[Paper]:
        emitted = 0
        token = self._resume_token()
        while True:
            params = (
                {"verb": "ListRecords", "resumptionToken": token}
                if token
                else {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": self.set_spec}
            )
            root = self._request(params)
            for record in root.findall(f".//{{{OAI_NS}}}record"):
                header = record.find(f"{{{OAI_NS}}}header")
                metadata = record.find(f"{{{OAI_NS}}}metadata/{{{ARXIV_NS}}}arXiv")
                if header is None or metadata is None or header.get("status") == "deleted":
                    continue
                try:
                    paper = self._parse(metadata)
                except ValueError as exc:
                    identifier = self._text(metadata, "id") or "unknown"
                    self.quarantined.append({"id": identifier, "reason": str(exc)})
                    continue
                yield paper
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return
            token_element = root.find(f".//{{{OAI_NS}}}resumptionToken")
            token = (
                token_element.text.strip()
                if token_element is not None and token_element.text
                else None
            )
            if self.checkpoint_path:
                atomic_write_text(self.checkpoint_path, f"{token or ''}\n")
            if not token:
                return
            time.sleep(self.request_interval_seconds)

    @staticmethod
    def _text(element: ET.Element, name: str) -> str | None:
        child = element.find(f"{{{ARXIV_NS}}}{name}")
        return child.text.strip() if child is not None and child.text else None

    def _parse(self, metadata: ET.Element) -> Paper:
        arxiv_id = self._text(metadata, "id")
        title = self._text(metadata, "title")
        abstract = self._text(metadata, "abstract")
        if not arxiv_id or not title or not abstract:
            raise ValueError("arXiv record is missing id, title, or abstract")
        authors = []
        for author in metadata.findall(f".//{{{ARXIV_NS}}}author"):
            keyname = self._text(author, "keyname")
            forenames = self._text(author, "forenames")
            name = " ".join(part for part in (forenames, keyname) if part)
            if name:
                authors.append(name)
        categories = (self._text(metadata, "categories") or "").split()
        created = self._text(metadata, "created")
        updated = self._text(metadata, "updated")
        doi = self._text(metadata, "doi")
        license_url = self._text(metadata, "license")
        return Paper(
            id=f"arxiv:{arxiv_id}",
            arxiv_id=arxiv_id,
            doi=doi,
            title=" ".join(title.split()),
            abstract=" ".join(abstract.split()),
            authors=authors,
            categories=categories,
            published_at=datetime.fromisoformat(created) if created else None,
            updated_at=datetime.fromisoformat(updated) if updated else None,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            source=self.name,
            license=license_url,
        )
