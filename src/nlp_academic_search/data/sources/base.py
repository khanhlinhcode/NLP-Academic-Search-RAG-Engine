"""Source adapter protocol."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from nlp_academic_search.data.loader import Paper


class PaperSource(Protocol):
    name: str

    def iter_papers(self, *, max_records: int | None = None) -> Iterator[Paper]: ...
