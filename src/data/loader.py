"""
Data loader module.

Loads papers from JSONL files and provides Paper dataclass.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.config import settings


@dataclass
class Paper:
    """Represents a single academic paper."""

    id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    category: str = ""
    year: Optional[int] = None

    @property
    def text(self) -> str:
        """Combined text field for embedding/search: title + abstract."""
        return f"{self.title}. {self.abstract}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "category": self.category,
            "year": self.year,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        """Create Paper from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            abstract=data["abstract"],
            authors=data.get("authors", []),
            category=data.get("category", ""),
            year=data.get("year"),
        )


def load_papers(path: Optional[Path] = None) -> List[Paper]:
    """
    Load papers from a JSONL file.

    Args:
        path: Path to the JSONL file. Defaults to data/raw/papers.jsonl.

    Returns:
        List of Paper objects.
    """
    if path is None:
        path = settings.data.raw_dir / "papers.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run 'python -m scripts.download_data' first."
        )

    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                papers.append(Paper.from_dict(data))

    print(f"📚 Loaded {len(papers)} papers from {path}")
    return papers
