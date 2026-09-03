"""Adapter that keeps the existing Ollama implementation behind a provider contract."""

from __future__ import annotations

from typing import Any


def build_ollama_provider(**kwargs: Any) -> Any:
    # Importing this module is restricted to the local provider factory path.
    from nlp_academic_search.rag.generator import RAGGenerator

    return RAGGenerator(**kwargs)
