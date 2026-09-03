from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from nlp_academic_search.api.services import ServiceContainer
from nlp_academic_search.data.loader import Paper
from nlp_academic_search.search.bm25_search import BM25Searcher
from nlp_academic_search.search.hybrid_search import HybridSearcher
from nlp_academic_search.search.semantic_search import SemanticSearcher


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            lowered = text.casefold()
            row = np.array(
                [
                    1.0 if "attention" in lowered or "transformer" in lowered else 0.05,
                    1.0 if "retrieval" in lowered or "evidence" in lowered else 0.05,
                    1.0 if "vision" in lowered or "image" in lowered else 0.05,
                ],
                dtype=np.float32,
            )
            rows.append(row / np.linalg.norm(row))
        return np.vstack(rows)


class FakeRAGGenerator:
    model_name = "fake-grounded-model"

    def generate(self, messages, temperature=0.2):
        assert messages[0]["role"] == "system"
        return "The Transformer uses attention instead of recurrence [1]."

    def generate_stream(self, messages, temperature=0.2):
        assert messages[0]["role"] == "system"
        yield "Grounded answer "
        yield "[1]."

    async def generate_stream_async(self, messages, temperature=0.2):
        assert messages[0]["role"] == "system"
        yield "Grounded answer "
        yield "[1]."


@pytest.fixture
def papers() -> list[Paper]:
    return [
        Paper(
            id="arxiv:1706.03762",
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="A Transformer uses attention instead of recurrence.",
            authors=["Ashish Vaswani"],
            categories=["cs.CL"],
            published_at=datetime(2017, 6, 12, tzinfo=UTC),
            source="arxiv",
        ),
        Paper(
            id="arxiv:2005.11401",
            arxiv_id="2005.11401",
            title="Retrieval-Augmented Generation",
            abstract="A retriever supplies evidence to a neural generator.",
            authors=["Patrick Lewis"],
            categories=["cs.CL"],
            published_at=datetime(2020, 5, 22, tzinfo=UTC),
            source="arxiv",
        ),
        Paper(
            id="local:vision",
            title="A Vision Model",
            abstract="Image recognition with a convolutional network.",
            authors=[],
            categories=["cs.CV"],
            source="test-fixture",
        ),
    ]


@pytest.fixture
def corpus_path(tmp_path: Path, papers: list[Paper]) -> Path:
    path = tmp_path / "papers.jsonl"
    path.write_text(
        "".join(json.dumps(paper.to_dict()) + "\n" for paper in papers), encoding="utf-8"
    )
    return path


@pytest.fixture
def semantic(papers: list[Paper], corpus_path: Path) -> SemanticSearcher:
    return SemanticSearcher(
        papers,
        model_name="fake-embedding",
        model_revision="fixture-v1",
        load_existing=False,
        model=FakeEmbeddingModel(),
        corpus_path=corpus_path,
    )


@pytest.fixture
def services(papers: list[Paper], semantic: SemanticSearcher) -> ServiceContainer:
    bm25 = BM25Searcher(papers)
    return ServiceContainer(
        papers=papers,
        bm25=bm25,
        semantic=semantic,
        hybrid=HybridSearcher(bm25, semantic),
        rag_generator=FakeRAGGenerator(),  # type: ignore[arg-type]
    )
