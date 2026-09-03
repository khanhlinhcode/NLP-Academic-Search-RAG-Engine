from pathlib import Path

import pytest

from nlp_academic_search.search.hybrid_search import FusionMethod
from nlp_academic_search.search.index_manifest import IndexCompatibilityError, IndexManifest
from nlp_academic_search.search.models import SearchFilters, SearchResult
from nlp_academic_search.search.reranker import Reranker
from nlp_academic_search.search.semantic_search import SemanticSearcher


def test_semantic_search_uses_mocked_embeddings(semantic):
    results = semantic.search("attention transformer", top_k=1)
    assert results[0].paper.arxiv_id == "1706.03762"
    assert results[0].score_type == "semantic_score"


def test_semantic_index_round_trip(tmp_path: Path, papers, corpus_path, semantic):
    index_dir = tmp_path / "active-index"
    index_dir.mkdir()
    import faiss
    import numpy as np

    np.save(index_dir / "paper_embeddings.npy", semantic.embeddings)
    faiss.write_index(semantic.index, str(index_dir / "faiss.index"))
    (index_dir / "index_manifest.json").write_text(
        semantic.manifest.model_dump_json(), encoding="utf-8"
    )
    loaded = SemanticSearcher(
        papers,
        model_name="fake-embedding",
        model_revision="fixture-v1",
        model=semantic.model,
        index_dir=index_dir,
        corpus_path=corpus_path,
    )
    assert loaded.index.ntotal == 3


def test_manifest_rejects_order_mismatch(semantic, papers, corpus_path):
    with pytest.raises(IndexCompatibilityError, match="ordered document-ID"):
        semantic.manifest.validate(
            corpus_path=corpus_path,
            papers=list(reversed(papers)),
            model_name="fake-embedding",
            model_revision="fixture-v1",
            embeddings=semantic.embeddings,
            index=semantic.index,
        )


def test_index_manifest_rejects_invalid_json(tmp_path):
    path = tmp_path / "index_manifest.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(IndexCompatibilityError):
        IndexManifest.load(path)


def test_rrf_weighted_filters_and_candidate_pool(services):
    rrf = services.hybrid.search("attention", top_k=2)
    weighted = services.hybrid.search("retrieval", top_k=2, method=FusionMethod.WEIGHTED)
    filtered = services.hybrid.search("model", top_k=2, filters=SearchFilters(category="cs.CV"))
    assert rrf[0].rrf_score is not None
    assert weighted[0].weighted_score is not None
    assert [item.paper.category for item in filtered] == ["cs.CV"]
    with pytest.raises(ValueError, match="candidate_pool"):
        services.hybrid.search("x", top_k=5, candidate_pool=2)


class FakeCrossEncoder:
    def predict(self, pairs, **_kwargs):
        return [0.1, 0.9]


def test_reranker_preserves_component_scores(papers):
    candidates = [
        SearchResult(paper=papers[0], rrf_score=0.03),
        SearchResult(paper=papers[1], rrf_score=0.02),
    ]
    reranked = Reranker(model=FakeCrossEncoder()).rerank("retrieval", candidates, top_k=1)
    assert reranked[0].paper == papers[1]
    assert reranked[0].reranker_score == pytest.approx(0.9)
