"""Unit tests for legacy index adoption and manifest compatibility validation."""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper
from nlp_academic_search.search import indexing
from nlp_academic_search.search.index_manifest import IndexCompatibilityError, IndexManifest
from nlp_academic_search.search.indexing import adopt_existing_index


def create_fake_corpus_and_index(tmp_path: Path, paper_count: int = 3, dim: int = 4):
    raw_dir = tmp_path / "raw"
    emb_dir = tmp_path / "embeddings"
    raw_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    papers = [
        Paper(
            id=f"paper_{i:04d}",
            title=f"Paper Title {i}",
            abstract=f"Abstract content for paper {i}.",
            authors=["Author A"],
            categories=["cs.AI"],
        )
        for i in range(paper_count)
    ]
    corpus_file = raw_dir / "papers.jsonl"
    corpus_lines = [json.dumps(p.to_dict()) for p in papers]
    corpus_file.write_text("\n".join(corpus_lines), encoding="utf-8")

    emb_matrix = np.random.randn(paper_count, dim).astype(np.float32)
    # Normalize
    faiss.normalize_L2(emb_matrix)
    np.save(emb_dir / "paper_embeddings.npy", emb_matrix)

    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    faiss.write_index(index, str(emb_dir / "faiss.index"))

    return raw_dir, emb_dir, corpus_file, papers, emb_matrix, index


def test_legacy_adoption_happy_path(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, emb_matrix, index = create_fake_corpus_and_index(
        tmp_path
    )

    manifest_path = adopt_existing_index(
        embeddings_dir=emb_dir,
        raw_dir=raw_dir,
        papers_list=papers,
        corpus_file_path=corpus_file,
    )

    assert manifest_path.is_file()
    manifest = IndexManifest.load(manifest_path)
    assert manifest.provenance == "legacy-adopted-unverified-model-weights"
    assert manifest.document_count == len(papers)
    assert manifest.embedding_dimension == 4
    assert (raw_dir / "corpus_manifest.json").is_file()

    # Validate against actual assets
    manifest.validate(
        corpus_path=corpus_file,
        papers=papers,
        model_name=settings.embedding.model_name,
        model_revision=settings.embedding.model_revision,
        embeddings=emb_matrix,
        index=index,
    )


def test_legacy_adoption_missing_embeddings(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, _, _ = create_fake_corpus_and_index(tmp_path)
    (emb_dir / "paper_embeddings.npy").unlink()

    with pytest.raises(FileNotFoundError, match="Legacy embeddings/faiss.index are missing"):
        adopt_existing_index(
            embeddings_dir=emb_dir,
            raw_dir=raw_dir,
            papers_list=papers,
            corpus_file_path=corpus_file,
        )


def test_legacy_adoption_missing_faiss_index(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, _, _ = create_fake_corpus_and_index(tmp_path)
    (emb_dir / "faiss.index").unlink()

    with pytest.raises(FileNotFoundError, match="Legacy embeddings/faiss.index are missing"):
        adopt_existing_index(
            embeddings_dir=emb_dir,
            raw_dir=raw_dir,
            papers_list=papers,
            corpus_file_path=corpus_file,
        )


def test_legacy_adoption_embedding_row_mismatch(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, _, _ = create_fake_corpus_and_index(
        tmp_path, paper_count=3
    )
    # Overwrite embeddings with 5 rows
    bad_emb = np.random.randn(5, 4).astype(np.float32)
    np.save(emb_dir / "paper_embeddings.npy", bad_emb)

    with pytest.raises(RuntimeError, match="Legacy index does not match the active corpus"):
        adopt_existing_index(
            embeddings_dir=emb_dir,
            raw_dir=raw_dir,
            papers_list=papers,
            corpus_file_path=corpus_file,
        )


def test_legacy_adoption_faiss_ntotal_mismatch(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, _, _ = create_fake_corpus_and_index(
        tmp_path, paper_count=3
    )
    # Overwrite faiss index with 2 items
    bad_index = faiss.IndexFlatIP(4)
    bad_index.add(np.random.randn(2, 4).astype(np.float32))
    faiss.write_index(bad_index, str(emb_dir / "faiss.index"))

    with pytest.raises(RuntimeError, match="Legacy index does not match the active corpus"):
        adopt_existing_index(
            embeddings_dir=emb_dir,
            raw_dir=raw_dir,
            papers_list=papers,
            corpus_file_path=corpus_file,
        )


def test_legacy_adoption_embedding_not_2d(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, _, _ = create_fake_corpus_and_index(tmp_path)
    bad_emb = np.random.randn(3, 4, 2).astype(np.float32)
    np.save(emb_dir / "paper_embeddings.npy", bad_emb)

    with pytest.raises(RuntimeError, match="Legacy index does not match the active corpus"):
        adopt_existing_index(
            embeddings_dir=emb_dir,
            raw_dir=raw_dir,
            papers_list=papers,
            corpus_file_path=corpus_file,
        )


def test_manifest_validation_duplicate_corpus_ids(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, emb_matrix, index = create_fake_corpus_and_index(
        tmp_path
    )

    adopt_existing_index(
        embeddings_dir=emb_dir, raw_dir=raw_dir, papers_list=papers, corpus_file_path=corpus_file
    )
    manifest = IndexManifest.load(emb_dir / "index_manifest.json")

    # Duplicate paper ID
    dup_papers = [papers[0], papers[0], papers[2]]
    with pytest.raises(IndexCompatibilityError, match="ordered document-ID hash differs"):
        manifest.validate(
            corpus_path=corpus_file,
            papers=dup_papers,
            model_name=settings.embedding.model_name,
            model_revision=settings.embedding.model_revision,
            embeddings=emb_matrix,
            index=index,
        )


def test_manifest_validation_corpus_order_change(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, emb_matrix, index = create_fake_corpus_and_index(
        tmp_path
    )

    adopt_existing_index(
        embeddings_dir=emb_dir, raw_dir=raw_dir, papers_list=papers, corpus_file_path=corpus_file
    )
    manifest = IndexManifest.load(emb_dir / "index_manifest.json")

    reversed_papers = list(reversed(papers))
    with pytest.raises(IndexCompatibilityError, match="ordered document-ID hash differs"):
        manifest.validate(
            corpus_path=corpus_file,
            papers=reversed_papers,
            model_name=settings.embedding.model_name,
            model_revision=settings.embedding.model_revision,
            embeddings=emb_matrix,
            index=index,
        )


def test_manifest_validation_embedding_dimension_mismatch(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, _, index = create_fake_corpus_and_index(tmp_path, dim=4)

    adopt_existing_index(
        embeddings_dir=emb_dir, raw_dir=raw_dir, papers_list=papers, corpus_file_path=corpus_file
    )
    manifest = IndexManifest.load(emb_dir / "index_manifest.json")

    bad_dim_matrix = np.random.randn(3, 8).astype(np.float32)
    with pytest.raises(IndexCompatibilityError, match="embedding array dimension differs"):
        manifest.validate(
            corpus_path=corpus_file,
            papers=papers,
            model_name=settings.embedding.model_name,
            model_revision=settings.embedding.model_revision,
            embeddings=bad_dim_matrix,
            index=index,
        )


def test_manifest_validation_model_or_revision_mismatch(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, emb_matrix, index = create_fake_corpus_and_index(
        tmp_path
    )

    adopt_existing_index(
        embeddings_dir=emb_dir, raw_dir=raw_dir, papers_list=papers, corpus_file_path=corpus_file
    )
    manifest = IndexManifest.load(emb_dir / "index_manifest.json")

    with pytest.raises(IndexCompatibilityError, match="embedding model differs"):
        manifest.validate(
            corpus_path=corpus_file,
            papers=papers,
            model_name="different-embedding-model",
            model_revision=settings.embedding.model_revision,
            embeddings=emb_matrix,
            index=index,
        )

    with pytest.raises(IndexCompatibilityError, match="model revision differs"):
        manifest.validate(
            corpus_path=corpus_file,
            papers=papers,
            model_name=settings.embedding.model_name,
            model_revision="different-revision",
            embeddings=emb_matrix,
            index=index,
        )


def test_legacy_adoption_rejects_duplicate_ids(tmp_path: Path):
    raw_dir, emb_dir, corpus_file, papers, _, _ = create_fake_corpus_and_index(tmp_path)
    duplicate_papers = [papers[0], papers[0], papers[2]]
    with pytest.raises(RuntimeError, match="duplicate paper IDs"):
        adopt_existing_index(
            embeddings_dir=emb_dir,
            raw_dir=raw_dir,
            papers_list=duplicate_papers,
            corpus_file_path=corpus_file,
        )


def test_legacy_adoption_rolls_back_all_outputs_on_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw_dir, emb_dir, corpus_file, papers, _, _ = create_fake_corpus_and_index(tmp_path)
    raw_current = raw_dir / "CURRENT"
    index_current = emb_dir / "CURRENT"
    raw_current.write_text("corpus-before\n", encoding="utf-8")
    index_current.write_text("index-before\n", encoding="utf-8")
    old_index_manifest = emb_dir / "index_manifest.json"
    old_corpus_manifest = raw_dir / "corpus_manifest.json"
    old_index_manifest.write_text("old-index-manifest", encoding="utf-8")
    old_corpus_manifest.write_text("old-corpus-manifest", encoding="utf-8")

    real_atomic_write = indexing.atomic_write_text
    calls = 0

    def fail_second_publish(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        real_atomic_write(path, content)

    monkeypatch.setattr(indexing, "atomic_write_text", fail_second_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        adopt_existing_index(
            embeddings_dir=emb_dir,
            raw_dir=raw_dir,
            papers_list=papers,
            corpus_file_path=corpus_file,
        )

    assert raw_current.read_text(encoding="utf-8") == "corpus-before\n"
    assert index_current.read_text(encoding="utf-8") == "index-before\n"
    assert old_index_manifest.read_text(encoding="utf-8") == "old-index-manifest"
    assert old_corpus_manifest.read_text(encoding="utf-8") == "old-corpus-manifest"
    assert not list(raw_dir.glob("*.tmp"))
    assert not list(emb_dir.glob("*.tmp"))


def test_corrupted_manifest_json(tmp_path: Path):
    bad_manifest_path = tmp_path / "index_manifest.json"
    bad_manifest_path.write_text("{corrupted json string", encoding="utf-8")

    with pytest.raises(IndexCompatibilityError, match="Invalid index manifest"):
        IndexManifest.load(bad_manifest_path)
