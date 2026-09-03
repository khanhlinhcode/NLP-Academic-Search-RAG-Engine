from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_academic_search.data import audit, ingestion, processing
from nlp_academic_search.data.loader import Paper, load_papers
from nlp_academic_search.data.sources.beir import convert_beir
from nlp_academic_search.evaluation.retrieval_runner import run_retrieval_evaluation
from nlp_academic_search.search import indexing


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_preprocess_corpus_versions_valid_rows(tmp_path: Path, monkeypatch, papers: list[Paper]):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(processing.settings, "data_raw_dir", raw_dir)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(papers[0].to_dict())
        + "\n"
        + json.dumps(papers[0].to_dict())
        + "\n{invalid json}\n",
        encoding="utf-8",
    )

    output = processing.preprocess_corpus(source)

    assert [paper.id for paper in load_papers(output)] == [papers[0].id]
    assert (raw_dir / "CURRENT").read_text().strip() == output.parent.name
    manifest = json.loads((output.parent / "corpus_manifest.json").read_text())
    assert manifest["document_count"] == 1
    assert manifest["quarantined_count"] == 1
    assert (output.parent / "quarantine.jsonl").is_file()


def test_preprocess_rejects_an_empty_invalid_corpus(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(processing.settings, "data_raw_dir", tmp_path / "raw")
    source = tmp_path / "invalid.jsonl"
    source.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No valid records"):
        processing.preprocess_corpus(source)


def test_ingest_arxiv_activates_a_version(tmp_path: Path, monkeypatch, papers: list[Paper]):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(ingestion.settings, "data_raw_dir", raw_dir)

    class FakeAdapter:
        name = "fake-arxiv"
        quarantined = [{"id": "bad", "reason": "invalid"}]

        def __init__(self, **_kwargs):
            pass

        def iter_papers(self):
            yield papers[0]
            yield papers[0]
            yield papers[1]

    monkeypatch.setattr(ingestion, "ArxivOAIAdapter", FakeAdapter)
    count, version = ingestion.ingest_arxiv(2, "cs")

    assert count == 2
    assert (raw_dir / "CURRENT").read_text().strip() == version
    version_dir = raw_dir / "versions" / version
    assert len(load_papers(version_dir / "papers.jsonl")) == 2
    manifest = json.loads((version_dir / "corpus_manifest.json").read_text())
    assert manifest["source"] == "fake-arxiv"
    assert manifest["quarantined_count"] == 2


def test_ingest_arxiv_rejects_invalid_limits():
    with pytest.raises(ValueError, match="positive"):
        ingestion.ingest_arxiv(0)


def test_data_audit_describes_corpus_without_index(
    tmp_path: Path, monkeypatch, papers: list[Paper]
):
    raw_dir = tmp_path / "raw"
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    _write_jsonl(raw_dir / "papers.jsonl", [paper.to_dict() for paper in papers])
    monkeypatch.setattr(audit.settings, "data_raw_dir", raw_dir)
    monkeypatch.setattr(audit.settings, "data_embeddings_dir", embeddings_dir)

    report = audit.build_data_audit()

    assert report["records"] == 3
    assert report["unique_ids"] == 3
    assert report["embeddings_shape"] is None
    assert report["faiss_ntotal"] is None


def test_convert_beir_preserves_relevance_judgments(tmp_path: Path):
    dataset = tmp_path / "scifact"
    _write_jsonl(
        dataset / "corpus.jsonl",
        [{"_id": "d1", "title": "A title", "text": "Scientific evidence text."}],
    )
    _write_jsonl(dataset / "queries.jsonl", [{"_id": "q1", "text": "evidence query"}])
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\n", encoding="utf-8")
    output = tmp_path / "benchmark.json"

    document_count, query_count = convert_beir(dataset, qrels, output, "scifact")
    payload = json.loads(output.read_text())

    assert (document_count, query_count) == (1, 1)
    assert payload["queries"][0]["qrels"] == {"beir:d1": 2}
    assert payload["documents"][0]["source"] == "BEIR/scifact"
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["counts"] == {"documents": 1, "queries": 1, "qrels": 1}
    assert manifest["judgments_transformed"] is False
    assert manifest["files"]["benchmark.json"]


def test_index_workflow_validates_missing_legacy_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(indexing.settings, "data_embeddings_dir", tmp_path / "embeddings")
    with pytest.raises(FileNotFoundError, match="Legacy embeddings"):
        indexing.adopt_existing_index()


def test_retrieval_runner_rejects_invalid_k(tmp_path: Path):
    with pytest.raises(ValueError, match="positive"):
        run_retrieval_evaluation(tmp_path / "unused.json", 0)
