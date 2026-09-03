import json

import pytest

from nlp_academic_search.data.loader import CorpusValidationError, Paper, load_papers
from nlp_academic_search.data.manifest import (
    activate_version,
    build_manifest,
    write_jsonl,
    write_manifest,
)
from nlp_academic_search.data.preprocessor import clean_text, tokenize_for_bm25


def test_paper_builds_verified_source_links():
    paper = Paper(
        id="arxiv:1706.03762",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="Transformer abstract",
    )
    assert paper.source_url == "https://arxiv.org/abs/1706.03762"
    assert paper.pdf_url == "https://arxiv.org/pdf/1706.03762"
    assert len(paper.content_hash) == 64


def test_legacy_placeholders_are_not_exposed_as_metadata():
    paper = Paper.from_dict(
        {
            "id": "paper_00001",
            "title": "Legacy row",
            "abstract": "Legacy abstract",
            "authors": ["Placeholder"],
            "category": "cs",
            "year": 2023,
        }
    )
    assert paper.authors == []
    assert paper.categories == []
    assert paper.year is None
    assert paper.source_url is None


def test_invalid_identifiers_are_rejected():
    with pytest.raises(ValueError, match="arXiv"):
        Paper(id="x", arxiv_id="paper_fake", title="T", abstract="A")


def test_loader_reports_malformed_json_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="bad.jsonl:1"):
        load_papers(path)


def test_loader_rejects_duplicate_ids(tmp_path, papers):
    path = tmp_path / "duplicates.jsonl"
    row = json.dumps(papers[0].to_dict())
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="duplicate"):
        load_papers(path)


def test_preprocessor_preserves_scientific_terms_and_unicode():
    text = r"A \text{cross-encoder} estimates $E=mc^2$ for naïve Bayes and BERT-base."
    cleaned = clean_text(text)
    tokens = tokenize_for_bm25(text)
    assert "cross-encoder" in tokens
    assert "naïve" in tokens
    assert "bert-base" in tokens
    assert "E=mc^2" in cleaned
    assert tokenize_for_bm25("") == []


def test_corpus_manifest_and_atomic_version_activation(tmp_path, papers):
    version = "fixture-v1"
    version_dir = tmp_path / "versions" / version
    corpus = version_dir / "papers.jsonl"
    write_jsonl(corpus, papers)
    manifest = build_manifest(corpus, papers, source="test", filtering_rules=["validated"])
    write_manifest(version_dir / "corpus_manifest.json", manifest)
    activate_version(tmp_path, version)
    assert (tmp_path / "CURRENT").read_text().strip() == version
    assert manifest.document_count == len(papers)
