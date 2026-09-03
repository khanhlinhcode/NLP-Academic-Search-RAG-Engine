"""Corpus and semantic-index integrity inventory."""

from __future__ import annotations

from collections import Counter

import faiss
import numpy as np

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import active_corpus_path, load_papers


def build_data_audit() -> dict[str, object]:
    """Return a serializable snapshot of corpus and index integrity."""
    corpus_path = active_corpus_path()
    papers = load_papers(corpus_path)
    ids = [paper.id for paper in papers]
    titles = [paper.title.casefold() for paper in papers]
    years = Counter(str(paper.year) if paper.year else "missing" for paper in papers)
    categories = Counter(category for paper in papers for category in paper.categories)
    embeddings_path = settings.data.embeddings_dir / "paper_embeddings.npy"
    index_path = settings.data.embeddings_dir / "faiss.index"
    embeddings = np.load(embeddings_path, mmap_mode="r") if embeddings_path.exists() else None
    index = faiss.read_index(str(index_path)) if index_path.exists() else None
    return {
        "corpus_path": str(corpus_path.relative_to(settings.data.raw_dir.parent)),
        "records": len(papers),
        "unique_ids": len(set(ids)),
        "duplicate_titles": len(titles) - len(set(titles)),
        "missing_authors": sum(not paper.authors for paper in papers),
        "missing_categories": sum(not paper.categories for paper in papers),
        "missing_published_at": sum(paper.published_at is None for paper in papers),
        "valid_arxiv_ids": sum(paper.arxiv_id is not None for paper in papers),
        "source_distribution": dict(Counter(paper.source for paper in papers)),
        "year_distribution": dict(years.most_common(20)),
        "category_distribution": dict(categories.most_common(20)),
        "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
        "faiss_ntotal": index.ntotal if index is not None else None,
        "faiss_dimension": index.d if index is not None else None,
    }
