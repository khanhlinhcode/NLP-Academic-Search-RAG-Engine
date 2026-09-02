"""
Build FAISS index and pre-compute embeddings.

This script downloads papers (if needed), computes SBERT embeddings
for all papers, builds a FAISS index, and saves everything to disk
for fast loading during search.

Usage:
    python -m scripts.build_index
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.loader import load_papers
from src.search.semantic_search import SemanticSearcher


def build_index():
    """Build and save FAISS index + embeddings."""

    settings.data.ensure_dirs()

    # Check if dataset exists
    dataset_path = settings.data.raw_dir / "papers.jsonl"
    if not dataset_path.exists():
        print("❌ Dataset not found! Run 'python -m scripts.download_data' first.")
        return

    # Load papers
    papers = load_papers()

    # Build semantic search index (computes embeddings)
    print(f"\n🚀 Building index for {len(papers)} papers...")
    start_time = time.time()

    searcher = SemanticSearcher(papers, load_existing=False)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Embedding computation took {elapsed:.1f}s")
    print(f"   ({elapsed / len(papers) * 1000:.1f}ms per paper)")

    # Save to disk
    searcher.save_index()

    print("\n✅ Index built and saved successfully!")
    print(f"   Embeddings: {settings.data.embeddings_dir / 'paper_embeddings.npy'}")
    print(f"   FAISS index: {settings.data.embeddings_dir / 'faiss.index'}")


if __name__ == "__main__":
    build_index()
