"""
Run evaluation benchmarks across all search methods.

Compares BM25, Semantic, Hybrid, and Hybrid + Reranker
using standard IR metrics.

Usage:
    python -m scripts.run_evaluation
"""

import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.loader import load_papers
from src.evaluation.metrics import evaluate_search_method, format_evaluation_table
from src.search.bm25_search import BM25Searcher
from src.search.hybrid_search import HybridSearcher
from src.search.reranker import Reranker
from src.search.semantic_search import SemanticSearcher


# ─── Sample evaluation queries ───────────────────────────────────
# In a real project, these would be manually annotated or generated
# For now, we create synthetic queries based on paper content

def create_evaluation_queries(papers, num_queries=50):
    """
    Create evaluation queries from the dataset.

    Strategy: Use paper titles as queries and mark the source paper
    + papers with similar categories as relevant.
    """
    import random

    random.seed(42)

    queries = []
    sample_papers = random.sample(papers, min(num_queries, len(papers)))

    for paper in sample_papers:
        # The query is derived from the paper's title/abstract
        # Use the first sentence of the abstract as the query
        abstract_sentences = paper.abstract.split(".")
        query = abstract_sentences[0].strip() if abstract_sentences else paper.title

        if len(query) < 10:
            query = paper.title

        # The source paper is definitely relevant
        relevant_ids = [paper.id]

        queries.append({
            "query": query,
            "relevant_ids": relevant_ids,
        })

    return queries


def run_evaluation():
    """Run full evaluation across all search methods."""

    print("\n" + "=" * 60)
    print("📊 NLP Academic Search — Evaluation Benchmark")
    print("=" * 60 + "\n")

    # Load papers
    papers = load_papers()

    # Create evaluation queries
    print("📝 Creating evaluation queries...")
    queries = create_evaluation_queries(papers, num_queries=20)
    print(f"   Generated {len(queries)} evaluation queries.\n")

    k = 10  # Evaluate @10

    # ─── BM25 ─────────────────────────────────────────────────────
    print("🔍 Evaluating BM25...")
    bm25 = BM25Searcher(papers)
    bm25_results = evaluate_search_method(bm25.search, queries, k=k)

    # ─── Semantic ─────────────────────────────────────────────────
    print("\n🧠 Evaluating Semantic Search...")
    semantic = SemanticSearcher(papers)
    semantic_results = evaluate_search_method(semantic.search, queries, k=k)

    # ─── Hybrid ───────────────────────────────────────────────────
    print("\n⚡ Evaluating Hybrid Search...")
    hybrid = HybridSearcher(bm25=bm25, semantic=semantic)
    hybrid_results = evaluate_search_method(hybrid.search, queries, k=k)

    # ─── Hybrid + Reranker ────────────────────────────────────────
    print("\n🔄 Evaluating Hybrid + Reranker...")
    try:
        reranker = Reranker(device="cpu")

        def hybrid_rerank_search(query, top_k=10):
            candidates = hybrid.search(query, top_k=top_k * 4)
            return reranker.rerank(query, candidates, top_k=top_k)

        reranker_results = evaluate_search_method(hybrid_rerank_search, queries, k=k)
    except Exception as e:
        import traceback
        print(f"   ⚠️ Reranker failed: {e}")
        traceback.print_exc()
        reranker_results = None

    # ─── Results ──────────────────────────────────────────────────
    all_results = {
        "BM25": bm25_results,
        "SBERT": semantic_results,
        "Hybrid": hybrid_results,
    }
    if reranker_results:
        all_results["Hybrid + Reranker"] = reranker_results

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60 + "\n")
    print(format_evaluation_table(all_results))
    print()

    # Save results to file
    output_path = settings.data.processed_dir / "evaluation_results.json"
    settings.data.ensure_dirs()
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"💾 Results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
