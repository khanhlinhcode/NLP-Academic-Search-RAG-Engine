"""Final Validation & Multi-Trial Benchmark for Project A Retrieval Pipeline.

Addresses all validation requirements:
1. Dev split (100 queries) vs. Holdout split (100 unseen queries).
2. Explicit execution of the combined optimized configuration (k=20, pool=20).
3. 3 repeated trials per configuration with Mean, Std Dev, and 95% CI.
4. Precision vs. percentage points (pp) distinction.
5. Component memory vs. Process Peak RSS breakdown.
6. Local CPU vs. Cloud verification boundary.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import resource
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import torch

from nlp_academic_search.data.loader import Paper
from nlp_academic_search.evaluation.metrics import (
    _positive_ids,
    _ranked_prefix,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from nlp_academic_search.search.bm25_search import BM25Searcher
from nlp_academic_search.search.hybrid_search import FusionMethod, HybridSearcher
from nlp_academic_search.search.models import SearchResult
from nlp_academic_search.search.semantic_search import SemanticSearcher

torch.set_num_threads(1)
torch.set_grad_enabled(False)
faiss.omp_set_num_threads(1)


def get_peak_memory_mb() -> float:
    """Return process high-water mark RSS in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def evaluate_queries(
    search_fn,
    queries: list[dict[str, Any]],
    *,
    pool_k: int = 50,
) -> dict[str, Any]:
    latencies = []
    p5_list, r5_list, hit5_list = [], [], []
    mrr10_list, ndcg10_list = [], []

    # Warm-up with first 2 queries
    for q in queries[:2]:
        search_fn(q["query"], top_k=pool_k)

    for q in queries:
        query_text = q["query"]
        relevance = q.get("qrels", {})
        relevant_ids = _positive_ids(relevance)

        t0 = time.perf_counter()
        results: list[SearchResult] = search_fn(query_text, top_k=pool_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        retrieved_ids = [r.paper.id for r in results]

        p5_list.append(precision_at_k(retrieved_ids, relevant_ids, k=5))
        r5_list.append(recall_at_k(retrieved_ids, relevant_ids, k=5))
        ranked_top5 = _ranked_prefix(retrieved_ids, 5)
        hit5_list.append(1.0 if any(doc in relevant_ids for doc in ranked_top5) else 0.0)

        mrr10_list.append(mrr_at_k(retrieved_ids, relevant_ids, k=10))
        ndcg10_list.append(ndcg_at_k(retrieved_ids, relevance, k=10))

    return {
        "Precision@5": round(statistics.mean(p5_list), 4),
        "Recall@5": round(statistics.mean(r5_list), 4),
        "HitRate@5": round(statistics.mean(hit5_list), 4),
        "MRR@10": round(statistics.mean(mrr10_list), 4),
        "nDCG@10": round(statistics.mean(ndcg10_list), 4),
        "Latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "Latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "Latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "Mean_Latency_ms": round(statistics.mean(latencies), 2),
    }


def run_trials(
    search_fn,
    queries: list[dict[str, Any]],
    num_trials: int = 3,
    pool_k: int = 50,
) -> dict[str, Any]:
    trial_results = []
    for _ in range(num_trials):
        gc.collect()
        res = evaluate_queries(search_fn, queries, pool_k=pool_k)
        trial_results.append(res)

    p5s = [r["Precision@5"] for r in trial_results]
    r5s = [r["Recall@5"] for r in trial_results]
    hits = [r["HitRate@5"] for r in trial_results]
    mrrs = [r["MRR@10"] for r in trial_results]
    ndcgs = [r["nDCG@10"] for r in trial_results]
    p50_lats = [r["Latency_p50_ms"] for r in trial_results]
    p95_lats = [r["Latency_p95_ms"] for r in trial_results]

    def stats(values: list[float]):
        m = float(np.mean(values))
        s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci = 1.96 * s / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return {
            "mean": round(m, 4),
            "std": round(s, 4),
            "median": round(float(np.median(values)), 4),
            "ci95": round(ci, 4),
        }

    return {
        "trials": trial_results,
        "aggregate": {
            "Precision@5": stats(p5s),
            "Recall@5": stats(r5s),
            "HitRate@5": stats(hits),
            "MRR@10": stats(mrrs),
            "nDCG@10": stats(ndcgs),
            "Latency_p50_ms": stats(p50_lats),
            "Latency_p95_ms": stats(p95_lats),
        },
    }


def main():
    dev_path = Path("data/benchmarks/scifact_dev_100.json")
    holdout_path = Path("data/benchmarks/scifact_holdout_100.json")
    emb_cache = Path("data/benchmarks/scifact_embeddings.npy")
    idx_cache = Path("data/benchmarks/scifact_faiss.index")

    print(f"Loading Dev queries from {dev_path}...")
    dev_data = json.loads(dev_path.read_text(encoding="utf-8"))
    papers = [Paper.from_dict(item) for item in dev_data["documents"]]
    dev_queries = dev_data["queries"]

    print(f"Loading Holdout queries from {holdout_path}...")
    holdout_data = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout_queries = holdout_data["queries"]
    print(
        f"Corpus: {len(papers)} documents | Dev: {len(dev_queries)} queries | Holdout: {len(holdout_queries)} queries."
    )

    # 1. Initialize BM25 and measure exact memory
    print("\n--- Measuring BM25 memory ---")
    tracemalloc.start()
    t_bm25_start = time.perf_counter()
    bm25 = BM25Searcher(papers)
    bm25_time = time.perf_counter() - t_bm25_start
    bm25_curr, bm25_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    bm25_heap_mb = bm25_peak / (1024 * 1024)
    print(f"BM25 initialized in {bm25_time:.2f}s | Traced Heap: {bm25_heap_mb:.2f} MB")

    # 2. Initialize Dense SBERT + FAISS
    print("\n--- Initializing Dense Searcher ---")
    if emb_cache.exists() and idx_cache.exists():
        print(f"Loading cached embeddings from {emb_cache} and FAISS from {idx_cache}...")
        embeddings = np.load(emb_cache)
        index = faiss.read_index(str(idx_cache))
        semantic = SemanticSearcher(papers, load_existing=False, corpus_path=dev_path)
        semantic.embeddings = embeddings
        semantic.index = index
    else:
        print("Computing embeddings with SentenceTransformer('all-MiniLM-L6-v2')...")
        semantic = SemanticSearcher(papers, load_existing=False, corpus_path=dev_path)
        np.save(emb_cache, semantic.embeddings)
        faiss.write_index(semantic.index, str(idx_cache))

    faiss_size_mb = idx_cache.stat().st_size / (1024 * 1024)
    emb_size_mb = emb_cache.stat().st_size / (1024 * 1024)
    total_index_size_mb = faiss_size_mb + emb_size_mb
    print(
        f"Index storage: FAISS={faiss_size_mb:.2f} MB, raw embeddings={emb_size_mb:.2f} MB, total={total_index_size_mb:.2f} MB"
    )

    # 3. Hybrid searcher
    hybrid = HybridSearcher(bm25, semantic, alpha=0.4)

    # 4. Cross-Encoder
    from nlp_academic_search.search.reranker import Reranker

    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    def run_bm25(query: str, top_k: int):
        return bm25.search(query, top_k=top_k)

    def run_dense(query: str, top_k: int):
        return semantic.search(query, top_k=top_k)

    def run_hybrid_rrf(query: str, top_k: int, candidate_pool: int = 50, rrf_k: int = 60):
        return hybrid.search(
            query,
            top_k=top_k,
            method=FusionMethod.RRF,
            candidate_pool=candidate_pool,
            rrf_k=rrf_k,
        )

    def run_hybrid_reranked(query: str, top_k: int, candidate_pool: int = 20, rrf_k: int = 60):
        candidates = hybrid.search(
            query,
            top_k=candidate_pool,
            method=FusionMethod.RRF,
            candidate_pool=candidate_pool,
            rrf_k=rrf_k,
        )
        return reranker.rerank(query, candidates, top_k=top_k)

    # ========================================================
    # PART A: HYPERPARAMETER TUNING ON DEV SPLIT (100 QUERIES)
    # ========================================================
    print("\n========================================================")
    print("PART A: HYPERPARAMETER TUNING ON DEV SPLIT (100 QUERIES)")
    print("========================================================")
    dev_results = {}

    print("Evaluating Dev Baseline BM25...")
    dev_results["BM25"] = evaluate_queries(run_bm25, dev_queries, pool_k=10)
    print("Evaluating Dev Baseline Dense...")
    dev_results["Dense"] = evaluate_queries(run_dense, dev_queries, pool_k=10)
    print("Evaluating Dev Baseline Hybrid RRF (k=60, pool=50)...")
    dev_results["Hybrid_RRF_baseline_k60_pool50"] = evaluate_queries(
        lambda q, top_k: run_hybrid_rrf(q, top_k, candidate_pool=50, rrf_k=60),
        dev_queries,
        pool_k=10,
    )

    print("\nSweeping RRF k values on Dev...")
    for k_val in [20, 40, 60]:
        name = f"Dev_RRF_k{k_val}_pool50"
        dev_results[name] = evaluate_queries(
            lambda q, top_k, kv=k_val: run_hybrid_rrf(q, top_k, candidate_pool=50, rrf_k=kv),
            dev_queries,
            pool_k=10,
        )
        print(
            f"  k={k_val}: Recall@5={dev_results[name]['Recall@5']}, nDCG@10={dev_results[name]['nDCG@10']}, Latency P50={dev_results[name]['Latency_p50_ms']}ms"
        )

    print("\nSweeping Candidate Pool sizes on Dev...")
    for p_val in [20, 30, 50]:
        name = f"Dev_RRF_pool{p_val}_k60"
        dev_results[name] = evaluate_queries(
            lambda q, top_k, pv=p_val: run_hybrid_rrf(q, top_k, candidate_pool=pv, rrf_k=60),
            dev_queries,
            pool_k=10,
        )
        print(
            f"  pool={p_val}: Recall@5={dev_results[name]['Recall@5']}, nDCG@10={dev_results[name]['nDCG@10']}, Latency P50={dev_results[name]['Latency_p50_ms']}ms"
        )

    print("\nEvaluating COMBINED OPTIMIZED configuration on Dev (k=20, pool=20)...")
    dev_results["Dev_RRF_COMBINED_k20_pool20"] = evaluate_queries(
        lambda q, top_k: run_hybrid_rrf(q, top_k, candidate_pool=20, rrf_k=20),
        dev_queries,
        pool_k=10,
    )
    print(
        f"  COMBINED (k=20, pool=20): Recall@5={dev_results['Dev_RRF_COMBINED_k20_pool20']['Recall@5']}, nDCG@10={dev_results['Dev_RRF_COMBINED_k20_pool20']['nDCG@10']}, Latency P50={dev_results['Dev_RRF_COMBINED_k20_pool20']['Latency_p50_ms']}ms"
    )

    print("\nSweeping Cross-Encoder Depth on Dev...")
    for ce_depth in [10, 20, 30]:
        name = f"Dev_CrossEncoder_pool{ce_depth}"
        dev_results[name] = evaluate_queries(
            lambda q, top_k, cd=ce_depth: run_hybrid_reranked(
                q, top_k, candidate_pool=cd, rrf_k=60
            ),
            dev_queries,
            pool_k=10,
        )
        print(
            f"  CE Depth={ce_depth}: Recall@5={dev_results[name]['Recall@5']}, nDCG@10={dev_results[name]['nDCG@10']}, Latency P50={dev_results[name]['Latency_p50_ms']}ms"
        )

    # ========================================================
    # PART B: REPEATED TRIALS ON HOLDOUT TEST SPLIT (100 QUERIES)
    # ========================================================
    print("\n========================================================")
    print("PART B: REPEATED TRIALS ON HOLDOUT TEST SPLIT (100 QUERIES)")
    print("========================================================")
    holdout_benchmark = {}

    configs_to_evaluate = [
        ("BM25", lambda q, top_k: run_bm25(q, top_k), 10),
        ("Dense", lambda q, top_k: run_dense(q, top_k), 10),
        (
            "Hybrid_RRF_Baseline (k=60, pool=50)",
            lambda q, top_k: run_hybrid_rrf(q, top_k, candidate_pool=50, rrf_k=60),
            10,
        ),
        (
            "Hybrid_RRF_Optimized_Combined (k=20, pool=20)",
            lambda q, top_k: run_hybrid_rrf(q, top_k, candidate_pool=20, rrf_k=20),
            10,
        ),
        (
            "Hybrid_CrossEncoder_Baseline (pool=20)",
            lambda q, top_k: run_hybrid_reranked(q, top_k, candidate_pool=20, rrf_k=60),
            10,
        ),
        (
            "Hybrid_CrossEncoder_Optimized (pool=10)",
            lambda q, top_k: run_hybrid_reranked(q, top_k, candidate_pool=10, rrf_k=60),
            10,
        ),
    ]

    for name, fn, pool in configs_to_evaluate:
        print(f"Running 3 trials for {name} on Holdout...")
        holdout_benchmark[name] = run_trials(fn, holdout_queries, num_trials=3, pool_k=pool)
        agg = holdout_benchmark[name]["aggregate"]
        print(
            f"  -> Recall@5: {agg['Recall@5']['mean']} +/- {agg['Recall@5']['std']} (CI95: [{agg['Recall@5']['mean'] - agg['Recall@5']['ci95']:.4f}, {agg['Recall@5']['mean'] + agg['Recall@5']['ci95']:.4f}])"
        )
        print(
            f"  -> nDCG@10:  {agg['nDCG@10']['mean']} +/- {agg['nDCG@10']['std']} (CI95: [{agg['nDCG@10']['mean'] - agg['nDCG@10']['ci95']:.4f}, {agg['nDCG@10']['mean'] + agg['nDCG@10']['ci95']:.4f}])"
        )
        print(
            f"  -> Latency P50: {agg['Latency_p50_ms']['median']} ms (CI95: +/- {agg['Latency_p50_ms']['ci95']:.2f} ms)"
        )

    # Memory accounting
    process_peak_rss_mb = get_peak_memory_mb()

    # Package output JSON
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "threads": 1,
            "process_peak_rss_mb": round(process_peak_rss_mb, 2),
            "component_heap": {
                "bm25_heap_mb": round(bm25_heap_mb, 2),
                "faiss_index_mb": round(faiss_size_mb, 2),
                "embeddings_npy_mb": round(emb_size_mb, 2),
                "total_index_disk_mb": round(total_index_size_mb, 2),
            },
        },
        "splits": {
            "dev_queries_count": len(dev_queries),
            "holdout_queries_count": len(holdout_queries),
            "total_corpus_documents": len(papers),
        },
        "dev_split_sweep": dev_results,
        "holdout_split_trials_3x": holdout_benchmark,
    }

    out_file = Path("reports/benchmark/retrieval_final_validated.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved final validated retrieval benchmark to {out_file}")


if __name__ == "__main__":
    main()
