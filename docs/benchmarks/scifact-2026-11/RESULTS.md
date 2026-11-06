# Comprehensive Benchmark Results — Project A (Final Validation)

**Project**: NLP Academic Search & RAG Engine
**Repository**: [https://github.com/khanhlinhcode/NLP-Academic-Search-RAG-Engine](https://github.com/khanhlinhcode/NLP-Academic-Search-RAG-Engine)
**Commit**: `cb8d0bc83b041f3ab04299b81c5015e44042a710`
**Dataset**: BEIR / SciFact (5,183 scientific papers; 100 Dev queries, 100 Holdout queries)
**Hardware**: Apple M1 (8 cores), 8GB Unified Memory, macOS Darwin 27.0.0

---

## 1. Multi-Trial Retrieval Benchmark (Holdout Test Split — 100 Unseen Queries)

Evaluated across 3 independent trials on unseen test queries. Deterministic ranking yielded identical rank lists across trials ($\text{std} = 0.0$ for accuracy metrics). Latency reports median and 95% confidence intervals:

| Architecture | Precision@5 | Recall@5 | HitRate@5 | MRR@10 | nDCG@10 | Latency P50 (ms) | Latency P95 (ms) | Peak RAM (MB) | Index Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **BM25 (Sparse Okapi)** | 0.1420 | 0.6817 | 0.7000 | 0.5935 | 0.6254 | $9.00 \pm 2.13$ | $18.32 \pm 3.83$ | 68.83 (heap) | 0.0 (in-memory) |
| **Dense (all-MiniLM-L6-v2)** | 0.1420 | 0.6753 | 0.7000 | 0.5404 | 0.5858 | $7.02 \pm 0.71$ | $8.95 \pm 1.83$ | 7.59 (heap) | 15.18 |
| **Hybrid RRF Baseline ($k=60, \text{pool}=50$)** | **0.1580** | **0.7387** | **0.7600** | **0.6063** | **0.6389** | **$15.94 \pm 5.33$** | **$27.48 \pm 12.86$** | 766.62 (process) | 15.18 |
| **Hybrid RRF Combined ($k=20, \text{pool}=20$)** | 0.1540 | 0.7187 | 0.7400 | 0.6055 | 0.6380 | $17.74 \pm 0.99$ | $30.69 \pm 1.67$ | 766.62 (process) | 15.18 |
| **Hybrid + Cross-Encoder (Depth 20)** | 0.1480 | 0.6920 | 0.7200 | 0.5925 | 0.6295 | $1,135.72 \pm 115.44$ | $1,590.58 \pm 200.75$ | 766.62 (process) | 15.18 |
| **Hybrid + Cross-Encoder (Depth 10)** | **0.1540** | **0.7220** | **0.7500** | **0.6144** | **0.6527** | **$664.11 \pm 24.96$** | **$867.83 \pm 27.88$** | 766.62 (process) | 15.18 |

---

## 2. Dev Split Exploration Table (100 Queries)

| Architecture / Config | Precision@5 | Recall@5 | HitRate@5 | MRR@10 | nDCG@10 | Latency P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **BM25** | 0.1640 | 0.7658 | 0.7900 | 0.6768 | 0.7135 | 7.39 |
| **Dense** | 0.1780 | 0.7860 | 0.8000 | 0.6682 | 0.7009 | 7.71 |
| **Hybrid RRF Baseline ($k=60, \text{pool}=50$)** | 0.1800 | 0.8182 | 0.8400 | 0.7476 | 0.7789 | 15.12 |
| **Hybrid RRF ($k=20, \text{pool}=50$)** | 0.1800 | 0.8182 | 0.8400 | 0.7507 | 0.7850 | 16.69 |
| **Hybrid RRF ($k=60, \text{pool}=20$)** | 0.1840 | 0.8382 | 0.8600 | 0.7514 | 0.7825 | 16.19 |
| **Hybrid RRF Combined ($k=20, \text{pool}=20$)** | **0.1840** | **0.8382** | **0.8600** | **0.7514** | **0.7828** | **17.52** |
| **Hybrid + Cross-Encoder (Depth 10)** | 0.1840 | 0.8265 | 0.8400 | 0.7084 | 0.7489 | 774.55 |

---

## 3. RAG Groundedness & Hallucination Evaluation (20 Cases)

### A. Case Breakdown & Taxonomy
* **Total Scenarios**: 20 cases (13 Dev split + 7 Holdout split).
* **Positive Ground-Truth Supported Cases**: 6 cases (supported claims, multi-source valid, Vietnamese valid, modality preservation, conflicting evidence).
* **Negative Unsupported Cases**: 12 cases (partially supported, fabricated quotes, wrong source, entity swap, negation inversion, fabricated p-value, epistemic overclaim, out-of-bounds index).
* **Canonical Refusal Cases**: 2 cases (empty retrieved context in English and Vietnamese).
* **Adversarial Prompt Injection Cases**: 2 cases (abstract-embedded system override instructions).

### B. Mode Evaluation Matrix

| Metric | Mode 1 (Structural Regex) | Mode 2 (Deterministic Exact-Quote) | Mode 3 (Bounded Repair + Refusal) | Notes |
|---|---:|---:|---:|---|
| **Unsupported Claim Slip Rate** | **83.3% (10/12)** | **0.0% (0/12)** | **0.0% (0/12)** | Fraction of negative claims accepted as valid |
| **Prompt Injection Resistance** | **0.0% (0/2)** | **100.0% (2/2)** | **100.0% (2/2)** | Both adversarial injections blocked in Mode 2 & 3 |
| **Refusal Accuracy** | **100.0% (2/2)** | **100.0% (2/2)** | **100.0% (2/2)** | Both empty-evidence cases triggered refusal |
| **P50 Local Verification Overhead** | **0.107 ms** | **0.297 ms** | **0.261 ms** | Python NFKC string search overhead only |
| **P95 Local Verification Overhead** | **0.259 ms** | **0.719 ms** | **0.451 ms** | Excludes external LLM generation / API time |

*Scope Clarification*: Latency represents local deterministic string normalization and boundary checking in Python, NOT live generative LLM inference.

---

## 4. Final Metric Verification Status Matrix

| Metric / Claim | Verification Status | Empirical Result | Evidence File / Location |
|---|---|---|---|
| **BM25 Retrieval (Holdout)** | `VERIFIED` | Recall@5: 0.6817, nDCG@10: 0.6254, P50: 9.00 ms | `reports/benchmark/retrieval_final_validated.json` |
| **Dense Retrieval (Holdout)** | `VERIFIED` | Recall@5: 0.6753, nDCG@10: 0.5858, P50: 7.02 ms | `reports/benchmark/retrieval_final_validated.json` |
| **Hybrid RRF Gain (Holdout)** | `VERIFIED` | Recall@5: 0.7387 (+5.70 pp / +8.36% rel over BM25), P50: 15.94 ms | `reports/benchmark/retrieval_final_validated.json` |
| **Combined Config (k=20, pool=20)** | `VERIFIED` | Dev Recall@5: 0.8382, Holdout Recall@5: 0.7187 | `reports/benchmark/retrieval_final_validated.json` |
| **Cross-Encoder Pruning Speedup** | `VERIFIED` | Depth 20 → 10: 41.5% latency reduction (1,135 ms → 664 ms) | `reports/benchmark/retrieval_final_validated.json` |
| **Index Disk Footprint** | `VERIFIED` | 15.18 MB total (FAISS 7.59 MB + npy 7.59 MB) | APFS filesystem stat on `data/benchmarks/` |
| **BM25 Heap Usage** | `VERIFIED` | 68.83 MB traced heap | `tracemalloc` in `benchmark_retrieval_audit.py` |
| **Process Peak RAM** | `VERIFIED` | 766.62 MB process high-water mark | `resource.getrusage` in macOS CPython |
| **Hallucination Slip Rate Reduction** | `VERIFIED` | 83.3% (Mode 1) → 0.0% (Mode 2/3) across 20 cases | `reports/benchmark/rag_final_validated.json` |
| **Prompt Injection Resistance** | `VERIFIED` | 100% blocked (2/2 adversarial test cases) | `reports/benchmark/rag_final_validated.json` |
| **Cross-Encoder Domain Mismatch** | `VERIFIED FINDING (HYPOTHESIS EXPLANATION)` | Degradation (-3.53 pp) verified; domain mismatch is literature causal explanation | SciFact benchmark vs MS MARCO pretraining |
| **Cloud 512 MB Container Stability** | `NOT RUN / THEORETICAL SPECIFICATION` | Lazy import architecture validated statically; live load test not run | `src/nlp_academic_search/providers/` |
