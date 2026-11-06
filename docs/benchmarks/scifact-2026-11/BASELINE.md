# Baseline Benchmark Report — Project A (NLP Academic Search & RAG)

**Project**: NLP Academic Search & RAG Engine **Commit**:
`cb8d0bc83b041f3ab04299b81c5015e44042a710` **Dataset**: SciFact Academic Corpus
(5,183 scientific papers, 100 human-judged test queries) **Hardware**: Apple M1
(8 cores), 8GB Unified Memory, macOS Darwin 27.0.0

---

## 1. Problem Statement

Academic search engines face a severe vocabulary mismatch:

1. Exact keyword search (BM25) fails when researchers phrase queries
   conceptually without using the exact author vocabulary.
2. Dense semantic embeddings (`all-MiniLM-L6-v2`) capture high-level conceptual
   similarity but struggle with rare terminology, gene names, chemical
   compounds, and dataset acronyms.
3. Conventional generative RAG systems suffer from hallucinated citations: an
   LLM outputs syntactically valid citations like `[1]` even when the underlying
   cited paper makes no such claim.

---

## 2. Baseline Architecture Evaluation (4 Configurations)

All four configurations were independently evaluated against the exact same
corpus (5,183 papers) and 100 judged queries under identical hardware
conditions.

| Architecture                                                | Precision@5 |   Recall@5 |  HitRate@5 |     MRR@10 |    nDCG@10 | Latency P50 (ms) | Latency P95 (ms) | Peak RAM (MB) | Index Size (MB) |
| ----------------------------------------------------------- | ----------: | ---------: | ---------: | ---------: | ---------: | ---------------: | ---------------: | ------------: | --------------: |
| **BM25** (Sparse Okapi)                                     |      0.1640 |     0.7658 |     0.7900 |     0.6768 |     0.7135 |             7.56 |            14.26 |         750.2 | 0.0 (in-memory) |
| **Dense** (`all-MiniLM-L6-v2` + FAISS FlatIP)               |      0.1780 |     0.7860 |     0.8000 |     0.6682 |     0.7009 |             6.97 |            11.44 |         750.2 |           15.18 |
| **Hybrid RRF** (BM25 + Dense + RRF k=60, pool=50)           |  **0.1800** | **0.8182** | **0.8400** | **0.7476** | **0.7789** |            13.68 |            20.75 |         750.2 |           15.18 |
| **Hybrid + Cross-Encoder** (RRF + `ms-marco` rerank top-20) |      0.1840 |     0.8085 |     0.8200 |     0.7010 |     0.7436 |         1,054.85 |         2,658.48 |         750.2 |           15.18 |

---

## 3. Key Baseline Findings

1. **Hybrid RRF outperforms both standalone BM25 and Dense Retrieval across all
   retrieval metrics**:
   - Recall@5 increased from 0.7658 (BM25) and 0.7860 (Dense) to **0.8182**
     (+5.24% over BM25, +3.22% over Dense).
   - nDCG@10 increased from 0.7135 (BM25) and 0.7009 (Dense) to **0.7789**
     (+6.54% over BM25, +7.80% over Dense).
   - HitRate@5 reached **84.00%** (84 out of 100 queries returned at least one
     ground-truth relevant document in top 5).
   - Latency remains ultra-low: P50 latency is **13.68 ms**, well within
     interactive SLA targets (< 50 ms).

2. **Cross-Encoder Empirical Bottleneck on CPU**:
   - Reranking top-20 candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`
     caused latency to surge from **13.68 ms to 1,054.85 ms (a 77x latency
     penalty)**.
   - Crucially, nDCG@10 dropped from **0.7789 to 0.7436** (-3.53%). This
     occurred because MS-MARCO was trained on general web search queries and
     penalized scientific abstract phrasing, proving why Cross-Encoder reranking
     should be deactivated by default in lightweight deployments.

---

## 4. RAG Quality Baseline Across 3 Verification Modes

Evaluated across 13 test scenarios including supported claims, multi-source
citations, fabricated quotes, out-of-order sources, out-of-bounds citations,
epistemic overclaims, and adversarial prompt injections.

| Verification Mode                                            | Unsupported Claim Slip Rate | Prompt Injection Resistance | Refusal Accuracy | P50 Overhead (ms) | P95 Overhead (ms) |
| ------------------------------------------------------------ | --------------------------: | --------------------------: | ---------------: | ----------------: | ----------------: |
| **Mode 1: Structural Citation Validation**                   |                       87.5% |                 0.0% (FAIL) |           100.0% |              0.07 |              0.76 |
| **Mode 2: Structural + Semantic Verification**               |                       12.5% |               100.0% (PASS) |           100.0% |              0.14 |              0.28 |
| **Mode 3: Semantic Verification + Bounded Repair + Refusal** |                    **0.0%** |           **100.0% (PASS)** |       **100.0%** |              0.14 |              0.18 |

### Mode Comparison Insights

- **Mode 1 Failure**: Structural regex citation validation alone fails to catch
  hallucinated content. 87.5% of unsupported claims and 100% of adversarial
  prompt injections slip past Mode 1 because citations look syntactically
  correct (e.g. `[1]`).
- **Mode 2 Protection**: Requiring exact evidence quote matching in
  title/abstract immediately eliminates prompt injection and drops claim
  slippage to 12.5%.
- **Mode 3 Elimination**: Bounded 1-pass repair combined with canonical
  fail-closed refusal reduces unsupported claim slip rate to **0.0%**.
