# Model Card: NLP Academic Search & RAG Engine

## System components

| Layer | Default identifier | Role |
|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Normalized dense vectors for FAISS retrieval |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional cross-encoder ordering of retrieved candidates |
| Generator | Ollama tag `qwen2.5:1.5b` (local default) / `qwen2.5:7b` | Local evidence-constrained answer synthesis |
| Cloud generator | Configured Groq model (`openai/gpt-oss-20b`) | Remote synthesis without local model weights in the cloud image |
| Semantic verifier | Deterministic exact-quote validator / Groq model | Structured claim-to-evidence assessment; disabled unless configured |

Exact licenses and acceptable uses must be checked against the selected upstream model revisions
when producing a release. An unpinned model name or Ollama tag is not a reproducible revision.

## Intended use

The system supports scientific-paper discovery and question answering over its active abstract
corpus. It is not designed for medical/legal decisions, autonomous literature review, factual
claims outside the corpus, or multi-tenant production use without additional controls.

## Inputs, outputs and grounding

- Embedding input: paper title plus abstract; output dimension is recorded in the active index
  manifest instead of assumed by documentation.
- Generator input: a system grounding policy and XML-delimited untrusted source excerpts.
- Generator output: text or SSE tokens with numbered citations.
- Layer 1 citation validation is sentence-scoped and structural. It detects invalid indices and
  uncited factual sentences, but does not prove semantic entailment.
- Layer 2 semantic verification decomposes factual content into claims, requires strict structured
  output and exact evidence quotes, and validates each quote against the cited title or abstract.
- Both enabled layers share one bounded repair budget. In fail-closed mode, an answer that remains
  invalid is replaced by a refusal rather than served as ready.
- Verification by the same model identifier as generation is explicitly marked non-independent.

Semantic support is scoped to the retrieved corpus. Exact quote validation proves that text exists
in a source; it does not prove that the source is factually correct, current or complete. The system
does not claim to eliminate hallucinations or establish real-world truth.

The default context budget is 24,000 characters. If a source exceeds the remaining budget, the
current implementation truncates its abstract by character count and appends an ellipsis; it does
not guarantee sentence-boundary truncation.

---

## Verified Empirical Benchmark (November 2026)

All figures below are strictly verified on Git commit `cb8d0bc83b041f3ab04299b81c5015e44042a710` on an **Apple M1 (8 cores), 8 GB Unified Memory**, evaluated against the **BEIR / SciFact** scientific corpus (5,183 peer-reviewed paper abstracts). Detailed reproduction artifacts reside in [`docs/benchmarks/scifact-2026-11/`](../benchmarks/scifact-2026-11/).

### 1. Holdout Retrieval Benchmark (100 Unseen Queries, 3-Trial Aggregate)

To avoid test-set overfitting, hyperparameter tuning ($k$, candidate pool depth) was conducted exclusively on a separate **Dev split (100 queries)**. The table below represents 3 independent trials on the **Holdout test split (100 unseen queries)**:

| Method | Recall@5 (Mean $\pm$ Std) | nDCG@10 (Mean $\pm$ Std) | HitRate@5 (Mean) | Latency P50 (Median $\pm$ CI95) |
|---|---:|---:|---:|---:|
| **BM25 (Sparse Okapi)** | $0.6817 \pm 0.0$ | $0.6254 \pm 0.0$ | 0.7000 | $9.00 \pm 2.13\text{ ms}$ |
| **Dense SBERT (`all-MiniLM-L6-v2`)** | $0.6753 \pm 0.0$ | $0.5858 \pm 0.0$ | 0.7000 | $7.02 \pm 0.71\text{ ms}$ |
| **Hybrid RRF ($k=60, \text{pool}=50$) [Default]** | **$0.7387 \pm 0.0$** | **$0.6389 \pm 0.0$** | **0.7600** | **$15.94 \pm 5.33\text{ ms}$** |
| **Hybrid RRF ($k=20, \text{pool}=20$)** | $0.7187 \pm 0.0$ | $0.6380 \pm 0.0$ | 0.7400 | $17.74 \pm 0.99\text{ ms}$ |
| **Hybrid + Cross-Encoder (Depth 20)** | $0.6920 \pm 0.0$ | $0.6295 \pm 0.0$ | 0.7200 | $1,135.72 \pm 115.44\text{ ms}$ |
| **Hybrid + Cross-Encoder (Depth 10)** | **$0.7220 \pm 0.0$** | **$0.6527 \pm 0.0$** | **0.7500** | **$664.11 \pm 24.96\text{ ms}$** |

*Key Findings*:
- Hybrid RRF achieved **+5.70 pp (+8.36% relative) Recall@5 gain** over BM25 and **+6.34 pp (+9.39% relative) gain** over Dense retrieval at 15.94 ms median latency.
- Pruning Cross-Encoder depth from 20 to 10 cut latency by **41.5%** ($1,135\text{ ms} \to 664\text{ ms}$) and improved nDCG@10 from 0.6295 to 0.6527 by filtering borderline noise.
- Causal hypothesis for Cross-Encoder underperformance: Pretrained MS-MARCO weights experience domain mismatch on biomedical literature without domain adaptation.

### 2. Grounding & Hallucination Mitigation (20 Target Scenarios)

Evaluated across 20 deterministic scenarios (13 Dev split + 7 Holdout split), encompassing 6 supported, 12 unsupported/fabricated/overclaimed, 2 canonical refusal, and 2 adversarial prompt injection cases:

| Metric | Mode 1 (Structural Regex) | Mode 2 (Deterministic Exact-Quote) | Mode 3 (Bounded Repair + Refusal) |
|---|---:|---:|---:|
| **Unsupported Claim Slip Rate** | **83.3% (10/12 passed)** | **0.0% (0/12 passed)** | **0.0% (0/12 passed)** |
| **Prompt Injection Resistance** | **0.0% (0/2 blocked)** | **100.0% (2/2 blocked)** | **100.0% (2/2 blocked)** |
| **Refusal Accuracy on Empty Context** | **100.0% (2/2)** | **100.0% (2/2)** | **100.0% (2/2)** |
| **Local Verification Latency P50** | **0.107 ms** | **0.297 ms** | **0.261 ms** |

*Scope Note*: Latency reflects local deterministic Python NFKC exact-quote verification only. Live LLM generation and network API calls are excluded.

### 3. Resource & Memory Profiling

- **Process Peak RSS (High-Water Mark)**: `766.62 MB` on macOS arm64 (CPython runtime + PyTorch engine + SentenceTransformers + FAISS).
- **BM25 Standalone Inverted Index**: `68.83 MB` traced Python heap (`tracemalloc`).
- **FAISS IndexFlatIP (C++ memory & disk)**: `7.59 MB` ($5,183 \times 384 \times 4\text{ bytes}$).
- **Total Persistent Index Storage**: `15.18 MB` (`scifact_faiss.index` + `scifact_embeddings.npy`).
- **Cloud 512 MB Container Stability**: `NOT VERIFIED / ARCHITECTURAL PROFILE ONLY`. The cloud profile uses lazy imports to omit PyTorch/FAISS, but has not been stress-tested under live multi-tenant traffic.

---

## Failure modes

1. Retrieval miss: relevant evidence may not enter the candidate pool.
2. Retrieval-bound verification: a correct claim may be withheld when supporting evidence is not
   retrieved, trading answer coverage for grounding safety.
3. Judge error: a semantic verifier may accept unsupported content or reject supported content;
   deterministic exact-quote checks constrain but do not eliminate this risk.
4. Source error: an exact quote may come from an incorrect or outdated paper.
5. Corpus staleness and domain bias: answer quality is limited by active abstracts.
6. Unpinned revision: results can change after an upstream model/tag update.
7. Provider failure: timeout, quota or circuit-breaker activation causes a typed unavailable state
   and, when required, a fail-closed refusal.

## Evaluation status

- Unit tests verify retrieval formulas, citation structure, exact quote checks, typed provider
  failures, bounded repair, refusal behavior and runner failure handling with mock transports.
- The committed three-query retrieval fixture is a pipeline regression fixture, not an external
  quality benchmark.
- SciFact results are fully published and reproducible under `docs/benchmarks/scifact-2026-11/`.
- Deterministic `faithfulness_proxy` is not semantic faithfulness and must not be presented as a
  RAGAS/LLM-judge score.
- Structured semantic verification adds provider calls, latency and quota usage. Report these costs
  alongside verification coverage; do not compare them as if they were retrieval latency.
