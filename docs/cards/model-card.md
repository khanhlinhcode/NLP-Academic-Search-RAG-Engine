# Model Card: NLP Academic Search & RAG Engine

## System components

| Layer | Default identifier | Role |
|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Normalized dense vectors for FAISS retrieval |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional cross-encoder ordering of retrieved candidates |
| Generator | Ollama tag `qwen2.5:7b` | Local evidence-constrained answer synthesis |
| Cloud generator | Configured Groq model | Remote synthesis without local model weights in the cloud image |
| Semantic verifier | Configured provider/model | Structured claim-to-evidence assessment; disabled unless configured |

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

## Runtime and performance status

Hardware requirements and latency vary with corpus size, model revision, device, quantization,
concurrency and output length. This card intentionally contains no universal latency or memory
claim. Use versioned retrieval/RAG/load reports under the ignored local `reports/` directory and
record hardware plus effective configuration for each measurement.

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
- SciFact/BEIR results are reportable only when an imported dataset manifest and experiment report
  exist for the run.
- Deterministic `faithfulness_proxy` is not semantic faithfulness and must not be presented as a
  RAGAS/LLM-judge score.
- Structured semantic verification adds provider calls, latency and quota usage. Report these costs
  alongside verification coverage; do not compare them as if they were retrieval latency.
