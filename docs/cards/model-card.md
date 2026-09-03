# Model Card: NLP Academic Search & RAG Engine

## System components

| Layer | Default identifier | Role |
|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Normalized dense vectors for FAISS retrieval |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional cross-encoder ordering of retrieved candidates |
| Generator | Ollama tag `qwen2.5:7b` | Local evidence-constrained answer synthesis |

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
- Citation validation is structural. It detects invalid indices and likely uncited sentences but
  does not prove semantic entailment.

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
2. Grounding failure: the generator may invent or misattribute claims despite instructions.
3. Structural-only citation check: a valid citation index may still not entail the sentence.
4. Corpus staleness and domain bias: answer quality is limited by active abstracts.
5. Unpinned revision: results can change after an upstream model/tag update.
6. Local-service failure: Ollama timeout/unavailability results in a typed error or refusal path.

## Evaluation status

- Unit tests verify retrieval formulas, citation structure, refusal behavior and runner failure
  handling with mock transports.
- The committed three-query retrieval fixture is a pipeline regression fixture, not an external
  quality benchmark.
- SciFact/BEIR results are reportable only when an imported dataset manifest and experiment report
  exist for the run.
- Deterministic `faithfulness_proxy` is not semantic faithfulness and must not be presented as a
  RAGAS/LLM-judge score.
