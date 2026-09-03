# System Architecture

## Architectural style

The project uses a Python src layout and separates offline data/index construction from online
retrieval and generation. `nlp_academic_search` contains reusable application logic; `scripts`
contains command-line adapters only.

## Dependency direction

```text
ui → api → services ─┬→ search → data/config
                     └→ rag ───→ data/config

scripts → data/search/evaluation workflows
evaluation → search/rag/data
```

The following reverse dependencies are forbidden:

- `data`, `search` or `rag` importing from `api` or `ui`;
- production modules importing from `evaluation`;
- UI loading the corpus/index or calling Ollama directly;
- API routes implementing retrieval algorithms or prompt construction.

## Offline data path

```text
arXiv OAI-PMH
  → schema validation
  → deduplication and quarantine
  → versioned corpus + corpus manifest
  → Sentence-BERT encoding
  → versioned FAISS index + index manifest
  → atomic CURRENT activation
```

Corpus and index manifests bind provenance, ordered document identity, model configuration and
vector shape. An incompatible active corpus/index pair must fail readiness instead of serving
silent retrieval corruption.

## Online request path

Search requests use BM25 and normalized Sentence-BERT/FAISS retrieval, then combine ranks with RRF.
Cross-Encoder reranking is optional. Ask requests reuse retrieval, budget the selected abstracts as
untrusted context, stream Ollama output over SSE and validate citation indices before completion.

## Runtime boundaries

- FastAPI owns service lifecycle, readiness, concurrency and error contracts.
- Streamlit is an HTTP/SSE client of FastAPI.
- Ollama is an external local dependency, not imported by the UI.
- Runtime corpus, indexes, caches and reports are not package or Git artifacts.
- Golden benchmark fixtures are version-controlled because they define evaluation inputs.

## Change rule

New domain logic belongs in the relevant package module. A CLI script may parse arguments, invoke
that module, serialize a report and choose an exit code; it must not duplicate domain algorithms.
