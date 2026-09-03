# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Inferred from the project brief: researchers, students, and technical practitioners who need to find relevant computer-science and AI papers and synthesize evidence-backed answers while exploring a literature question.

## Product Purpose

The NLP Academic Search & RAG Engine provides scientific-paper discovery and question answering over an indexed arXiv corpus. Success means users can retrieve relevant papers with keyword or meaning-based queries, understand why results are relevant, and receive answers grounded in identifiable source papers.

## Positioning

The product combines BM25 sparse retrieval, Sentence-BERT dense retrieval, Reciprocal Rank Fusion, optional Cross-Encoder reranking, and local Ollama generation in one inspectable pipeline with citation-backed output.

## Operating Context

The product is currently operated locally through a FastAPI service and command-line workflows. The requested UI adds an interactive search and streaming RAG workspace. Data preparation, FAISS indexing, evaluation, API serving, and Ollama remain separate local services or commands.

## Capabilities and Constraints

- The runtime corpus is local and may be either the 15,000-record legacy summarization corpus or a
  newly ingested arXiv corpus. Legacy text is searchable, but its historical bibliographic metadata
  and embedding-weight provenance are not verified.
- Search modes are BM25, semantic, and hybrid retrieval; RAG uses hybrid retrieval with optional reranking.
- The local LLM is configured as `qwen2.5:7b` through Ollama.
- The FastAPI service is the source of truth for UI data and model readiness.
- Abstracts are available. Authors, categories, dates, DOI/arXiv identifiers, and source links are
  displayed only when validated metadata supplies them.
- Inferred for this UI task: the experience is desktop-first, responsive, and must explain missing API, index, or Ollama availability without failing silently.
- Language localization beyond the existing English academic content is undecided.

## Brand Commitments

The confirmed product name is “NLP Academic Search & RAG Engine.” The established voice is technical, evidence-led, and transparent about retrieval methods, latency, model availability, and citations.

## Evidence on Hand

- The former self-retrieval benchmark is retained only as invalid historical output under ignored
  local data. The checked golden fixture and current limitations are documented in `README.md`.
- Search and RAG implementation under `src/nlp_academic_search/search/`,
  `src/nlp_academic_search/rag/`, and `src/nlp_academic_search/api/`.
- A local paper corpus and FAISS/embedding assets under `data/`.
- No testimonials, customer logos, production usage evidence, or visual brand assets are present; future work must not fabricate them.

## Product Principles

1. Keep evidence inspectable: answers, citations, retrieval method, and source details remain visible.
2. Prefer useful relevance over algorithm theater: advanced controls are available without obstructing the main query flow.
3. Fail informatively when local dependencies are unavailable.
4. Preserve a local-first workflow and avoid requiring external hosted model services.
5. Treat measured benchmark data as evidence, not as a promise beyond the evaluated corpus and query set.

## Accessibility & Inclusion

Inferred baseline for the new web interface: keyboard-operable controls, visible focus, semantic labels, sufficient contrast, reduced-motion respect, and responsive reading widths.
