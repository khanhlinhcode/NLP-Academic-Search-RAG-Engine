---
name: nlp-model-design
description: Complete architectural standards and best practices for building scalable Natural Language Processing (NLP), Information Retrieval (IR), and Large Language Model (LLM) search/RAG applications.
version: 1.0.0
---

# NLP Model Design & Architecture Skill

This skill defines the technical standards for designing, evaluating, and operating production-grade **NLP**, **Information Retrieval (IR)**, and **RAG (Retrieval-Augmented Generation)** systems.

---

## 1. Core NLP & RAG Architecture Blueprint

A robust Production-Grade NLP Search & RAG system must adhere to a modular, decoupled pipeline:

```
[Query Input] ──> [Query Preprocessing & Intent]
                       │
                       ▼
        [Hybrid Retrieval Engine]
        ├── Sparse Retrieval (BM25 / Okapi)
        └── Dense Retrieval (Embeddings / Vector DB: Qdrant)
                       │
                       ▼
         [Reciprocal Rank Fusion (RRF)]
                       │
                       ▼
        [Context Builder & Prompt Sanitizer]
                       │
                       ▼
           [LLM Generator Provider]
           (Groq / Ollama / OpenAI API)
                       │
                       ▼
       [Verification & Safety Pipeline]
       ├── Layer 1: Structural Citation Validator
       └── Layer 2: Semantic Evidence Verifier
                       │
                       ▼
            [SSE Streaming / API Output]
```

---

## 2. Information Retrieval (IR) Best Practices

### A. Hybrid Search & Re-ranking
- **Sparse (BM25):** Ideal for exact keyword matching, technical terms, paper IDs, author names, and specific acronyms.
- **Dense (Vector Search):** Ideal for semantic similarity, natural language questions, and cross-lingual concept matching.
- **Reciprocal Rank Fusion (RRF):** Combine rankings without needing score normalization:
  $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$ (where $k \approx 60$).

### B. Text Chunking & Preprocessing
- Maintain document boundaries (Title, Abstract, Metadata, Citations).
- Use sentence-aware boundaries rather than fixed token slices to prevent broken facts.
- Normalization: Apply Unicode NFKC normalization and collapse whitespace before indexing or vectorizing.

---

## 3. Resource & Memory Footprint Optimization (Cloud MLOps)

- **Strict RAM Limits:** For cloud hosts with strict memory ceilings (e.g., Render Free 512 MB RAM), **NEVER** embed heavy local model weights (`torch`, `spacy`, `sentence-transformers`) into the cloud API container.
- **Provider-Neutral API Abstraction:** Offload heavy embedding generation and LLM inference to cloud providers (Groq, OpenAI, Cohere) or lightweight vector DB endpoints (Qdrant Cloud).
- **Separation of Environments:** Keep local GPU/ML model experiments isolated in `notebooks/` or separate worker containers.

---

## 4. Prompt Engineering & Guardrails

- **System Prompts:** Instruct models strictly to base factual answers *only* on provided context chunks.
- **Prompt Injection Defense:** Treat all retrieved paper abstracts and external text as **UNTRUSTED** user input. Never execute code or allow instructions embedded inside retrieved documents to alter system behavior.
- **Refusal Protocol:** Instruct LLMs to explicitly return standard refusal statements (e.g., `"Not enough verified evidence in the retrieved sources."`) if retrieved documents lack sufficient context.

---

## 5. NLP Model Card & Evaluation Standards

Every production NLP system must maintain documented model cards (`docs/cards/model-card.md`) covering:

1. **Intended Domain & Scope:** Academic literature, technical manuals, domain-specific text.
2. **Retrieval Performance Metrics:**
   - **MRR (Mean Reciprocal Rank)**
   - **NDCG@K (Normalized Discounted Cumulative Gain)**
   - **Precision@K & Recall@K**
3. **Generation & Verification Metrics:**
   - **Citation Coverage:** Percentage of factual claims backed by valid indices.
   - **Source Utilization:** Percentage of retrieved sources referenced in the answer.
   - **Evidence Quote Validity:** Exact match ratio of claimed quotes against retrieved source text.
4. **Latency & Cost Profiling:** p50/p95 latency breakdown across Retrieval, Generation, and Verification stages.
