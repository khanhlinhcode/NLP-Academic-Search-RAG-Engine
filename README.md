# 🔬 NLP Academic Search & RAG Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A production-ready **semantic search system** for scientific papers with **LLM-powered question answering** (RAG). Combines BM25 keyword search, SBERT embeddings, FAISS vector search, and Ollama LLM to deliver accurate, citation-backed answers.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌───────────────┐
│ Query Process │
└───────┬───────┘
        │
 ┌──────┴───────┐
 ▼              ▼
BM25         SBERT
Keyword      Semantic
Search       Search
 │              │
 └──────┬───────┘
        ▼
Hybrid Retrieval (RRF)
        │
        ▼
Cross-Encoder Reranker
        │
        ▼
     Top-K Papers
        │
        ▼
   Ollama LLM (RAG)
        │
        ▼
 Answer + Sources
```

### Pipeline Steps

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Query Processing** | Tokenize and prepare user input |
| 2 | **BM25 Search** | Sparse retrieval via keyword matching |
| 3 | **Semantic Search** | Dense retrieval via SBERT + FAISS |
| 4 | **Hybrid Fusion** | Combine results using Reciprocal Rank Fusion |
| 5 | **Reranking** | Cross-Encoder re-scores top candidates |
| 6 | **RAG** | LLM generates answer with citations |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~2GB disk space for models and data

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nlp-academic-search.git
cd nlp-academic-search

# Create virtual environment & install dependencies
make setup
source venv/bin/activate
```

### 2. Download Dataset

```bash
# Download ~15,000 arXiv papers
make download

# Or specify a custom count
python -m scripts.download_data --max-papers 5000
```

### 3. Build Search Index

```bash
# Compute SBERT embeddings and build FAISS index
make index
```

### 4. Start Ollama

```bash
# Pull a model (if you haven't already)
ollama pull qwen2.5:7b

# Ollama should be running at http://localhost:11434
```

### 5. Start the API

```bash
make api
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Pull the LLM model inside the Ollama container
docker exec nlp-search-ollama ollama pull qwen2.5:7b
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | `GET` | **Hybrid search** (BM25 + Semantic) |
| `/search/bm25` | `GET` | BM25 keyword search only |
| `/search/semantic` | `GET` | SBERT semantic search only |
| `/ask` | `POST` | **RAG** — Question answering with sources |
| `/ask/stream` | `POST` | Streaming RAG response |
| `/health` | `GET` | Health check |
| `/stats` | `GET` | Dataset & model statistics |
| `/docs` | `GET` | Swagger UI documentation |

### Example: Search

```bash
curl "http://localhost:8000/search?q=transformer+attention+mechanism&top_k=5"
```

```json
{
  "query": "transformer attention mechanism",
  "method": "hybrid",
  "total_results": 5,
  "results": [
    {
      "id": "paper_00042",
      "title": "Attention Is All You Need",
      "abstract": "The dominant sequence transduction models...",
      "score": 0.0328
    }
  ],
  "latency_ms": 18.45
}
```

### Example: RAG

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the transformer architecture improve machine translation?", "top_k": 5}'
```

```json
{
  "question": "How does the transformer architecture improve machine translation?",
  "answer": "The transformer architecture improves machine translation through several key innovations [1][3]...",
  "sources": [
    {"index": 1, "title": "Attention Is All You Need", "id": "paper_00042"},
    {"index": 2, "title": "Neural Machine Translation...", "id": "paper_00187"}
  ],
  "retrieval_method": "hybrid + reranker",
  "latency_ms": 2340.12
}
```

---

## 📊 Evaluation

Run benchmarks across all search methods:

```bash
make eval
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of retrieved results that are relevant |
| **Recall@K** | Fraction of relevant documents found |
| **MRR@K** | How early the first relevant result appears |
| **nDCG@K** | Overall ranking quality |
| **Latency** | Response time in milliseconds |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | Core language |
| ML Framework | PyTorch | Neural model inference |
| Transformers | Hugging Face | Pre-trained model loading |
| Embeddings | Sentence-BERT (`all-MiniLM-L6-v2`) | Semantic embeddings |
| Keyword Search | BM25 (rank-bm25) | Sparse retrieval |
| Vector Search | FAISS | Approximate nearest neighbor |
| Hybrid Fusion | Reciprocal Rank Fusion | Score combination |
| Reranking | Cross-Encoder (`ms-marco-MiniLM`) | Result refinement |
| LLM | Qwen/Llama via Ollama | RAG answer generation |
| API | FastAPI + Uvicorn | REST API backend |
| Data | JSONL + Pandas | Dataset processing |
| Evaluation | Precision, Recall, MRR, nDCG | IR metrics |
| Deployment | Docker + Docker Compose | Containerization |

---

## 📁 Project Structure

```
├── src/
│   ├── config.py                 # App configuration
│   ├── data/
│   │   ├── loader.py             # Dataset loading
│   │   └── preprocessor.py       # Text preprocessing
│   ├── search/
│   │   ├── bm25_search.py        # BM25 keyword search
│   │   ├── semantic_search.py    # SBERT + FAISS
│   │   ├── hybrid_search.py      # Fusion logic
│   │   └── reranker.py           # Cross-Encoder
│   ├── rag/
│   │   ├── prompt_builder.py     # RAG prompt construction
│   │   └── generator.py          # Ollama LLM integration
│   ├── api/
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Pydantic models
│   │   └── routes/
│   │       ├── search.py         # Search endpoints
│   │       └── rag.py            # RAG endpoints
│   └── evaluation/
│       └── metrics.py            # IR evaluation metrics
├── scripts/
│   ├── download_data.py          # Dataset download
│   ├── build_index.py            # Build FAISS index
│   └── run_evaluation.py         # Run benchmarks
├── tests/                        # Unit & integration tests
├── data/                         # Dataset & embeddings (gitignored)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🎯 Skills Demonstrated

| Skill | Evidence |
|-------|---------|
| **NLP** | Text preprocessing, tokenization, embeddings |
| **Transformers** | SBERT, Cross-Encoder models |
| **Information Retrieval** | BM25, dense retrieval, hybrid search |
| **Embeddings** | Sentence embeddings, vector similarity |
| **Vector Database** | FAISS indexing, ANN search |
| **RAG** | Retrieval-Augmented Generation pipeline |
| **LLM** | Ollama integration, prompt engineering |
| **API Engineering** | FastAPI, REST design, Pydantic schemas |
| **Evaluation** | IR metrics (Precision, Recall, MRR, nDCG) |
| **Software Engineering** | Modular design, typing, Docker |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
