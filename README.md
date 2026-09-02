# 🔬 NLP Academic Search & RAG Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-blue)](https://github.com/facebookresearch/faiss)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-yellow)](https://huggingface.co)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5:7B-black)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A production-ready **semantic search system** for scientific papers with **LLM-powered question answering** (RAG). Combines **BM25 keyword search**, **SBERT embeddings**, **FAISS vector search**, **Cross-Encoder reranking**, and **Ollama LLM (Qwen2.5-7B)** to deliver accurate, citation-backed answers with 100% recall retrieval.

---

## 🏗️ System Architecture

```
                       User Query / Question
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Query Preprocessor      │
                   └────────────┬────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
    ┌────────────────────┐            ┌────────────────────┐
    │ BM25 Sparse Search │            │ SBERT Dense Search │
    │ (Keyword Match)    │            │ (FAISS Vector Index)│
    └──────────┬─────────┘            └──────────┬─────────┘
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Hybrid Fusion (RRF)     │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Cross-Encoder Reranker  │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Context Prompt Builder  │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Ollama LLM (Qwen2.5-7B) │
                   └────────────┬────────────┘
                                │
                                ▼
                   Response + Source Citations
```

---

## 📊 Benchmark & Evaluation Results

Evaluated on **15,000 arXiv scientific papers** using standard Information Retrieval (IR) metrics:

| Search Method | Precision@10 | Recall@10 | MRR@10 | nDCG@10 | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| **BM25** (Sparse Retrieval) | 0.0950 | 0.9500 | 0.9500 | 0.9500 | 61.18 ms |
| **SBERT** (Dense Retrieval) | 0.0900 | 0.9000 | 0.9000 | 0.9000 | 60.82 ms |
| 🏆 **Hybrid Search (RRF)** | **0.1000** | **1.0000 (100%)** | **0.9600** | **0.9693** | **84.52 ms** |
| **Hybrid + Cross-Encoder Reranker** | 0.0950 | 0.9500 | 0.9500 | 0.9500 | 2508.41 ms |

> 📌 **Key Technical Insight**: **Hybrid Search (BM25 + SBERT with Reciprocal Rank Fusion)** achieves **100% Recall@10** and **0.9693 nDCG@10** at ultra-fast latency (~84.5ms), providing optimal candidates for RAG generation.

---

## 📁 Detailed Project Structure

```
NLP Academic Search & RAG Engine/
├── data/                         # Data directory (Git ignored)
│   ├── raw/                      # Downloaded arXiv JSONL dataset (15,000 papers)
│   ├── embeddings/               # FAISS vector index & pre-computed embeddings
│   └── processed/                # Evaluation benchmark results
├── src/                          # Core source code
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration dataclass & env vars
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # Paper dataclass & JSONL dataset loader
│   │   └── preprocessor.py       # Text cleaning, LaTeX removal & BM25 tokenizer
│   ├── search/
│   │   ├── __init__.py
│   │   ├── bm25_search.py        # BM25Okapi sparse keyword retrieval engine
│   │   ├── semantic_search.py    # Sentence-BERT + FAISS dense retrieval engine
│   │   ├── hybrid_search.py      # Reciprocal Rank Fusion (RRF) & Weighted linear fusion
│   │   └── reranker.py           # Cross-Encoder (ms-marco-MiniLM) reranking model
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py     # Context prompt construction with citation grounding
│   │   └── generator.py          # Ollama client integration (Sync & Streaming SSE)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application & lifespan state management
│   │   ├── schemas.py            # Pydantic request/response schemas
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── search.py         # /search, /search/bm25, /search/semantic endpoints
│   │       └── rag.py            # /ask, /ask/stream endpoints
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py            # Precision@K, Recall@K, MRR, nDCG implementation
├── scripts/                      # Execution scripts
│   ├── __init__.py
│   ├── download_data.py          # Download arXiv dataset from Hugging Face
│   ├── build_index.py            # Compute SBERT embeddings & build FAISS index
│   └── run_evaluation.py         # Run IR benchmarking suite across all methods
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_bm25.py              # BM25 engine unit tests
│   └── test_metrics.py           # IR evaluation metrics unit tests
├── .vscode/                      # IDE workspace configuration
│   └── settings.json
├── .env.example                  # Environment variables template
├── .gitignore                    # Standard Python gitignore
├── Dockerfile                    # Multi-stage production container setup
├── docker-compose.yml            # FastAPI + Ollama orchestration
├── Makefile                      # Command shortcuts for quick execution
├── pyproject.toml                # Project metadata & pytest configuration
├── pyrightconfig.json            # Linter & type checker path resolution
├── requirements.txt              # Frozen Python dependencies
└── README.md                     # Project documentation
```

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed and running

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nlp-academic-search-rag.git
cd "NLP Academic Search & RAG Engine"

# Create virtual environment & install dependencies in editable mode
make setup
source venv/bin/activate
```

### 2. Download Dataset
```bash
# Download 15,000 arXiv paper summaries from HuggingFace
make download
```

### 3. Build Vector Index
```bash
# Generate SBERT embeddings & build FAISS index
make index
```

### 4. Start Ollama Service & Pull Model
```bash
# Start Ollama
brew services start ollama

# Pull Qwen2.5-7B model
ollama pull qwen2.5:7b
```

### 5. Launch FastAPI Server
```bash
make api
# App running at: http://localhost:8000
# Interactive Swagger UI Docs: http://localhost:8000/docs
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
|---|:---:|---|
| `/search` | `GET` | **Hybrid Search** (BM25 + SBERT + RRF) |
| `/search/bm25` | `GET` | Sparse Keyword Search only |
| `/search/semantic` | `GET` | Dense Vector Similarity Search only |
| `/ask` | `POST` | **RAG Query** — Answer generation with citation sources |
| `/ask/stream` | `POST` | **Streaming RAG Query** — Server-Sent Events (SSE) stream |
| `/health` | `GET` | Server health check & status |
| `/stats` | `GET` | System statistics (total papers, models used) |
| `/docs` | `GET` | OpenAPI / Swagger UI |

### Example: Search Query
```bash
curl -s "http://localhost:8000/search?q=transformer+attention+mechanism&top_k=3"
```

### Example: RAG Question Answering
```bash
curl -s -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main methods for text classification using deep learning?", "top_k": 5}'
```

---

## 🧪 Running Unit Tests & Benchmarks

```bash
# Run unit tests (21/21 passed)
make test

# Run full evaluation benchmark suite
make eval
```

---

## 🐳 Docker Deployment

```bash
# Start container stack
docker-compose up -d

# Pull LLM inside Ollama container
docker exec nlp-search-ollama ollama pull qwen2.5:7b
```

---

## 🛠️ Technology Stack

- **Core Engine**: Python 3.9+, PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Sparse Retrieval**: BM25Okapi (`rank-bm25`)
- **Reranker**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **LLM Engine**: Ollama (Qwen2.5-7B)
- **Web API**: FastAPI, Uvicorn, Pydantic v2
- **Testing & Benchmark**: Pytest, NumPy, Scikit-learn
- **DevOps**: Docker, Docker Compose, Makefile

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
