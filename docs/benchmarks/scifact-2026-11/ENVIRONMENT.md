# Benchmark Environment Specification — Project A (Final Validation)

**Project**: NLP Academic Search & RAG Engine
**Repository**: [https://github.com/khanhlinhcode/NLP-Academic-Search-RAG-Engine](https://github.com/khanhlinhcode/NLP-Academic-Search-RAG-Engine)
**Git Commit SHA**: `cb8d0bc83b041f3ab04299b81c5015e44042a710` (Branch: `main`)
**Audit Timestamp**: 2026-11-06T09:25:06Z

---

## 1. Hardware Specification & CPU Throttling Characteristics

| Parameter | Measured Specification |
|---|---|
| **Host Machine** | Apple MacBook Air (M1, 2020) |
| **Operating System** | macOS Darwin 27.0.0 (`arm64-apple-darwin`) |
| **CPU Architecture** | Apple M1 (8 cores: 4 Performance cores @ 3.2 GHz, 4 Efficiency cores @ 2.0 GHz) |
| **Thermal Design** | Passive / Fanless cooling (under sustained single-threaded compute, thermal throttling introduces ~1-2 ms latency variance) |
| **System Memory (RAM)** | 8.00 GB Unified Memory (`hw.memsize = 8589934592 bytes`) |
| **Hardware Threads** | CPU single-threaded execution configured for deterministic evaluation (`torch.set_num_threads(1)`, `faiss.omp_set_num_threads(1)`) |
| **OpenMP Runtime Safety** | `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` configured to avoid OpenMP library collisions on macOS arm64 |

---

## 2. Memory Accounting Breakdown

| Scope | Measured Memory | Accounting Method & Boundaries |
|---|---:|---|
| **Process Peak High-Water Mark (RSS)** | **766.62 MB** | Measured via `resource.getrusage(RUSAGE_SELF).ru_maxrss`. Includes CPython runtime, PyTorch C++ static engine, Accelerate framework allocations, SentenceTransformers model weights, and FAISS. Monotonically non-decreasing lifetime peak. |
| **BM25 Standalone Inverted Index** | **68.83 MB** | Measured via `tracemalloc`. Python dictionary, tokenized vocabulary, and document posting lists for 5,183 documents. |
| **FAISS IndexFlatIP (C++ memory)** | **7.59 MB** | Exact theoretical & disk size: $5,183 \times 384 \times 4\text{ bytes} = 7,961,088\text{ bytes}$. |
| **Raw Embeddings Matrix (`npy`)** | **7.59 MB** | Persisted float32 NumPy array on disk. |
| **Total Semantic Disk Footprint** | **15.18 MB** | Sum of `scifact_faiss.index` (7.59 MB) and `scifact_embeddings.npy` (7.59 MB). |
| **Cloud 512 MB Container Limit** | **NOT RUN / THEORETICAL ONLY** | Architectural profile uses lazy imports to omit PyTorch/FAISS on cloud API, but live cluster execution was not performed. |

---

## 3. Dataset Splits & Provenance

Corpus source: **BEIR / SciFact** (Scientific Fact Checking benchmark from PubMed abstracts; Wadden et al., EMNLP 2020; Thakur et al., NeurIPS 2021).

| Split | Document Count | Query Count | Purpose & Leakage Prevention |
|---|---:|---:|---|
| **Full Corpus** | 5,183 papers | - | Common indexing target for all retrieval architectures. |
| **Dev Split** (`scifact_dev_100.json`) | 5,183 papers | 100 queries | Queries 0..99 (Avg 1.16 qrels/query). Used exclusively for parameter exploration ($k$, candidate pool, reranker depth). |
| **Holdout Test Split** (`scifact_holdout_100.json`) | 5,183 papers | 100 queries | Queries 100..199 (Avg 1.10 qrels/query). Untouched unseen test set evaluated across 3 independent trials. |
| **RAG Dev Split** | 13 scenarios | - | Golden cases testing citation structure and exact-quote matching. |
| **RAG Holdout Split** | 7 scenarios | - | Unseen test cases testing roleplay injection, statistical contradiction, negation inversion, and Vietnamese refusal. |

---

## 4. SHA-256 Checksums for Scientific Reproducibility

```text
acc102140659276720fb5a9a7e071f468567572c123ac8f787f5c4c495159fb4  data/benchmarks/scifact_dev_100.json
31d0617507c22112c68c754a357b2a3183b280a5a9dfd89da6c8ea87a106766a  data/benchmarks/scifact_holdout_100.json
2fd14ed86f300c6e3c2a56ce017631c5978b790625264269776f70a51844ba36  benchmarks/rag/semantic_verification_golden.json
b89f5b17fae57e748c81de1459833bf5cba24e998e4f0cf097179f5bbff03265  benchmarks/rag/holdout_verification_golden.json
2ce98c38fa63810b88c0923b5f65e90ac545b4d74b4e6629d230570ffe10549c  scripts/benchmark_retrieval_audit.py
0d418ca5b99918eb163f5d10b55815ad1dfdd112acfc2470d8a7c82c7fc083eb  scripts/benchmark_rag_modes.py
```
