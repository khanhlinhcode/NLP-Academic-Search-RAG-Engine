# NLP Academic Search & RAG Engine

**Retrieve · Read · Reason**

Hệ thống local-first để tìm kiếm bài báo khoa học và hỏi đáp có dẫn nguồn. Pipeline kết hợp BM25,
Sentence-Transformers, FAISS, Reciprocal Rank Fusion (RRF), Cross-Encoder tùy chọn, FastAPI,
Streamlit và Ollama.

> **Trạng thái:** hardened local MVP. Dự án đã có cấu hình typed, corpus/index manifest, timeout,
> giới hạn concurrency, lỗi có cấu trúc, kiểm tra citation, test tự động, CI và container. Dự án
> chưa được tuyên bố production-ready vì chưa hoàn tất benchmark chuẩn SciFact/BEIR, đánh giá
> faithfulness theo semantic entailment và kiểm thử tải/bảo mật ở quy mô production.

## Mục lục

- [1. Mục tiêu và phạm vi](#1-mục-tiêu-và-phạm-vi)
- [2. Những gì có thể tin cậy](#2-những-gì-có-thể-tin-cậy)
- [3. Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
- [4. Phương pháp truy xuất và RAG](#4-phương-pháp-truy-xuất-và-rag)
- [5. Cấu trúc repository](#5-cấu-trúc-repository)
- [6. Cài đặt và chạy nhanh](#6-cài-đặt-và-chạy-nhanh)
- [7. Cách sử dụng](#7-cách-sử-dụng)
- [8. Dữ liệu, index và khả năng tái lập](#8-dữ-liệu-index-và-khả-năng-tái-lập)
- [9. API](#9-api)
- [10. Đánh giá](#10-đánh-giá)
- [11. Kiểm tra chất lượng](#11-kiểm-tra-chất-lượng)
- [12. Docker](#12-docker)
- [13. Cấu hình và bảo mật](#13-cấu-hình-và-bảo-mật)
- [14. Giới hạn hiện tại](#14-giới-hạn-hiện-tại)
- [15. Tài liệu liên quan](#15-tài-liệu-liên-quan)

## 1. Mục tiêu và phạm vi

Dự án giải quyết hai tác vụ riêng biệt nhưng dùng chung một nền tảng dữ liệu:

| Tác vụ | Mục tiêu | Đầu ra |
|---|---|---|
| **Search** | Tìm và xếp hạng papers phù hợp với một chủ đề. | Danh sách papers, metadata, score và liên kết nguồn. |
| **Ask/RAG** | Tổng hợp câu trả lời chỉ từ evidence đã truy xuất. | Câu trả lời streaming, citation và Evidence ledger. |

Thiết kế local-first giữ corpus, embeddings và LLM trên máy người dùng. FastAPI là lớp dịch vụ
trung tâm; Streamlit chỉ giao tiếp với API và không trực tiếp tải index hoặc gọi model.

## 2. Những gì có thể tin cậy

- Ingestion mới dùng endpoint metadata arXiv OAI-PMH chính thức và giữ lại arXiv ID, tác giả,
  category, ngày, URL, DOI và license khi nguồn cung cấp.
- Các bản ghi cũ có ID dạng `paper_XXXXX` được gắn nguồn
  `legacy-ccdv-arxiv-summarization`. Hệ thống không tự tạo metadata hoặc URL arXiv giả.
- Semantic index phải có `index_manifest.json` ràng buộc corpus hash, thứ tự document ID, số lượng
  và chiều vector, embedding model/revision, chuẩn hóa, loại FAISS và phiên bản thư viện.
- Nội dung paper được coi là **untrusted evidence**, không phải instruction cho LLM.
- Citation validator kiểm tra index nguồn và độ phủ citation một cách xác định. Đây là kiểm tra cấu
  trúc, chưa phải xác minh đầy đủ rằng mọi mệnh đề được nguồn hỗ trợ về mặt ngữ nghĩa.
- Cross-Encoder mặc định tắt cho đến khi một validation set độc lập chứng minh nó cải thiện chất
  lượng đủ để bù chi phí latency.

Corpus runtime nằm trong `data/`, không được commit vào Git. Vì vậy số papers thực tế có thể là
corpus legacy 15.000 bản ghi hoặc một corpus arXiv mới với metadata đã xác minh.

## 3. Kiến trúc hệ thống

### 3.1 Sơ đồ tổng thể

```text
                         OFFLINE / BUILD-TIME

arXiv OAI-PMH
      │
      ▼
validate ── deduplicate ── quarantine
      │
      ▼
versioned corpus + corpus_manifest.json
      │
      ├── BM25 corpus
      └── Sentence-BERT ── FAISS index + index_manifest.json

                         ONLINE / REQUEST-TIME

Browser ──► Streamlit UI ──HTTP/SSE──► FastAPI
                                         │
                         ┌───────────────┴────────────────┐
                         ▼                                ▼
                  Retrieval service                 RAG service
            BM25 + FAISS → RRF → rerank?      retrieve → context → Ollama
                         │                                │
                         ▼                                ▼
                 ranked papers                  answer + citations
```

### 3.2 Trách nhiệm của từng lớp

| Lớp | Trách nhiệm |
|---|---|
| `data` | Mô hình `Paper`, ingestion adapter, làm sạch, validation và manifest. |
| `search` | BM25, embedding/FAISS, fusion, filter và reranking. |
| `rag` | Xây context/prompt, gọi Ollama, streaming và kiểm tra citation. |
| `evaluation` | Metrics retrieval và RAG có thể chạy lặp lại. |
| `api` | Contract HTTP, dependency lifecycle, timeout, concurrency và lỗi có cấu trúc. |
| `ui` | Trình bày Search/Ask, trạng thái hệ thống và Evidence ledger. |
| `scripts` | Entry point cho ingestion, indexing, audit và evaluation; không chứa business logic lõi. |

Hướng phụ thuộc chính là `UI → API → domain services → data/config`. Các module retrieval và RAG
không phụ thuộc giao diện, nhờ đó có thể kiểm thử độc lập và tái sử dụng qua API hoặc CLI.

## 4. Phương pháp truy xuất và RAG

### 4.1 Luồng Search

```text
query
  ├── Unicode-safe tokenize ──► BM25 ─────────────┐
  └── Sentence-BERT ──► normalized vector ─► FAISS├──► RRF ─► reranker? ─► top-k
                                                   ┘
```

- **BM25** ưu tiên từ khóa chính xác, tên phương pháp, từ viết tắt và thuật ngữ hiếm.
- **Dense retrieval** tìm papers diễn đạt cùng ý bằng từ khác thông qua cosine-equivalent inner
  product trên vector đã chuẩn hóa.
- **RRF** hợp nhất thứ hạng thay vì cộng trực tiếp hai score khác thang đo:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

- **Cross-Encoder** có thể chấm lại một candidate pool nhỏ nhưng tăng latency đáng kể.

`rrf_score` là tín hiệu **xếp hạng**, không phải xác suất paper đúng hay phần trăm liên quan.

### 4.2 Luồng Ask/RAG

```text
question
   ↓
hybrid retrieval → top-k evidence → relevance gate
   ↓
context budget: title + metadata + abstract + stable source index
   ↓
system policy + untrusted evidence + user question
   ↓
Ollama / qwen2.5:7b
   ↓
SSE: stage → sources → token → warning/error → done
   ↓
citation validation → answer + Evidence ledger
```

Nếu evidence không vượt ngưỡng phù hợp, hệ thống nên từ chối trả lời thay vì để model đoán. Mỗi
nguồn có index ổn định như `[1]`, `[2]`; cùng index đó được dùng trong prompt, câu trả lời và
Evidence ledger.

## 5. Cấu trúc repository

Dự án dùng **Python src layout**: tên package là `nlp_academic_search`, còn `src/` chỉ là thư mục
chứa source code. Cách này ngăn test vô tình import code từ working directory thay vì package đã
được cài.

```text
.
├── src/
│   └── nlp_academic_search/
│       ├── api/                  # FastAPI app, routes, schemas, service container
│       ├── data/                 # Paper model, preprocessing, manifests, source adapters
│       │   └── sources/          # arXiv OAI-PMH và interface nguồn dữ liệu
│       ├── evaluation/           # Retrieval/RAG metrics
│       ├── rag/                  # Prompt, generator, streaming, citations
│       ├── search/               # BM25, FAISS, fusion, filters, reranker
│       ├── ui/                   # Streamlit client, views, styles, assets
│       └── config.py             # Typed settings từ environment
├── scripts/                      # CLI/operational entry points
├── tests/
│   ├── unit/                     # Deterministic tests, không cần service/model thật
│   └── integration/              # Local service và end-to-end tests có marker
├── benchmarks/
│   ├── retrieval/                # Golden retrieval queries, documents và qrels
│   └── rag/                      # Golden RAG questions và expected evidence
├── data/                         # Corpus/index runtime; không commit
├── docs/
│   ├── architecture.md           # Ranh giới module và dependency rules
│   ├── product.md                # Phạm vi và nguyên tắc sản phẩm
│   └── design.md                 # Design system của UI
├── .github/workflows/ci.yml      # Quality gate trên CI
├── CONTRIBUTING.md               # Quy trình phát triển và kiểm tra thay đổi
├── pyproject.toml                # Package metadata, dependencies và tool configuration
├── uv.lock                       # Dependency lock để tái lập môi trường
├── Makefile                      # Giao diện lệnh thống nhất
├── Dockerfile
└── docker-compose.yml
```

## 6. Cài đặt và chạy nhanh

### 6.1 Yêu cầu

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) nếu sử dụng Ask/RAG

### 6.2 Chuẩn bị môi trường và dữ liệu

```bash
cp .env.example .env
make setup

# Lần chạy thử đầu tiên nên dùng corpus nhỏ.
ARXIV_MAX_RECORDS=1000 make download
make index

# Chỉ cần pull một lần. Nếu `ollama list` hoạt động thì Ollama server đã chạy.
ollama pull qwen2.5:7b
```

Muốn ingest mặc định 15.000 records:

```bash
make download
make index
```

### 6.3 Khởi động ứng dụng

Terminal 1:

```bash
make api
```

Terminal 2:

```bash
make ui
```

- Swagger/OpenAPI: <http://localhost:8000/docs>
- Streamlit UI: <http://localhost:8501>

Không chạy thêm `ollama serve` nếu lệnh đó báo `address already in use` và `ollama list` vẫn trả
về danh sách model; điều đó có nghĩa service đã hoạt động trên cổng `11434`.

## 7. Cách sử dụng

### Search

Nhập một chủ đề hoặc mô tả nhu cầu thông tin, ví dụ:

```text
information retrieval evaluation using precision and recall
```

Chọn `Hybrid · RRF` làm mặc định. Dùng BM25 khi cần khớp thuật ngữ chính xác và Semantic khi câu
query mang tính diễn giải. Metadata filter nên được dùng sau khi đã kiểm tra corpus có category,
năm và tác giả đáng tin cậy.

### Ask

Đặt một câu hỏi cần tổng hợp evidence, ví dụ:

```text
Why do the authors propose novelty-based evaluation in addition to precision and recall?
```

Một câu trả lời ngắn nhưng bám sát abstract và dẫn đúng nguồn tốt hơn một câu trả lời dài chứa kiến
thức ngoài corpus. Evidence ledger cho biết nguồn nào đã truy xuất và nguồn nào thực sự được trích
dẫn.

## 8. Dữ liệu, index và khả năng tái lập

### 8.1 Versioning và atomic activation

Mỗi lần ingestion hoặc build index tạo một version mới. Version đang phục vụ không bị ghi đè. Chỉ
sau khi output mới vượt validation, con trỏ `CURRENT` mới được cập nhật atomically.

Khi API khởi động, index manifest được đối chiếu với:

- SHA-256 của corpus;
- hash thứ tự document ID;
- số lượng và chiều vector;
- embedding model và revision;
- normalization và FAISS metric/type.

Mismatch làm startup fail rõ ràng, tránh trả kết quả sai âm thầm.

### 8.2 Corpus legacy

Không coi `data/raw/papers.jsonl` từ workflow Hugging Face cũ là metadata thư mục học thuật đã xác
minh. Có thể tạo compatibility manifest cho index cũ bằng:

```bash
uv run python -m scripts.build_index --adopt-existing
```

Manifest này chỉ xác minh shape, order và hash; nó không khẳng định provenance của model weights
lịch sử. Để có provenance đầy đủ, hãy ingest lại arXiv và chạy `make index`.

## 9. API

Các route ổn định nằm dưới `/api/v1`; route không version được giữ làm alias tương thích ngược.

| Route | Chức năng |
|---|---|
| `GET /api/v1/search` | Hybrid RRF/weighted search, filter và pagination. |
| `GET /api/v1/search/bm25` | Sparse retrieval. |
| `GET /api/v1/search/semantic` | Dense FAISS retrieval. |
| `POST /api/v1/ask` | Grounded answer không streaming. |
| `POST /api/v1/ask/stream` | SSE gồm `stage`, `sources`, `token`, `warning`, `error`, `done`. |
| `GET /health/live` | Process liveness. |
| `GET /health/ready` | Readiness của corpus, index và RAG dependency. |
| `GET /stats` | Metadata của corpus, index và model. |

Ví dụ:

```bash
curl 'http://localhost:8000/api/v1/search?q=hybrid+retrieval&top_k=10&category=cs.CL'

curl -X POST 'http://localhost:8000/api/v1/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does retrieval-augmented generation use evidence?","top_k":5,"use_reranker":false}'
```

## 10. Đánh giá

### 10.1 Nguyên tắc

- Tách query, corpus và qrels; không tạo query bằng cách chép nguyên văn relevant document.
- Không tune hyperparameter trên test split.
- Báo cáo cả effectiveness và latency.
- Ghi rõ corpus version, model/revision, `k`, seed và cấu hình retrieval.
- Không suy rộng kết luận từ fixture nhỏ sang chất lượng production.

Benchmark 20 query trước đây đã bị loại vì mỗi query được chép từ document liên quan và chỉ có một
relevant item. Các số 100% Recall@10 và 0.9693 nDCG@10 không còn được dùng làm quality claim.

### 10.2 Retrieval evaluation

```bash
make eval-retrieval
```

Fixture `benchmarks/retrieval/in_domain_golden.json` kiểm tra pipeline và metrics. Lần chạy ghi nhận ngày
2026-09-03 chỉ có 3 queries và 3 documents:

| Method | Recall@3 | MRR@3 | MAP@3 | nDCG@3 | p50 latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.6667 | 1.0000 | 0.6667 | 0.8842 | 0.05 ms |
| Dense | 1.0000 | 1.0000 | 0.8889 | 0.9760 | 5.91 ms |
| RRF | 1.0000 | 1.0000 | 0.8889 | 0.9760 | 5.43 ms |

Kết quả này xác nhận hành vi của code, không phải benchmark đủ mạnh. Có thể import BEIR/SciFact mà
không sửa judgments:

```bash
uv run python -m scripts.import_beir path/to/scifact path/to/scifact/qrels/test.tsv \
  data/benchmarks/scifact-test.json --name scifact
uv run python -m scripts.run_evaluation \
  --benchmark data/benchmarks/scifact-test.json -k 10
```

### 10.3 RAG evaluation

```bash
make eval-rag
```

Script đánh giá context precision/recall, answer relevance, citation coverage, refusal correctness,
latency và error rate trên API đang chạy. Nó ghi lại generator model và không giả vờ sử dụng LLM
judge. Semantic entailment-based faithfulness vẫn là hạng mục chưa hoàn tất.

## 11. Kiểm tra chất lượng

```bash
make test            # unit tests, không cần network hoặc model thật
make coverage        # coverage gate 70%
make lint
make format-check
make typecheck
make package
make check           # chạy quality gate cục bộ
uv run pytest -m integration
```

CI chạy lint, format, type checking, unit coverage, package build, Compose validation và kiểm tra
secret/generated data không bị commit.

## 12. Docker

Image dùng multi-stage Python 3.11 build và chạy bằng non-root user. Corpus/index và model cache
được mount thay vì đóng vào image.

```bash
docker compose config --quiet
docker compose up --build

# Smoke test đầy đủ, tốn nhiều tài nguyên hơn:
make docker-smoke
```

Qwen2.5 7B cần vài GB lưu trữ và khoảng 8 GB RAM khả dụng trở lên. GPU acceleration phụ thuộc môi
trường và không được bật trong Compose portable mặc định.

## 13. Cấu hình và bảo mật

Mọi biến môi trường và ràng buộc được mô tả trong `.env.example`. Khi triển khai ngoài máy cá nhân:

- đặt `ENVIRONMENT=production`;
- dùng allowlist cụ thể cho `CORS_ORIGINS`; wildcard bị từ chối;
- không expose Ollama trực tiếp ra mạng không tin cậy;
- đặt authentication/rate limiting ở reverse proxy đáng tin cậy;
- giữ `EMBEDDING_DEVICE=cpu` và `EMBEDDING_NATIVE_THREADS=1` làm mặc định ổn định;
- pin `EMBEDDING_MODEL_REVISION` trước khi build release index;
- không commit `.env`, corpus, embeddings, model weights hoặc evaluation reports cục bộ.

Ứng dụng có bounded workers, generation concurrency limit, deadline, request ID, structured logs và
sanitized errors. Ứng dụng chưa có multi-tenant authorization hoặc distributed job queue.

## 14. Giới hạn hiện tại

- Corpus legacy 15.000 dòng không có metadata tác giả/category/ngày/arXiv đã xác minh.
- Adopted FAISS manifest không chứng minh được historical model-weight provenance.
- Citation validation chưa phải full semantic entailment.
- Chất lượng Ask phụ thuộc trực tiếp vào độ phủ và độ mới của corpus.
- UI là research workspace cục bộ, chưa phải hệ thống hội thoại đa người dùng có persistent memory.
- SciFact/BEIR, reranker calibration, load test và security test production vẫn chưa hoàn tất.

## 15. Tài liệu liên quan

- [Architecture](docs/architecture.md): ranh giới module và quy tắc dependency.
- [Product](docs/product.md): mục tiêu, người dùng và nguyên tắc sản phẩm.
- [Design](docs/design.md): hệ thống thiết kế và hành vi giao diện.
- [Data Card](docs/cards/data-card.md): nguồn dữ liệu, quy trình thu thập, schema và giới hạn.
- [Model Card](docs/cards/model-card.md): mô hình embedding, reranker, generator và latency.
- [Threat Model & Security](docs/security/threat-model.md): phân tích mối đe dọa và kiểm thử an toàn.
- [Experiment Configurations](configs/experiments/): cấu hình thử nghiệm có thể tái lập (TOML).
- [Contributing](CONTRIBUTING.md): cách thiết lập môi trường và quality gate cho thay đổi mới.
- [LICENSE](LICENSE): giấy phép MIT.
