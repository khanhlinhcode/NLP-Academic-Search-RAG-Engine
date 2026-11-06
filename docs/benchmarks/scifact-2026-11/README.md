# SciFact Benchmark Package (November 2026)

**Benchmark Target**:
[NLP Academic Search & RAG Engine](https://github.com/khanhlinhcode/NLP-Academic-Search-RAG-Engine)
**Evaluated Commit SHA**: `cb8d0bc83b041f3ab04299b81c5015e44042a710`
**Dataset**: BEIR / SciFact (Scientific Fact Checking) **Corpus**: 5,183
scientific paper abstracts from PubMed **Query Splits**: 100 Dev Queries
(Hyperparameter Sweep) / 100 Holdout Queries (3-Trial Validation)

---

## 1. Directory Structure

```text
docs/benchmarks/scifact-2026-11/
├── README.md                      # This package overview and reproduction guide
├── ENVIRONMENT.md                 # Detailed hardware, OS, Python and memory specification
├── BASELINE.md                    # Baseline measurements for 4 retrieval architectures and 3 RAG modes
├── RESULTS.md                     # Final verified multi-trial results and verification status matrix
├── OPTIMIZATION.md                # Detailed sweeps (candidate pool, RRF k, Cross-Encoder depth)
├── metrics.json                   # Machine-readable consolidated JSON metrics
├── results.csv                    # Consolidated CSV table for spreadsheet/chart import
├── retrieval_final_validated.json # Complete raw execution output across all 3 trials
└── rag_final_validated.json       # Complete raw execution output across 20 RAG scenarios
```

---

## 2. File Checksums (SHA-256)

```text
acc102140659276720fb5a9a7e071f468567572c123ac8f787f5c4c495159fb4  data/benchmarks/scifact_dev_100.json
31d0617507c22112c68c754a357b2a3183b280a5a9dfd89da6c8ea87a106766a  data/benchmarks/scifact_holdout_100.json
2fd14ed86f300c6e3c2a56ce017631c5978b790625264269776f70a51844ba36  benchmarks/rag/semantic_verification_golden.json
b89f5b17fae57e748c81de1459833bf5cba24e998e4f0cf097179f5bbff03265  benchmarks/rag/holdout_verification_golden.json
2ce98c38fa63810b88c0923b5f65e90ac545b4d74b4e6629d230570ffe10549c  scripts/benchmark_retrieval_audit.py
0d418ca5b99918eb163f5d10b55815ad1dfdd112acfc2470d8a7c82c7fc083eb  scripts/benchmark_rag_modes.py
```

---

## 3. Reproduction Instructions

### Step 1: Install Dependencies

```bash
git checkout cb8d0bc83b041f3ab04299b81c5015e44042a710
uv sync --all-extras --python 3.11
```

### Step 2: Acquire Official SciFact Dataset

```bash
mkdir -p data/benchmarks
curl -sSL -o data/benchmarks/scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip -q -o data/benchmarks/scifact.zip -d data/benchmarks/

# Convert official TSV qrels to project format
uv run python -m scripts.import_beir \
  data/benchmarks/scifact \
  data/benchmarks/scifact/qrels/test.tsv \
  data/benchmarks/scifact-test.json \
  --name scifact
```

### Step 3: Run Multi-Trial Retrieval Benchmark

```bash
uv run python scripts/benchmark_retrieval_audit.py
```

### Step 4: Run RAG Groundedness & Hallucination Benchmark

```bash
uv run python scripts/benchmark_rag_modes.py
```

---

## 4. Reproduction Expectations & Boundaries

- **Exact Numerical Reproduction**: Ranking metrics (`Recall@5`, `nDCG@10`,
  `HitRate@5`, `MRR@10`) and verification decisions are 100% deterministic given
  identical model weights and single-threaded CPU configuration
  (`torch.set_num_threads(1)`).
- **Environment-Dependent Latency**: Latency figures (`P50`, `P95`, `P99`)
  depend on CPU architecture, memory bandwidth, and thermal throttling
  (especially on fanless devices like the Apple M1 MacBook Air). Expect
  approximately $\pm 10\text{–}20\%$ latency variance on different hardware.
