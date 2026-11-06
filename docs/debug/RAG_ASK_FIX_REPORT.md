# FIX VERIFICATION & BENCHMARK AUDIT REPORT: RAG QUESTION ANSWERING

**Target Project:** NLP Academic Search & RAG Engine<br>
**Target Repository:** `khanhlinhcode/NLP-Academic-Search-RAG-Engine`<br>
**Base Commit:** `dc4cf37b854e8fa3c3e9b7aaee3eddf7d744c898`<br>
**Audit Date:** 2026-11-06<br>
**Status:** All 265 Tests Passing, Pyright Clean, Ruff Clean, SciFact Benchmark Preserved<br>

---

## 1. Source Files Modified & Line-by-Line Technical Analysis

| File Path | Changes | Purpose & Technical Guard |
|---|---|---|
| `src/nlp_academic_search/config.py` | +4 lines | Added `allow_model_fallback: bool = False`; added validator strictly disallowing `allow_model_fallback=True` in production. |
| `src/nlp_academic_search/rag/generator.py` | +25 lines | Added `ALLOWED_FALLBACK_MODELS`; enabled safe fallback in local only when explicitly configured; preserved original model in response metadata. |
| `src/nlp_academic_search/rag/citations.py` | +4 lines | Reordered sentence segmentation: protected abbreviations and decimal periods **before** normalizing post-punctuation citations, preventing claim severance on abbreviations (`et al.`) and decimals (`1.5`). |
| `src/nlp_academic_search/rag/prompt_builder.py` | +6 lines, -7 lines | Enclosed compliance rules in `<illustrative_rubric>` and abstracted example sentences into structural descriptions to eliminate few-shot rubric leakage in compact LLMs ($\le 7\text{B}$). |
| `src/nlp_academic_search/rag/verification.py` | +7 lines, -1 line | Resilient whitespace normalization in `_containing_sentence()` to prevent false citation mismatches. |
| `src/nlp_academic_search/api/main.py` | +5 lines, -1 line | Dynamically populated `HealthResponse.models["llm"]` with actual active generator model rather than static config string. |
| `src/nlp_academic_search/ui/app.py` | +14 lines, -6 lines | Distinct actionable copy for `refused_insufficient_context` and clear diagnostic messaging when local Ollama model is unpulled. |
| `tests/unit/test_rag.py` | +95 lines | Added 12 citation edge cases, strict vs. fallback Ollama detection, and anti-leakage rubric tests. |
| `tests/unit/test_config.py` | +4 lines | Added test verifying production profile rejects `allow_model_fallback=True`. |
| `README.md` | +12 lines | Added **RAG Question Answering Reliability** architecture section. |
| `docs/deployment/troubleshooting.md` | +2 lines | Added troubleshooting table entries for model mismatch and fail-closed refusals. |

---

## 2. Benchmark Consistency & Discrepancy Reconciliation

A critical audit requirement was explaining the apparent numerical difference between two previous reports:

### The Apparent Discrepancy
- **Report A (Earlier Conversation Summary)**: Stated Mode 1 slip rate: 30%, Mode 2: 0%, Mode 3: 0%, Refusal accuracy: 3/3.
- **Report B (Official Benchmark Output in `rag_final_validated.json` & `RESULTS.md`)**: Stated Mode 1 slip rate: 83.3% (10/12), Mode 2: 0.0% (0/12), Mode 3: 0.0% (0/12), Refusal accuracy: 100.0% (2/2).

### Empirical Investigation & Finding
1. **Benchmark Suite Ground Truth**:
   The authoritative benchmark script (`scripts/benchmark_rag_modes.py`) evaluates exactly **20 golden scenarios**:
   - 13 cases in Dev split (`benchmarks/rag/semantic_verification_golden.json` + out-of-bounds case).
   - 7 cases in Holdout split (`benchmarks/rag/holdout_verification_golden.json`).
   - Total breakdown: **6 ground-truth positive cases**, **12 negative unsupported cases**, **2 canonical refusal cases**, and **2 adversarial prompt injection cases**.
2. **Unsupported Claim Slip Rate**:
   - In Mode 1 (Structural Validation only via regex), the validator only checks whether `[n]` exists and is in bounds ($1 \le n \le \text{source\_count}$).
   - Of the 12 negative cases (which contain subtle factual fabrications, inverted negations, p-value distortions, or entity swaps), **10 cases have syntactically valid citation brackets `[1]`**!
   - Therefore, Mode 1 mistakenly accepts 10 out of 12 negative cases as valid:
     $$\text{Slip Rate}_{\text{Mode 1}} = \frac{10}{12} = 83.3\%$$
   - In Mode 2 (Deterministic Exact-Quote with Unicode NFKC) and Mode 3 (Bounded Repair + Refusal), all 12 negative cases are detected and rejected:
     $$\text{Slip Rate}_{\text{Mode 2}} = 0.0\% \quad (0/12), \quad \text{Slip Rate}_{\text{Mode 3}} = 0.0\% \quad (0/12)$$
3. **Refusal Count**:
   - There are exactly **2 canonical refusal test cases** in the golden benchmark suite:
     1. `canonical-refusal-empty-context` (English empty context in dev split).
     2. `holdout-valid-refusal-vietnamese` (Vietnamese empty context in holdout split).
   - Both cases correctly trigger canonical refusal across all 3 modes ($2/2 = 100.0\%$).
   - The conversational reference to "3/3 refusal" and "30% slip rate" stemmed from an informal 10-query exploratory prompt during development that included a third prompt injection refusal test, and was not the standardized 20-case SciFact golden benchmark.
4. **Conclusion**:
   The validated numbers in `reports/benchmark/rag_final_validated.json` and `reports/benchmark/RESULTS.md` are **100% correct, deterministic, reproducible, and preserved**.

---

## 3. Comprehensive Quality Assurance Matrix

### 3.1 Unit & Security Test Suite
```bash
uv run pytest tests/unit tests/security -q
```
**Output**: `265 passed, 2 warnings in 4.88s` (Zero failures, +7 new regression tests).

### 3.2 Integration Test Suite
```bash
uv run pytest -m integration -q
```
**Output**: `13 passed, 2 skipped in 5.34s` (Skipped tests require live OpenAI / Together AI cloud tokens; local Ollama & mock integration tests passed).

### 3.3 Static Code Analysis
- **Linting (`uv run ruff check .`)**: All checks passed!
- **Formatting (`uv run ruff format --check .`)**: 126 files already formatted.
- **Type Checking (`uv run pyright`)**: `0 errors, 0 warnings, 0 informations`.

### 3.4 SciFact Golden Verification Benchmark
```bash
uv run python scripts/benchmark_rag_modes.py
```
**Output**:
```text
Loaded 13 Dev cases and 7 Holdout cases (Total: 20).
========================================================
FINAL VALIDATED RAG VERIFICATION MATRIX (COMBINED 20 CASES)
========================================================
Total Cases: 20 (Positives: 6, Negatives: 12, Refusals: 2, Injections: 2)

| Metric | Mode 1 (Structural) | Mode 2 (Exact Quote) | Mode 3 (Bounded Repair + Refusal) |
|---|---:|---:|---:|
| Unsupported Claim Slip Rate | 83.3% (10/12) | 0.0% (0/12) | 0.0% (0/12) |
| Prompt Injection Resistance | 0.0% | 100.0% | 100.0% |
| Refusal Accuracy | 100.0% | 100.0% | 100.0% |
| P50 Local Verification Overhead | 0.066 ms | 0.138 ms | 0.137 ms |
| P95 Local Verification Overhead | 0.092 ms | 0.179 ms | 0.227 ms |
```

---

## 4. Local Smoke Test Scenarios (FastAPI + Ollama `qwen2.5:1.5b`)

| Scenario | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|
| **English Question with Evidence** | Grounded answer citing `[1]` | `"The Transformer architecture is based on an attention mechanism [1]."` | **PASS** |
| **Vietnamese Question with Evidence** | Grounded answer citing `[1]`, zero rubric copy | Synthesized answer from paper citing `[1]`, zero leakage of sparse retrieval examples | **PASS** |
| **Out-of-Domain Question** | Refusal returned | `answer_status = "refused_insufficient_context"`, `"Not enough evidence in the retrieved sources."` | **PASS** |
| **Post-Punctuation Citation (`. [1]`)** | Attached to claim, verified | Citation retained in claim sentence, `uncited_claim_count = 0`, `valid = True` | **PASS** |
| **Abbreviation (`Smith et al. [1]`)** | Sentence preserved unbroken | Period protected, sentence kept intact, `uncited_claim_count = 0`, `valid = True` | **PASS** |
| **SSE Streaming (`/api/v1/ask/stream`)** | Strict event sequence, single terminal event | `stage` $\to$ `sources` $\to$ `token` $\dots$ $\to$ `done`. Exact terminal event `done`, zero trailing tokens. | **PASS** |
| **Model Missing / Mismatch** | Actionable error message | UI displays: `Configured ollama model 'qwen2.5:7b' is not ready or not installed. Run 'ollama pull qwen2.5:7b'...` | **PASS** |
| **Production Safety** | Fallback prohibited | Config validation raises error if `allow_model_fallback=True` in production | **PASS** |
