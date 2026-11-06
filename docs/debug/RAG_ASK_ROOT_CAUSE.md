# ROOT CAUSE ANALYSIS: RAG QUESTION ANSWERING DEFECTS

**Component:** NLP Academic Search & RAG Engine (`nlp_academic_search.rag`, `nlp_academic_search.ui`, `nlp_academic_search.api`)<br>
**Investigation Date:** 2026-11-06<br>
**Status:** Resolved, Regression-Tested & Live-Verified<br>
**Target Branch:** `main`<br>

---

## 1. Executive Summary & Scope

A thorough investigation of the end-to-end RAG pipeline (**User Question $\to$ Hybrid Retrieval $\to$ Context Building $\to$ LLM Generation $\to$ Citation Parsing $\to$ Grounding Verification $\to$ Streamlit UI**) revealed four distinct defects affecting the reliability of Question Answering:

1. **BUG-1 (Model Readiness & Inactive Ask Button)**: Streamlit disabled the form submit button whenever the default model (`qwen2.5:7b`) was absent, even when compatible local Ollama models (e.g. `qwen2.5:1.5b`) were ready.
2. **BUG-2 (Citation Segmentation & False Refusal)**: Post-punctuation citations (`Sentence. [1]`, `[1, 2]`) and abbreviations (`Smith et al. [1]`) were severed from factual claims by sentence splitting, triggering unwarranted hallucination refusals.
3. **BUG-3 (Prompt Rubric Leaking on Small Models)**: Unbracketed few-shot rubric examples in `SYSTEM_PROMPT` were regurgitated verbatim by smaller open-weight models ($\le 7\text{B}$) when answering multilingual/Vietnamese queries.
4. **BUG-4 (Ambiguous UI Diagnostics)**: Failures caused by insufficient retrieval context were conflated with draft verification rejections, confusing users.

> [!NOTE]
> Codebase audit confirmed that only **Feature A (Question Answering)** is implemented. Feature B (Question Generation) was never part of the repository. All fixes strictly target Feature A.

---

## 2. Bug Breakdown & Evidence Traces

### BUG-1: Ollama Model Mismatch & Silent Button Disablement
- **Severity**: HIGH (Blocks QA feature entirely in environments without 7B model)
- **Status**: **LIVE VERIFIED** (Tested with local Ollama `qwen2.5:1.5b` and FastAPI health endpoint)
- **Symptom**: "Ask" submit button in Streamlit UI remained permanently disabled (`disabled=True`).
- **Reproduction Steps**:
  1. Install `qwen2.5:1.5b` in local Ollama (`ollama pull qwen2.5:1.5b`).
  2. Launch FastAPI service without modifying default `LLM_MODEL_NAME="qwen2.5:7b"`.
  3. Open Streamlit UI $\to$ Notice Ask button is disabled without specific error copy.
- **Root Cause**:
  1. `RAGGenerator.is_available()` performed an exact-string match against `self.model_name` (`"qwen2.5:7b"`).
  2. Streamlit evaluated `rag_ready = bool(health.get("generation_available"))`, which evaluated to `False`.
  3. No explicit fallback mechanism existed in configuration.
- **Affected File & Function**:
  - `src/nlp_academic_search/rag/generator.py`: `RAGGenerator.is_available()`
  - `src/nlp_academic_search/config.py`: `Settings`
  - `src/nlp_academic_search/ui/app.py`: `render_ask()`
- **Fix Implemented**:
  1. Added `allow_model_fallback: bool = False` to `Settings`.
  2. Added validator in `config.py` preventing `allow_model_fallback=True` in `environment="production"`.
  3. Added `ALLOWED_FALLBACK_MODELS = frozenset({"qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:latest", "qwen2.5"})`.
  4. Updated `api/main.py` so `HealthResponse` dynamically reports actual generator model in response metadata.
  5. Enhanced UI copy to display exact missing model and actionable resolution instructions (`ollama pull` or settings update).
- **Regression Test**:
  - `tests/unit/test_rag.py`: `test_ollama_generator_model_fallback_when_default_unpulled()`
  - `tests/unit/test_config.py`: `test_settings_reject_invalid_values()`

---

### BUG-2: Citation Segmentation Severing Claims Causing False Refusals
- **Severity**: CRITICAL (Causes verified grounded answers to be rejected as hallucinations)
- **Status**: **LIVE VERIFIED** (Tested across 12 comprehensive edge-case scenarios)
- **Symptom**: Valid answers generated with bracketed citations placed after terminal punctuation (e.g. `"Method improves recall. [1]"`) or containing abbreviations (`"Smith et al. [1] showed..."`) were classified as unverified, resulting in `answer_status = "refused_unverified"`.
- **Reproduction Steps**:
  1. Query: `"What does the Transformer architecture rely on?"`
  2. LLM response: `"The Transformer relies on multi-head attention. [1] Recurrent layers are eschewed. [1]"`
  3. Call `validate_citations(answer, source_count=2)`.
  4. Sentence segmentation splits at `(?<=[.!?])\s+`.
  5. Claim sentence loses citation; `[1]` becomes orphan fragment in next segment.
  6. `uncited_claim_count >= 1` $\to$ Bounded repair fails $\to$ Refusal returned.
- **Root Cause**:
  - `_protect_periods` was executing after or missing post-punctuation citation normalization.
  - Regex lookbehind `(?<=[.!?])\s+` split immediately between terminal period and bracket `[1]`.
  - When citation normalization was placed before abbreviation protection, `et al.` had its period transformed, causing the sentence to split after `et al [1].`.
- **Affected File & Function**:
  - `src/nlp_academic_search/rag/citations.py`: `segment_sentences()`, `_protect_periods()`
  - `src/nlp_academic_search/rag/verification.py`: `_containing_sentence()`
- **Fix Implemented**:
  1. Ran `_protect_periods()` FIRST to protect abbreviations (`et al.`, `e.g.`, `i.e.`) and decimal numbers (`1.5`, `95.5%`).
  2. Normalized post-punctuation citations (`re.sub(r"([.!?])\s*((?:\[(?:\d+\s*,\s*)*\d+\])+)\.?(\s*)", r" \2\1\3", protected)`), relocating citation tags inside sentence boundaries without modifying punctuation semantics or creating synthetic indices.
  3. Restored protected markers seamlessly after sentence splitting.
- **Regression Test**:
  - `tests/unit/test_rag.py`: `test_sentence_segmentation_preserves_post_punctuation_citations()`
  - `tests/unit/test_rag.py`: `test_citation_validation_twelve_edge_cases()` (12 edge cases: before punct, after punct, multi-citation, uncited claim, markdown bullets, decimals, abbreviations, Vietnamese punct, invalid index, unsupported claim, wrong sentence, malformed citations).

---

### BUG-3: Prompt Rubric Leaking on Small Open-Weight Models
- **Severity**: HIGH (Causes nonsensical/unrelated rubric text in generated answers)
- **Status**: **LIVE VERIFIED** (Verified on 10 live test questions using `qwen2.5:1.5b`)
- **Symptom**: When prompting compact models ($\le 7\text{B}$) with cross-lingual or Vietnamese questions (e.g. `"Mô hình Transformer dựa trên cơ chế gì?"`), the model outputted `"The method uses sparse retrieval [1]. It improves recall [1]."` instead of synthesizing from retrieved paper context.
- **Reproduction Steps**:
  1. Provide paper abstract for "Attention Is All You Need".
  2. Ask question in Vietnamese: `"Mô hình Transformer dựa trên cơ chế gì?"`.
  3. Small model fails to bridge cross-lingual context and regurgitates illustrative prose example from system prompt.
- **Root Cause**:
  - `SYSTEM_PROMPT` contained realistic English scientific sentences as compliance examples (`"The method uses sparse retrieval [1]. It improves recall [1]."`).
  - Compact models treated this as few-shot completion data rather than meta-instructions.
- **Affected File & Function**:
  - `src/nlp_academic_search/rag/prompt_builder.py`: `SYSTEM_PROMPT`, `build_rag_messages()`
- **Fix Implemented**:
  1. Encapsulated formatting rules inside `<illustrative_rubric>` tags.
  2. Replaced concrete scientific example sentences with structural descriptions (`"Factual sentences missing citations"`, `"Every factual sentence ends with citation tags before terminal punctuation [1]"`).
  3. Preserved required unit-test assertion strings (`"Citation non-compliant"`, `"Strength non-compliant"`, `"guarantees better retrieval"`, `"may improve retrieval"`).
  4. Strengthened `user_message` instruction to ground answers strictly on `<retrieved_sources>`.
- **Regression Test**:
  - `tests/unit/test_rag.py`: `test_prompt_rubric_delimiters_prevent_example_leakage()`
  - Live 10-query validation on `qwen2.5:1.5b` confirming 0% leakage across English and Vietnamese queries.

---

### BUG-4: Ambiguous UI Diagnostics for Refusal States
- **Severity**: MEDIUM (Poor user experience and misleading error messages)
- **Status**: **LIVE VERIFIED** (Tested through Streamlit UI rendering and TestClient)
- **Symptom**: When no papers met the relevance threshold for a query, the UI displayed: *"Verification issue: The generated draft could not be verified against the retrieved evidence"*, implying draft verification failed rather than empty retrieval.
- **Root Cause**:
  - `verification_failure_detail()` in `ui/app.py` lacked explicit branching for `refused_insufficient_context`.
- **Affected File & Function**:
  - `src/nlp_academic_search/ui/app.py`: `verification_failure_detail()`
- **Fix Implemented**:
  - Added dedicated branch returning `"No indexed paper met the evidence threshold for this question"` when `answer_status == "refused_insufficient_context"`.
  - Added specific actionable messaging for uninstalled local models vs. cloud provider errors.
- **Regression Test**:
  - `tests/unit/test_rag.py`: `test_ui_verification_failure_detail_handles_insufficient_context()`

---

## 3. Summary of Verification Status

| Bug ID | Description | Resolution Status | Evidence Method |
|---|---|---|---|
| **BUG-1** | Ollama Model Mismatch / UI Disabled | **LIVE VERIFIED** | Unit tests + Local Ollama dynamic check + Settings validation |
| **BUG-2** | Citation Segmentation False Refusal | **LIVE VERIFIED** | 12-case regression test suite + SciFact benchmark execution |
| **BUG-3** | Prompt Rubric Leaking on Small Models | **LIVE VERIFIED** | 10 live queries on local `qwen2.5:1.5b` + Delimiter tests |
| **BUG-4** | Misleading UI Refusal Diagnostics | **LIVE VERIFIED** | Unit tests + Streamlit diagnostic helper verification |
