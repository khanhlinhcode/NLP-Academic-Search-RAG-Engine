---
name: semantically-verified-rag
description: Technical architecture and implementation rules for Semantically Verified Retrieval-Augmented Generation (RAG) systems. Enforces structural citation validation, provider-neutral semantic evidence verification, exact evidence quote verification, strict JSON schema output, and one-pass answer repair.
version: 1.0.0
---

# Semantically Verified RAG System Skill

This skill defines the technical specification and verification rules for elevating a RAG system to **Production Pilot with Semantically Verified Grounding**.

A RAG system is **NOT** verified merely because citation indices (e.g., `[1]`, `[2]`) match syntactic format. Every factual claim must be backed by true evidence quotes existing in retrieved documents.

---

## 1. Core Verification Data Models

### A. EvidenceSpan
- `source_index`: `int` (0-indexed position of source document)
- `quote`: `str` (Exact substring quote extracted from source title or abstract)

### B. ClaimAssessment
- `claim_text`: `str` (Atomic factual claim from generated answer)
- `factual`: `bool` (Whether sentence carries factual assertion requiring citation)
- `cited_indices`: `list[int]` (Citation indices associated with claim)
- `verdict`: `"supported"` | `"unsupported"` | `"insufficient"`
- `evidence`: `list[EvidenceSpan]`
- `explanation`: `str` (Short summary without exposing internal reasoning/chain-of-thought)

### C. SemanticValidation
- `valid`: `bool` (True only if `unsupported_claim_count == 0` and `evidence_quote_validity == 1.0`)
- `total_factual_claims`: `int`
- `supported_claim_count`: `int`
- `unsupported_claim_count`: `int`
- `insufficient_claim_count`: `int`
- `semantic_claim_coverage`: `float`
- `evidence_quote_validity`: `float`
- `claims`: `list[ClaimAssessment]`
- `verifier_provider`: `str`
- `verifier_model`: `str`
- `verifier_independent`: `bool` (False if verifier model matches generator model)
- `warnings`: `list[str]`

---

## 2. Server-Side Evidence Quote Verification Protocol

The backend server **MUST** independently verify every `EvidenceSpan` returned by the verifier:

1. **Index Check:** `source_index` must be within range of retrieved source list (`0 <= source_index < len(sources)`).
2. **Citation Match:** `source_index + 1` must be included in the claim's `cited_indices`.
3. **Exact Substring Search:** The quote must exist verbatim in the source's `title` or `abstract`.
   - **Normalization:** Apply Unicode NFKC normalization, collapse consecutive whitespace, and case-folding.
   - **No Fuzzy Hallucination:** Do not accept fabricated quotes or vector similarity scores as logical entailment proof.
4. **Invalidation:** If quote does not exist in source text, mark `EvidenceSpan` invalid $\rightarrow$ set `verdict = unsupported` $\rightarrow$ set `valid = False`.

---

## 3. Provider-Neutral Verification & Groq Strict Schema

- Verification protocols must be independent of generator providers (`src/nlp_academic_search/providers/verification/`).
- When using Groq or Structured JSON providers:
  - Set `response_format = {"type": "json_schema", "json_schema": {"strict": True, ...}}`.
  - Set `temperature = 0.0`.
  - Set `additionalProperties: False` on all object schemas.
  - Require all properties in `required`.
  - Never request chain-of-thought or reasoning text.

---

## 4. Unified Validation & One-Pass Repair Pipeline

```
[Initial Draft Generation]
            │
            ▼
[Layer 1: Structural Validation] ──(Fail)──┐
            │                             │
         (Pass)                           │
            ▼                             ▼
[Layer 2: Semantic Verification] ──(Fail)─> [One-Pass RAG Repair Engine]
            │                                         │
         (Pass)                                  (Re-validate)
            ▼                                         │
    [Answer Verified] <──────────(Pass)───────────────┤
                                                      │
                                                   (Fail)
                                                      ▼
                                       [Refusal: Not enough verified evidence]
```

### Repair Constraints:
- Maximum repair attempts = **1** (`MAX_RAG_REPAIR_ATTEMPTS = 1`). No infinite retry loops.
- Repair prompt is restricted:
  - Keep supported claims.
  - Remove unsupported claims.
  - Fix citations to match verified evidence.
  - Do NOT invent new facts or sources.
  - Do NOT append artificial citation tags via string concatenation.
- If final validation still fails: Fail-closed with explicit refusal: `"Not enough verified evidence in the retrieved sources."`

---

## 5. UI & Telemetry Contract

- **`verified`:** Displayed ONLY when Layer 1 (Structural) AND Layer 2 (Semantic) pass.
- **`structurally_valid`:** Display `"Citation format valid"` (Do NOT display `"Evidence verified"`).
- **`refused_unverified` / `Answer withheld`:** Displayed when draft cannot be verified against retrieved evidence.
- **Observability:** Log safe metadata (`request_id`, `provider`, `model`, `answer_status`, `latency_ms`). Never log user API keys or prompt credentials.
