# 🛡️ Threat Model & Security Architecture

## 1. Executive Summary & Security Philosophy

The **NLP Academic Search & RAG Engine** operates as a local-first search and RAG engine over indexed scientific papers. All paper content retrieved during search/RAG is treated as **untrusted data**. System prompts enforce strict grounding, input validation, context boundaries, error sanitization, and non-root execution.

Threat mitigation status is categorized into:
- **Implemented Mitigation**: Code controls are present in the repository.
- **Verified Mitigation**: Automated security tests in `tests/security/test_security.py` execute and pass against the control.
- **Planned Mitigation**: Future architecture control scheduled for subsequent iterations.
- **Accepted Residual Risk**: Documented operational risk accepted under current single-tenant local deployment.

---

## 2. Threat Vector Matrix & Status

| Threat ID | Threat Vector / Category | Attack Scenario | Mitigation Control | Mitigation Status | Verification Test / Method |
|---|---|---|---|---|---|
| **TM-01** | **Indirect Prompt Injection** | arXiv abstract contains instructions ("SYSTEM OVERRIDE: Ignore instructions"). | Untrusted evidence is bounded in tagged XML blocks. Prompt instructs model to read evidence as text data only. | **Verified Mitigation** | `test_01_prompt_injection_isolation` |
| **TM-02** | **Malicious Citation / Forgery** | Model outputs citation `[99]`, leaves a factual sentence uncited, or relies on a citation in a later sentence. | Sentence-scoped post-generation validation rejects invalid indices and uncited factual sentences. At most one bounded repair pass may remove unsupported claims or correct citations without adding facts. | **Verified Mitigation** | Citation validation and API repair tests |
| **TM-03** | **Path Traversal via Pointer** | Attacker tampers with `CURRENT` pointer to target `/etc/passwd`. | `active_corpus_path()` resolves realpath and asserts path is within `data/raw/`. | **Verified Mitigation** | `test_03_path_traversal_pointer_protection` |
| **TM-04** | **CORS Wildcard Misconfiguration** | Production config receives untrusted Origin header. | CORS middleware rejects untrusted origin headers. | **Verified Mitigation** | `test_09_cors_headers_strict` |
| **TM-05** | **Resource Exhaustion / DoS** | Attacker sends oversized string (> 2000 chars) as query/question. | Pydantic schemas enforce max string length (max 2000 chars). Rejection with 422. | **Verified Mitigation** | `test_oversized_query_returns_422` (in API test suite) |
| **TM-06** | **XSS in Bibliographic Metadata** | Paper title contains `<script>alert('xss')</script>`. | Streamlit UI escapes metadata strings via `escape_html()` before rendering. | **Verified Mitigation** | `test_04_ui_xss_escaping` |
| **TM-07** | **Internal Stack Trace Leakage** | Internal exception occurs during retrieval or FAISS query. | FastAPI global exception handlers catch unhandled errors, return sanitized JSON without stack/paths. | **Verified Mitigation** | `test_08_internal_exception_no_path_leakage` |
| **TM-08** | **SSE Event Injection** | Paper text contains `event: done\ndata: fake`. | SSE stream encoder serializes JSON payload escaping raw control newlines. | **Verified Mitigation** | `test_05_sse_event_injection_prevention` |
| **TM-09** | **Unsafe External URL Injection** | Attacker injects `javascript:alert(1)` in paper source URL. | `safe_external_url()` allows only `https://` protocol URLs matching known domains. | **Verified Mitigation** | `test_06_unsafe_url_rejection` |
| **TM-10** | **Index Poisoning / Mismatch** | Corpus file modified without re-building FAISS index. | Readiness probe checks Ollama availability and index compatibility. Fail with 503. | **Verified Mitigation** | `test_07_corpus_index_mismatch_readiness_failure` |
| **TM-11** | **Secret / Artifact Leakage** | Sensitive `.env` or data files committed to Docker context. | Multi-stage Docker build copies only build wheel; `.dockerignore` blocks `.env` and `data/`. | **Verified Mitigation** | `test_10_secrets_excluded_from_dockerignore` |
| **TM-12** | **Container Privilege Escalation** | Container runs as root user inside Docker host. | Dockerfile creates system user `app:app` and executes entrypoint under non-root USER. | **Verified Mitigation** | `test_11_container_runs_non_root` |
| **TM-13** | **Unsupported Semantic Claim** | A response uses a syntactically valid citation whose paper does not support the claim, or a verifier fabricates an evidence quote. | A provider-neutral semantic verifier returns strict typed claim assessments. The server independently checks exact normalized quotes against the cited title or abstract, permits one repair pass and can fail closed. Same-model verification is marked non-independent. | **Verified Mitigation** | Semantic verifier, provider taxonomy, repair and golden-fixture unit tests. |
| **TM-14** | **Unauthenticated Network Access** | A network user calls search or ask endpoints without authorization. | When configured, constant-time Bearer-token middleware protects non-public routes; bounded per-subject in-memory rate limits reduce simple abuse. Production cloud configuration requires a backend token. | **Verified Mitigation** | Authentication, CORS and rate-limit API/security tests. |
| **TM-15** | **Verifier Contract Rejection** | A structured-output provider rejects unsupported schema annotations and no semantic assessment is produced. | The provider adapter sends a strict, allow-listed transport schema while retaining full local Pydantic validation. HTTP 400 is classified as a non-retryable invalid request, logged without response content, and fails closed without answer repair. | **Verified Mitigation** | Groq schema, HTTP taxonomy, API fail-closed and UI tests. |

---

## 3. Environment & Configuration Controls

- **API Request Concurrency**: Bounded via `asyncio.Semaphore`.
- **Generation Deadline**: Hard timeout (`timeout_seconds=30.0`) prevents hung sockets.
- **Model Endpoint Isolation**: Ollama API accessed only over local loopback or container network bridge.
- **Verification Deadline and Circuit Breaker**: Remote semantic verification uses a configured
  timeout and a bounded in-memory failure circuit; production fail-closed readiness degrades when
  required verification is unavailable.
- **Safe Provider Diagnostics**: Verification logs contain only provider/model identifiers, status,
  error category, provider request ID and latency. They exclude credentials, prompts, answers,
  evidence text and raw provider response bodies.

## 4. Residual verification risk

Semantic verification establishes whether retrieved text supports generated claims within the
configured policy; it does not establish factual truth outside the corpus. Exact quote matching
shows that evidence exists in a retrieved paper, not that the paper is correct. An LLM verifier can
also make entailment errors, and using the same model as generation is not independent verification.
Fail-closed mode reduces exposure to unverified answers at the cost of lower answer coverage,
additional latency and provider quota.
