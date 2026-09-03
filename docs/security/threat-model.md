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
| **TM-02** | **Malicious Citation / Forgery** | Model outputs citation `[99]` or index not in retrieved candidate sources. | Post-generation citation validator checks `1 <= index <= len(sources)` and flags invalid indices. | **Verified Mitigation** | `test_02_citation_validation_detects_invalid_indices` |
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
| **TM-13** | **LLM Semantic Entailment Judge** | Hallucination in complex multi-hop claims. | Entailment verification via LLM judge or cross-encoder entailment model. | **Planned Mitigation** | Benchmarked on offline golden sets via `evaluate_rag.py`. |
| **TM-14** | **Unauthenticated Local Network Access** | Local network user accesses local API without API key. | API authentication middleware (OAuth2 / API Keys). | **Accepted Residual Risk** | Accepted for local single-tenant developer deployment; production deployment requires API Gateway auth. |

---

## 3. Environment & Configuration Controls

- **API Request Concurrency**: Bounded via `asyncio.Semaphore`.
- **Generation Deadline**: Hard timeout (`timeout_seconds=30.0`) prevents hung sockets.
- **Model Endpoint Isolation**: Ollama API accessed only over local loopback or container network bridge.
