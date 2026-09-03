# Deployment architecture

The repository supports two explicit profiles; neither silently falls back to the other.

```text
local:  Streamlit -> FastAPI -> BM25 + SBERT/FAISS -> Ollama
cloud:  Streamlit Cloud -> Render FastAPI -> Qdrant dense/BM25/RRF -> Groq
```

The local profile remains the reproducible research path. It owns corpus files, manifests, FAISS
artifacts and local model weights. The cloud profile is a lightweight orchestration API: Qdrant
stores paper payloads and generates dense and sparse vectors, while Groq produces the grounded
answer. The public HTTP and SSE contracts are shared by both profiles.

Cloud imports are isolated deliberately. `nlp_academic_search.api.main` must not import Torch,
Sentence-Transformers, FAISS, Ollama, or Streamlit when `DEPLOYMENT_PROFILE=cloud`. Provider
selection happens in `providers.factory`; local-only packages are imported inside the local branch.

Render's filesystem is not a data store. Migration is run from a trusted local workstation, creates
a versioned Qdrant collection, verifies its manifest, then switches the stable collection alias.

## Trust boundaries

- The browser talks only to Streamlit.
- Streamlit stores only the Render URL and shared backend token.
- Qdrant and Groq credentials exist only in Render and the migration workstation.
- Paper content remains untrusted evidence and is XML-delimited before generation.
- `/health/live` is public; search, stats and RAG endpoints require bearer authentication when
  `BACKEND_API_TOKEN` is configured.

