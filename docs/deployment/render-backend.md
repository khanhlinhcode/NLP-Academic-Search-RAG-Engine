# Render backend deployment

Render runs only the cloud FastAPI image in `deploy/Dockerfile.api`. That image intentionally omits
Torch, Sentence-Transformers, FAISS, Ollama, Streamlit, corpus files and model caches.

## Deploy the Blueprint

1. Push the repository to GitHub.
2. In Render choose **New → Blueprint** and select the repository's `render.yaml`.
3. Use the Free web-service plan and a region close to the Qdrant cluster.
4. Enter every `sync: false` value when prompted:

```text
QDRANT_URL
QDRANT_API_KEY
QDRANT_DENSE_MODEL
QDRANT_EXPECTED_CORPUS_SHA256
GROQ_API_KEY
CORS_ORIGINS=https://<your-app>.streamlit.app
```

Render generates `BACKEND_API_TOKEN`. Copy its value securely to Streamlit Secrets after the first
deployment. For an existing Blueprint, newly added `sync: false` variables must be entered manually
in the Render dashboard.

The container binds `0.0.0.0:${PORT:-10000}` with one Uvicorn worker. `/health/live` performs no
network checks and is the Render health path. `/health/ready` verifies the Qdrant manifest and reports
generation readiness separately.

## Verify

```bash
export API_URL='https://nlp-academic-search-api.onrender.com'
export BACKEND_API_TOKEN='...'

curl -fsS "$API_URL/health/live"
curl -i "$API_URL/health/ready"
curl -fsS "$API_URL/api/v1/search?q=information+retrieval&top_k=3" \
  -H "Authorization: Bearer $BACKEND_API_TOKEN"
```

Render Free has 512 MB RAM, an ephemeral filesystem, 750 included instance hours per workspace per
month, and spins down after 15 minutes without inbound traffic. Waking can take about one minute.
These are accepted demo limitations, not production guarantees.
