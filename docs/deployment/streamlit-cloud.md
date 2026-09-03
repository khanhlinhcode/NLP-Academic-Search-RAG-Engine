# Streamlit Community Cloud deployment

The Streamlit deployment is an HTTP/SSE client only. It never imports FastAPI services, local data,
FAISS, Sentence-Transformers, Qdrant, Groq or Ollama.

1. Open <https://share.streamlit.io/> and choose the GitHub repository and branch.
2. Set the entrypoint to `scripts/streamlit_app.py`.
3. Select Python 3.11.
4. In **Advanced settings → Secrets**, enter:

```toml
API_BASE_URL = "https://nlp-academic-search-api.onrender.com"
BACKEND_API_TOKEN = "<same token as Render>"
```

5. Deploy or reboot the app.

`scripts/requirements.txt` is intentionally adjacent to the entrypoint so Community Cloud chooses
the lightweight UI dependencies instead of the repository's complete development lockfile. Never
commit `.streamlit/secrets.toml`.

If the UI still displays `localhost:8000`, the deployed branch is stale or `API_BASE_URL` was not
saved. If it says the backend is waking, wait for Render's cold start and retry. A 401 means the two
copies of `BACKEND_API_TOKEN` differ. Community Cloud apps also hibernate after inactivity and have
resource limits that can change.

