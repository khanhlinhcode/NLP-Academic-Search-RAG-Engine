# Cloud deployment troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `FastAPI is not reachable at http://localhost:8000` | Streamlit has no remote API URL | Set `API_BASE_URL` in Streamlit Secrets and reboot. |
| Render reports no open ports | Process failed before binding or wrong command | Use the cloud Dockerfile and bind `0.0.0.0:$PORT`; inspect the first exception. |
| Render exits with status 137 | Memory limit exceeded | Verify cloud image contains no local ML packages and uses one worker. |
| `/health/live` fails | Container/process problem | Inspect Render logs; this endpoint has no external dependencies. |
| `/health/ready` returns 503 | Qdrant manifest or generation provider unavailable | Check `checks`, provider keys, collection alias, checksum, model and quota. |
| Search returns 401 | Missing/mismatched backend token | Copy the Render token to Streamlit Secrets. |
| `retrieval_unavailable` | Qdrant unavailable, suspended, or incompatible | Wake/check cluster, then run `make qdrant-audit`. |
| `generation_rate_limited` | Groq free quota reached | Wait for reset; Search remains usable. |
| Empty RAG evidence | Corpus coverage or filtering issue | Inspect Search with the same question and broaden filters/corpus. |
| Wrong semantic results after model change | Papers and queries use different vector spaces | Create a new collection and reindex all papers with the new exact model. |

Do not fix port errors by increasing memory-intensive startup timeouts. Cloud startup must remain
lightweight, and external state belongs in Qdrant rather than Render's ephemeral filesystem.

