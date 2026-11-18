# Collect synchronous cloud Ask observations

`scripts/collect_cloud_ask_observations.py` calls the final, non-streaming
`POST /api/v1/ask` route. It records API observations in a Git-ignored copy of
the review sheet. It does **not** grade answers or claim a locked holdout. The
Streamlit UI uses `/api/v1/ask/stream`; keep its SSE and draft-withholding smoke
checks separate from these synchronous API timings.

## Before any request

Complete the pre-run human review in
`reports/experiments/cloud_holdout_v1/review_sheet.csv` as described in
[CLOUD_HOLDOUT_REVIEW.md](CLOUD_HOLDOUT_REVIEW.md). Every row needs a human
answerability label, source IDs, an expected answer and evidence quote for
answerable cases, an insufficiency explanation for unanswerable cases, and a
reviewer alias/date. The CLI rejects a blank or partly reviewed sheet **before
calling the API**. Leave `answer_correct`, `factual_claims_json` and
`refusal_appropriate` blank until all observations finish. It hashes questions
and these pre-run labels in a run manifest; editing them between resume calls
requires a new output directory.

Provide `BACKEND_API_TOKEN` through the environment without putting its value
in a command, file or Git commit. Confirm the deployed commit separately; the
API readiness endpoint does not attest its Git SHA. To run a small first batch:

```bash
DEPLOYED_COMMIT=$(git rev-parse HEAD) # Replace if Render runs another commit.
PYTHONPATH=src uv run --no-sync python -m scripts.collect_cloud_ask_observations \
  --app-commit "$DEPLOYED_COMMIT" --limit 3 --delay-seconds 6
```

The CLI requires `--limit` (1–100), checks `/health/ready`, confirms the
Qdrant corpus version, paper count and required verifier, and then sends
requests one at a time. `--limit` counts only new requests. Repeating the
command resumes at the first row without an observation. There is no automatic retry or overwrite of
observed rows. Failed HTTP/provider requests remain in the sheet as errors;
retain them in the error-rate denominator and investigate them in a separate
run. The collector stops after the first 401, 403, 429, 502, 503 or 504 so an
auth, quota or provider outage cannot consume the rest of the batch.
The API URL is restricted to the known Render service or local loopback to keep
the Bearer token from being sent to an arbitrary host.

Outputs under `reports/experiments/cloud_holdout_v1/`:

- `review_sheet.csv`: the original questions/human labels plus final answer,
  refusal/error flag, wall-clock latency, cited source IDs, model IDs, status,
  HTTP status, request ID and error kind.
- `ask_run_manifest.json`: candidate-file hash, pre-run label hash, asserted
  app commit, corpus provenance, models and API URL. It contains no token.

`cited_source_ids` maps numbered citations in the **final answer** to source
IDs; it is not the full list of retrieved sources or proof that a citation
supports a claim. A human reviewer still checks each factual claim and its
source before using `convert_cloud_review_sheet.py` and the scorer.
