# Cloud operations

## Publish a new corpus

1. Ingest and validate locally.
2. Keep the previous Qdrant alias target intact.
3. Run `make qdrant-migrate` with the new active corpus.
4. Migration validates the versioned collection and switches the alias only on success.
5. Run the labelled retrieval benchmark against the cloud provider before publishing quality claims.

Rollback is an alias switch to the last audited collection. Deletion is deliberately not part of
the migration command.

## Rotate credentials

1. Create the replacement Qdrant or Groq key.
2. Update Render environment variables and deploy.
3. Confirm `/health/ready` and an authenticated Search/Ask request.
4. Revoke the old provider key.

For the backend token, update Render and Streamlit Secrets in the same maintenance window. Health
endpoints remain available while a mismatched token makes application routes return 401.

## Quota and incident handling

- Groq 429 responses become `generation_rate_limited`; Search remains available.
- Qdrant failure becomes `retrieval_unavailable`. Optional degraded retrieval is explicit and named
  `bm25_degraded`; it is disabled by default.
- Inspect Render request IDs and structured logs without logging prompts, papers, or credentials.
- Check Qdrant cluster suspension after prolonged inactivity.
- Reboot Streamlit from its workspace if dependency or secret changes were not picked up.

