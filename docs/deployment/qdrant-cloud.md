# Qdrant Cloud setup and corpus migration

Qdrant Cloud is the cloud profile's durable paper store and retrieval engine. The free tier is
appropriate for a research demo, not an availability SLA. Free hosted inference models are shown
with a **Cost: Free** label in the cluster's Inference tab and may be region-dependent.

## Provision

1. Create a free Qdrant Cloud cluster.
2. Prefer a US cluster when using free hosted inference; verify availability in the console.
3. Open the Inference tab and enable Cloud Inference if necessary.
4. Select an exact dense model ID marked **Cost: Free**. MiniLM is suitable when offered.
5. Create a database API key with only the access needed by this application.
6. Export credentials locally without writing them to Git:

```bash
export QDRANT_URL='https://...cloud.qdrant.io'
export QDRANT_API_KEY='...'
export QDRANT_DENSE_MODEL='sentence-transformers/all-MiniLM-L6-v2'
export QDRANT_DENSE_VECTOR_SIZE=384
export QDRANT_SPARSE_MODEL='qdrant/bm25'
export QDRANT_COLLECTION_ALIAS='academic-papers-current'
```

The model value above is an example, not a promise that the model is free on every cluster. Use the
ID displayed by your cluster. Papers and queries must always use the same model. The migration knows
that `all-MiniLM-L6-v2` has 384 dimensions; for another model, set `QDRANT_DENSE_VECTOR_SIZE` to the
dimension published for that exact model. It deliberately does not install FastEmbed locally.

## Migrate and validate

```bash
make qdrant-migrate
make qdrant-audit
make qdrant-smoke
```

Migration creates `academic-papers-<corpus-version>`, two named vectors (`dense`, `sparse`), payload
indexes, deterministic UUIDs and one excluded manifest record. It uploads idempotent batches,
validates count/checksum/model/schema, runs three representative searches, and only then updates
the `academic-papers-current` alias atomically. It never deletes older collections.

Set the audited checksum in Render for strict readiness:

```text
QDRANT_EXPECTED_CORPUS_SHA256=<value printed by qdrant-audit>
```

To roll back, update the alias to the preceding validated collection in the Qdrant console. Never
delete the active collection. Free clusters can be suspended after inactivity; inspect the Qdrant
console if readiness changes after a long idle period.
