# Cloud RAG evaluation protocol

The live demo currently serves a Qdrant corpus of 1,000 arXiv papers. The local BEIR/SciFact
retrieval benchmark uses 5,183 different papers and must remain a separate result.

## Current evidence

`benchmarks/rag/cloud_smoke_v1.json` records two **agent-reviewed UI smoke cases** from
2026-11-18 after rebooting Streamlit to load commit `b42cdff`: one answer supported by
`arxiv:cs/0107020`, and one out-of-domain refusal. The sample is too small and not independently
human labelled. Its rates and one measured latency are **not CV quality metrics**.

Score only the recorded labels, without calling providers:

```bash
PYTHONPATH=src uv run --no-sync python -m scripts.score_cloud_rag_review \
  benchmarks/rag/cloud_smoke_v1.json \
  --output reports/experiments/cloud_smoke_v1/cloud_review.json
```

The scorer does not treat the API's `verified` status, citation syntax, or keyword overlap as
ground truth. A reviewer must mark answer correctness and every factual claim's evidence and
citation support against the original paper. It reports exact numerators and denominators:

- **Grounded answer accuracy:** correct, fully supported answers / all answerable questions.
- **Correct refusal rate:** appropriate refusals / all unanswerable questions.
- **False refusal rate:** refusals / all answerable questions.
- **Supported claim rate:** claims supported by the cited source / all factual claims in
  non-refused answers to answerable questions.
- **Latency P50/P95:** request-to-final-answer time, only when at least five timings exist.
- **Error rate:** failed requests / all questions; errors also count against answer or refusal
  success and are excluded from successful-request latency percentiles.

## Before a CV quality claim

The [100-question candidate bank and review sheet](CLOUD_HOLDOUT_REVIEW.md) are a starting
point for human review. They are drafts, not a locked or scored holdout.
After pre-run labels are frozen, use the [cloud Ask observation collector](CLOUD_ASK_COLLECTION.md)
to record synchronous API responses without changing reviewer labels.

1. Freeze the Qdrant corpus version and app commit. Create at least 100 **new** questions over
   this corpus, including answerable, insufficient-evidence and adversarial cases. Keep their
   labels out of prompt tuning and model selection.
2. Have an independent human reviewer record the expected answer, evidence source IDs and
   support of each factual claim. Do not derive labels from the generator or its verifier; the
   live deployment currently uses the same Groq model ID for both.
3. Run the full Ask flow and record the final answer, source IDs, refusal, latency, errors,
   model IDs and verification status for every case. Review answers against original sources.
4. Report sample sizes, counts, rates, uncertainty, P50/P95, model and corpus versions, and
   failure examples. Separate warm and cold starts. Do not merge these results with local
   SciFact retrieval metrics.

After deployment, also check authenticated and unauthorized requests, Qdrant/Groq outage and
quota handling, concurrency, logs, and restoration from the Qdrant snapshot. The two smoke cases
above establish only that the public UI reaches a final answer or safe refusal.
