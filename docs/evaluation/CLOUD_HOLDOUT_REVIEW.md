# Cloud RAG candidate review

`benchmarks/rag/cloud_candidates_v1.json` is a **draft question bank**, and
`benchmarks/rag/cloud_review_sheet_v1.csv` is its editable review sheet. Neither is a
locked holdout or a scored benchmark. Do not use its counts as answer-quality evidence.

## Source trace

The questions were prepared from a read-only Qdrant Console scroll of
`academic-papers-current` on 2026-11-18. The export contained 1,001 points:
1,000 `record_type=paper` payloads and one manifest. All selected anchor IDs,
titles, URLs and abstract hashes were checked against those paper payloads.
The SHA256 recorded in the JSON identifies the **local export file**, not the
Qdrant collection itself. The full export and its abstracts are not committed.

The 60 distinct anchor papers were selected by filtering for abstracts of at
least 350 characters, titles of at most 160 characters and titles without
`proceedings` (case-insensitive). This left 908 papers. The first 60 after
sorting by SHA256 digest bytes of `cloud-holdout-v2:` plus the paper ID became
anchors. An independent check reproduced the exact 60 IDs and their order.
This is a deterministic convenience sample, not a probability sample or a
topic/year-stratified evaluation set. The remaining anchored questions reuse
these papers; a reviewer should rebalance the final holdout if needed.

The 100 drafts comprise 60 proposed answerable questions, 20 proposed
insufficient-evidence or out-of-domain questions, and 20 adversarial user
instructions. Ten adversarial questions have a proposed answerable core and ten
ask for unsupported facts. Some questions use a paper title, which may make
retrieval easier; reviewers should revise wording and add title-free variants
before locking the holdout. Several remaining questions use templates and need
specificity review.

## Human review and execution

Keep the tracked CSV as an empty template. Copy it to the Git-ignored
`reports/experiments/cloud_holdout_v1/` directory before filling reviewer
identifiers, answers or observations:

```bash
mkdir -p reports/experiments/cloud_holdout_v1
cp benchmarks/rag/cloud_review_sheet_v1.csv \
  reports/experiments/cloud_holdout_v1/review_sheet.csv
```

1. Open each `source_url` and inspect the source abstract or paper. Confirm the
   `anchor_paper_id` and whether the question is answerable **from the indexed
   abstract**. For out-of-domain questions, confirm absence from the corpus;
   change or discard ambiguous cases. Mark `human_expected_answerability` as
   `answerable` or `unanswerable`. Do not simply copy the proposed value.
2. For answerable cases, write `expected_answer`, `reference_source_ids` as a
   JSON array such as `["arxiv:cs/0109012"]`, and an `evidence_quote` from the
   source. For unanswerable cases, use `[]` for `reference_source_ids` and
   explain the insufficiency in `review_notes`.
3. Freeze the reviewed questions and corpus version **before** running Ask.
   Record the deployed app commit and model IDs. Do not use any reviewed answer
   to tune prompts or select a model before this evaluation is complete.
4. Run each question through the final Ask flow. Record `observed_answer`,
   `observed_refused`, `observed_error`, `total_latency_ms`, `cited_source_ids`,
   `generation_model`, `verification_model` and `verification_status` from the
   runtime. Use `true` or `false` for boolean columns; use JSON arrays for IDs.
5. A human reviewer who did not generate or verify the API answer records
   `answer_correct` and `factual_claims_json` for every non-refused answerable
   response. The claims field is an array such as
   `[{"claim":"...","supported":true,"citation_correct":true}]`.
   For unanswerable cases, record `refusal_appropriate`. Set `reviewer_kind`
   to `human` only for genuine human review, and record a reviewer ID/date.

The sheet is a staging artefact. It is intentionally not accepted directly by
`score_cloud_rag_review.py`. Once **all** rows are complete, convert it with:

```bash
DEPLOYED_COMMIT=$(git rev-parse HEAD) # Replace if the deployed commit differs from local HEAD.
PYTHONPATH=src uv run --no-sync python -m scripts.convert_cloud_review_sheet \
  benchmarks/rag/cloud_candidates_v1.json \
  reports/experiments/cloud_holdout_v1/review_sheet.csv \
  --app-commit "$DEPLOYED_COMMIT" \
  --output reports/experiments/cloud_holdout_v1/reviewed_unlocked.json
```

The converter rejects missing human labels or runtime observations and calls
the scorer for schema validation. Its output remains
`dataset_role=reviewed_candidate_unlocked`; it cannot certify label
independence or sampling quality. Audit those and freeze the reviewed JSON
before setting `dataset_role` to `locked_holdout` and reporting metrics. A
reviewer should reject or replace weak template questions rather than filling
optimistic labels.
