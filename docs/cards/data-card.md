# Data Card: NLP Academic Search Corpus

## Purpose

The runtime corpus contains public scientific-paper metadata and abstracts used as untrusted
evidence for sparse/dense retrieval and citation-constrained RAG. Full-text PDFs are not indexed.

## Sources and provenance

| Corpus | Source | Adapter | Provenance status |
|---|---|---|---|
| Versioned arXiv metadata | `https://oaipmh.arxiv.org/oai` with metadata prefix `arXiv` | `nlp_academic_search.data.sources.arxiv_oai` | Per-run corpus manifest records checksums, count, collection time and filtering rules. Per-record license is retained when supplied. |
| Legacy summary corpus | Historical `ccdv/arxiv-summarization` export | Adoption only; the current downloader does not recreate it | Original snapshot/model-weight provenance is incomplete. Do not redistribute it based on this repository alone. |

## Record schema

Records are validated by the Pydantic `Paper` model. The complete schema also contains
`updated_at`, `source_url`, `pdf_url`, `content_hash` and `schema_version`.

```json
{
  "id": "arxiv:2301.00001",
  "arxiv_id": "2301.00001",
  "doi": null,
  "title": "Example title",
  "abstract": "Example abstract.",
  "authors": ["Example Author"],
  "categories": ["cs.IR"],
  "published_at": "2023-01-02T00:00:00Z",
  "source": "arxiv-oai-pmh",
  "license": null
}
```

Required non-empty fields are `id`, `title` and `abstract`. IDs, arXiv IDs, DOI values and
content hashes participate in validation or deduplication.

## Collection and processing

1. Fetch OAI-PMH records with redirect handling, bounded retries and a resumable token.
2. Normalize whitespace and validate each record with `Paper`.
3. Deduplicate by ID, arXiv ID, DOI and title/abstract content hash.
4. Store rejected records, when present, under the run-specific
   `data/raw/versions/<version>/quarantine.jsonl`.
5. Write `papers.jsonl` and `corpus_manifest.json`, then atomically update `data/raw/CURRENT`.

Runtime corpus files are ignored by Git. The active local version is environment-specific and is
not a versioned project claim.

## Biases, privacy and limitations

- The default arXiv set is `cs`, creating strong computer-science and English-language bias.
- Abstracts are author-provided summaries and do not reproduce full methods, tables or caveats.
- Public author names are bibliographic data but still require appropriate privacy handling.
- Paper text may contain adversarial instructions and is always treated as untrusted evidence.
- The legacy corpus lacks verified author/date/category/source-link metadata.

## Reproduction

```bash
ARXIV_MAX_RECORDS=15000 make download
make preprocess
make index
```

These commands create new version directories. They must not overwrite an already active version
until validation and manifest generation succeed.
