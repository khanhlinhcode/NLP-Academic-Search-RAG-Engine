# Contributing

## Environment

Use Python 3.11 and the locked `uv` environment:

```bash
uv sync --frozen --all-extras --python 3.11
```

Do not commit `.env`, corpus files, indexes, model weights, caches or local evaluation reports.

## Architecture

Application code belongs under `src/nlp_academic_search/`. Keep `scripts/` as thin command-line
adapters and follow the dependency rules in [docs/architecture.md](docs/architecture.md).

Use imports such as:

```python
from nlp_academic_search.search.hybrid_search import HybridSearcher
```

Never import from a package named `src`.

## Tests

- Put deterministic tests that need no network, model or service in `tests/unit/`.
- Put local service/model or end-to-end tests in `tests/integration/`.
- Mark every integration test with `@pytest.mark.integration`.
- Do not alter golden qrels merely to improve a score.

Run the local quality gate before submitting a change:

```bash
make check
```

For end-to-end changes, also run:

```bash
uv run pytest -m integration
docker compose config --quiet
```

## Evaluation changes

Record the benchmark provenance, split, corpus version, embedding model/revision, `k`, relevant
configuration and limitations. Never promote fixture results to production quality claims.
