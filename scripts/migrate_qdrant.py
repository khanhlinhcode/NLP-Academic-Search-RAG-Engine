"""Version, upload, validate, and smoke-test an academic corpus in Qdrant Cloud."""

from __future__ import annotations

import argparse
import re
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from nlp_academic_search.config import settings
from nlp_academic_search.data.loader import Paper, active_corpus_path, load_papers
from nlp_academic_search.data.manifest import CorpusManifest, sha256_file

T = TypeVar("T")
MIGRATION_SCHEMA_VERSION = 1
POINT_NAMESPACE = uuid.UUID("f286bf75-38c0-49d1-b3c7-855eff8a2a35")
MANIFEST_ID = uuid.UUID("0e855fb4-2578-4e41-a0dd-0387bf45ef11")
KNOWN_DENSE_VECTOR_SIZES = {
    # Qdrant Cloud's official free-inference quickstart model.
    "sentence-transformers/all-minilm-l6-v2": 384,
}


def _require_cloud_settings() -> tuple[str, str, str]:
    config = settings.qdrant
    missing = [
        name
        for name, value in (
            ("QDRANT_URL", config.url),
            ("QDRANT_API_KEY", config.api_key),
            ("QDRANT_DENSE_MODEL", config.dense_model),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")
    return str(config.url), str(config.api_key), str(config.dense_model)


def _client() -> Any:
    from qdrant_client import QdrantClient

    url, api_key, _ = _require_cloud_settings()
    return QdrantClient(
        url=url,
        api_key=api_key,
        cloud_inference=True,
        timeout=max(1, int(settings.qdrant.timeout_seconds)),
        check_compatibility=False,
    )


def _corpus_manifest(corpus_path: Path, papers: list[Paper]) -> CorpusManifest:
    path = corpus_path.with_name("corpus_manifest.json")
    if path.is_file():
        manifest = CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.document_count != len(papers) or manifest.corpus_sha256 != sha256_file(
            corpus_path
        ):
            raise SystemExit("Active corpus does not match corpus_manifest.json")
        return manifest
    raise SystemExit(f"Corpus manifest is missing at {path}")


def _collection_name(corpus_path: Path) -> str:
    version = corpus_path.parent.name if corpus_path.parent.parent.name == "versions" else "legacy"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", version).strip("-").lower()
    return f"academic-papers-{safe}"


def _paper_payload(paper: Paper, *, corpus_version: str, corpus_sha256: str) -> dict[str, Any]:
    payload = paper.to_dict()
    payload.update(
        {
            "record_type": "paper",
            "paper_id": paper.id,
            "text": paper.text,
            "year": paper.year,
            "url": paper.source_url,
            "corpus_version": corpus_version,
            "corpus_sha256": corpus_sha256,
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "paper_schema_version": paper.schema_version,
        }
    )
    return payload


def _chunks(values: list[T], size: int) -> Iterable[list[T]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _retry(operation: Callable[[], T], attempts: int = 3) -> T:
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


def _paper_filter() -> Any:
    from qdrant_client import models

    return models.Filter(
        must=[models.FieldCondition(key="record_type", match=models.MatchValue(value="paper"))]
    )


def _manifest_filter() -> Any:
    from qdrant_client import models

    return models.Filter(
        must=[models.FieldCondition(key="record_type", match=models.MatchValue(value="manifest"))]
    )


def _dense_vector_size(model_name: str, configured_size: int | None = None) -> int:
    """Resolve collection dimensions without importing client-side FastEmbed."""
    if configured_size is not None:
        return configured_size
    if known_size := KNOWN_DENSE_VECTOR_SIZES.get(model_name.casefold()):
        return known_size
    raise SystemExit(
        "Unknown dense model vector size. Set QDRANT_DENSE_VECTOR_SIZE to the "
        "dimension shown for QDRANT_DENSE_MODEL in Qdrant Cloud."
    )


def _ensure_collection(client: Any, collection: str, dense_model: str) -> None:
    from qdrant_client import models

    if client.collection_exists(collection):
        return
    dense_size = _dense_vector_size(dense_model, settings.qdrant.dense_vector_size)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        on_disk_payload=True,
    )
    schemas: dict[str, Any] = {
        "record_type": models.PayloadSchemaType.KEYWORD,
        "paper_id": models.PayloadSchemaType.KEYWORD,
        "categories": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
        "authors": models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
        ),
        "source": models.PayloadSchemaType.KEYWORD,
    }
    for field_name, field_schema in schemas.items():
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def _upsert_corpus(
    client: Any,
    collection: str,
    papers: list[Paper],
    manifest: CorpusManifest,
    corpus_version: str,
    dense_model: str,
    batch_size: int,
) -> None:
    from qdrant_client import models

    for batch_number, batch in enumerate(_chunks(papers, batch_size), start=1):
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(POINT_NAMESPACE, paper.id)),
                vector={
                    "dense": models.Document(text=paper.text, model=dense_model),
                    "sparse": models.Document(text=paper.text, model=settings.qdrant.sparse_model),
                },
                payload=_paper_payload(
                    paper,
                    corpus_version=corpus_version,
                    corpus_sha256=manifest.corpus_sha256,
                ),
            )
            for paper in batch
        ]
        _retry(
            lambda points=points: client.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )
        )
        print(
            f"Uploaded batch {batch_number}: {min(batch_number * batch_size, len(papers))}/{len(papers)}"
        )

    manifest_text = f"Corpus manifest for {corpus_version}"
    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(MANIFEST_ID),
                vector={
                    "dense": models.Document(text=manifest_text, model=dense_model),
                    "sparse": models.Document(
                        text=manifest_text, model=settings.qdrant.sparse_model
                    ),
                },
                payload={
                    "record_type": "manifest",
                    "schema_version": MIGRATION_SCHEMA_VERSION,
                    "paper_schema_version": manifest.schema_version,
                    "corpus_version": corpus_version,
                    "corpus_sha256": manifest.corpus_sha256,
                    "paper_count": len(papers),
                    "dense_model": dense_model,
                    "dense_vector_size": _dense_vector_size(
                        dense_model, settings.qdrant.dense_vector_size
                    ),
                    "sparse_model": settings.qdrant.sparse_model,
                    "source": manifest.source,
                    "migrated_at": datetime.now(UTC).isoformat(),
                },
            )
        ],
        wait=True,
    )


def audit(
    client: Any, collection: str, expected_manifest: CorpusManifest | None = None
) -> dict[str, Any]:
    records, _ = client.scroll(
        collection_name=collection,
        scroll_filter=_manifest_filter(),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if len(records) != 1:
        raise SystemExit("Qdrant collection must contain exactly one manifest record")
    payload = dict(records[0].payload or {})
    count = int(
        client.count(collection_name=collection, count_filter=_paper_filter(), exact=True).count
    )
    if count != int(payload.get("paper_count", -1)):
        raise SystemExit(
            f"Point count mismatch: papers={count}, manifest={payload.get('paper_count')}"
        )
    if int(payload.get("schema_version", 0)) != MIGRATION_SCHEMA_VERSION:
        raise SystemExit("Unsupported Qdrant migration schema version")
    if payload.get("dense_model") != settings.qdrant.dense_model:
        raise SystemExit("Dense model does not match QDRANT_DENSE_MODEL")
    if payload.get("sparse_model") != settings.qdrant.sparse_model:
        raise SystemExit("Sparse model does not match QDRANT_SPARSE_MODEL")
    if expected_manifest:
        if count != expected_manifest.document_count:
            raise SystemExit("Qdrant paper count does not match the active corpus")
        if payload.get("corpus_sha256") != expected_manifest.corpus_sha256:
            raise SystemExit("Qdrant checksum does not match the active corpus")
    print(
        f"Audit passed: collection={collection} papers={count} "
        f"corpus_sha256={payload.get('corpus_sha256')}"
    )
    return payload


def _activate_alias(client: Any, collection: str) -> None:
    from qdrant_client import models

    alias = settings.qdrant.collection_alias
    existing = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    operations: list[Any] = []
    if alias in existing:
        if existing[alias] == collection:
            print(f"Alias already active: {alias} -> {collection}")
            return
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
    print(f"Activated alias: {alias} -> {collection}")


def smoke(client: Any, collection: str) -> None:
    from qdrant_client import models

    _, _, dense_model = _require_cloud_settings()
    query = "information retrieval evaluation using precision and recall"
    common = {
        "collection_name": collection,
        "query_filter": _paper_filter(),
        "limit": 3,
        "with_payload": True,
    }
    dense = client.query_points(
        query=models.Document(text=query, model=dense_model),
        using="dense",
        **common,
    )
    sparse = client.query_points(
        query=models.Document(text=query, model=settings.qdrant.sparse_model),
        using="sparse",
        **common,
    )
    hybrid = client.query_points(
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=dense_model),
                using="dense",
                filter=_paper_filter(),
                limit=20,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=settings.qdrant.sparse_model),
                using="sparse",
                filter=_paper_filter(),
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        **common,
    )
    for name, response in (("dense", dense), ("bm25", sparse), ("hybrid", hybrid)):
        if not response.points:
            raise SystemExit(f"Smoke query returned no {name} results")
        print(
            f"{name}: {len(response.points)} results; top={response.points[0].payload.get('paper_id')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("migrate", "audit", "smoke"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--collection")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 512:
        raise SystemExit("--batch-size must be between 1 and 512")

    client = _client()
    corpus_path = active_corpus_path()
    collection = args.collection
    try:
        if args.command == "migrate":
            papers = load_papers(corpus_path)
            manifest = _corpus_manifest(corpus_path, papers)
            collection = collection or _collection_name(corpus_path)
            _, _, dense_model = _require_cloud_settings()
            _ensure_collection(client, collection, dense_model)
            _upsert_corpus(
                client,
                collection,
                papers,
                manifest,
                corpus_path.parent.name,
                dense_model,
                args.batch_size,
            )
            audit(client, collection, manifest)
            smoke(client, collection)
            _activate_alias(client, collection)
        elif args.command == "audit":
            audit(client, collection or settings.qdrant.collection_alias)
        else:
            smoke(client, collection or settings.qdrant.collection_alias)
    finally:
        client.close()


if __name__ == "__main__":
    main()
