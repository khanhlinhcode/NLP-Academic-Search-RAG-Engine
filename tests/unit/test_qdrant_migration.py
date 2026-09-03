"""Unit tests for lightweight Qdrant Cloud migration helpers."""

import pytest
from qdrant_client import models

from scripts.migrate_qdrant import _dense_vector_size, _ensure_collection


def test_known_cloud_model_size_does_not_require_fastembed():
    assert _dense_vector_size("sentence-transformers/all-MiniLM-L6-v2") == 384


def test_explicit_cloud_model_size_supports_other_models():
    assert _dense_vector_size("provider/custom-model", 768) == 768


def test_unknown_cloud_model_requires_explicit_size():
    with pytest.raises(SystemExit, match="QDRANT_DENSE_VECTOR_SIZE"):
        _dense_vector_size("provider/unknown-model")


def test_collection_creation_uses_cloud_model_metadata_without_fastembed():
    class FakeClient:
        def __init__(self) -> None:
            self.collection: dict = {}

        def collection_exists(self, _name: str) -> bool:
            return False

        def create_collection(self, **kwargs) -> None:
            self.collection = kwargs

        def create_payload_index(self, **_kwargs) -> None:
            return None

    client = FakeClient()
    _ensure_collection(client, "papers", "sentence-transformers/all-MiniLM-L6-v2")

    assert client.collection["vectors_config"]["dense"].size == 384
    sparse = client.collection["sparse_vectors_config"]["sparse"]
    assert sparse.modifier == models.Modifier.IDF
