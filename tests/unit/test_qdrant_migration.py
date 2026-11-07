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


def test_migrate_command_does_not_activate_alias_without_flag(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import migrate_qdrant

    alias_activated = False
    fake_client = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(migrate_qdrant, "_client", lambda: fake_client)
    monkeypatch.setattr(
        migrate_qdrant, "active_corpus_path", lambda: Path("/fake/versions/v1/papers.jsonl")
    )
    monkeypatch.setattr(migrate_qdrant, "load_papers", lambda _p: [])
    monkeypatch.setattr(migrate_qdrant, "_corpus_manifest", lambda _p, _d: None)
    monkeypatch.setattr(migrate_qdrant, "_collection_name", lambda _p: "academic-papers-test")
    monkeypatch.setattr(migrate_qdrant, "_require_cloud_settings", lambda: ("url", "key", "model"))
    monkeypatch.setattr(migrate_qdrant, "_ensure_collection", lambda *_: None)
    monkeypatch.setattr(migrate_qdrant, "_upsert_corpus", lambda *_: None)
    monkeypatch.setattr(migrate_qdrant, "audit", lambda *_: {})
    monkeypatch.setattr(migrate_qdrant, "smoke", lambda *_: None)

    def spy_activate(*_):
        nonlocal alias_activated
        alias_activated = True

    monkeypatch.setattr(migrate_qdrant, "_activate_alias", spy_activate)

    # Run migrate without --activate
    monkeypatch.setattr(sys, "argv", ["migrate_qdrant.py", "migrate"])
    migrate_qdrant.main()
    assert alias_activated is False, (
        "migrate without --activate must NOT switch the production alias"
    )

    # Run migrate with --activate
    monkeypatch.setattr(sys, "argv", ["migrate_qdrant.py", "migrate", "--activate"])
    migrate_qdrant.main()
    assert alias_activated is True, "migrate with --activate MUST switch the production alias"


def test_activate_command_switches_alias_when_explicitly_called(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import migrate_qdrant

    alias_activated = False
    fake_client = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(migrate_qdrant, "_client", lambda: fake_client)
    monkeypatch.setattr(
        migrate_qdrant, "active_corpus_path", lambda: Path("/fake/versions/v1/papers.jsonl")
    )
    monkeypatch.setattr(migrate_qdrant, "_collection_name", lambda _p: "academic-papers-test")
    monkeypatch.setattr(migrate_qdrant, "audit", lambda *_: {})
    monkeypatch.setattr(migrate_qdrant, "smoke", lambda *_: None)

    def spy_activate(*_):
        nonlocal alias_activated
        alias_activated = True

    monkeypatch.setattr(migrate_qdrant, "_activate_alias", spy_activate)

    monkeypatch.setattr(
        sys, "argv", ["migrate_qdrant.py", "activate", "--collection", "academic-papers-v2"]
    )
    migrate_qdrant.main()
    assert alias_activated is True


def test_activate_command_fails_without_explicit_collection(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import migrate_qdrant

    fake_client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(migrate_qdrant, "_client", lambda: fake_client)
    monkeypatch.setattr(
        migrate_qdrant, "active_corpus_path", lambda: Path("/fake/versions/v1/papers.jsonl")
    )

    monkeypatch.setattr(sys, "argv", ["migrate_qdrant.py", "activate"])
    with pytest.raises(SystemExit, match="explicit --collection"):
        migrate_qdrant.main()


def test_activate_does_not_switch_alias_when_audit_fails(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import migrate_qdrant

    alias_activated = False
    fake_client = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(migrate_qdrant, "_client", lambda: fake_client)
    monkeypatch.setattr(
        migrate_qdrant, "active_corpus_path", lambda: Path("/fake/versions/v1/papers.jsonl")
    )

    def failing_audit(*_):
        raise SystemExit("Audit validation failed: count mismatch")

    def spy_activate(*_):
        nonlocal alias_activated
        alias_activated = True

    monkeypatch.setattr(migrate_qdrant, "audit", failing_audit)
    monkeypatch.setattr(migrate_qdrant, "smoke", lambda *_: None)
    monkeypatch.setattr(migrate_qdrant, "_activate_alias", spy_activate)

    monkeypatch.setattr(
        sys, "argv", ["migrate_qdrant.py", "activate", "--collection", "academic-papers-broken"]
    )
    with pytest.raises(SystemExit, match="Audit validation failed"):
        migrate_qdrant.main()

    assert alias_activated is False, "Alias must NOT be switched if audit fails"


def test_activate_does_not_switch_alias_when_smoke_fails(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import migrate_qdrant

    alias_activated = False
    fake_client = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(migrate_qdrant, "_client", lambda: fake_client)
    monkeypatch.setattr(
        migrate_qdrant, "active_corpus_path", lambda: Path("/fake/versions/v1/papers.jsonl")
    )

    def failing_smoke(*_):
        raise SystemExit("Smoke test query returned no results")

    def spy_activate(*_):
        nonlocal alias_activated
        alias_activated = True

    monkeypatch.setattr(migrate_qdrant, "audit", lambda *_: {})
    monkeypatch.setattr(migrate_qdrant, "smoke", failing_smoke)
    monkeypatch.setattr(migrate_qdrant, "_activate_alias", spy_activate)

    monkeypatch.setattr(
        sys, "argv", ["migrate_qdrant.py", "activate", "--collection", "academic-papers-broken"]
    )
    with pytest.raises(SystemExit, match="Smoke test query returned no results"):
        migrate_qdrant.main()

    assert alias_activated is False, "Alias must NOT be switched if smoke test fails"
