from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_academic_search.config import Settings, load_settings


def test_settings_accept_valid_values():
    config = Settings(_env_file=None, environment="test", api_port=9000)  # type: ignore[call-arg]
    assert config.api.port == 9000
    assert config.reranker.enabled is False
    assert config.embedding.device == "cpu"
    assert config.embedding.native_threads == 1
    assert config.api.cors_origins == ["http://localhost:8501", "http://127.0.0.1:8501"]


def test_relative_data_paths_are_resolved_from_runtime_root():
    config = Settings(_env_file=None, data_raw_dir=Path("data/raw"))  # type: ignore[call-arg]
    assert config.data.raw_dir == (Path.cwd() / "data/raw").resolve()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"api_port": 70000}, "less than or equal"),
        ({"bm25_weight": 0.2, "semantic_weight": 0.2}, "must sum"),
        ({"search_candidate_pool": 2, "default_top_k": 10}, "must be"),
        ({"llm_model_name": " "}, "must not be empty"),
        ({"environment": "production", "cors_origins": "*"}, "cannot contain"),
        ({"embedding_device": "metal"}, "Input should be"),
        ({"embedding_native_threads": 0}, "greater than or equal"),
    ],
)
def test_settings_reject_invalid_values(values, message):
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None, **values)  # type: ignore[call-arg]


def test_load_settings_names_invalid_environment_variable(monkeypatch):
    monkeypatch.setenv("API_PORT", "not-a-port")
    with pytest.raises(RuntimeError, match="API_PORT"):
        load_settings()
