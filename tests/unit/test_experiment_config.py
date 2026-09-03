"""Unit tests for experiment configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_academic_search.evaluation.experiment_config import ExperimentConfig


def test_load_valid_experiment_config():
    config_path = Path("configs/experiments/retrieval_baseline.toml")
    config = ExperimentConfig.load(config_path)

    assert config.name == "retrieval_baseline"
    assert config.dataset == "scifact"
    assert config.bm25_weight == 0.4
    assert config.semantic_weight == 0.6
    assert config.top_k == 10
    assert config.candidate_pool >= config.top_k
    assert config.output_dir.is_absolute()
    assert len(config.effective()["config_sha256"]) == 64


def test_load_missing_file(tmp_path: Path):
    missing_path = tmp_path / "non_existent.toml"
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.load(missing_path)


def test_load_invalid_table(tmp_path: Path):
    bad_file = tmp_path / "bad.toml"
    bad_file.write_text("[other]\nkey = 'value'", encoding="utf-8")
    with pytest.raises(ValueError, match="missing \\[experiment\\] table"):
        ExperimentConfig.load(bad_file)


def test_extra_field_forbidden(tmp_path: Path):
    bad_file = tmp_path / "extra.toml"
    content = """
[experiment]
name = "test"
dataset = "scifact"
output_dir = "reports/test"
unknown_key = "forbidden"
"""
    bad_file.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError):
        ExperimentConfig.load(bad_file)


def test_candidate_pool_smaller_than_top_k(tmp_path: Path):
    bad_file = tmp_path / "small_pool.toml"
    content = """
[experiment]
name = "test"
dataset = "scifact"
top_k = 20
candidate_pool = 5
output_dir = "reports/test"
"""
    bad_file.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError):
        ExperimentConfig.load(bad_file)


def test_bm25_weight_out_of_range(tmp_path: Path):
    bad_file = tmp_path / "bad_weight.toml"
    content = """
[experiment]
name = "test"
dataset = "scifact"
bm25_weight = 1.5
output_dir = "reports/test"
"""
    bad_file.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError):
        ExperimentConfig.load(bad_file)


def test_fusion_weights_must_sum_to_one(tmp_path: Path):
    bad_file = tmp_path / "bad_weight_sum.toml"
    bad_file.write_text(
        """
[experiment]
name = "test"
dataset = "fixture"
bm25_weight = 0.4
semantic_weight = 0.5
output_dir = "reports/test"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        ExperimentConfig.load(bad_file)


def test_unknown_retrieval_method_rejected(tmp_path: Path):
    bad_file = tmp_path / "bad_method.toml"
    bad_file.write_text(
        """
[experiment]
name = "test"
dataset = "fixture"
retrieval_method = "magic"
output_dir = "reports/test"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        ExperimentConfig.load(bad_file)
