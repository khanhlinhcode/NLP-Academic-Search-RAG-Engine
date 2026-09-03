"""Strict, reproducible experiment configuration loading."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

RetrievalMethod = Literal["bm25", "dense", "rrf", "weighted", "hybrid", "hybrid_reranked"]


def _repository_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent.resolve()
    return Path.cwd().resolve()


class ExperimentConfig(BaseModel):
    """Validated effective configuration; relative paths resolve from repository root."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    split: str = Field("test", min_length=1)
    corpus_path: Path | None = None
    index_path: Path | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_revision: str | None = None
    reranker_model: str | None = None
    retrieval_method: RetrievalMethod = "hybrid"
    bm25_weight: float = Field(0.4, ge=0.0, le=1.0)
    semantic_weight: float = Field(0.6, ge=0.0, le=1.0)
    rrf_k: int = Field(60, gt=0)
    candidate_pool: int = Field(50, gt=0)
    top_k: int = Field(10, gt=0)
    random_seed: int = Field(42, ge=0)
    timeout_seconds: float = Field(30.0, gt=0.0)
    concurrency: int = Field(1, gt=0)
    generator_model: str | None = None
    prompt_version: str | None = None
    output_dir: Path

    _config_path: Path | None = PrivateAttr(default=None)
    _config_sha256: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_contract(self) -> ExperimentConfig:
        if self.candidate_pool < self.top_k:
            raise ValueError("candidate_pool must be greater than or equal to top_k")
        if abs((self.bm25_weight + self.semantic_weight) - 1.0) > 1e-6:
            raise ValueError("bm25_weight and semantic_weight must sum to 1.0")
        return self

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        resolved_path = path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Experiment config file not found: {resolved_path}")
        raw = resolved_path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
        experiment_section = data.get("experiment")
        if not isinstance(experiment_section, dict):
            raise ValueError(f"Configuration file {resolved_path} missing [experiment] table")
        config = cls.model_validate(experiment_section)
        root = _repository_root(resolved_path)
        for field_name in ("corpus_path", "index_path", "output_dir"):
            value = getattr(config, field_name)
            if value is not None and not value.is_absolute():
                setattr(config, field_name, (root / value).resolve())
        config._config_path = resolved_path
        config._config_sha256 = hashlib.sha256(raw).hexdigest()
        return config

    def effective(self) -> dict[str, Any]:
        values = self.model_dump(mode="json")
        values["config_path"] = str(self._config_path) if self._config_path else None
        values["config_sha256"] = self._config_sha256
        return values
