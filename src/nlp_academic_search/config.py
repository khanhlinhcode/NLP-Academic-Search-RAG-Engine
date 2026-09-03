"""Typed, validated application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Relative runtime paths are anchored to the process working directory. This works
# for editable installs and wheels alike, provided commands are launched from the
# project/deployment root as documented by the Makefile and container WORKDIR.
PROJECT_ROOT = Path.cwd().resolve()


class OllamaConfig(BaseModel):
    base_url: str
    model_name: str
    timeout_seconds: float
    concurrency_limit: int


class EmbeddingConfig(BaseModel):
    model_name: str
    model_revision: str | None
    device: Literal["cpu", "cuda", "mps"]
    native_threads: int


class SearchConfig(BaseModel):
    bm25_weight: float
    semantic_weight: float
    default_top_k: int
    candidate_pool: int
    timeout_seconds: float


class RerankerConfig(BaseModel):
    model_name: str
    enabled: bool
    timeout_seconds: float


class DataConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    embeddings_dir: Path

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.embeddings_dir):
            path.mkdir(parents=True, exist_ok=True)


class APIConfig(BaseModel):
    host: str
    port: int
    base_url: str
    cors_origins: list[str]
    request_timeout_seconds: float
    concurrency_limit: int


class Settings(BaseSettings):
    """Single source of truth for process configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    environment: Literal["local", "test", "production"] = "local"
    ollama_base_url: AnyHttpUrl = "http://localhost:11434"  # type: ignore[assignment]
    llm_model_name: str = "qwen2.5:7b"
    generation_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    generation_concurrency_limit: int = Field(default=1, ge=1, le=32)
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_model_revision: str | None = None
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_native_threads: int = Field(default=1, ge=1, le=32)
    bm25_weight: float = Field(default=0.4, ge=0, le=1)
    semantic_weight: float = Field(default=0.6, ge=0, le=1)
    default_top_k: int = Field(default=10, ge=1, le=50)
    search_candidate_pool: int = Field(default=50, ge=1, le=500)
    search_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = False
    reranker_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_embeddings_dir: Path = Path("data/embeddings")
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: AnyHttpUrl = "http://localhost:8000"  # type: ignore[assignment]
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    api_request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    api_concurrency_limit: int = Field(default=4, ge=1, le=128)
    rag_min_relevance_score: float = Field(default=0.2, ge=-1, le=1)
    rag_max_context_chars: int = Field(default=24000, ge=1000, le=200000)
    rag_enabled: bool = True

    @field_validator("llm_model_name", "embedding_model_name", "reranker_model_name", mode="before")
    @classmethod
    def model_names_must_not_be_blank(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("embedding_model_revision", mode="before")
    @classmethod
    def empty_revision_must_be_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("data_raw_dir", "data_processed_dir", "data_embeddings_dir")
    @classmethod
    def resolve_data_path(cls, value: Path) -> Path:
        path = value.expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @model_validator(mode="after")
    def validate_search_contract(self) -> Settings:
        if abs((self.bm25_weight + self.semantic_weight) - 1.0) > 1e-6:
            raise ValueError("BM25_WEIGHT and SEMANTIC_WEIGHT must sum to 1.0")
        if self.search_candidate_pool < self.default_top_k:
            raise ValueError("SEARCH_CANDIDATE_POOL must be >= DEFAULT_TOP_K")
        if self.environment == "production" and "*" in self.cors_origin_list:
            raise ValueError("CORS_ORIGINS cannot contain '*' in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [origin.strip().rstrip("/") for origin in self.cors_origins.split(",")]
        return [origin for origin in origins if origin]

    @property
    def ollama(self) -> OllamaConfig:
        return OllamaConfig(
            base_url=str(self.ollama_base_url).rstrip("/"),
            model_name=self.llm_model_name,
            timeout_seconds=self.generation_timeout_seconds,
            concurrency_limit=self.generation_concurrency_limit,
        )

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_name=self.embedding_model_name,
            model_revision=self.embedding_model_revision,
            device=self.embedding_device,
            native_threads=self.embedding_native_threads,
        )

    @property
    def search(self) -> SearchConfig:
        return SearchConfig(
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            default_top_k=self.default_top_k,
            candidate_pool=self.search_candidate_pool,
            timeout_seconds=self.search_timeout_seconds,
        )

    @property
    def reranker(self) -> RerankerConfig:
        return RerankerConfig(
            model_name=self.reranker_model_name,
            enabled=self.use_reranker,
            timeout_seconds=self.reranker_timeout_seconds,
        )

    @property
    def data(self) -> DataConfig:
        return DataConfig(
            raw_dir=self.data_raw_dir,
            processed_dir=self.data_processed_dir,
            embeddings_dir=self.data_embeddings_dir,
        )

    @property
    def api(self) -> APIConfig:
        return APIConfig(
            host=self.api_host,
            port=self.api_port,
            base_url=str(self.api_base_url).rstrip("/"),
            cors_origins=self.cors_origin_list,
            request_timeout_seconds=self.api_request_timeout_seconds,
            concurrency_limit=self.api_concurrency_limit,
        )


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        errors = []
        for error in exc.errors(include_url=False, include_input=False):
            variable = ".".join(str(part).upper() for part in error["loc"])
            errors.append(f"{variable}: {error['msg']}")
        raise RuntimeError(f"Invalid application configuration: {'; '.join(errors)}") from None


settings = load_settings()
