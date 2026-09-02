"""
Application configuration.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""

    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name: str = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b")


@dataclass
class EmbeddingConfig:
    """Sentence embedding model configuration."""

    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


@dataclass
class SearchConfig:
    """Search parameters."""

    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "10"))


@dataclass
class RerankerConfig:
    """Cross-encoder reranker configuration."""

    model_name: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    enabled: bool = os.getenv("USE_RERANKER", "true").lower() == "true"


@dataclass
class DataConfig:
    """Data directory paths."""

    raw_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
    )
    processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    )
    embeddings_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_EMBEDDINGS_DIR", "data/embeddings")
    )

    def ensure_dirs(self):
        """Create data directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """FastAPI server configuration."""

    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


@dataclass
class Settings:
    """Application settings — single source of truth."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global settings instance
settings = Settings()
