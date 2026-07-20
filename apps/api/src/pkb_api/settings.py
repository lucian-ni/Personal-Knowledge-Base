from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    database_url: str = "postgresql+psycopg://pkb:pkb@localhost:5432/pkb"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pkb_chunks"
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "pkb_chunks"
    storage_docs_path: Path = Path("storage/docs")

    # Embedding. "local" runs bge-small-zh-v1.5 in-process (no network after the
    # one-time model download); "openai" routes to an OpenAI-compatible
    # /embeddings endpoint (set EMBEDDING_API_* and EMBEDDING_MODEL to the API
    # model name). EMBEDDING_DIMENSIONS must match the model so the Qdrant
    # collection is created with the right vector size.
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimensions: int = 512
    embedding_api_base_url: str | None = None
    embedding_api_key: str | None = None

    # Cross-encoder reranker (Qwen3-Reranker-0.6B). Disable to skip the model
    # load for fast smoke tests; retrieval then returns RRF-fused hits directly.
    reranker_enabled: bool = True
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"

    # Retrieval tuning: RRF constant k, and how many hits each backend fetches
    # before reranking.
    rrf_k: int = 60
    rerank_top_k: int = 20

    llm_api_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    # Comma-separated list of allowed browser origins for the API.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


settings = Settings()
