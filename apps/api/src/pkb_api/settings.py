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
    storage_derived_path: Path = Path("storage/derived")

    # Embedding provider selection. "hash" is a deterministic local provider used so
    # tests never need a network model; "openai" routes to an OpenAI-compatible
    # /embeddings endpoint. When using "openai", set EMBEDDING_DIMENSIONS to match
    # the model so the Qdrant collection is created with the correct vector size.
    embedding_provider: str = "hash"
    embedding_dimensions: int = 1024
    embedding_api_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None

    llm_api_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    # Comma-separated list of allowed browser origins for the API.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


settings = Settings()
