from __future__ import annotations

from pathlib import Path

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
    embedding_provider: str = "hash"
    embedding_dimensions: int = 1024
    llm_api_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None


settings = Settings()
