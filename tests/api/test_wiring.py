from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pkb_api.db import get_session
from pkb_api.main import app
from pkb_api.models import Base
from pkb_api.retrieval import RetrievalHit, SimpleAnswerGenerator
from pkb_api.services import Services, get_services, make_embedding_provider
from pkb_api.settings import settings
from pkb_api.storage import DocumentStorage
from pkb_ingestion.embeddings import HashEmbeddingProvider
from pkb_ingestion.models import QdrantPoint
from pkb_ingestion.pipeline import IngestionPipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class FakeStores:
    """In-memory capture of what the indexers write, shared with the search backends."""

    def __init__(self) -> None:
        self.qdrant_points: dict[str, QdrantPoint] = {}
        self.opensearch_docs: dict[str, dict] = {}
        self.collection_ensured = False
        self.index_ensured = False


class FakeQdrantIndexer:
    def __init__(self, stores: FakeStores, *, fail_upsert: bool = False) -> None:
        self.stores = stores
        self.fail_upsert = fail_upsert

    def ensure_collection(self) -> None:
        self.stores.collection_ensured = True

    def upsert(self, points: list[QdrantPoint]) -> None:
        if self.fail_upsert:
            raise RuntimeError("qdrant unreachable")
        for point in points:
            self.stores.qdrant_points[point.id] = point


class FakeOpenSearchIndexer:
    def __init__(self, stores: FakeStores) -> None:
        self.stores = stores

    def ensure_index(self) -> None:
        self.stores.index_ensured = True

    def bulk_index(self, documents: list[dict]) -> None:
        for doc in documents:
            self.stores.opensearch_docs[doc["_id"]] = doc["document"]


class FakeVectorBackend:
    def __init__(self, stores: FakeStores) -> None:
        self.stores = stores

    def search(self, vector: list[float], limit: int) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for point in list(self.stores.qdrant_points.values())[:limit]:
            payload = point.payload
            hits.append(
                RetrievalHit(
                    document_id=payload["docId"],
                    chunk_id=payload["chunkId"],
                    title=payload["title"],
                    section=payload["section"],
                    page=payload["page"],
                    text=payload["text"],
                    score=0.9,
                    source="vector",
                )
            )
        return hits


class FakeKeywordBackend:
    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        return []


def _build_fake_services(
    tmp_path: Path, *, fail_qdrant: bool = False
) -> tuple[Services, FakeStores]:
    stores = FakeStores()
    services = Services(
        pipeline=IngestionPipeline(embedding_provider=HashEmbeddingProvider(dimensions=16)),
        storage=DocumentStorage(tmp_path),
        qdrant_indexer=FakeQdrantIndexer(stores, fail_upsert=fail_qdrant),
        opensearch_indexer=FakeOpenSearchIndexer(stores),
        vector_backend=FakeVectorBackend(stores),
        keyword_backend=FakeKeywordBackend(),
        answer_generator=SimpleAnswerGenerator(),
    )
    return services, stores


def _install_overrides(services: Services) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def get_session_override() -> object:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_services] = lambda: services


@pytest.fixture(autouse=True)
def _reset_overrides() -> None:
    yield
    app.dependency_overrides.clear()


_MARKDOWN = b"""# Notes

## Locks

ReentrantLock is reentrant and useful for concurrency.
"""


def test_upload_then_list_then_search_round_trip(tmp_path: Path) -> None:
    services, stores = _build_fake_services(tmp_path)
    _install_overrides(services)

    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("notes.md", _MARKDOWN, "text/markdown")},
    )

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "completed"
    assert job["document_id"]

    document_id = job["document_id"]
    assert stores.collection_ensured
    assert stores.index_ensured
    # Same stable_chunk_id is used as Qdrant point id and OpenSearch doc id.
    assert list(stores.qdrant_points) == list(stores.opensearch_docs)
    chunk_id = f"{document_id}:000001"
    assert chunk_id in stores.qdrant_points
    assert stores.qdrant_points[chunk_id].payload["text"].startswith("ReentrantLock")

    listed = client.get("/documents").json()
    assert len(listed) == 1
    assert listed[0]["status"] == "ready"
    assert listed[0]["original_filename"] == "notes.md"

    result = client.post("/search", json={"query": "reentrant", "limit": 5}).json()
    assert result["citations"], result
    assert result["citations"][0]["chunk_id"] == chunk_id
    assert "reentrant" in result["citations"][0]["text"].lower()


def test_unsupported_file_type_returns_415(tmp_path: Path) -> None:
    services, _ = _build_fake_services(tmp_path)
    _install_overrides(services)

    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("picture.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )

    assert response.status_code == 415, response.text
    # No document row should be created for an unsupported type.
    assert client.get("/documents").json() == []


def test_store_failure_marks_document_failed(tmp_path: Path) -> None:
    services, _ = _build_fake_services(tmp_path, fail_qdrant=True)
    _install_overrides(services)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/documents",
        files={"file": ("notes.md", _MARKDOWN, "text/markdown")},
    )

    assert response.status_code == 500, response.text
    listed = client.get("/documents").json()
    assert len(listed) == 1
    assert listed[0]["status"] == "failed"


def test_make_embedding_provider_defaults_to_hash() -> None:
    provider = make_embedding_provider(settings)
    assert isinstance(provider, HashEmbeddingProvider)


def test_make_embedding_provider_returns_hash_when_openai_unconfigured() -> None:
    from pkb_api.settings import Settings

    provider = make_embedding_provider(
        Settings(embedding_provider="openai", embedding_dimensions=64)
    )
    assert isinstance(provider, HashEmbeddingProvider)


def test_make_embedding_provider_builds_http_when_configured() -> None:
    from pkb_api.settings import Settings
    from pkb_ingestion.embeddings import OpenAICompatibleEmbeddingProvider

    config = Settings(
        embedding_provider="openai",
        embedding_dimensions=64,
        embedding_api_base_url="https://embed.example.com",
        embedding_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )
    provider = make_embedding_provider(config)
    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
