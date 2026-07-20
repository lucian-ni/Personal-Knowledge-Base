from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pkb_api.db import get_session
from pkb_api.main import app
from pkb_api.models import Base
from pkb_api.retrieval import LLMAnswerGenerator, RetrievalHit, SimpleAnswerGenerator
from pkb_api.services import Services, get_services
from pkb_api.storage import DocumentStorage
from pkb_ingestion.pipeline import IngestionPipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _hit(chunk_id: str, text: str = "ReentrantLock is reentrant.") -> RetrievalHit:
    return RetrievalHit("doc", chunk_id, "Title", "Lock", 12, text, 0.9, "vector")


class FakeEmbed:
    dimensions = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 16 for _ in texts]


class FakeVectorBackend:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits

    def search(self, vector: list[float], limit: int) -> list[RetrievalHit]:
        return self._hits[:limit]


class FakeKeywordBackend:
    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        return []


def _install_services(services: Services) -> None:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def get_session_override() -> object:
        with factory() as session:
            yield session

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_services] = lambda: services


def _parse_sse(text: str) -> list[dict]:
    events: list[dict] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = {"type": "message", "data": ""}
        for line in block.split("\n"):
            if line.startswith("event: "):
                event["type"] = line[len("event: ") :]
            elif line.startswith("data: "):
                event["data"] += line[len("data: ") :]
        events.append(event)
    return events


@pytest.fixture(autouse=True)
def _reset_overrides() -> None:
    yield
    app.dependency_overrides.clear()


def test_search_stream_emits_citations_delta_done(tmp_path: Path) -> None:
    hits = [_hit("doc:000001")]
    services = Services(
        pipeline=IngestionPipeline(embedding_provider=FakeEmbed()),
        storage=DocumentStorage(tmp_path),
        qdrant_indexer=SimpleNamespace(),
        opensearch_indexer=SimpleNamespace(),
        vector_backend=FakeVectorBackend(hits),
        keyword_backend=FakeKeywordBackend(),
        answer_generator=SimpleAnswerGenerator(),
        reranker=None,
    )
    _install_services(services)

    client = TestClient(app)
    response = client.post("/search/stream", json={"query": "reentrant", "limit": 5})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)

    assert events[0]["type"] == "citations"
    citations = json.loads(events[0]["data"])
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "doc:000001"

    deltas = [json.loads(e["data"])["content"] for e in events if e["type"] == "delta"]
    assert deltas  # at least one delta
    assert "ReentrantLock is reentrant." in "".join(deltas)

    assert events[-1]["type"] == "done"


def test_search_stream_empty_when_nothing_indexed(tmp_path: Path) -> None:
    services = Services(
        pipeline=IngestionPipeline(embedding_provider=FakeEmbed()),
        storage=DocumentStorage(tmp_path),
        qdrant_indexer=SimpleNamespace(),
        opensearch_indexer=SimpleNamespace(),
        vector_backend=FakeVectorBackend([]),
        keyword_backend=FakeKeywordBackend(),
        answer_generator=SimpleAnswerGenerator(),
        reranker=None,
    )
    _install_services(services)

    client = TestClient(app)
    events = _parse_sse(client.post("/search/stream", json={"query": "x", "limit": 5}).text)

    assert events[0]["type"] == "citations"
    assert json.loads(events[0]["data"]) == []
    assert events[-1]["type"] == "done"


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        pass

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeStreamClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.captured: dict = {}

    def stream(self, method: str, url: str, **kwargs):
        self.captured["url"] = url
        self.captured["json"] = kwargs.get("json")
        return _FakeStreamResponse(self._lines)


def test_llm_stream_answer_yields_token_deltas() -> None:
    """Mocks the OpenAI SSE stream: deltas are extracted until [DONE]."""
    lines = [
        'data: {"choices":[{"delta":{"content":"Reentrant"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":"Lock"}}]}',
        "",
        'data: [DONE]',
    ]
    gen = LLMAnswerGenerator(base_url="https://llm.example.com", api_key="sk-test", model="m")
    gen._client = _FakeStreamClient(lines)

    chunks = list(gen.stream_answer("What is a lock?", [_hit("c1")]))

    assert chunks == ["Reentrant", "Lock"]
    assert gen._client.captured["json"]["stream"] is True


def test_llm_stream_answer_falls_back_when_unreachable() -> None:
    class BrokenClient:
        def stream(self, method, url, **kwargs):
            raise RuntimeError("llm unreachable")

    gen = LLMAnswerGenerator(base_url="https://llm.example.com", api_key="sk-test", model="m")
    gen._client = BrokenClient()

    chunks = "".join(gen.stream_answer("q", [_hit("c1", "ReentrantLock is reentrant.")]))

    assert "unreachable" in chunks
    assert "ReentrantLock is reentrant." in chunks  # raw context surfaced as fallback
