from types import SimpleNamespace

from fastapi.testclient import TestClient
from pkb_api.db import get_session
from pkb_api.main import app
from pkb_api.models import Base
from pkb_api.services import get_services
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _install_health_overrides(*, qdrant: bool, opensearch: bool) -> None:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def get_session_override() -> object:
        with factory() as session:
            yield session

    fake_services = SimpleNamespace(
        qdrant_indexer=SimpleNamespace(health=lambda: qdrant),
        opensearch_indexer=SimpleNamespace(health=lambda: opensearch),
    )
    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_services] = lambda: fake_services


def test_health_endpoint_reports_ok_when_all_stores_up() -> None:
    _install_health_overrides(qdrant=True, opensearch=True)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["stores"] == {"postgres": True, "qdrant": True, "opensearch": True}
    app.dependency_overrides.clear()


def test_health_endpoint_reports_degraded_when_a_store_is_down() -> None:
    _install_health_overrides(qdrant=False, opensearch=True)
    client = TestClient(app)

    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["stores"]["qdrant"] is False
    assert body["stores"]["opensearch"] is True
    assert body["stores"]["postgres"] is True
    app.dependency_overrides.clear()
