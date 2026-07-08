from fastapi.testclient import TestClient
from pkb_api.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_endpoint_returns_empty_citation_result_before_indexes_exist() -> None:
    client = TestClient(app)

    response = client.post("/search", json={"query": "lock", "limit": 3})

    assert response.status_code == 200
    assert response.json()["citations"] == []
