from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app


MOCK_QUERY_RESPONSE = {
    "query": "test query",
    "answer": "ok",
    "citations": [],
    "retrieved_chunk_ids": [],
    "retrieval_strategy": "vector",
    "query_type": "LOOKUP",
    "sub_queries": ["test query"],
    "conflicts": [],
    "refused": False,
    "trace_spans": [],
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_pipeline(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    pipeline = MagicMock()
    response = MagicMock()
    response.to_dict.return_value = MOCK_QUERY_RESPONSE
    pipeline.run.return_value = response
    pipeline.pipeline.claims_store.list_conflicts.return_value = []
    monkeypatch.setattr("api.main.get_pipeline", lambda: pipeline)
    monkeypatch.setattr(
        "api.main.predict_for_eval",
        lambda payload, variant="full_pipeline": MOCK_QUERY_RESPONSE,
    )
    return pipeline


def test_health_does_not_require_api_key(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_allowed_without_key_when_auth_disabled(
    client: TestClient,
    mock_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDED_REQUIRE_API_KEY", "false")
    monkeypatch.delenv("GROUNDED_API_KEY", raising=False)

    response = client.post("/query", json={"query": "test query"})

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"


def test_query_rejects_missing_key_when_auth_enabled(
    client: TestClient,
    mock_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDED_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("GROUNDED_API_KEY", "server-secret")

    response = client.post("/query", json={"query": "test query"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    mock_pipeline.run.assert_not_called()


def test_query_rejects_wrong_key_when_auth_enabled(
    client: TestClient,
    mock_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDED_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("GROUNDED_API_KEY", "server-secret")

    response = client.post(
        "/query",
        json={"query": "test query"},
        headers={"X-API-Key": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    mock_pipeline.run.assert_not_called()


def test_query_accepts_valid_key_when_auth_enabled(
    client: TestClient,
    mock_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDED_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("GROUNDED_API_KEY", "server-secret")

    response = client.post(
        "/query",
        json={"query": "test query"},
        headers={"X-API-Key": "server-secret"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
    mock_pipeline.run.assert_called_once()


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("get", "/conflicts", None),
        ("post", "/eval/predict", {"query": "test query"}),
    ],
)
def test_protected_routes_require_valid_key(
    client: TestClient,
    mock_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    payload: dict[str, str] | None,
) -> None:
    monkeypatch.setenv("GROUNDED_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("GROUNDED_API_KEY", "server-secret")

    request = getattr(client, method)
    kwargs: dict = {}
    if payload is not None:
        kwargs["json"] = payload

    unauthorized = request(path, **kwargs)
    assert unauthorized.status_code == 401

    authorized = request(path, headers={"X-API-Key": "server-secret"}, **kwargs)
    assert authorized.status_code == 200
