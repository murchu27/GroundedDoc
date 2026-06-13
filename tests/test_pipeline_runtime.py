from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from grounded_doc_agent.agents import pipeline as pipeline_module
from grounded_doc_agent.agents.pipeline import get_pipeline, reset_pipeline
from grounded_doc_agent.agents.tools import get_conflict_report, query_documents


@pytest.fixture(autouse=True)
def clear_pipeline_singleton() -> None:
    reset_pipeline()
    yield
    reset_pipeline()


def test_get_pipeline_returns_same_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_pipeline = MagicMock(name="DocumentPipeline")
    monkeypatch.setattr(pipeline_module, "DocumentPipeline", mock_pipeline)

    first = get_pipeline()
    second = get_pipeline()

    assert first is second
    mock_pipeline.assert_called_once()


def test_reset_pipeline_creates_new_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = [MagicMock(name=f"DocumentPipeline-{index}") for index in range(2)]
    call_count = {"value": 0}

    def factory(*args, **kwargs):
        instance = instances[call_count["value"]]
        call_count["value"] += 1
        return instance

    monkeypatch.setattr(pipeline_module, "DocumentPipeline", factory)

    first = get_pipeline()
    reset_pipeline()
    second = get_pipeline()

    assert first is not second


def test_tools_use_shared_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    document_pipeline = MagicMock()
    ingestion_pipeline = MagicMock()
    document_pipeline.pipeline = ingestion_pipeline
    response = MagicMock()
    response.to_dict.return_value = {"answer": "grounded"}
    document_pipeline.run.return_value = response
    ingestion_pipeline.claims_store.list_conflicts.return_value = []

    monkeypatch.setattr("grounded_doc_agent.agents.tools.get_pipeline", lambda: document_pipeline)

    query_result = query_documents("What is GDPR retention?")
    conflict_result = get_conflict_report()

    assert query_result == {"answer": "grounded"}
    document_pipeline.run.assert_called_once_with("What is GDPR retention?")
    assert conflict_result == {"conflicts": []}