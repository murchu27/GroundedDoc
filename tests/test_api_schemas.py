from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import (
    AnalyzeResponse,
    ConflictsResponse,
    HealthResponse,
    IngestResponse,
    QueryResponse,
)


def test_health_response_validates():
    model = HealthResponse.model_validate({"status": "ok"})
    assert model.status == "ok"


def test_query_response_validates_minimal_payload():
    payload = {
        "query": "Compare GDPR vs PIPEDA retention",
        "answer": "Based on the indexed documents...",
        "citations": [
            {
                "chunk_id": "abc123",
                "doc_id": "gdpr_summary",
                "section_path": "data-retention",
                "source": "gdpr_summary",
            }
        ],
        "retrieved_chunk_ids": ["abc123"],
        "retrieval_strategy": "claims",
        "query_type": "COMPARE",
        "sub_queries": ["GDPR requirements", "PIPEDA requirements"],
        "conflicts": [
            {
                "subject": "data_retention",
                "description": "Sources disagree on retention period.",
                "sources": ["gdpr_summary §data-retention", "pipeda_summary §data-retention"],
            }
        ],
        "refused": False,
        "trace_spans": [
            {"name": "router", "attributes": {"query_type": "COMPARE"}},
        ],
    }

    model = QueryResponse.model_validate(payload)

    assert model.query.startswith("Compare GDPR")
    assert len(model.citations) == 1
    assert model.citations[0].doc_id == "gdpr_summary"
    assert len(model.conflicts) == 1
    assert model.trace_spans[0].name == "router"


def test_ingest_response_validates_skipped_shape():
    model = IngestResponse.model_validate(
        {"status": "skipped", "reason": "index already exists"}
    )

    assert model.status == "skipped"
    assert model.reason == "index already exists"
    assert model.documents is None


def test_ingest_response_validates_completed_shape():
    model = IngestResponse.model_validate(
        {
            "status": "completed",
            "documents": 9,
            "sections": 24,
            "parent_chunks": 24,
            "child_chunks": 24,
            "claims": 32,
            "conflicts": [
                {
                    "subject": "data_retention",
                    "description": "Retention periods differ.",
                    "sources": ["gdpr_summary §data-retention"],
                }
            ],
        }
    )

    assert model.status == "completed"
    assert model.documents == 9
    assert model.conflicts is not None
    assert len(model.conflicts) == 1


def test_conflicts_response_validates():
    model = ConflictsResponse.model_validate(
        {
            "conflicts": [
                {
                    "subject": "data_retention",
                    "description": "Retention periods differ.",
                    "sources": ["gdpr_summary §data-retention"],
                }
            ]
        }
    )

    assert len(model.conflicts) == 1
    assert model.conflicts[0].subject == "data_retention"


def test_analyze_response_validates_success_shape():
    model = AnalyzeResponse.model_validate(
        {
            "doc_id": "policy:abc123",
            "url": "https://example.com/privacy",
            "disclaimer": "Informational analysis only; not legal advice.",
            "findings": [
                {
                    "topic": "data_retention",
                    "question": "Does retention align with GDPR?",
                    "status": "needs_review",
                    "summary": "The policy retains data for 90 days.",
                    "policy_citation": {
                        "doc_id": "policy:abc123",
                        "section_path": "data-retention",
                        "source": "policy:abc123",
                        "text": "We retain data for 90 days.",
                    },
                    "regulation_citation": {
                        "doc_id": "gdpr_retention",
                        "section_path": "requirement",
                        "source": "gdpr_retention",
                        "text": "Retention must be limited.",
                    },
                    "conflicts": [],
                    "refused": False,
                }
            ],
            "conflicts": [],
            "ingest": {
                "doc_id": "policy:abc123",
                "sections": 3,
                "parent_chunks": 3,
                "child_chunks": 3,
                "claims": 2,
                "conflicts": [],
            },
            "refused_count": 0,
            "gap_count": 0,
            "review_count": 1,
        }
    )

    assert model.doc_id == "policy:abc123"
    assert len(model.findings) == 1
    assert model.findings[0].status == "needs_review"
    assert model.findings[0].policy_citation is not None
    assert model.review_count == 1


def test_analyze_response_validates_error_shape():
    model = AnalyzeResponse.model_validate(
        {
            "doc_id": "",
            "url": "https://example.com/privacy",
            "findings": [],
            "conflicts": [],
            "disclaimer": "Informational analysis only; not legal advice.",
            "error": "Page text is too short to analyze.",
        }
    )

    assert model.error == "Page text is too short to analyze."
    assert model.findings == []
    assert model.ingest is None


def test_analyze_response_rejects_invalid_finding_status():
    with pytest.raises(Exception):
        AnalyzeResponse.model_validate(
            {
                "doc_id": "policy:abc123",
                "url": "",
                "disclaimer": "Informational analysis only; not legal advice.",
                "findings": [
                    {
                        "topic": "data_retention",
                        "question": "q",
                        "status": "not_a_real_status",
                        "summary": "answer",
                    }
                ],
            }
        )


def test_openapi_includes_response_schemas():
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    assert "HealthResponse" in components
    assert "QueryResponse" in components
    assert "AnalyzeResponse" in components
    assert "IngestResponse" in components
    assert "ConflictsResponse" in components