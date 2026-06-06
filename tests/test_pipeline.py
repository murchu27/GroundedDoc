from __future__ import annotations

from pathlib import Path

import pytest

from grounded_doc_agent.agents.pipeline import DocumentPipeline
from grounded_doc_agent.agents.planner import build_retrieval_plan, classify_query
from grounded_doc_agent.config.settings import CORPUS_DIR, INDEX_DIR
from grounded_doc_agent.ingestion.claims import detect_conflicts, extract_claims_from_section
from grounded_doc_agent.ingestion.parser import parse_markdown_text
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline
from grounded_doc_agent.models import DocumentSection, QueryType


@pytest.fixture(scope="session")
def indexed_pipeline() -> IngestionPipeline:
    report_path = INDEX_DIR / "ingestion_report.json"
    if report_path.exists():
        return IngestionPipeline.load_existing(INDEX_DIR)
    pipeline = IngestionPipeline(corpus_dir=CORPUS_DIR, index_dir=INDEX_DIR)
    pipeline.run()
    return pipeline


def test_parse_markdown_sections():
    text = "# Retention\nKeep data for 30 days.\n\n# Sharing\nUse contracts."
    sections = parse_markdown_text(text, doc_id="demo")
    assert len(sections) == 2
    assert sections[0].section_path == "retention"


def test_retention_pattern_does_not_split_multi_digit_values():
    sections = [
        DocumentSection(
            "gdpr_summary",
            "data-retention",
            "Data Retention",
            "Typical guidance indicates personal data retention should not exceed 30 days "
            "unless a longer period is legally justified.",
        ),
        DocumentSection(
            "pipeda_summary",
            "data-retention",
            "Data Retention",
            "For many consumer services, personal data retention is commonly set to 90 days "
            "after account closure unless law requires longer storage.",
        ),
    ]
    claims = []
    for section in sections:
        claims.extend(extract_claims_from_section(section))
    retention_values = {
        claim.value for claim in claims if claim.subject == "data_retention"
    }
    assert "0 days" not in retention_values
    assert "30 days" in retention_values
    assert "90 days" in retention_values


def test_conflict_detection():
    sections = [
        DocumentSection("gdpr_summary", "data-retention", "Retention", "retention is 30 days"),
        DocumentSection("pipeda_summary", "data-retention", "Retention", "retention is 90 days"),
    ]
    claims = []
    for section in sections:
        claims.extend(extract_claims_from_section(section))
    conflicts = detect_conflicts(claims)
    assert conflicts
    assert conflicts[0].subject == "data_retention"


def test_query_planner_compare():
    plan = build_retrieval_plan("Compare data retention requirements in GDPR vs PIPEDA.")
    assert plan.query_type == QueryType.COMPARE
    assert len(plan.sub_queries) >= 2


def test_out_of_scope_refusal(indexed_pipeline):
    pipeline = DocumentPipeline(indexed_pipeline)
    response = pipeline.run("What is the weather in Toronto today?")
    assert response.refused is True


def test_compare_query_surfaces_multiple_docs(indexed_pipeline):
    pipeline = DocumentPipeline(indexed_pipeline)
    response = pipeline.run("Compare data retention requirements in GDPR vs PIPEDA.")
    docs = {citation["doc_id"] for citation in response.citations}
    assert "gdpr_summary" in docs or "pipeda_summary" in docs
    assert response.query_type == "COMPARE"


def test_fastapi_lookup(indexed_pipeline):
    pipeline = DocumentPipeline(indexed_pipeline)
    response = pipeline.run("How do I define dependencies in FastAPI?")
    assert response.refused is False
    assert response.retrieved_chunk_ids
