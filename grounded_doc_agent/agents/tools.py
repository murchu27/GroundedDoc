from __future__ import annotations

from grounded_doc_agent.agents.pipeline import get_pipeline


def query_documents(query: str) -> dict:
    """Answer a question over the indexed document corpus with citations."""
    response = get_pipeline().run(query)
    return response.to_dict()


def get_conflict_report() -> dict:
    """Return detected cross-document claim conflicts from the indexed corpus."""
    pipeline = get_pipeline().pipeline
    conflicts = pipeline.claims_store.list_conflicts()
    return {
        "conflicts": [
            {
                "subject": conflict.subject,
                "description": conflict.description,
                "sources": [claim.source_label for claim in conflict.claims],
            }
            for conflict in conflicts
        ]
    }


def get_ingestion_status() -> dict:
    """Return ingestion statistics for the current index."""
    from grounded_doc_agent.config.settings import INDEX_DIR
    import json

    report_path = INDEX_DIR / "ingestion_report.json"
    if not report_path.exists():
        return {"status": "not_indexed"}
    return {"status": "ready", **json.loads(report_path.read_text(encoding="utf-8"))}
