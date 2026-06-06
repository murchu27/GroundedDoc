import pytest

from grounded_doc_agent.config.settings import INDEX_DIR


@pytest.fixture(scope="session", autouse=True)
def ensure_index_exists():
    if not (INDEX_DIR / "ingestion_report.json").exists():
        from grounded_doc_agent.ingestion.pipeline import IngestionPipeline

        IngestionPipeline().run()
