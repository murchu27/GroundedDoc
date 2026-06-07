import pytest

from grounded_doc_agent.config.settings import CORPUS_DIR, INDEX_DIR
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline


@pytest.fixture(scope="session", autouse=True)
def ensure_index_exists():
    if not (INDEX_DIR / "ingestion_report.json").exists():
        IngestionPipeline().run()


@pytest.fixture(scope="session")
def indexed_pipeline() -> IngestionPipeline:
    report_path = INDEX_DIR / "ingestion_report.json"
    if report_path.exists():
        return IngestionPipeline.load_existing(INDEX_DIR)
    pipeline = IngestionPipeline(corpus_dir=CORPUS_DIR, index_dir=INDEX_DIR)
    pipeline.run()
    return pipeline
