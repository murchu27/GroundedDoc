from __future__ import annotations

import argparse
from pathlib import Path

from grounded_doc_agent.config.settings import CORPUS_DIR, INDEX_DIR
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus into GroundedDoc indexes")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
    )
    report = pipeline.run()
    print("Ingestion complete:")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
