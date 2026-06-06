from __future__ import annotations

import json
from pathlib import Path

from grounded_doc_agent.config.settings import CLAIMS_DB_PATH, CORPUS_DIR, INDEX_DIR
from grounded_doc_agent.ingestion.chunker import build_hierarchical_chunks
from grounded_doc_agent.ingestion.claims import (
    detect_conflicts,
    extract_claims_from_child,
    extract_claims_from_section,
)
from grounded_doc_agent.ingestion.parser import load_corpus
from grounded_doc_agent.models import ChildChunk, ParentChunk
from grounded_doc_agent.retrieval.bm25 import BM25Index
from grounded_doc_agent.retrieval.claims_store import ClaimsStore
from grounded_doc_agent.retrieval.vector_index import VectorIndex


class IngestionPipeline:
    def __init__(
        self,
        *,
        corpus_dir: Path = CORPUS_DIR,
        index_dir: Path = INDEX_DIR,
        claims_db_path: Path = CLAIMS_DB_PATH,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.index_dir = index_dir
        self.claims_store = ClaimsStore(claims_db_path)
        self.vector_index = VectorIndex(index_dir)
        self.child_bm25 = BM25Index()
        self.parent_bm25 = BM25Index()

    def run(self) -> dict:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_index = VectorIndex(self.index_dir)
        self.claims_store.clear()

        sections = load_corpus(self.corpus_dir)
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        claims = []

        for section in sections:
            parent, section_children = build_hierarchical_chunks(section)
            parents.append(parent)
            children.extend(section_children)
            claims.extend(extract_claims_from_section(section))
            for child in section_children:
                claims.extend(extract_claims_from_child(child))

        conflicts = detect_conflicts(claims)
        self.vector_index.add_parents(parents)
        self.vector_index.add_children(children)
        self.child_bm25.build(children, level="child")
        self.parent_bm25.build(parents, level="parent")
        self.claims_store.upsert_claims(claims)
        self.claims_store.upsert_conflicts(conflicts)

        metadata_path = self.index_dir / "chunk_metadata.json"
        self.vector_index.save_metadata(metadata_path)
        self.child_bm25.save(self.index_dir / "bm25_child.json")
        self.parent_bm25.save(self.index_dir / "bm25_parent.json")

        report = {
            "documents": len({section.doc_id for section in sections}),
            "sections": len(sections),
            "parent_chunks": len(parents),
            "child_chunks": len(children),
            "claims": len(claims),
            "conflicts": [
                {
                    "subject": conflict.subject,
                    "description": conflict.description,
                    "sources": [claim.source_label for claim in conflict.claims],
                }
                for conflict in conflicts
            ],
        }
        (self.index_dir / "ingestion_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return report

    @classmethod
    def load_existing(cls, index_dir: Path = INDEX_DIR) -> "IngestionPipeline":
        pipeline = cls(index_dir=index_dir)
        pipeline.vector_index.load_metadata(index_dir / "chunk_metadata.json")
        child_bm25_path = index_dir / "bm25_child.json"
        parent_bm25_path = index_dir / "bm25_parent.json"
        if child_bm25_path.exists():
            pipeline.child_bm25.load(child_bm25_path)
        if parent_bm25_path.exists():
            pipeline.parent_bm25.load(parent_bm25_path)
        return pipeline
