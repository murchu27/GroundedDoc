from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    LOOKUP = "LOOKUP"
    COMPARE = "COMPARE"
    MULTI_HOP = "MULTI_HOP"
    SUMMARIZE = "SUMMARIZE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    CLAIMS = "claims"
    MULTI_HOP = "multi_hop"
    PARENT = "parent"


@dataclass
class DocumentSection:
    doc_id: str
    section_path: str
    title: str
    content: str
    source_url: str = ""
    page: int | None = None


@dataclass
class ParentChunk:
    chunk_id: str
    doc_id: str
    section_path: str
    title: str
    summary: str
    child_ids: list[str] = field(default_factory=list)
    source_url: str = ""


@dataclass
class ChildChunk:
    chunk_id: str
    doc_id: str
    parent_id: str
    section_path: str
    text: str
    page: int | None = None
    source_url: str = ""


@dataclass
class Claim:
    claim_id: str
    subject: str
    value: str
    doc_id: str
    chunk_id: str
    section_path: str
    source_label: str


@dataclass
class ClaimConflict:
    subject: str
    claims: list[Claim]
    description: str


@dataclass
class RetrievalPlan:
    query_type: QueryType
    sub_queries: list[str]
    strategies: list[RetrievalStrategy]


@dataclass
class RetrievedContext:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_path: str
    source_url: str
    chunk_level: str = "child"


@dataclass
class PipelineTraceSpan:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    query: str
    answer: str
    citations: list[dict[str, str]]
    retrieved_chunk_ids: list[str]
    retrieval_strategy: str
    query_type: str
    sub_queries: list[str]
    conflicts: list[dict[str, Any]]
    refused: bool
    trace_spans: list[PipelineTraceSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieval_strategy": self.retrieval_strategy,
            "query_type": self.query_type,
            "sub_queries": self.sub_queries,
            "conflicts": self.conflicts,
            "refused": self.refused,
            "trace_spans": [
                {"name": span.name, "attributes": span.attributes}
                for span in self.trace_spans
            ],
        }
