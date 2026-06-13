from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Shared components
class ConflictSummary(BaseModel):
    subject: str
    description: str
    sources: list[str]

class Citation(BaseModel):
    chunk_id: str = ""
    doc_id: str
    section_path: str
    source: str = ""

class TraceSpan(BaseModel):
    name: str
    attributes: dict[str, str] = Field(default_factory=dict)

class CitationDetail(BaseModel):
    doc_id: str
    section_path: str
    source: str = ""
    text: str = ""

FindingStatus = Literal[
    "aligned", "potential_gap", "insufficient_evidence", "needs_review"
]


# Request models
## /query, /eval/predict
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    variant: str = Field(default="full_pipeline")

## /ingest
class IngestRequest(BaseModel):
    rebuild: bool = True

## /analyze
class AnalyzeRequest(BaseModel):
    page_text: str = Field(..., min_length=50)
    url: str = Field(default="")
    questions: list[str] = Field(default_factory=list)


# Response models
## /health
class HealthResponse(BaseModel):
    status: str

## /query, /eval/predict
class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    retrieval_strategy: str
    query_type: str
    sub_queries: list[str]
    conflicts: list[ConflictSummary]
    refused: bool
    trace_spans: list[TraceSpan] = Field(default_factory=list)

## /ingest
class IngestResponse(BaseModel):
    status: Literal["completed", "skipped"]
    reason: str | None = None          # when skipped
    documents: int | None = None       # when completed
    sections: int | None = None
    parent_chunks: int | None = None
    child_chunks: int | None = None
    claims: int | None = None
    conflicts: list[ConflictSummary] | None = None

## /conflicts
class ConflictsResponse(BaseModel):
    conflicts: list[ConflictSummary]


class AnalyzeFinding(BaseModel):
    topic: str
    question: str
    status: FindingStatus
    summary: str
    policy_citation: CitationDetail | None = None
    regulation_citation: CitationDetail | None = None
    conflicts: list[ConflictSummary] = Field(default_factory=list)
    refused: bool = False

class OverlayIngestReport(BaseModel):
    doc_id: str
    sections: int
    parent_chunks: int
    child_chunks: int
    claims: int
    conflicts: list[ConflictSummary] = Field(default_factory=list)

## /analyze
class AnalyzeResponse(BaseModel):
    doc_id: str
    url: str
    disclaimer: str
    findings: list[AnalyzeFinding] = Field(default_factory=list)
    conflicts: list[ConflictSummary] = Field(default_factory=list)
    ingest: OverlayIngestReport | None = None
    refused_count: int | None = None
    gap_count: int | None = None
    review_count: int | None = None
    error: str | None = None