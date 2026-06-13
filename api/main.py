from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from grounded_doc_agent.agents.analyze import analyze_policy
from grounded_doc_agent.agents.pipeline import get_pipeline, reset_pipeline, predict_for_eval
from grounded_doc_agent.config.settings import INDEX_DIR
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ConflictsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)        

app = FastAPI(
    title="GroundedDoc Agent API",
    description="Conflict-aware document intelligence agent",
    version="0.1.0",
)

cors_origins = [
    origin.strip() for origin in os.getenv("GROUNDED_CORS_ORIGINS", "*").split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    required = os.getenv("GROUNDED_REQUIRE_API_KEY", "false").lower() == "true"
    if not required:
        return
    expected = os.getenv("GROUNDED_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    _: None = Depends(_check_api_key),
) -> QueryResponse:
    response = get_pipeline().run(request.query, variant=request.variant)
    return QueryResponse.model_validate(response.to_dict())


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    _: None = Depends(_check_api_key),
) -> AnalyzeResponse:
    questions = request.questions or None
    payload = analyze_policy(
        get_pipeline(),
        page_text=request.page_text,
        url=request.url,
        questions=questions,
    )
    return AnalyzeResponse.model_validate(payload)

@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    _: None = Depends(_check_api_key),
) -> IngestResponse:
    if not request.rebuild:
        report_path = INDEX_DIR / "ingestion_report.json"
        if report_path.exists():
            return IngestResponse(status="skipped", reason="index already exists")
    report = IngestionPipeline().run()
    reset_pipeline()
    return IngestResponse.model_validate({"status": "completed", **report})


@app.get("/conflicts", response_model=ConflictsResponse)
def conflicts(_: None = Depends(_check_api_key)) -> ConflictsResponse:
    pipeline = get_pipeline().pipeline
    items = pipeline.claims_store.list_conflicts()
    return ConflictsResponse(
        conflicts=[
            {
                "subject": item.subject,
                "description": item.description,
                "sources": [claim.source_label for claim in item.claims],
            }
            for item in items
        ]
    )

@app.post("/eval/predict", response_model=QueryResponse)
def eval_predict(
    request: QueryRequest,
    _: None = Depends(_check_api_key),
) -> QueryResponse:
    payload = predict_for_eval({"query": request.query}, variant=request.variant)
    return QueryResponse.model_validate(payload)
