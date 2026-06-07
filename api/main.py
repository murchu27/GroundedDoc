from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from grounded_doc_agent.agents.analyze import analyze_policy
from grounded_doc_agent.agents.pipeline import DocumentPipeline, predict_for_eval
from grounded_doc_agent.config.settings import INDEX_DIR
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline
from grounded_doc_agent.storage.backend import maybe_sync_index

_pipeline: DocumentPipeline | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    variant: str = Field(default="full_pipeline")


class IngestRequest(BaseModel):
    rebuild: bool = True


class AnalyzeRequest(BaseModel):
    page_text: str = Field(..., min_length=50)
    url: str = Field(default="")
    questions: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(_: FastAPI):
    maybe_sync_index(INDEX_DIR)
    yield


app = FastAPI(
    title="GroundedDoc Agent API",
    description="Conflict-aware document intelligence agent",
    version="0.1.0",
    lifespan=lifespan,
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


def get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        if not (INDEX_DIR / "ingestion_report.json").exists():
            IngestionPipeline().run()
        _pipeline = DocumentPipeline()
    return _pipeline


def _check_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    required = os.getenv("GROUNDED_REQUIRE_API_KEY", "false").lower() == "true"
    if not required:
        return
    expected = os.getenv("GROUNDED_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(
    request: QueryRequest,
    _: None = Depends(_check_api_key),
) -> dict[str, Any]:
    response = get_pipeline().run(request.query, variant=request.variant)
    return response.to_dict()


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
    _: None = Depends(_check_api_key),
) -> dict[str, Any]:
    questions = request.questions or None
    return analyze_policy(
        get_pipeline(),
        page_text=request.page_text,
        url=request.url,
        questions=questions,
    )


@app.post("/ingest")
def ingest(
    request: IngestRequest,
    _: None = Depends(_check_api_key),
) -> dict[str, Any]:
    if not request.rebuild:
        report_path = INDEX_DIR / "ingestion_report.json"
        if report_path.exists():
            return {"status": "skipped", "reason": "index already exists"}
    report = IngestionPipeline().run()
    global _pipeline
    _pipeline = DocumentPipeline()
    return {"status": "completed", **report}


@app.get("/conflicts")
def conflicts(_: None = Depends(_check_api_key)) -> dict[str, Any]:
    pipeline = get_pipeline().pipeline
    items = pipeline.claims_store.list_conflicts()
    return {
        "conflicts": [
            {
                "subject": item.subject,
                "description": item.description,
                "sources": [claim.source_label for claim in item.claims],
            }
            for item in items
        ]
    }


@app.post("/eval/predict")
def eval_predict(
    request: QueryRequest,
    _: None = Depends(_check_api_key),
) -> dict[str, Any]:
    return predict_for_eval({"query": request.query}, variant=request.variant)
