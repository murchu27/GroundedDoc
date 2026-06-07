# GroundedDoc Agent

Agentic document intelligence with hierarchical indexing, query decomposition, conflict detection, and MLflow-evaluated grounding.

## Highlights

- **Ingestion engineering:** structure-aware parsing, parent/child indexes, claim extraction, conflict detection
- **Agentic query layer:** query planner, adaptive retrieval (vector/BM25/claims/multi-hop), citation verifier loop
- **Evaluation:** MLflow `genai.evaluate` with custom scorers and A/B experiment runs
- **Deployment:** FastAPI service + Cloud Run Dockerfile (~$0/month on GCP free tiers)
- **Privacy policy extension:** browser extension + `POST /analyze` for grounded GDPR review of arbitrary pages

## Architecture

```text
Corpus -> IngestionPipeline -> {parent index, child index, BM25, claims DB}
Query -> Router -> QueryPlanner -> AdaptiveRetriever -> Synthesizer -> Verifier -> Response
All runs traced in MLflow with custom scorers for CI gating
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install .

# Build indexes from sample corpus
python -m grounded_doc_agent.ingestion.cli

# Run a query locally
python -c "from grounded_doc_agent.agents.pipeline import DocumentPipeline; print(DocumentPipeline().run('Compare GDPR vs PIPEDA retention').answer)"

# Run MLflow evaluation
python -m grounded_doc_agent.eval.run_eval --variant full_pipeline

# Compare baseline vs full pipeline
python -m grounded_doc_agent.eval.run_eval --compare-ab
```

## ADK Agent

The ADK entrypoint lives in `agents/grounded_doc/agent.py` and exposes a sequential multi-agent workflow. The `grounded_doc_agent/` directory is the Python library package; `agents/grounded_doc/` is the ADK web entrypoint.

```bash
# Prerequisites: pip install ., indexes built (ingestion CLI), API key set
export GOOGLE_API_KEY=your-key

adk web agents --port 8000
# Open http://localhost:8000 and select "grounded_doc" (not "grounded_doc_agent")
```

## API / Cloud Run

```bash
uvicorn api.main:app --reload --port 8080
```

Analyze a privacy policy page (used by the browser extension):

```bash
curl -X POST http://localhost:8080/analyze \
  -H 'Content-Type: application/json' \
  -d '{"page_text":"## Data Retention\nWe retain data for 90 days.","url":"https://example.com/privacy"}'
```

## Browser Extension

Load `extension/` as an unpacked extension in Chrome:

1. Build indexes (`python -m grounded_doc_agent.ingestion.cli`)
2. Start the API (`uvicorn api.main:app --reload --port 8080`)
3. Open `chrome://extensions`, enable Developer mode, load unpacked, select `extension/`
4. Open a privacy policy page and click the extension icon

The popup extracts page text, calls `POST /analyze`, and renders cited findings. Results are informational only, not legal advice.

Docker:

```bash
docker build -f infra/Dockerfile -t grounded-doc-agent .
docker run -p 8080:8080 grounded-doc-agent
```

Deploy script (requires gcloud auth + secrets):

```bash
GCP_PROJECT_ID=your-project ./scripts/deploy_cloud_run.sh
```

## MLflow Scorers

- `retrieval_recall`
- `citation_fidelity`
- `retrieval_strategy_match`
- `refusal_correctness`
- `conflict_surfaced`

CI fails if core metrics drop below thresholds (0.85 recall, 0.90 citation fidelity, 1.0 refusal correctness).

## Cost Notes

- Local/open embeddings via `sentence-transformers`
- Lightweight JSON vector store on disk (no managed vector DB required)
- Gemini Flash optional for synthesis (extractive fallback works without API key)
- MLflow tracking via local SQLite (`mlflow.db`)
- Cloud Run + GCS free tiers suitable for portfolio traffic

## Sample Corpus

- `data/corpus/gdpr_summary.md`
- `data/corpus/pipeda_summary.md`
- `data/corpus/acme_privacy_policy.md`
- `data/corpus/regulatory/` (structured GDPR requirement cards)
- `data/corpus/fastapi_current.md`
- `data/corpus/fastapi_migration.md`

Intentional retention conflicts are included for demo/eval purposes.

## Optional GCS Persistence

For Cloud Run deployments with dynamic policy ingest:

```bash
pip install '.[gcp]'
export GROUNDED_STORAGE_BACKEND=gcs
export GROUNDED_GCS_BUCKET=your-bucket
```

On startup the API syncs `index/base/` from GCS. Ingested policies are stored under `corpus/policies/{hash}/`.
