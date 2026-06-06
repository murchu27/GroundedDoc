# Architecture Notes

## Ingestion

1. Parse markdown corpus into sections
2. Build parent summaries + child chunks (hierarchical index)
3. Extract atomic claims and detect cross-document conflicts (SQLite claims store)
4. Embed and persist vectors to `data/index/vector_store.json`
5. Build BM25 indexes for keyword retrieval

## Query Pipeline

1. **Router** — out-of-scope detection
2. **QueryPlanner** — classify query + decompose sub-queries
3. **AdaptiveRetriever** — vector / BM25 / claims / multi-hop routing
4. **Synthesizer** — extractive answer with citations (optional Gemini)
5. **Verifier** — citation check + re-retrieval loop

## Evaluation

- Golden dataset: `grounded_doc_agent/eval/golden_dataset.json`
- MLflow scorers in `grounded_doc_agent/eval/scorers.py`
- CI gate via `.github/workflows/ci.yml`

## Deployment

- API: `api/main.py` (FastAPI)
- Container: `infra/Dockerfile`
- Cloud Run script: `scripts/deploy_cloud_run.sh`
