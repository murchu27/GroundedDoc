# API Reference

The GroundedDoc FastAPI service is defined in [api/main.py](../api/main.py). Run
it locally with:

```bash
uvicorn api.main:app --reload --port 8080
```

All examples below assume `http://localhost:8080`.

## Authentication

Authentication is off by default. Set `GROUNDED_REQUIRE_API_KEY=true` to require
a key, and set `GROUNDED_API_KEY` to the expected value. When enabled, every
endpoint except `GET /health` requires an `X-API-Key` header:

```bash
curl -H 'X-API-Key: your-key' http://localhost:8080/conflicts
```

A missing or incorrect key returns `401 Invalid API key`.

## Endpoints

### `GET /health`

Liveness check. Always unauthenticated.

```bash
curl http://localhost:8080/health
```

Response:

```json
{ "status": "ok" }
```

### `POST /query`

Ask a question against the indexed corpus and get a cited, verified answer.

Request body:

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `query` | string | yes | -- | The question (minimum 3 characters). |
| `variant` | string | no | `full_pipeline` | Pipeline variant to run. |

```bash
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Compare GDPR vs PIPEDA retention"}'
```

Returns the serialized response (answer text, citations, retrieval metadata, and
any surfaced conflicts).

### `POST /analyze`

Analyze a single page of text (for example a privacy policy) against the corpus.
This powers the browser extension. Results are informational only, not legal
advice.

Request body:

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `page_text` | string | yes | -- | The page text to analyze (minimum 50 characters). |
| `url` | string | no | `""` | Source URL of the page, for labeling. |
| `questions` | string[] | no | `[]` | Optional specific questions; defaults are used when empty. |

```bash
curl -X POST http://localhost:8080/analyze \
  -H 'Content-Type: application/json' \
  -d '{"page_text":"## Data Retention\nWe retain data for 90 days.","url":"https://example.com/privacy"}'
```

Returns cited findings comparing the page against indexed regulations.

### `POST /ingest`

Rebuild the document indexes from the corpus.

Request body:

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `rebuild` | boolean | no | `true` | When `false`, skips work if an index already exists. |

```bash
curl -X POST http://localhost:8080/ingest \
  -H 'Content-Type: application/json' \
  -d '{"rebuild":true}'
```

Returns `{"status":"completed", ...}` with the ingestion report, or
`{"status":"skipped"}` when `rebuild` is `false` and an index is already present.

### `GET /conflicts`

List cross-document conflicts detected during ingestion.

```bash
curl http://localhost:8080/conflicts
```

Response shape:

```json
{
  "conflicts": [
    {
      "subject": "data retention",
      "description": "...",
      "sources": ["gdpr_summary.md", "acme_privacy_policy.md"]
    }
  ]
}
```

### `POST /eval/predict`

Prediction entry point used by the MLflow evaluation harness. Takes the same body
as `/query` and returns a prediction payload shaped for the evaluation scorers.

```bash
curl -X POST http://localhost:8080/eval/predict \
  -H 'Content-Type: application/json' \
  -d '{"query":"How long can data be retained under GDPR?"}'
```
