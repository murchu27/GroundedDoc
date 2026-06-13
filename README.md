![GitHub License](https://img.shields.io/github/license/murchu27/GroundedDoc)

# GroundedDoc Agent

GroundedDoc answers questions over a collection of documents and backs every
answer with citations you can check. It is built for situations where a wrong or
unsupported answer is worse than no answer: it surfaces disagreements between
sources instead of papering over them, and it refuses to answer when the corpus
does not contain the evidence. The included privacy-policy use case puts this to
work, reviewing arbitrary web pages against regulations like GDPR and PIPEDA.

In short, GroundedDoc is about *trustworthy* document answers. Three properties
follow from that goal and shape the whole system:

- **Grounding.** Every factual statement maps to a cited source passage; nothing
  is asserted without evidence.
- **Conflict awareness.** When two sources disagree (for example, on a data
  retention period), the disagreement is reported rather than silently resolved.
- **Honest refusal.** When the corpus cannot support an answer, the agent says so
  instead of guessing.

## Architecture

A document corpus is ingested into hierarchical indexes (parent summaries, child
chunks, a keyword index, and an extracted-claims store). At query time a router
decides whether the question is in scope, a planner decomposes it, an adaptive
retriever pulls evidence from the most relevant indexes, a synthesizer drafts a
cited answer, and a verifier checks every citation before the answer is returned.
Every run is traced in MLflow so quality can be measured and gated in CI.

See [docs/architecture.md](docs/architecture.md) for the component breakdown and
diagrams.

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

## Optional: ADK web UI

For interactive exploration there is an optional Agent Development Kit (ADK)
front end in `agents/grounded_doc/agent.py`. It is a dev/demo surface only -- the
product itself is the Python library (`grounded_doc_agent/`) and the API below,
neither of which requires ADK.

```bash
# Prerequisites: pip install ., indexes built (ingestion CLI), Google API key set
export GOOGLE_API_KEY=your-key
adk web agents --port 8000
# Open http://localhost:8000 and select "grounded_doc"
```

## API / Cloud Run

Run the FastAPI service locally:

```bash
uvicorn api.main:app --reload --port 8080
```

The service exposes endpoints for querying the corpus, analyzing a page against
regulations (used by the browser extension), rebuilding indexes, and listing
detected conflicts. See [docs/api.md](docs/api.md) for the full endpoint
reference, request/response shapes, auth, and `curl` examples.

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

Each query in the golden dataset is scored on the properties the project cares
about, so quality regressions show up as numbers rather than vibes:

- **`retrieval_recall`** -- fraction of the expected source documents that were
  actually retrieved. Measures whether the right evidence was found.
- **`citation_fidelity`** -- fraction of an answer's citations that point to
  passages that were genuinely retrieved. Catches fabricated or mismatched
  citations.
- **`retrieval_strategy_match`** -- whether the planner chose the expected
  retrieval strategy (or a compatible one) for a given query type.
- **`refusal_correctness`** -- whether the agent refused exactly when it should
  have: refusing unanswerable questions and answering answerable ones.
- **`conflict_surfaced`** -- whether known cross-document conflicts are explicitly
  reported in the answer instead of being hidden.

CI fails if core metrics drop below thresholds (0.85 recall, 0.90 citation
fidelity, 1.0 refusal correctness).

## Why this runs for ~$0

GroundedDoc deliberately avoids paid, managed services so it can run on a laptop
or a free cloud tier with no recurring bill:

- **Embeddings run locally** with open `sentence-transformers` models, so there
  is no embedding API to pay for.
- **The vector store is a plain JSON file on disk**, so no managed vector
  database is needed.
- **Answer synthesis is optional.** Gemini Flash can be used if you provide an
  API key, but an extractive fallback produces cited answers with no key and no
  cost.
- **Experiment tracking uses a local SQLite file** (`mlflow.db`) rather than a
  hosted MLflow server.
- **Hosting fits free tiers.** Cloud Run plus GCS free allowances comfortably
  cover portfolio-level traffic.

## Sample corpus

A small example corpus ships with the repo so you can build indexes and run
queries immediately, and so the evaluation suite has known-good expected answers.
It lives under `data/corpus/`:

- `gdpr_summary.md`, `pipeda_summary.md` -- plain-language summaries of two
  privacy regulations.
- `acme_privacy_policy.md` -- a fictional company privacy policy to analyze
  against those regulations.
- `regulatory/` -- structured GDPR requirement "cards" used for claim-level
  retrieval.
- `fastapi_current.md`, `fastapi_migration.md` -- a non-privacy topic, included
  to show retrieval works across unrelated subject matter.

The corpus intentionally contains conflicting data-retention statements (for
example, between the regulations and the sample policy) so the conflict-detection
features have something real to surface.

## Optional GCS Persistence

For Cloud Run deployments with dynamic policy ingest:

```bash
pip install '.[gcp]'
export GROUNDED_STORAGE_BACKEND=gcs
export GROUNDED_GCS_BUCKET=your-bucket
```

On startup the API syncs `index/base/` from GCS. Ingested policies are stored under `corpus/policies/{hash}/`.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
