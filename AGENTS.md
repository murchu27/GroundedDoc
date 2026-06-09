# AGENTS.md

## Cursor Cloud specific instructions

GroundedDoc Agent is a **Python 3.10+** project (CI uses 3.12). There is no `package.json`, Docker Compose stack, or Makefile — local development is venv + pip.

### System prerequisite (one-time on fresh VMs)

Ubuntu images may lack `python3-venv`. Install before creating `.venv`:

```bash
sudo apt-get install -y python3.12-venv
```

### Activate the environment

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

The update script creates `.venv` and installs `".[dev]"` on startup; always activate before running commands.

### Core workflows

| Task | Command |
|------|---------|
| Install / refresh deps | `pip install ".[dev]"` (handled by update script) |
| Build indexes | `python -m grounded_doc_agent.ingestion.cli` |
| Lint | `ruff check .` (5 known pre-existing issues in repo; not CI-gated) |
| Tests | `pytest -q` (conftest can build index; faster if ingestion already ran) |
| Eval gate | `python -m grounded_doc_agent.eval.run_eval --variant full_pipeline` |
| API server | `uvicorn api.main:app --reload --port 8080` |
| ADK web UI (optional) | `GOOGLE_API_KEY=... adk web agents --port 8000` |

### Services

- **FastAPI on `:8080`** is the primary runnable service. Auto-builds indexes on first request if `data/index/` is missing.
- **No external DB** — indexes are JSON files under `data/index/`, claims in SQLite (`claims.db`), MLflow in `mlflow.db`.
- **`GOOGLE_API_KEY`** is optional; extractive synthesis works without it. Required only for Gemini-powered answers and the ADK web UI.
- **Chrome extension** (`extension/`) needs the API running and manual load in `chrome://extensions` — not practical in headless cloud VMs.

### Gotchas

- First ingest/query downloads `sentence-transformers` model `all-MiniLM-L6-v2` (~90MB); needs network and is slow once.
- `data/index/` is gitignored; run ingestion CLI after clone or let the API build on startup.
- CI lint step is import-health only (`python -c "from ... import DocumentPipeline"`), not `ruff check`.
- Long-running servers: use tmux (e.g. session `grounded-api-server`).

See `README.md` and `docs/DEVELOPER_GUIDE.md` for architecture and API reference.
