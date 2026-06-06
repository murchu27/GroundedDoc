FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY grounded_doc_agent ./grounded_doc_agent
COPY agents ./agents
COPY api ./api
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir .

ENV GROUNDED_DATA_DIR=/app/data
ENV GROUNDED_CORPUS_DIR=/app/data/corpus
ENV GROUNDED_INDEX_DIR=/app/data/index
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV GROUNDED_REQUIRE_API_KEY=false
ENV PORT=8080

RUN python -m grounded_doc_agent.ingestion.cli

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
