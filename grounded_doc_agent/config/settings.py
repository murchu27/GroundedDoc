from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("GROUNDED_DATA_DIR", PROJECT_ROOT / "data"))
CORPUS_DIR = Path(os.getenv("GROUNDED_CORPUS_DIR", DATA_DIR / "corpus"))
INDEX_DIR = Path(os.getenv("GROUNDED_INDEX_DIR", DATA_DIR / "index"))
CLAIMS_DB_PATH = Path(os.getenv("GROUNDED_CLAIMS_DB", INDEX_DIR / "claims.db"))

EMBEDDING_MODEL = os.getenv("GROUNDED_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GROUNDED_GEMINI_MODEL", "gemini-2.5-flash")
TOP_K_CHILD = int(os.getenv("GROUNDED_TOP_K_CHILD", "5"))
TOP_K_PARENT = int(os.getenv("GROUNDED_TOP_K_PARENT", "3"))
CHUNK_SIZE = int(os.getenv("GROUNDED_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("GROUNDED_CHUNK_OVERLAP", "80"))

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "grounded-doc-agent")
