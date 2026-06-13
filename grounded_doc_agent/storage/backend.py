from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from grounded_doc_agent.config.settings import GCS_BUCKET, INDEX_DIR, STORAGE_BACKEND


class StorageBackend(ABC):
    @abstractmethod
    def sync_index_from_remote(self, index_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def persist_policy(self, doc_hash: str, text: str, metadata: dict) -> None:
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    def sync_index_from_remote(self, index_dir: Path) -> None:
        return

    def persist_policy(self, doc_hash: str, text: str, metadata: dict) -> None:
        policy_dir = INDEX_DIR.parent / "policies" / doc_hash
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "source.md").write_text(text, encoding="utf-8")
        (policy_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class GcsStorageBackend(StorageBackend):
    def __init__(self, bucket_name: str) -> None:
        from google.cloud import storage

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def sync_index_from_remote(self, index_dir: Path) -> None:
        prefix = "index/base/"
        index_dir.mkdir(parents=True, exist_ok=True)
        for blob in self._client.list_blobs(self._bucket, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            relative = blob.name.removeprefix(prefix)
            target = index_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(target)

    def persist_policy(self, doc_hash: str, text: str, metadata: dict) -> None:
        source_blob = self._bucket.blob(f"corpus/policies/{doc_hash}/source.md")
        meta_blob = self._bucket.blob(f"corpus/policies/{doc_hash}/meta.json")
        source_blob.upload_from_string(text, content_type="text/markdown")
        meta_blob.upload_from_string(json.dumps(metadata, indent=2), content_type="application/json")


def get_storage_backend() -> StorageBackend:
    if STORAGE_BACKEND == "gcs":
        if not GCS_BUCKET:
            raise ValueError("GROUNDED_GCS_BUCKET is required when GROUNDED_STORAGE_BACKEND=gcs")
        return GcsStorageBackend(GCS_BUCKET)
    return LocalStorageBackend()


def maybe_sync_index(index_dir: Path = INDEX_DIR) -> None:
    if os.getenv("GROUNDED_SKIP_STORAGE_SYNC", "false").lower() == "true":
        return
    backend = get_storage_backend()
    backend.sync_index_from_remote(index_dir)


def persist_policy_snapshot(doc_hash: str, text: str, *, url: str) -> None:
    backend = get_storage_backend()
    backend.persist_policy(
        doc_hash,
        text,
        {
            "url": url,
            "ingested_at": datetime.now(UTC).isoformat(),
        },
    )
