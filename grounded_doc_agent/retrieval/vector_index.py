from __future__ import annotations

import json
from pathlib import Path

from grounded_doc_agent.config.settings import EMBEDDING_MODEL, TOP_K_CHILD, TOP_K_PARENT
from grounded_doc_agent.models import ChildChunk, ParentChunk, RetrievedContext


class _SimpleEmbedder:
    _model = None

    @classmethod
    def embed(cls, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(EMBEDDING_MODEL)
        vectors = cls._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class VectorIndex:
    """Lightweight on-disk vector index (Chroma-free for portability)."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.index_dir / "vector_store.json"
        self._entries: list[dict] = []
        self._child_map: dict[str, ChildChunk] = {}
        self._parent_map: dict[str, ParentChunk] = {}
        if self.store_path.exists():
            self._load_store()

    def reset(self) -> None:
        self._entries = []
        self._child_map.clear()
        self._parent_map.clear()
        if self.store_path.exists():
            self.store_path.unlink()

    def add_children(self, children: list[ChildChunk]) -> None:
        texts = [child.text for child in children]
        vectors = _SimpleEmbedder.embed(texts)
        for child, vector in zip(children, vectors, strict=False):
            self._child_map[child.chunk_id] = child
            self._entries.append(
                {
                    "chunk_id": child.chunk_id,
                    "doc_id": child.doc_id,
                    "section_path": child.section_path,
                    "source_url": child.source_url,
                    "text": child.text,
                    "level": "child",
                    "vector": vector,
                }
            )

    def remove_doc(self, doc_id: str) -> None:
        self._entries = [entry for entry in self._entries if entry["doc_id"] != doc_id]
        self._child_map = {
            chunk_id: child
            for chunk_id, child in self._child_map.items()
            if child.doc_id != doc_id
        }
        self._parent_map = {
            chunk_id: parent
            for chunk_id, parent in self._parent_map.items()
            if parent.doc_id != doc_id
        }
        self._persist_store()

    def persist(self) -> None:
        self._persist_store()

    def add_parents(self, parents: list[ParentChunk]) -> None:
        texts = [parent.summary for parent in parents]
        vectors = _SimpleEmbedder.embed(texts)
        for parent, vector in zip(parents, vectors, strict=False):
            self._parent_map[parent.chunk_id] = parent
            self._entries.append(
                {
                    "chunk_id": parent.chunk_id,
                    "doc_id": parent.doc_id,
                    "section_path": parent.section_path,
                    "source_url": parent.source_url,
                    "text": parent.summary,
                    "level": "parent",
                    "vector": vector,
                }
            )
        self._persist_store()

    def search_children(self, query: str, top_k: int = TOP_K_CHILD) -> list[RetrievedContext]:
        return self._search(query, top_k=top_k, level="child")

    def search_parents(self, query: str, top_k: int = TOP_K_PARENT) -> list[RetrievedContext]:
        return self._search(query, top_k=top_k, level="parent")

    def get_child(self, chunk_id: str) -> ChildChunk | None:
        return self._child_map.get(chunk_id)

    def get_chunk_text(self, chunk_id: str) -> str:
        child = self._child_map.get(chunk_id)
        if child:
            return child.text
        parent = self._parent_map.get(chunk_id)
        if parent:
            return parent.summary
        for entry in self._entries:
            if entry["chunk_id"] == chunk_id:
                return entry["text"]
        return ""

    def save_metadata(self, path: Path) -> None:
        payload = {
            "children": [child.__dict__ for child in self._child_map.values()],
            "parents": [parent.__dict__ for parent in self._parent_map.values()],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load_metadata(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._child_map = {
            item["chunk_id"]: ChildChunk(**item) for item in payload.get("children", [])
        }
        self._parent_map = {
            item["chunk_id"]: ParentChunk(**item) for item in payload.get("parents", [])
        }

    def _search(self, query: str, *, top_k: int, level: str) -> list[RetrievedContext]:
        if not self._entries:
            self._load_store()
        candidates = [entry for entry in self._entries if entry["level"] == level]
        if not candidates:
            return []
        query_vector = _SimpleEmbedder.embed([query])[0]
        scored = [
            (entry, _cosine(query_vector, entry["vector"]))
            for entry in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[RetrievedContext] = []
        for entry, score in scored[:top_k]:
            if score <= 0:
                continue
            results.append(
                RetrievedContext(
                    chunk_id=entry["chunk_id"],
                    doc_id=entry["doc_id"],
                    text=entry["text"],
                    score=float(score),
                    section_path=entry["section_path"],
                    source_url=entry["source_url"],
                    chunk_level=level,
                )
            )
        return results

    def _persist_store(self) -> None:
        self.store_path.write_text(json.dumps(self._entries), encoding="utf-8")

    def _load_store(self) -> None:
        if not self.store_path.exists():
            return
        self._entries = json.loads(self.store_path.read_text(encoding="utf-8"))
