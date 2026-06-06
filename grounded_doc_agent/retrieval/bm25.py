from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from grounded_doc_agent.models import ChildChunk, ParentChunk, RetrievedContext


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[ChildChunk | ParentChunk] = []
        self._bm25: BM25Okapi | None = None
        self._level = "child"

    def build(self, chunks: list[ChildChunk | ParentChunk], *, level: str = "child") -> None:
        self._chunks = chunks
        self._level = level
        tokenized = [self._tokenize(self._chunk_text(chunk)) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 5) -> list[RetrievedContext]:
        if not self._bm25 or not self._chunks:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            zip(self._chunks, scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        results: list[RetrievedContext] = []
        for chunk, score in ranked:
            if score <= 0:
                continue
            results.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=self._chunk_text(chunk),
                    score=float(score),
                    section_path=getattr(chunk, "section_path", ""),
                    source_url=getattr(chunk, "source_url", ""),
                    chunk_level=self._level,
                )
            )
        return results

    def save(self, path: Path) -> None:
        payload = {
            "level": self._level,
            "chunks": [self._serialize_chunk(chunk) for chunk in self._chunks],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._level = payload["level"]
        chunks: list[ChildChunk | ParentChunk] = []
        for item in payload["chunks"]:
            if item["kind"] == "parent":
                chunks.append(ParentChunk(**item["data"]))
            else:
                chunks.append(ChildChunk(**item["data"]))
        self.build(chunks, level=self._level)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in text.split() if token.strip()]

    @staticmethod
    def _chunk_text(chunk: ChildChunk | ParentChunk) -> str:
        if isinstance(chunk, ParentChunk):
            return chunk.summary
        return chunk.text

    @staticmethod
    def _serialize_chunk(chunk: ChildChunk | ParentChunk) -> dict:
        if isinstance(chunk, ParentChunk):
            return {"kind": "parent", "data": chunk.__dict__}
        return {"kind": "child", "data": chunk.__dict__}
