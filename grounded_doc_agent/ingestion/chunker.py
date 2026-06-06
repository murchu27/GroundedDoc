from __future__ import annotations

import hashlib
import re

from grounded_doc_agent.config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from grounded_doc_agent.models import ChildChunk, DocumentSection, ParentChunk


def _make_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:16]


def summarize_section(title: str, content: str, max_chars: int = 500) -> str:
    normalized = re.sub(r"\s+", " ", content).strip()
    lead = f"{title}: {normalized}"
    if len(lead) <= max_chars:
        return lead
    return lead[: max_chars - 3].rstrip() + "..."


def split_into_child_chunks(section: DocumentSection) -> list[ChildChunk]:
    parent_id = _make_id(section.doc_id, section.section_path, "parent")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.content) if p.strip()]
    chunks: list[ChildChunk] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer.strip():
            return
        chunk_id = _make_id(section.doc_id, section.section_path, buffer[:80])
        chunks.append(
            ChildChunk(
                chunk_id=chunk_id,
                doc_id=section.doc_id,
                parent_id=parent_id,
                section_path=section.section_path,
                text=buffer.strip(),
                source_url=section.source_url,
                page=section.page,
            )
        )
        buffer = ""

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= CHUNK_SIZE:
            buffer = candidate
            continue
        flush_buffer()
        if len(paragraph) <= CHUNK_SIZE:
            buffer = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + CHUNK_SIZE, len(paragraph))
            piece = paragraph[start:end]
            chunk_id = _make_id(section.doc_id, section.section_path, piece[:80], str(start))
            chunks.append(
                ChildChunk(
                    chunk_id=chunk_id,
                    doc_id=section.doc_id,
                    parent_id=parent_id,
                    section_path=section.section_path,
                    text=piece.strip(),
                    source_url=section.source_url,
                    page=section.page,
                )
            )
            if end == len(paragraph):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    flush_buffer()
    return chunks


def build_hierarchical_chunks(section: DocumentSection) -> tuple[ParentChunk, list[ChildChunk]]:
    children = split_into_child_chunks(section)
    parent_id = _make_id(section.doc_id, section.section_path, "parent")
    summary_source = section.content if section.content else section.title
    parent = ParentChunk(
        chunk_id=parent_id,
        doc_id=section.doc_id,
        section_path=section.section_path,
        title=section.title,
        summary=summarize_section(section.title, summary_source),
        child_ids=[child.chunk_id for child in children],
        source_url=section.source_url,
    )
    return parent, children
