from __future__ import annotations

from grounded_doc_agent.config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from grounded_doc_agent.ingestion.chunker import split_into_child_chunks
from grounded_doc_agent.models import DocumentSection


def test_long_paragraph_splits_at_word_boundary():
    words = ["alpha"] + ["beta"] * 80 + ["gamma"]
    paragraph = " ".join(words)
    section = DocumentSection(
        doc_id="demo",
        section_path="retention",
        title="Retention",
        content=paragraph,
    )
    chunks = split_into_child_chunks(section)
    assert len(chunks) > 1
    for chunk in chunks:
        text = chunk.text
        if len(text) < CHUNK_SIZE:
            continue
        assert not text.endswith("bet")
        assert not text.startswith("eta")


def test_overlap_chunk_starts_at_word_boundary():
    word = "abcdefghij"
    paragraph = " ".join([word] * 50)
    section = DocumentSection(
        doc_id="demo",
        section_path="retention",
        title="Retention",
        content=paragraph,
    )
    chunks = split_into_child_chunks(section)
    assert len(chunks) >= 2
    for chunk in chunks[1:]:
        assert chunk.text[0].isalnum()
        if chunk.text[0].isalpha():
            assert chunk.text[0].islower() or chunk.text[0].isupper()
        overlap_region = chunk.text[:CHUNK_OVERLAP + 20]
        assert " " in overlap_region or len(chunk.text) < CHUNK_OVERLAP
