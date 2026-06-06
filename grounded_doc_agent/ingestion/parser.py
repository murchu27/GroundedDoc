from __future__ import annotations

import re
from pathlib import Path

from grounded_doc_agent.models import DocumentSection


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def parse_markdown_file(path: Path) -> list[DocumentSection]:
    text = path.read_text(encoding="utf-8")
    doc_id = path.stem
    source_url = f"file://{path.name}"
    return parse_markdown_text(text, doc_id=doc_id, source_url=source_url)


def parse_markdown_text(
    text: str,
    *,
    doc_id: str,
    source_url: str = "",
) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        sections.append(
            DocumentSection(
                doc_id=doc_id,
                section_path="root",
                title=doc_id,
                content=text.strip(),
                source_url=source_url,
            )
        )
        return sections

    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append(
            DocumentSection(
                doc_id=doc_id,
                section_path="intro",
                title="Introduction",
                content=prefix,
                source_url=source_url,
            )
        )

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[start:end].strip()
        if not body:
            continue
        section_path = heading.lower().replace(" ", "-")
        sections.append(
            DocumentSection(
                doc_id=doc_id,
                section_path=section_path,
                title=heading,
                content=body,
                source_url=source_url,
            )
        )
    return sections


def load_corpus(corpus_dir: Path) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    for path in sorted(corpus_dir.glob("**/*")):
        if path.suffix.lower() in {".md", ".txt"}:
            sections.extend(parse_markdown_file(path))
    return sections
