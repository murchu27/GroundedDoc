from __future__ import annotations

import re
import uuid

from grounded_doc_agent.models import ChildChunk, Claim, ClaimConflict, DocumentSection

RETENTION_PATTERN = re.compile(
    r"(?P<subject>(?:data retention|retention period|personal data retention|retention)[^.:\n]*)"
    r"[^.\n]{0,80}?(?P<value>\d+\s*(?:days|months|years))",
    re.IGNORECASE,
)
SIMPLE_RETENTION_PATTERN = re.compile(
    r"retention\s+is\s+(?P<value>\d+\s*(?:days|months|years))",
    re.IGNORECASE,
)
GENERIC_CLAIM_PATTERN = re.compile(
    r"(?P<subject>[A-Z][^.:\n]{5,80}?)\s+(?:is|are|must be|shall be|requires?)\s+"
    r"(?P<value>[^.\n]{3,80})\.",
    re.MULTILINE,
)
SUBJECT_ALIASES = {
    "data retention": "data_retention",
    "retention period": "data_retention",
    "personal data retention": "data_retention",
}


def normalize_subject(subject: str) -> str:
    cleaned = re.sub(r"\s+", " ", subject.strip().lower())
    for alias, canonical in SUBJECT_ALIASES.items():
        if alias in cleaned:
            return canonical
    return re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")


def extract_claims_from_section(section: DocumentSection) -> list[Claim]:
    claims: list[Claim] = []
    text = section.content
    patterns = [RETENTION_PATTERN, SIMPLE_RETENTION_PATTERN, GENERIC_CLAIM_PATTERN]
    seen: set[tuple[str, str, str]] = set()

    for pattern in patterns:
        for match in pattern.finditer(text):
            if pattern is SIMPLE_RETENTION_PATTERN:
                subject = "data_retention"
                value = match.group("value").strip().lower()
            else:
                subject = normalize_subject(match.group("subject"))
                value = match.group("value").strip().lower()
            key = (subject, value, section.doc_id)
            if key in seen:
                continue
            seen.add(key)
            chunk_id = f"{section.doc_id}:{section.section_path}"
            claims.append(
                Claim(
                    claim_id=str(uuid.uuid4()),
                    subject=subject,
                    value=value,
                    doc_id=section.doc_id,
                    chunk_id=chunk_id,
                    section_path=section.section_path,
                    source_label=f"{section.doc_id} §{section.section_path}",
                )
            )
    return claims


def extract_claims_from_child(child: ChildChunk) -> list[Claim]:
    section = DocumentSection(
        doc_id=child.doc_id,
        section_path=child.section_path,
        title=child.section_path,
        content=child.text,
        source_url=child.source_url,
        page=child.page,
    )
    claims = extract_claims_from_section(section)
    for claim in claims:
        claim.chunk_id = child.chunk_id
    return claims


def detect_conflicts(claims: list[Claim]) -> list[ClaimConflict]:
    grouped: dict[str, dict[tuple[str, str], Claim]] = {}
    for claim in claims:
        if claim.value in {"0 days", "0 months", "0 years"}:
            continue
        grouped.setdefault(claim.subject, {})
        grouped[claim.subject][(claim.doc_id, claim.value)] = claim

    conflicts: list[ClaimConflict] = []
    for subject, by_doc_value in grouped.items():
        values = {value for _, value in by_doc_value}
        doc_ids = {doc_id for doc_id, _ in by_doc_value}
        if len(values) > 1 and len(doc_ids) > 1:
            subject_claims = list(by_doc_value.values())
            description = "; ".join(
                f"{claim.source_label} states '{claim.value}'" for claim in subject_claims
            )
            conflicts.append(
                ClaimConflict(
                    subject=subject,
                    claims=subject_claims,
                    description=description,
                )
            )
    return conflicts
