from __future__ import annotations

import hashlib
import re


def policy_doc_id(url: str, *, prefix: str = "policy") -> str:
    normalized = url.strip().lower() or "unknown"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def infer_topic(question: str) -> str:
    lowered = question.lower()
    if "retention" in lowered:
        return "data_retention"
    if "third-party" in lowered or "processor" in lowered or "sharing" in lowered:
        return "third_party_sharing"
    if "rights" in lowered or "contact" in lowered:
        return "data_subject_rights"
    if "lawful basis" in lowered or "legal basis" in lowered:
        return "lawful_basis"
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug[:48] or "general"
