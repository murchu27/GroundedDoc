from __future__ import annotations

from mlflow.genai.scorers import scorer


REFUSAL_HINT = "don't have sufficient evidence"


@scorer
def retrieval_recall(outputs, expectations):
    expected_docs = set(expectations.get("expected_doc_ids", []))
    if not expected_docs:
        return 1.0 if not outputs.get("retrieved_chunk_ids") else 0.0
    retrieved_docs = _extract_doc_ids(outputs)
    overlap = expected_docs.intersection(retrieved_docs)
    return len(overlap) / len(expected_docs)


@scorer
def citation_fidelity(outputs, expectations):
    if expectations.get("should_refuse"):
        return 1.0
    citations = outputs.get("citations", [])
    retrieved = set(outputs.get("retrieved_chunk_ids", []))
    if not citations:
        return 0.0
    valid = sum(1 for item in citations if item.get("chunk_id") in retrieved)
    return valid / len(citations)


@scorer
def retrieval_strategy_match(outputs, expectations):
    expected = expectations.get("expected_retrieval_strategy")
    actual = outputs.get("retrieval_strategy")
    if expected in {None, "none"}:
        return 1.0 if actual in {None, "none"} else 0.0
    if actual == expected:
        return 1.0
    compatible = {
        "claims": {"claims", "multi_hop"},
        "multi_hop": {"multi_hop", "claims", "vector"},
        "vector": {"vector", "bm25", "multi_hop"},
        "bm25": {"bm25", "vector"},
        "parent": {"parent", "vector"},
    }
    return 1.0 if actual in compatible.get(expected, {expected}) else 0.0


@scorer
def refusal_correctness(outputs, expectations):
    should_refuse = expectations.get("should_refuse", False)
    refused = outputs.get("refused", False) or REFUSAL_HINT in outputs.get("answer", "").lower()
    return 1.0 if refused == should_refuse else 0.0


@scorer
def conflict_surfaced(outputs, expectations):
    if not expectations.get("has_conflict"):
        return 1.0
    answer = outputs.get("answer", "").lower()
    conflicts = outputs.get("conflicts", [])
    if conflicts:
        return 1.0
    keywords = ("conflict", "disagree", "differs", "while", "whereas")
    return 1.0 if any(keyword in answer for keyword in keywords) else 0.0


def _extract_doc_ids(outputs: dict) -> set[str]:
    docs = set()
    for chunk_id in outputs.get("retrieved_chunk_ids", []):
        if ":" in chunk_id:
            docs.add(chunk_id.split(":", 1)[0])
    for citation in outputs.get("citations", []):
        if citation.get("doc_id"):
            docs.add(citation["doc_id"])
    return docs
