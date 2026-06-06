from grounded_doc_agent.eval.scorers import (
    citation_fidelity,
    conflict_surfaced,
    refusal_correctness,
    retrieval_recall,
    retrieval_strategy_match,
)


def test_refusal_scorer():
    outputs = {"answer": "I don't have sufficient evidence", "refused": True}
    expectations = {"should_refuse": True}
    assert refusal_correctness(outputs=outputs, expectations=expectations) == 1.0


def test_retrieval_recall_scorer():
    outputs = {
        "citations": [{"doc_id": "gdpr_summary", "chunk_id": "abc"}],
        "retrieved_chunk_ids": ["abc"],
    }
    expectations = {"expected_doc_ids": ["gdpr_summary"]}
    assert retrieval_recall(outputs=outputs, expectations=expectations) == 1.0


def test_strategy_match_scorer():
    outputs = {"retrieval_strategy": "claims"}
    expectations = {"expected_retrieval_strategy": "claims"}
    assert retrieval_strategy_match(outputs=outputs, expectations=expectations) == 1.0


def test_citation_fidelity_scorer():
    outputs = {
        "citations": [{"chunk_id": "abc", "doc_id": "gdpr_summary"}],
        "retrieved_chunk_ids": ["abc"],
    }
    expectations = {"should_refuse": False}
    assert citation_fidelity(outputs=outputs, expectations=expectations) == 1.0


def test_conflict_scorer():
    outputs = {"answer": "Sources conflict on retention", "conflicts": []}
    expectations = {"has_conflict": True}
    assert conflict_surfaced(outputs=outputs, expectations=expectations) == 1.0
