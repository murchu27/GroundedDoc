from __future__ import annotations

from fastapi.testclient import TestClient

from grounded_doc_agent.agents.analyze import (
    DEFAULT_COMPLIANCE_QUESTIONS,
    _conflicts_for_analyze,
    _context_from_answer_citations,
    _finding_status,
    _format_citation_snippet,
    _infer_status_from_summary,
    _policy_context_score,
    _retrieve_dual_source,
    analyze_policy,
    build_finding,
    run_analyze_query,
)
from grounded_doc_agent.agents.synthesizer import clean_analyze_answer
from grounded_doc_agent.agents.pipeline import DocumentPipeline
from grounded_doc_agent.ingestion.overlay import infer_topic, policy_doc_id
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline
from grounded_doc_agent.models import AgentResponse, RetrievedContext


SAMPLE_POLICY = """# Sample Privacy Policy

## Data Retention

We retain personal data for free tier users for 90 days after account deactivation.

## Third-Party Sharing

We may share personal data with analytics vendors under standard contractual clauses.

## Data Subject Rights

You may request access or deletion by contacting privacy@example.com.
"""


def test_policy_doc_id_is_stable():
    assert policy_doc_id("https://example.com/privacy") == policy_doc_id(
        "https://example.com/privacy"
    )
    assert policy_doc_id("https://example.com/privacy").startswith("policy:")


def test_infer_topic_maps_align_retention_question():
    assert (
        infer_topic("Does this policy's data retention approach align with GDPR retention guidance?")
        == "data_retention"
    )


def test_default_questions_use_align_phrasing():
    assert all("align" in question.lower() or "requires" in question.lower() for question in DEFAULT_COMPLIANCE_QUESTIONS)


def test_finding_status_needs_review_without_regulation_citation():
    response = AgentResponse(
        query="q",
        answer="answer",
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    assert (
        _finding_status(
            response,
            has_policy_citation=True,
            has_regulation_citation=False,
            policy_relevant_conflicts=[],
            summary="answer",
        )
        == "needs_review"
    )


def test_finding_status_aligned_requires_both_citations():
    response = AgentResponse(
        query="q",
        answer="answer",
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    assert (
        _finding_status(
            response,
            has_policy_citation=True,
            has_regulation_citation=True,
            policy_relevant_conflicts=[],
            summary="This policy aligns with GDPR retention guidance.",
        )
        == "aligned"
    )


def test_finding_status_potential_gap_only_with_policy_relevant_conflicts():
    response = AgentResponse(
        query="q",
        answer="answer",
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[{"subject": "data_retention", "description": "demo", "sources": []}],
        refused=False,
    )
    assert (
        _finding_status(
            response,
            has_policy_citation=True,
            has_regulation_citation=True,
            policy_relevant_conflicts=response.conflicts,
            summary="answer",
        )
        == "potential_gap"
    )


def test_infer_status_from_summary_detects_gaps():
    summary = (
        "The provided policy context does not explicitly state that retention "
        "aligns with GDPR guidance."
    )
    assert _infer_status_from_summary(summary) == "needs_review"


def test_finding_status_uses_summary_gap_language_over_aligned_citations():
    response = AgentResponse(
        query="q",
        answer=(
            "The policy does not explicitly explain data subject rights "
            "[policy:abc §data-subject-rights]. "
            "GDPR requires contact details [gdpr_art13_transparency §requirement]."
        ),
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    assert (
        _finding_status(
            response,
            has_policy_citation=True,
            has_regulation_citation=True,
            policy_relevant_conflicts=[],
            summary=response.answer,
        )
        == "needs_review"
    )


def test_format_citation_snippet_trims_mid_word_start():
    snippet = _format_citation_snippet("ocess to make sure we delete data promptly.")
    assert snippet.startswith("to make sure")
    assert not snippet.startswith("ocess")


def test_format_citation_snippet_truncates_at_word_boundary():
    long_text = "word " * 200
    snippet = _format_citation_snippet(long_text, max_len=100)
    assert snippet.endswith("...")
    assert " " not in snippet[-5:]


def test_format_citation_snippet_drops_trailing_incomplete_sentence():
    text = (
        "We try to ensure that services protect information. "
        "Because of this, there may be delays between deletion and backup cleanup. "
        "You can read more about Google's data retention"
    )
    snippet = _format_citation_snippet(text)
    assert snippet.endswith(".")
    assert "data retention" not in snippet


def test_format_citation_snippet_prefers_sentence_boundary_when_truncating():
    sentences = ["This is sentence one."] + ["This is another sentence."] * 40
    text = " ".join(sentences)
    snippet = _format_citation_snippet(text, max_len=120)
    assert snippet.endswith(".")
    assert "another sentence." in snippet


def test_policy_context_score_deprioritizes_personal_info_for_rights():
    personal_info = RetrievedContext(
        chunk_id="c1",
        doc_id="policy:abc",
        text="Manage your contact information, such as your name, email, and phone number.",
        score=2.0,
        section_path="your-personal-information",
        source_url="",
    )
    privacy_controls = RetrievedContext(
        chunk_id="c2",
        doc_id="policy:abc",
        text="You may export or delete your data.",
        score=1.0,
        section_path="your-privacy-controls",
        source_url="",
    )
    assert _policy_context_score(personal_info, "data_subject_rights") < _policy_context_score(
        privacy_controls,
        "data_subject_rights",
    )


def test_context_from_answer_citations_skips_deprioritized_policy_section():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="Manage your contact information.",
            score=2.0,
            section_path="your-personal-information",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="p2",
            doc_id="policy:abc",
            text="You may export or delete your data.",
            score=1.0,
            section_path="your-privacy-controls",
            source_url="",
        ),
    ]
    answer = "Rights are missing [policy:abc §your-personal-information]."
    matched = _context_from_answer_citations(
        answer,
        contexts,
        policy_doc_id="policy:abc",
        prefer_regulatory=False,
        topic="data_subject_rights",
    )
    assert matched is None


def test_build_finding_prefers_privacy_controls_over_personal_info_for_rights():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="Manage your contact information, such as your name, email, and phone number.",
            score=2.0,
            section_path="your-personal-information",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="p2",
            doc_id="policy:abc",
            text="You may export or delete your data and manage privacy settings.",
            score=1.0,
            section_path="your-privacy-controls",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r1",
            doc_id="gdpr_art13_transparency",
            text="Controllers must provide contact details and explain rights.",
            score=1.0,
            section_path="good-disclosure",
            source_url="",
        ),
    ]
    response = AgentResponse(
        query="q",
        answer=(
            "The policy does not explain data subject rights "
            "[policy:abc §your-personal-information]. "
            "GDPR requires contact details [gdpr_art13_transparency §good-disclosure]."
        ),
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    finding = build_finding(
        "Does this policy explain data subject rights and contact details as GDPR requires?",
        response,
        "policy:abc",
        contexts,
    )
    assert finding["policy_citation"]["section_path"] == "your-privacy-controls"


def test_clean_analyze_answer_strips_chunk_markers():
    answer = (
        "Rights are unclear [policy:abc §rights | chunk=abc123]. "
        "GDPR requires contact [gdpr_art13_transparency §requirement | chunk=xyz]."
    )
    cleaned = clean_analyze_answer(answer)
    assert "| chunk=" not in cleaned
    assert "[policy:abc §rights]" in cleaned
    assert "[gdpr_art13_transparency §requirement]" in cleaned


def test_context_from_answer_citations_prefers_topic_regulatory_doc():
    contexts = [
        RetrievedContext(
            chunk_id="r1",
            doc_id="gdpr_retention",
            text="Retention must be limited.",
            score=2.0,
            section_path="requirement",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r2",
            doc_id="gdpr_art13_transparency",
            text="Controllers must provide contact details.",
            score=1.0,
            section_path="requirement",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r3",
            doc_id="gdpr_art17_erasure",
            text="Data subjects may request erasure.",
            score=0.5,
            section_path="requirement",
            source_url="",
        ),
    ]
    answer = (
        "Retention is mentioned [gdpr_retention §requirement] but rights need review "
        "[gdpr_art13_transparency §requirement]."
    )
    matched = _context_from_answer_citations(
        answer,
        contexts,
        policy_doc_id="policy:abc",
        prefer_regulatory=True,
        topic="data_subject_rights",
    )
    assert matched is not None
    assert matched.doc_id == "gdpr_art13_transparency"


def test_build_finding_picks_topic_regulatory_doc_over_incidental_mention():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="You may request deletion.",
            score=1.0,
            section_path="your-privacy-controls",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r1",
            doc_id="gdpr_retention",
            text="Retention must be limited.",
            score=2.0,
            section_path="requirement",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r2",
            doc_id="gdpr_art13_transparency",
            text="Controllers must provide contact details.",
            score=1.0,
            section_path="requirement",
            source_url="",
        ),
    ]
    response = AgentResponse(
        query="q",
        answer=(
            "Retention is cited [gdpr_retention §requirement] but contact details are missing "
            "[policy:abc §your-privacy-controls] [gdpr_art13_transparency §requirement]."
        ),
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    finding = build_finding(
        "Does this policy explain data subject rights and contact details as GDPR requires?",
        response,
        "policy:abc",
        contexts,
    )
    assert finding["regulation_citation"]["doc_id"] == "gdpr_art13_transparency"


def test_infer_status_from_summary_detects_do_not_explicitly():
    summary = (
        "The provided policy excerpts do not explicitly state a lawful basis "
        "for processing personal data."
    )
    assert _infer_status_from_summary(summary) == "needs_review"


def test_infer_status_from_summary_detects_hedged_retention_language():
    summary = (
        "The policy aligns with general principles, but without specific details "
        "it is not possible to fully determine if retention meets the 30-day guidance."
    )
    assert _infer_status_from_summary(summary) == "needs_review"


def test_finding_status_needs_review_without_explicit_align_language():
    response = AgentResponse(
        query="q",
        answer="answer",
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    assert (
        _finding_status(
            response,
            has_policy_citation=True,
            has_regulation_citation=True,
            policy_relevant_conflicts=[],
            summary=(
                "The provided policy excerpts do not explicitly state a lawful basis. "
                "GDPR requires disclosure [gdpr_art13_transparency §requirement]."
            ),
        )
        == "needs_review"
    )


def test_context_from_answer_citations_prefers_consent_over_scope_for_sharing():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="This Privacy Policy applies to services on third-party sites.",
            score=2.0,
            section_path="when-this-policy-applies",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="p2",
            doc_id="policy:abc",
            text="We'll share personal information outside of Google when we have your consent.",
            score=1.0,
            section_path="with-your-consent",
            source_url="",
        ),
    ]
    answer = (
        "Sharing with consent [policy:abc §with-your-consent]. "
        "Also applies on third-party sites [policy:abc §when-this-policy-applies]."
    )
    matched = _context_from_answer_citations(
        answer,
        contexts,
        policy_doc_id="policy:abc",
        prefer_regulatory=False,
        topic="third_party_sharing",
    )
    assert matched is not None
    assert matched.section_path == "with-your-consent"


def test_build_finding_prefers_consent_section_for_third_party_sharing():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="This Privacy Policy applies to services on third-party sites.",
            score=2.0,
            section_path="when-this-policy-applies",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="p2",
            doc_id="policy:abc",
            text="We'll share personal information when we have your consent.",
            score=1.0,
            section_path="with-your-consent",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r1",
            doc_id="gdpr_processors",
            text="Name categories of processors and reference safeguards.",
            score=1.0,
            section_path="good-disclosure",
            source_url="",
        ),
    ]
    response = AgentResponse(
        query="q",
        answer=(
            "Sharing occurs with consent [policy:abc §with-your-consent] and on third-party "
            "sites [policy:abc §when-this-policy-applies]. Processors require disclosure "
            "[gdpr_processors §good-disclosure]."
        ),
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    finding = build_finding(
        "Does this policy's third-party sharing align with GDPR processor requirements?",
        response,
        "policy:abc",
        contexts,
    )
    assert finding["policy_citation"]["section_path"] == "with-your-consent"


def test_context_from_answer_citations_prefers_policy_marker():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="Scope section",
            score=2.0,
            section_path="when-this-policy-applies",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="p2",
            doc_id="policy:abc",
            text="Consent sharing section",
            score=1.0,
            section_path="with-your-consent",
            source_url="",
        ),
    ]
    answer = "Sharing is described under consent [policy:abc §with-your-consent]."
    matched = _context_from_answer_citations(
        answer,
        contexts,
        policy_doc_id="policy:abc",
        prefer_regulatory=False,
    )
    assert matched is not None
    assert matched.section_path == "with-your-consent"


def test_policy_context_score_deprioritizes_changes_section_for_rights():
    ctx = RetrievedContext(
        chunk_id="c1",
        doc_id="policy:abc",
        text="We change this policy from time to time.",
        score=2.0,
        section_path="changes-to-this-policy",
        source_url="",
    )
    assert _policy_context_score(ctx, "data_subject_rights") < _policy_context_score(
        RetrievedContext(
            chunk_id="c2",
            doc_id="policy:abc",
            text="You may export or delete your data.",
            score=1.0,
            section_path="your-privacy-controls",
            source_url="",
        ),
        "data_subject_rights",
    )


def test_overlay_ingest_and_remove(indexed_pipeline: IngestionPipeline):
    doc_id = "policy:test-overlay"
    report = indexed_pipeline.ingest_document(SAMPLE_POLICY, doc_id=doc_id, source_url="https://x.test")
    assert report["doc_id"] == doc_id
    assert report["sections"] >= 1

    child_results = indexed_pipeline.child_bm25.search(
        "free tier users 90 days account deactivation",
        top_k=10,
    )
    doc_ids = {result.doc_id for result in child_results}
    assert doc_id in doc_ids

    indexed_pipeline.remove_document(doc_id)
    child_results = indexed_pipeline.child_bm25.search(
        "free tier users 90 days account deactivation",
        top_k=10,
    )
    doc_ids = {result.doc_id for result in child_results}
    assert doc_id not in doc_ids


def test_dual_source_retrieval_includes_regulatory(indexed_pipeline: IngestionPipeline):
    doc_id = "policy:test-dual"
    indexed_pipeline.ingest_document(SAMPLE_POLICY, doc_id=doc_id, source_url="https://x.test")
    try:
        contexts = _retrieve_dual_source(
            indexed_pipeline,
            "Does this policy's data retention approach align with GDPR retention guidance?",
            doc_id,
        )
        doc_ids = {ctx.doc_id for ctx in contexts}
        assert doc_id in doc_ids
        assert any(ctx.doc_id != doc_id for ctx in contexts)
    finally:
        indexed_pipeline.remove_document(doc_id)


def test_conflicts_for_analyze_excludes_demo_acme_without_policy(indexed_pipeline: IngestionPipeline):
    doc_id = "policy:external-google"
    indexed_pipeline.ingest_document(SAMPLE_POLICY, doc_id=doc_id, source_url="https://google.test")
    try:
        contexts = _retrieve_dual_source(
            indexed_pipeline,
            "Does this policy's data retention approach align with GDPR retention guidance?",
            doc_id,
        )
        conflicts = _conflicts_for_analyze(
            indexed_pipeline,
            contexts,
            policy_doc_id=doc_id,
            topic="data_retention",
        )
        for conflict in conflicts:
            claim_doc_ids = {claim.doc_id for claim in conflict.claims}
            assert doc_id in claim_doc_ids
            assert "acme_privacy_policy" not in claim_doc_ids
    finally:
        indexed_pipeline.remove_document(doc_id)


def test_analyze_policy_returns_regulation_citation(indexed_pipeline: IngestionPipeline):
    pipeline = DocumentPipeline(indexed_pipeline)
    report = analyze_policy(
        pipeline,
        page_text=SAMPLE_POLICY,
        url="https://example.com/privacy",
        questions=[
            "Does this policy's data retention approach align with GDPR retention guidance?"
        ],
    )
    assert report["findings"]
    finding = report["findings"][0]
    assert finding["topic"] == "data_retention"
    assert finding["regulation_citation"] is not None
    assert finding["regulation_citation"]["doc_id"] != report["doc_id"]
    assert finding["policy_citation"] is not None


def test_build_finding_selects_regulation_from_contexts():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="We retain data for 90 days.",
            score=1.0,
            section_path="data-retention",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r1",
            doc_id="gdpr_retention",
            text="Retention must be limited to what is necessary.",
            score=0.8,
            section_path="requirement",
            source_url="",
        ),
    ]
    response = AgentResponse(
        query="q",
        answer=(
            "Retention aligns with GDPR guidance in the policy [policy:abc §data-retention] "
            "and regulation [gdpr_retention §requirement]."
        ),
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    finding = build_finding(
        "Does this policy's data retention approach align with GDPR retention guidance?",
        response,
        "policy:abc",
        contexts,
    )
    assert finding["regulation_citation"]["doc_id"] == "gdpr_retention"
    assert finding["policy_citation"]["section_path"] == "data-retention"
    assert finding["status"] == "aligned"


def test_build_finding_marks_needs_review_when_summary_describes_gap():
    contexts = [
        RetrievedContext(
            chunk_id="p1",
            doc_id="policy:abc",
            text="We retain data for 90 days.",
            score=1.0,
            section_path="data-retention",
            source_url="",
        ),
        RetrievedContext(
            chunk_id="r1",
            doc_id="gdpr_retention",
            text="Retention must be limited to what is necessary.",
            score=0.8,
            section_path="requirement",
            source_url="",
        ),
    ]
    response = AgentResponse(
        query="q",
        answer=(
            "The policy does not explicitly address the 30-day guideline "
            "[policy:abc §data-retention] [gdpr_summary §data-retention]."
        ),
        citations=[],
        retrieved_chunk_ids=[],
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[],
        conflicts=[],
        refused=False,
    )
    finding = build_finding(
        "Does this policy's data retention approach align with GDPR retention guidance?",
        response,
        "policy:abc",
        contexts,
    )
    assert finding["status"] == "needs_review"
    assert finding["policy_citation"]["section_path"] == "data-retention"


def test_run_analyze_query_uses_dual_source(indexed_pipeline: IngestionPipeline):
    doc_id = "policy:test-run"
    indexed_pipeline.ingest_document(SAMPLE_POLICY, doc_id=doc_id, source_url="https://x.test")
    pipeline = DocumentPipeline(indexed_pipeline)
    try:
        response, contexts = run_analyze_query(
            pipeline,
            "Does this policy's data retention approach align with GDPR retention guidance?",
            doc_id,
        )
        assert response.retrieval_strategy == "analyze_dual_source"
        cited_docs = {citation["doc_id"] for citation in response.citations}
        assert doc_id in cited_docs
        assert any(doc != doc_id for doc in cited_docs)
        assert any(ctx.doc_id != doc_id for ctx in contexts)
    finally:
        indexed_pipeline.remove_document(doc_id)


def test_analyze_endpoint(indexed_pipeline: IngestionPipeline):
    from api.main import app

    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={
            "page_text": SAMPLE_POLICY,
            "url": "https://example.com/privacy",
            "questions": [
                "Does this policy's data retention approach align with GDPR retention guidance?"
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["findings"]
    assert payload["findings"][0]["regulation_citation"] is not None
    assert payload["disclaimer"]
    assert "review_count" in payload
