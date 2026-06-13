from __future__ import annotations

import re
import threading
from typing import Any
from collections.abc import Callable

from grounded_doc_agent.agents.pipeline import DocumentPipeline
from grounded_doc_agent.agents.synthesizer import (
    REFUSAL_PHRASE,
    build_agent_response,
    synthesize_analyze_answer,
    verify_answer,
)
from grounded_doc_agent.config.settings import TOP_K_CHILD
from grounded_doc_agent.ingestion.overlay import infer_topic, policy_doc_id
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline
from grounded_doc_agent.models import AgentResponse, ClaimConflict, RetrievedContext
from grounded_doc_agent.storage.backend import persist_policy_snapshot

DISCLAIMER = "Informational analysis only; not legal advice."

DEFAULT_COMPLIANCE_QUESTIONS = [
    "Does this policy's data retention approach align with GDPR retention guidance?",
    "Does this policy's third-party sharing align with GDPR processor requirements?",
    "Does this policy explain data subject rights and contact details as GDPR requires?",
    "Does this policy state a lawful basis for processing as GDPR requires?",
]

TOPIC_REGULATORY_HINTS: dict[str, list[str]] = {
    "data_retention": ["gdpr_retention", "gdpr_summary", "pipeda_summary"],
    "third_party_sharing": ["gdpr_processors", "gdpr_summary"],
    "data_subject_rights": ["gdpr_art17_erasure", "gdpr_art13_transparency"],
    "lawful_basis": ["gdpr_art13_transparency", "gdpr_summary"],
}

TOPIC_SECTION_KEYWORDS: dict[str, list[str]] = {
    "data_retention": ["retain", "retention", "delete", "storage", "information"],
    "third_party_sharing": ["sharing", "share", "third-party", "processor", "partner", "consent"],
    "data_subject_rights": ["rights", "privacy-controls", "export", "delete", "contact", "erasure"],
    "lawful_basis": ["lawful", "basis", "legal", "purpose", "collects", "why"],
}

EXCLUDED_SECTION_PATHS = frozenset({"contents", "intro"})

TOPIC_DEPRIORITIZED_SECTIONS: dict[str, frozenset[str]] = {
    "data_subject_rights": frozenset(
        {
            "changes-to-this-policy",
            "when-this-policy-applies",
            "your-personal-information",
        }
    ),
    "lawful_basis": frozenset({"changes-to-this-policy", "when-this-policy-applies"}),
    "third_party_sharing": frozenset({"when-this-policy-applies"}),
}

GAP_SUMMARY_PHRASES = (
    "does not explicitly",
    "do not explicitly",
    "does not explain",
    "does not state",
    "do not state",
    "does not address",
    "does not detail",
    "does not name",
    "does not reference",
    "does not align",
    "do not align",
    "not align with",
    "not possible to",
    "cannot fully determine",
    "without specific details",
    "provided context does not",
    "provided sections of the policy do not",
    "provided policy context does not",
    "provided policy does not",
    "provided policy excerpts do not",
    "however, the provided",
    "however, gdpr",
)

ALIGN_SUMMARY_PHRASES = (
    "aligns with gdpr",
    "appears to align with gdpr",
    "is consistent with gdpr",
    "meets the gdpr requirement",
    "satisfies the gdpr requirement",
)

CITATION_MARKER_PATTERN = re.compile(r"\[([^\]§]+?)\s+§([^\]]+?)\]")

_overlay_lock = threading.Lock()
_ANALYZE_TOP_K_EACH = max(2, TOP_K_CHILD // 2)


def _last_sentence_end(text: str) -> int:
    return max(text.rfind(". "), text.rfind("? "), text.rfind("! "))


def _advance_to_sentence_start(text: str) -> str:
    if not text or text[0].isupper():
        return text
    boundary = _last_sentence_end(text[:120])
    if boundary >= 0:
        return text[boundary + 2 :].lstrip()
    space_idx = text.find(" ")
    if 0 < space_idx < 48:
        return text[space_idx + 1 :].lstrip()
    return text


def _trim_trailing_incomplete_sentence(text: str, *, min_keep: int = 40) -> str:
    if not text or text[-1] in ".!?":
        return text
    boundary = _last_sentence_end(text)
    if boundary >= min_keep:
        return text[: boundary + 1].rstrip()
    return text


def _format_citation_snippet(text: str, *, max_len: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""
    cleaned = _advance_to_sentence_start(cleaned)
    if len(cleaned) <= max_len:
        return _trim_trailing_incomplete_sentence(cleaned)
    window = cleaned[:max_len]
    boundary = _last_sentence_end(window)
    if boundary >= max_len // 3:
        return window[: boundary + 1].rstrip()
    truncated = window
    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        truncated = truncated[:last_space]
    return _trim_trailing_incomplete_sentence(truncated.rstrip() + "...")


def _citation_detail(
    citation: dict[str, str] | None,
    *,
    chunk_text: str = "",
) -> dict[str, str] | None:
    if not citation:
        return None
    return {
        "doc_id": citation.get("doc_id", ""),
        "section_path": citation.get("section_path", ""),
        "source": citation.get("source", ""),
        "text": _format_citation_snippet(chunk_text),
    }


def _context_to_citation(ctx: RetrievedContext) -> dict[str, str]:
    return {
        "chunk_id": ctx.chunk_id,
        "doc_id": ctx.doc_id,
        "section_path": ctx.section_path,
        "source": ctx.source_url or ctx.doc_id,
    }


def _is_regulatory_doc_id(doc_id: str, current_policy_doc_id: str) -> bool:
    return doc_id != current_policy_doc_id


def _is_toc_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    short_lines = sum(1 for line in lines if len(line) < 60 and "." not in line)
    return short_lines / len(lines) > 0.7


def _policy_context_score(ctx: RetrievedContext, topic: str) -> float:
    score = ctx.score
    section_path = ctx.section_path.lower()
    if section_path in EXCLUDED_SECTION_PATHS:
        return score - 20.0
    if section_path in TOPIC_DEPRIORITIZED_SECTIONS.get(topic, frozenset()):
        return score - 25.0
    if _is_toc_like(ctx.text):
        return score - 15.0
    for keyword in TOPIC_SECTION_KEYWORDS.get(topic, []):
        if keyword in section_path:
            score += 5.0
    return score


def _infer_status_from_summary(summary: str) -> str | None:
    lowered = summary.lower()
    if any(phrase in lowered for phrase in GAP_SUMMARY_PHRASES):
        return "needs_review"
    if any(phrase in lowered for phrase in ALIGN_SUMMARY_PHRASES):
        return "aligned"
    return None


def _context_from_answer_citations(
    answer: str,
    contexts: list[RetrievedContext],
    *,
    policy_doc_id: str,
    prefer_regulatory: bool,
    topic: str | None = None,
) -> RetrievedContext | None:
    from grounded_doc_agent.agents.synthesizer import clean_analyze_answer

    context_index = {(ctx.doc_id, ctx.section_path): ctx for ctx in contexts}
    normalized_answer = clean_analyze_answer(answer)
    matches: list[RetrievedContext] = []
    for doc_id, section_path in CITATION_MARKER_PATTERN.findall(normalized_answer):
        doc_id = doc_id.strip()
        section_path = section_path.strip().rstrip(".,;")
        if prefer_regulatory:
            if not _is_regulatory_doc_id(doc_id, policy_doc_id):
                continue
        elif doc_id != policy_doc_id:
            continue
        matched = context_index.get((doc_id, section_path))
        if matched:
            matches.append(matched)

    if not matches:
        return None
    if prefer_regulatory and topic:
        return max(matches, key=lambda ctx: _regulatory_context_score(ctx, topic))
    if topic:
        best = max(matches, key=lambda ctx: _policy_context_score(ctx, topic))
        deprioritized = TOPIC_DEPRIORITIZED_SECTIONS.get(topic, frozenset())
        if best.section_path in deprioritized:
            return None
        return best
    return matches[0]


def _regulatory_context_score(ctx: RetrievedContext, topic: str) -> float:
    score = ctx.score
    for index, hint in enumerate(TOPIC_REGULATORY_HINTS.get(topic, [])):
        if hint in ctx.doc_id:
            score += 10.0 - index
    return score


def _select_best_context(
    contexts: list[RetrievedContext],
    *,
    policy_doc_id: str,
    topic: str,
    prefer_regulatory: bool,
) -> RetrievedContext | None:
    if prefer_regulatory:
        candidates = [
            ctx for ctx in contexts if _is_regulatory_doc_id(ctx.doc_id, policy_doc_id)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda ctx: _regulatory_context_score(ctx, topic))

    deprioritized = TOPIC_DEPRIORITIZED_SECTIONS.get(topic, frozenset())
    candidates = [
        ctx
        for ctx in contexts
        if ctx.doc_id == policy_doc_id
        and ctx.section_path not in EXCLUDED_SECTION_PATHS
        and ctx.section_path not in deprioritized
        and not _is_toc_like(ctx.text)
    ]
    if not candidates:
        candidates = [
            ctx
            for ctx in contexts
            if ctx.doc_id == policy_doc_id and ctx.section_path not in EXCLUDED_SECTION_PATHS
        ]
    if not candidates:
        candidates = [ctx for ctx in contexts if ctx.doc_id == policy_doc_id]
    if not candidates:
        return None
    return max(candidates, key=lambda ctx: _policy_context_score(ctx, topic))


def _finding_status(
    response: AgentResponse,
    *,
    has_policy_citation: bool,
    has_regulation_citation: bool,
    policy_relevant_conflicts: list[dict[str, Any]],
    summary: str,
) -> str:
    if response.refused or not has_policy_citation:
        return "insufficient_evidence"
    if policy_relevant_conflicts:
        return "potential_gap"

    summary_status = _infer_status_from_summary(summary)
    if summary_status == "needs_review":
        return "needs_review"
    if (
        summary_status == "aligned"
        and has_policy_citation
        and has_regulation_citation
    ):
        return "aligned"
    return "needs_review"


def _conflicts_for_analyze(
    pipeline: IngestionPipeline,
    contexts: list[RetrievedContext],
    *,
    policy_doc_id: str,
    topic: str,
) -> list[ClaimConflict]:
    regulatory_doc_ids = {
        ctx.doc_id for ctx in contexts if _is_regulatory_doc_id(ctx.doc_id, policy_doc_id)
    }
    conflicts: list[ClaimConflict] = []
    for conflict in pipeline.claims_store.list_conflicts():
        if conflict.subject != topic:
            continue
        claim_doc_ids = {claim.doc_id for claim in conflict.claims}
        if policy_doc_id not in claim_doc_ids:
            continue
        if not claim_doc_ids.intersection(regulatory_doc_ids):
            continue
        conflicts.append(conflict)
    return conflicts


def _rank_policy_contexts(
    contexts: list[RetrievedContext],
    topic: str,
) -> list[RetrievedContext]:
    return sorted(contexts, key=lambda ctx: _policy_context_score(ctx, topic), reverse=True)


def _collect_filtered(
    results: list[RetrievedContext],
    *,
    doc_filter: Callable[[str], bool],
    top_k: int,
    seen: set[str],
    collected: list[RetrievedContext],
) -> None:
    for ctx in results:
        if not doc_filter(ctx.doc_id):
            continue
        if ctx.chunk_id in seen:
            continue
        collected.append(ctx)
        seen.add(ctx.chunk_id)
        if len(collected) >= top_k:
            break


def _retrieve_dual_source(
    pipeline: IngestionPipeline,
    query: str,
    policy_doc_id: str,
    *,
    top_k_each: int = _ANALYZE_TOP_K_EACH,
) -> list[RetrievedContext]:
    vector = pipeline.vector_index
    topic = infer_topic(query)
    topic_label = topic.replace("_", " ")
    policy_contexts: list[RetrievedContext] = []
    regulatory_contexts: list[RetrievedContext] = []
    policy_seen: set[str] = set()
    regulatory_seen: set[str] = set()

    policy_queries = [query, f"privacy policy {topic_label}", f"this policy {topic_label}"]
    regulatory_hints = TOPIC_REGULATORY_HINTS.get(topic, ["gdpr_summary"])
    regulatory_queries = [
        f"GDPR {topic_label} guidance",
        f"GDPR requirement {topic_label}",
        " ".join(regulatory_hints),
    ]

    bm25 = pipeline.child_bm25

    for policy_query in policy_queries:
        for search_results in (
            bm25.search(policy_query, top_k=top_k_each * 8),
            vector.search_children(policy_query, top_k=top_k_each * 8),
        ):
            _collect_filtered(
                search_results,
                doc_filter=lambda doc_id: doc_id == policy_doc_id,
                top_k=top_k_each * 2,
                seen=policy_seen,
                collected=policy_contexts,
            )
        if len(policy_contexts) >= top_k_each:
            break

    policy_contexts = _rank_policy_contexts(policy_contexts, topic)[:top_k_each]

    for regulatory_query in regulatory_queries:
        for search_results in (
            bm25.search(regulatory_query, top_k=top_k_each * 8),
            vector.search_children(regulatory_query, top_k=top_k_each * 8),
        ):
            _collect_filtered(
                search_results,
                doc_filter=lambda doc_id: _is_regulatory_doc_id(doc_id, policy_doc_id),
                top_k=top_k_each * 2,
                seen=regulatory_seen,
                collected=regulatory_contexts,
            )
        if len(regulatory_contexts) >= top_k_each:
            break

    regulatory_contexts = sorted(
        regulatory_contexts,
        key=lambda ctx: _regulatory_context_score(ctx, topic),
        reverse=True,
    )[:top_k_each]

    merged: dict[str, RetrievedContext] = {}
    for ctx in policy_contexts + regulatory_contexts:
        merged[ctx.chunk_id] = ctx
    return list(merged.values())


def run_analyze_query(
    pipeline: DocumentPipeline,
    question: str,
    policy_doc_id: str,
) -> tuple[AgentResponse, list[RetrievedContext]]:
    topic = infer_topic(question)
    contexts = _retrieve_dual_source(pipeline.pipeline, question, policy_doc_id)
    conflicts = _conflicts_for_analyze(
        pipeline.pipeline,
        contexts,
        policy_doc_id=policy_doc_id,
        topic=topic,
    )
    answer, citations = synthesize_analyze_answer(
        question,
        contexts,
        conflicts,
        policy_doc_id=policy_doc_id,
        regulatory_doc_hints=TOPIC_REGULATORY_HINTS.get(topic, []),
    )
    verified, _reason = verify_answer(answer, citations, contexts)
    if not verified:
        extra = _retrieve_dual_source(pipeline.pipeline, question, policy_doc_id)
        merged = {ctx.chunk_id: ctx for ctx in contexts}
        for ctx in extra:
            merged.setdefault(ctx.chunk_id, ctx)
        contexts = list(merged.values())
        conflicts = _conflicts_for_analyze(
            pipeline.pipeline,
            contexts,
            policy_doc_id=policy_doc_id,
            topic=topic,
        )
        answer, citations = synthesize_analyze_answer(
            question,
            contexts,
            conflicts,
            policy_doc_id=policy_doc_id,
            regulatory_doc_hints=TOPIC_REGULATORY_HINTS.get(topic, []),
        )

    refused = REFUSAL_PHRASE.split(".")[0].lower() in answer.lower()
    response = build_agent_response(
        query=question,
        answer=answer,
        citations=citations,
        contexts=contexts,
        retrieval_strategy="analyze_dual_source",
        query_type="ANALYZE",
        sub_queries=[question],
        conflicts=conflicts,
        refused=refused,
        trace_spans=[],
    )
    return response, contexts


def build_finding(
    question: str,
    response: AgentResponse,
    policy_doc_id: str,
    contexts: list[RetrievedContext],
) -> dict[str, Any]:
    topic = infer_topic(question)
    policy_ctx = _context_from_answer_citations(
        response.answer,
        contexts,
        policy_doc_id=policy_doc_id,
        prefer_regulatory=False,
        topic=topic,
    ) or _select_best_context(
        contexts,
        policy_doc_id=policy_doc_id,
        topic=topic,
        prefer_regulatory=False,
    )
    regulation_ctx = _context_from_answer_citations(
        response.answer,
        contexts,
        policy_doc_id=policy_doc_id,
        prefer_regulatory=True,
        topic=topic,
    ) or _select_best_context(
        contexts,
        policy_doc_id=policy_doc_id,
        topic=topic,
        prefer_regulatory=True,
    )

    policy_citation = _context_to_citation(policy_ctx) if policy_ctx else None
    regulation_citation = _context_to_citation(regulation_ctx) if regulation_ctx else None

    return {
        "topic": topic,
        "question": question,
        "status": _finding_status(
            response,
            has_policy_citation=bool(policy_citation),
            has_regulation_citation=bool(regulation_citation),
            policy_relevant_conflicts=response.conflicts,
            summary=response.answer,
        ),
        "summary": response.answer,
        "policy_citation": _citation_detail(
            policy_citation,
            chunk_text=policy_ctx.text if policy_ctx else "",
        ),
        "regulation_citation": _citation_detail(
            regulation_citation,
            chunk_text=regulation_ctx.text if regulation_ctx else "",
        ),
        "conflicts": response.conflicts,
        "refused": response.refused,
    }


def analyze_policy(
    pipeline: DocumentPipeline,
    *,
    page_text: str,
    url: str = "",
    questions: list[str] | None = None,
) -> dict[str, Any]:
    if len(page_text.strip()) < 50:
        return {
            "doc_id": "",
            "url": url,
            "findings": [],
            "conflicts": [],
            "disclaimer": DISCLAIMER,
            "error": "Page text is too short to analyze.",
        }

    doc_id = policy_doc_id(url or page_text[:120])
    question_list = questions or DEFAULT_COMPLIANCE_QUESTIONS

    with _overlay_lock:
        ingest_report = pipeline.pipeline.ingest_document(
            page_text,
            doc_id=doc_id,
            source_url=url,
        )
        persist_policy_snapshot(doc_id.split(":", 1)[-1], page_text, url=url)
        findings = []
        all_conflicts: list[dict[str, Any]] = []
        try:
            for question in question_list:
                response, contexts = run_analyze_query(pipeline, question, doc_id)
                finding = build_finding(question, response, doc_id, contexts)
                findings.append(finding)
                for conflict in response.conflicts:
                    if conflict not in all_conflicts:
                        all_conflicts.append(conflict)
        finally:
            pipeline.pipeline.remove_document(doc_id)

    return {
        "doc_id": doc_id,
        "url": url,
        "ingest": ingest_report,
        "findings": findings,
        "conflicts": all_conflicts,
        "disclaimer": DISCLAIMER,
        "refused_count": sum(
            1 for finding in findings if finding["status"] == "insufficient_evidence"
        ),
        "gap_count": sum(1 for finding in findings if finding["status"] == "potential_gap"),
        "review_count": sum(1 for finding in findings if finding["status"] == "needs_review"),
    }

