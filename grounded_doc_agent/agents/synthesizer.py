from __future__ import annotations

import os
import re

from grounded_doc_agent.config.settings import GEMINI_MODEL
from grounded_doc_agent.models import AgentResponse, ClaimConflict, PipelineTraceSpan, RetrievedContext


REFUSAL_PHRASE = "I don't have sufficient evidence in the indexed documents to answer that question."

CHUNK_MARKER_IN_CITATION = re.compile(
    r"\[([^\]|]+?\s+§[^\]|]+?)\s*\|\s*chunk=[^\]]+\]",
    re.IGNORECASE,
)


def clean_analyze_answer(answer: str) -> str:
    cleaned = CHUNK_MARKER_IN_CITATION.sub(r"[\1]", answer)
    return re.sub(r"\s+\]", "]", cleaned)


def synthesize_answer(
    query: str,
    contexts: list[RetrievedContext],
    conflicts: list[ClaimConflict],
    *,
    use_llm: bool | None = None,
) -> tuple[str, list[dict[str, str]]]:
    if not contexts:
        return REFUSAL_PHRASE, []

    if use_llm is None:
        use_llm = bool(os.getenv("GOOGLE_API_KEY"))

    if use_llm:
        answer, citations = _synthesize_with_gemini(query, contexts, conflicts)
        if answer:
            return answer, citations

    return _synthesize_extractive(query, contexts, conflicts)


def _score_regulatory_context(ctx: RetrievedContext, regulatory_doc_hints: list[str]) -> float:
    score = ctx.score
    for index, hint in enumerate(regulatory_doc_hints):
        if hint in ctx.doc_id:
            score += 10.0 - index
    return score


def synthesize_analyze_answer(
    query: str,
    contexts: list[RetrievedContext],
    conflicts: list[ClaimConflict],
    *,
    policy_doc_id: str,
    regulatory_doc_hints: list[str] | None = None,
    use_llm: bool | None = None,
) -> tuple[str, list[dict[str, str]]]:
    if not contexts:
        return REFUSAL_PHRASE, []

    policy_contexts = sorted(
        [ctx for ctx in contexts if ctx.doc_id == policy_doc_id],
        key=lambda item: item.score,
        reverse=True,
    )[:2]
    hints = regulatory_doc_hints or []
    regulatory_contexts = sorted(
        [ctx for ctx in contexts if ctx.doc_id != policy_doc_id],
        key=lambda item: _score_regulatory_context(item, hints),
        reverse=True,
    )[:1]
    selected = policy_contexts + regulatory_contexts
    if not selected:
        return REFUSAL_PHRASE, []

    if use_llm is None:
        use_llm = bool(os.getenv("GOOGLE_API_KEY"))

    if use_llm:
        answer, citations = _synthesize_with_gemini(
            query,
            selected,
            conflicts,
            citation_format="doc_section_only",
        )
        if answer:
            return clean_analyze_answer(answer), citations

    return _synthesize_analyze_extractive(
        query,
        policy_contexts,
        regulatory_contexts,
        conflicts,
    )


def _synthesize_analyze_extractive(
    query: str,
    policy_contexts: list[RetrievedContext],
    regulatory_contexts: list[RetrievedContext],
    conflicts: list[ClaimConflict],
) -> tuple[str, list[dict[str, str]]]:
    ordered = policy_contexts[:2] + regulatory_contexts[:1]
    citations = [
        {
            "chunk_id": ctx.chunk_id,
            "doc_id": ctx.doc_id,
            "section_path": ctx.section_path,
            "source": ctx.source_url or ctx.doc_id,
        }
        for ctx in ordered
    ]
    answer_parts = [f"Based on the indexed documents for '{query}':"]
    for idx, ctx in enumerate(ordered, start=1):
        answer_parts.append(
            f"{idx}. [{ctx.doc_id} §{ctx.section_path}] {ctx.text.strip()[:350]}"
        )
    if conflicts:
        answer_parts.append("Conflicting sources detected:")
        for conflict in conflicts:
            answer_parts.append(f"- {conflict.description}")
    return "\n".join(answer_parts), citations


def _synthesize_extractive(
    query: str,
    contexts: list[RetrievedContext],
    conflicts: list[ClaimConflict],
) -> tuple[str, list[dict[str, str]]]:
    citations = [
        {
            "chunk_id": ctx.chunk_id,
            "doc_id": ctx.doc_id,
            "section_path": ctx.section_path,
            "source": ctx.source_url or ctx.doc_id,
        }
        for ctx in contexts[:3]
    ]
    snippets = [ctx.text.strip() for ctx in contexts[:3]]
    answer_parts = [f"Based on the indexed documents for '{query}':"]
    for idx, snippet in enumerate(snippets, start=1):
        citation = citations[idx - 1]
        answer_parts.append(
            f"{idx}. [{citation['doc_id']} §{citation['section_path']}] {snippet[:350]}"
        )
    if conflicts:
        answer_parts.append("Conflicting sources detected:")
        for conflict in conflicts:
            answer_parts.append(f"- {conflict.description}")
    return "\n".join(answer_parts), citations


def _synthesize_with_gemini(
    query: str,
    contexts: list[RetrievedContext],
    conflicts: list[ClaimConflict],
    *,
    citation_format: str = "doc_section_or_chunk",
) -> tuple[str, list[dict[str, str]]]:
    try:
        from google import genai
    except ImportError:
        return "", []

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    context_block = "\n\n".join(
        f"[{ctx.doc_id} §{ctx.section_path} | chunk={ctx.chunk_id}] {ctx.text}"
        for ctx in contexts[:5]
    )
    conflict_block = "\n".join(conflict.description for conflict in conflicts)
    citation_instruction = (
        "Include bracket citations using only [doc_id §section_path]. "
        "Do not include chunk IDs or internal markers in citations."
        if citation_format == "doc_section_only"
        else "Include bracket citations like [doc_id §section_path]. "
    )
    prompt = (
        "Answer the user question using ONLY the provided context. "
        f"{citation_instruction} "
        "If sources conflict, explicitly mention both sides.\n\n"
        f"Question: {query}\n\nContext:\n{context_block}\n\n"
        f"Known conflicts:\n{conflict_block or 'None'}"
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    answer = response.text or ""
    citations = _extract_citations(answer, contexts)
    return answer, citations


def _extract_citations(
    answer: str,
    contexts: list[RetrievedContext],
) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    for ctx in contexts:
        marker = f"{ctx.doc_id} §{ctx.section_path}"
        if marker in answer or ctx.chunk_id in answer:
            citations.append(
                {
                    "chunk_id": ctx.chunk_id,
                    "doc_id": ctx.doc_id,
                    "section_path": ctx.section_path,
                    "source": ctx.source_url or ctx.doc_id,
                }
            )
    if citations:
        return citations
    return [
        {
            "chunk_id": ctx.chunk_id,
            "doc_id": ctx.doc_id,
            "section_path": ctx.section_path,
            "source": ctx.source_url or ctx.doc_id,
        }
        for ctx in contexts[:3]
    ]


def verify_answer(
    answer: str,
    citations: list[dict[str, str]],
    contexts: list[RetrievedContext],
    *,
    should_refuse: bool = False,
) -> tuple[bool, str]:
    if should_refuse:
        refused = REFUSAL_PHRASE.split(".")[0].lower() in answer.lower()
        return refused, "expected refusal" if refused else "missing refusal"

    if REFUSAL_PHRASE.split(".")[0].lower() in answer.lower():
        return not contexts, "unexpected refusal"

    if not citations:
        return False, "missing citations"

    retrieved_ids = {ctx.chunk_id for ctx in contexts}
    for citation in citations:
        if citation["chunk_id"] not in retrieved_ids:
            return False, f"orphan citation {citation['chunk_id']}"

    unsupported = re.findall(r"\[([^\]]+)\]", answer)
    if unsupported and not citations:
        return False, "citation markers without mapped chunks"
    return True, "verified"


def build_agent_response(
    query: str,
    answer: str,
    citations: list[dict[str, str]],
    contexts: list[RetrievedContext],
    *,
    retrieval_strategy: str,
    query_type: str,
    sub_queries: list[str],
    conflicts: list[ClaimConflict],
    refused: bool,
    trace_spans: list[PipelineTraceSpan],
) -> AgentResponse:
    return AgentResponse(
        query=query,
        answer=answer,
        citations=citations,
        retrieved_chunk_ids=[ctx.chunk_id for ctx in contexts],
        retrieval_strategy=retrieval_strategy,
        query_type=query_type,
        sub_queries=sub_queries,
        conflicts=[
            {
                "subject": conflict.subject,
                "description": conflict.description,
                "sources": [claim.source_label for claim in conflict.claims],
            }
            for conflict in conflicts
        ],
        refused=refused,
        trace_spans=trace_spans,
    )
