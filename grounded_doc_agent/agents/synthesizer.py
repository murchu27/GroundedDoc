from __future__ import annotations

import os
import re

from grounded_doc_agent.config.settings import GEMINI_MODEL
from grounded_doc_agent.models import AgentResponse, ClaimConflict, PipelineTraceSpan, RetrievedContext


REFUSAL_PHRASE = "I don't have sufficient evidence in the indexed documents to answer that question."


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
    prompt = (
        "Answer the user question using ONLY the provided context. "
        "Include bracket citations like [doc_id §section_path]. "
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
