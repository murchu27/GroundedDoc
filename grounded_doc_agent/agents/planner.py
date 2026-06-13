from __future__ import annotations

import re

from grounded_doc_agent.models import QueryType, RetrievalPlan, RetrievalStrategy

COMPARE_KEYWORDS = ("compare", "versus", " vs ", "difference between", "contrast")
MULTI_HOP_KEYWORDS = ("align", "changed between", "both", "and also", "as well as")
SUMMARIZE_KEYWORDS = ("summarize", "overview", "summary", "explain")
REFUSE_KEYWORDS = ("stock price", "weather", "recipe", "sports score")
DOC_HINTS = ("gdpr", "pipeda", "privacy", "fastapi", "retention", "policy")


def classify_query(query: str) -> QueryType:
    lowered = query.lower()
    if any(keyword in lowered for keyword in REFUSE_KEYWORDS):
        return QueryType.OUT_OF_SCOPE
    if any(keyword in lowered for keyword in COMPARE_KEYWORDS):
        return QueryType.COMPARE
    if any(keyword in lowered for keyword in MULTI_HOP_KEYWORDS):
        return QueryType.MULTI_HOP
    if any(keyword in lowered for keyword in SUMMARIZE_KEYWORDS):
        return QueryType.SUMMARIZE
    if not any(hint in lowered for hint in DOC_HINTS):
        return QueryType.LOOKUP
    return QueryType.LOOKUP


def _split_compare_query(query: str) -> list[str]:
    parts = re.split(r"\b(?:vs\.?|versus|compare|and)\b", query, flags=re.IGNORECASE)
    cleaned = [part.strip(" ?.") for part in parts if part.strip(" ?.")]
    if len(cleaned) >= 2:
        return [f"{cleaned[0]} requirements", f"{cleaned[1]} requirements"]
    return [query]


def _split_multi_hop_query(query: str) -> list[str]:
    if " and " in query.lower():
        parts = re.split(r"\band\b", query, flags=re.IGNORECASE)
        return [part.strip(" ?.") for part in parts if part.strip(" ?.")]
    return [query]


def build_retrieval_plan(query: str) -> RetrievalPlan:
    query_type = classify_query(query)
    if query_type == QueryType.OUT_OF_SCOPE:
        return RetrievalPlan(query_type=query_type, sub_queries=[], strategies=[])

    if query_type == QueryType.COMPARE:
        sub_queries = _split_compare_query(query)
        return RetrievalPlan(
            query_type=query_type,
            sub_queries=sub_queries,
            strategies=[RetrievalStrategy.CLAIMS, RetrievalStrategy.MULTI_HOP],
        )

    if query_type == QueryType.MULTI_HOP:
        sub_queries = _split_multi_hop_query(query)
        return RetrievalPlan(
            query_type=query_type,
            sub_queries=sub_queries,
            strategies=[RetrievalStrategy.MULTI_HOP, RetrievalStrategy.VECTOR],
        )

    if query_type == QueryType.SUMMARIZE:
        return RetrievalPlan(
            query_type=query_type,
            sub_queries=[query],
            strategies=[RetrievalStrategy.PARENT, RetrievalStrategy.VECTOR],
        )

    if re.search(r"\b(section|§|error|api)\b", query.lower()) or re.search(
        r"\b[A-Z_]{3,}\b", query
    ):
        return RetrievalPlan(
            query_type=query_type,
            sub_queries=[query],
            strategies=[RetrievalStrategy.BM25, RetrievalStrategy.VECTOR],
        )

    return RetrievalPlan(
        query_type=query_type,
        sub_queries=[query],
        strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.BM25],
    )
