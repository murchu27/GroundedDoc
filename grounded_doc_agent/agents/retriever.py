from __future__ import annotations

from grounded_doc_agent.config.settings import TOP_K_CHILD, TOP_K_PARENT
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline
from grounded_doc_agent.models import ClaimConflict, RetrievedContext, RetrievalPlan, RetrievalStrategy
from grounded_doc_agent.retrieval.claims_store import ClaimsStore


class AdaptiveRetriever:
    def __init__(self, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline
        self.claims_store: ClaimsStore = pipeline.claims_store

    def retrieve(self, plan: RetrievalPlan) -> tuple[list[RetrievedContext], str, list[ClaimConflict]]:
        if not plan.sub_queries:
            return [], RetrievalStrategy.VECTOR.value, []

        primary = plan.strategies[0] if plan.strategies else RetrievalStrategy.VECTOR
        contexts: list[RetrievedContext] = []
        seen_ids: set[str] = set()

        for sub_query in plan.sub_queries:
            batch = self._retrieve_single(sub_query, primary, plan)
            for ctx in batch:
                if ctx.chunk_id not in seen_ids:
                    seen_ids.add(ctx.chunk_id)
                    contexts.append(ctx)

        if len(contexts) < TOP_K_CHILD and len(plan.strategies) > 1:
            fallback = plan.strategies[1]
            for sub_query in plan.sub_queries:
                batch = self._retrieve_single(sub_query, fallback, plan)
                for ctx in batch:
                    if ctx.chunk_id not in seen_ids:
                        seen_ids.add(ctx.chunk_id)
                        contexts.append(ctx)

        contexts.sort(key=lambda item: item.score, reverse=True)
        subjects = {token for token in sub_query_tokens(plan) if len(token) > 4}
        conflicts = self.claims_store.get_conflicts_for_subjects(subjects)
        if not conflicts:
            conflicts = self._conflicts_from_contexts(contexts)
        return contexts[: TOP_K_CHILD + 2], primary.value, conflicts

    def _retrieve_single(
        self,
        query: str,
        strategy: RetrievalStrategy,
        plan: RetrievalPlan,
    ) -> list[RetrievedContext]:
        if strategy == RetrievalStrategy.BM25:
            return self.pipeline.child_bm25.search(query, top_k=TOP_K_CHILD)
        if strategy == RetrievalStrategy.PARENT:
            return self.pipeline.parent_bm25.search(query, top_k=TOP_K_PARENT)
        if strategy == RetrievalStrategy.CLAIMS:
            return self._retrieve_from_claims(query)
        if strategy == RetrievalStrategy.MULTI_HOP:
            claim_contexts = self._retrieve_from_claims(query)
            vector_contexts = self.pipeline.vector_index.search_children(query, top_k=TOP_K_CHILD)
            return self._merge_contexts(claim_contexts + vector_contexts)
        return self.pipeline.vector_index.search_children(query, top_k=TOP_K_CHILD)

    def _retrieve_from_claims(self, query: str) -> list[RetrievedContext]:
        claims = self.claims_store.search_claims(query)
        contexts: list[RetrievedContext] = []
        for claim in claims:
            child = self.pipeline.vector_index.get_child(claim.chunk_id)
            text = child.text if child else f"{claim.subject}: {claim.value}"
            contexts.append(
                RetrievedContext(
                    chunk_id=claim.chunk_id,
                    doc_id=claim.doc_id,
                    text=text,
                    score=1.0,
                    section_path=claim.section_path,
                    source_url=claim.source_label,
                    chunk_level="claim",
                )
            )
        return contexts

    def _conflicts_from_contexts(self, contexts: list[RetrievedContext]) -> list[ClaimConflict]:
        doc_ids = {ctx.doc_id for ctx in contexts}
        conflicts = self.claims_store.list_conflicts()
        return [
            conflict
            for conflict in conflicts
            if len({claim.doc_id for claim in conflict.claims}.intersection(doc_ids)) > 1
        ]

    @staticmethod
    def _merge_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        merged: dict[str, RetrievedContext] = {}
        for ctx in contexts:
            existing = merged.get(ctx.chunk_id)
            if existing is None or ctx.score > existing.score:
                merged[ctx.chunk_id] = ctx
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def sub_query_tokens(plan: RetrievalPlan) -> set[str]:
    tokens: set[str] = set()
    for sub_query in plan.sub_queries:
        for token in sub_query.lower().replace("_", " ").split():
            tokens.add(token.strip(".,?"))
    return tokens
