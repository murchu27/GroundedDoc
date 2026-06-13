from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from grounded_doc_agent.agents.planner import build_retrieval_plan, classify_query
from grounded_doc_agent.agents.retriever import AdaptiveRetriever
from grounded_doc_agent.agents.synthesizer import (
    REFUSAL_PHRASE,
    build_agent_response,
    synthesize_answer,
    verify_answer,
)
from grounded_doc_agent.config.settings import INDEX_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline
from grounded_doc_agent.models import AgentResponse, PipelineTraceSpan, QueryType
from grounded_doc_agent.storage.backend import maybe_sync_index


_pipeline: DocumentPipeline | None = None
_index_synced = False


class DocumentPipeline:
    def __init__(self, pipeline: IngestionPipeline | None = None) -> None:
        self.pipeline = pipeline or IngestionPipeline.load_existing(INDEX_DIR)
        self.retriever = AdaptiveRetriever(self.pipeline)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    @mlflow.trace(name="grounded_doc_query", span_type=SpanType.CHAIN)
    def run(self, query: str, *, variant: str = "full_pipeline") -> AgentResponse:
        trace_spans: list[PipelineTraceSpan] = []
        mlflow.update_current_trace(metadata={"variant": variant})

        query_type = classify_query(query)
        trace_spans.append(
            PipelineTraceSpan(
                name="router",
                attributes={"query_type": query_type.value},
            )
        )

        if query_type == QueryType.OUT_OF_SCOPE:
            response = build_agent_response(
                query=query,
                answer=REFUSAL_PHRASE,
                citations=[],
                contexts=[],
                retrieval_strategy="none",
                query_type=query_type.value,
                sub_queries=[],
                conflicts=[],
                refused=True,
                trace_spans=trace_spans,
            )
            self._log_span("response", response.to_dict())
            return response

        plan = build_retrieval_plan(query)
        trace_spans.append(
            PipelineTraceSpan(
                name="query_planner",
                attributes={
                    "query_type": plan.query_type.value,
                    "sub_queries": plan.sub_queries,
                    "strategies": [strategy.value for strategy in plan.strategies],
                },
            )
        )

        if variant == "baseline_flat_rag":
            contexts = self.pipeline.vector_index.search_children(query)
            retrieval_strategy = "vector"
            conflicts = []
        else:
            contexts, retrieval_strategy, conflicts = self.retriever.retrieve(plan)

        trace_spans.append(
            PipelineTraceSpan(
                name="adaptive_retriever",
                attributes={
                    "retrieval_strategy": retrieval_strategy,
                    "retrieved_chunk_ids": [ctx.chunk_id for ctx in contexts],
                    "num_results": len(contexts),
                },
            )
        )

        answer, citations = synthesize_answer(query, contexts, conflicts)
        verified, reason = verify_answer(answer, citations, contexts)
        if not verified and variant == "full_pipeline":
            extra_contexts, _, extra_conflicts = self.retriever.retrieve(plan)
            merged = _merge_contexts(contexts, extra_contexts)
            conflicts = extra_conflicts or conflicts
            answer, citations = synthesize_answer(query, merged, conflicts)
            verified, reason = verify_answer(answer, citations, merged)
            contexts = merged

        trace_spans.append(
            PipelineTraceSpan(
                name="citation_verifier",
                attributes={"verified": verified, "reason": reason},
            )
        )

        refused = REFUSAL_PHRASE.split(".")[0].lower() in answer.lower()
        response = build_agent_response(
            query=query,
            answer=answer,
            citations=citations,
            contexts=contexts,
            retrieval_strategy=retrieval_strategy,
            query_type=plan.query_type.value,
            sub_queries=plan.sub_queries,
            conflicts=conflicts,
            refused=refused,
            trace_spans=trace_spans,
        )
        self._log_span("response", response.to_dict())
        return response

    @staticmethod
    def _log_span(name: str, attributes: dict) -> None:
        with mlflow.start_span(name=name) as span:
            span.set_attributes({k: str(v) for k, v in attributes.items()})


def get_pipeline() -> DocumentPipeline:
    global _pipeline, _index_synced
    if not _index_synced:
        maybe_sync_index(INDEX_DIR)
        _index_synced = True
    if _pipeline is None:
        if not (INDEX_DIR / "ingestion_report.json").exists():
            IngestionPipeline().run()
        _pipeline = DocumentPipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None


def _merge_contexts(
    left: list,
    right: list,
) -> list:
    merged = {ctx.chunk_id: ctx for ctx in left}
    for ctx in right:
        merged.setdefault(ctx.chunk_id, ctx)
    return list(merged.values())


def predict_for_eval(inputs: dict | str, variant: str = "full_pipeline") -> dict:
    query = inputs["query"] if isinstance(inputs, dict) else str(inputs)
    response = get_pipeline().run(query, variant=variant)
    return response.to_dict()
