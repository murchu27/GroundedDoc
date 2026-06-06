from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.genai import evaluate

from grounded_doc_agent.agents.pipeline import predict_for_eval
from grounded_doc_agent.config.settings import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, INDEX_DIR, PROJECT_ROOT
from grounded_doc_agent.eval.scorers import (
    citation_fidelity,
    conflict_surfaced,
    refusal_correctness,
    retrieval_recall,
    retrieval_strategy_match,
)
from grounded_doc_agent.ingestion.pipeline import IngestionPipeline


DEFAULT_DATASET = PROJECT_ROOT / "grounded_doc_agent" / "eval" / "golden_dataset.json"
THRESHOLDS = {
    "retrieval_recall": 0.85,
    "citation_fidelity": 0.90,
    "refusal_correctness": 1.0,
}


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    variant: str = "full_pipeline",
    run_name: str | None = None,
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    if not (INDEX_DIR / "ingestion_report.json").exists():
        IngestionPipeline().run()

    dataset = load_dataset(dataset_path)

    def predict_fn(query: str) -> dict:
        return predict_for_eval({"query": query}, variant=variant)

    with mlflow.start_run(run_name=run_name or variant):
        mlflow.log_param("variant", variant)
        results = evaluate(
            data=dataset,
            predict_fn=predict_fn,
            scorers=[
                retrieval_recall,
                citation_fidelity,
                retrieval_strategy_match,
                refusal_correctness,
                conflict_surfaced,
            ],
        )

    metrics = {}
    for key, value in results.metrics.items():
        metric_name = key.split("/")[0]
        metrics[metric_name] = max(metrics.get(metric_name, 0.0), float(value))

    failures = [
        metric
        for metric, threshold in THRESHOLDS.items()
        if metrics.get(metric, 0.0) < threshold
    ]
    metrics["passed"] = not failures
    metrics["failures"] = failures
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLflow evaluation for GroundedDoc Agent")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--variant", type=str, default="full_pipeline")
    parser.add_argument("--compare-ab", action="store_true")
    args = parser.parse_args()

    if args.compare_ab:
        variants = [
            "baseline_flat_rag",
            "full_pipeline",
        ]
        summary = {}
        for variant in variants:
            summary[variant] = run_evaluation(
                dataset_path=Path(args.dataset),
                variant=variant,
                run_name=variant,
            )
        print(json.dumps(summary, indent=2))
        return

    metrics = run_evaluation(
        dataset_path=Path(args.dataset),
        variant=args.variant,
        run_name=args.variant,
    )
    print(json.dumps(metrics, indent=2))
    if not metrics.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
