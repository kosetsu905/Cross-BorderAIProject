from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import desc

from database import SessionLocal
from db_models import JobRecord
from models import JobStatus
from runtime_config import load_runtime_config
from utils.observability import evaluation_span, flush_observability, init_observability, set_span_attributes


logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _has_grounding_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"source", "sources", "source_urls", "evidence", "evidence_ids", "citations"}:
                if item:
                    return True
            if _has_grounding_evidence(item):
                return True
    if isinstance(value, list):
        return any(_has_grounding_evidence(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return "http://" in lowered or "https://" in lowered or "source:" in lowered
    return False


def _score_job(record: JobRecord) -> dict[str, Any]:
    result = record.result or {}
    has_evidence = _has_grounding_evidence(result)
    has_error = bool(record.error)
    groundedness_score = 1.0 if has_evidence and not has_error else 0.5 if not has_error else 0.0
    hallucination_risk = round(1.0 - groundedness_score, 3)
    rag_relevance_score = groundedness_score if record.workflow_type in {"support", "content", "analytics"} else None
    return {
        "job_id": record.job_id,
        "workflow_type": record.workflow_type,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "scores": {
            "groundedness": groundedness_score,
            "hallucination_risk": hallucination_risk,
            "rag_relevance": rag_relevance_score,
        },
        "metrics": {
            "prompt_tokens": record.prompt_tokens or 0,
            "completion_tokens": record.completion_tokens or 0,
            "total_tokens": record.total_tokens or 0,
            "cost_usd": float(record.cost_usd or 0.0),
            "duration_seconds": float(record.duration_seconds or 0.0),
        },
        "flags": {
            "has_grounding_evidence": has_evidence,
            "has_error": has_error,
            "cache_hit": bool(record.cache_hit),
        },
    }


def _completed_jobs(limit: int, workflow_type: str | None) -> list[JobRecord]:
    with SessionLocal() as session:
        query = session.query(JobRecord).filter(JobRecord.status == JobStatus.COMPLETED.value)
        if workflow_type:
            query = query.filter(JobRecord.workflow_type == workflow_type)
        return list(query.order_by(desc(JobRecord.updated_at)).limit(limit).all())


def _log_mlflow(record: dict[str, Any], config_context: dict[str, Any]) -> None:
    tracking_uri = config_context.get("mlflow_tracking_uri")
    if not tracking_uri:
        return
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow is not installed; skipping MLflow eval logging.")
        return

    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(str(config_context.get("mlflow_experiment_name") or "cross-border-ai"))
    with mlflow.start_run(run_name=f"eval_{record['workflow_type']}_{record['job_id']}"):
        mlflow.log_params(
            {
                "job_id": record["job_id"],
                "workflow_type": record["workflow_type"],
                "eval_source": "offline_job_store",
            }
        )
        mlflow.log_metrics(
            {
                key: value
                for key, value in {
                    **record["metrics"],
                    **{
                        f"{key}_score": value
                        for key, value in record["scores"].items()
                        if value is not None
                    },
                }.items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.log_dict(_json_safe(record), "observability_eval.json")


def run_evaluations(limit: int, workflow_type: str | None, dry_run: bool) -> list[dict[str, Any]]:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config_context = load_runtime_config().as_context()
    init_observability("cross-border-observability-evals", config_context=config_context)

    records: list[dict[str, Any]] = []
    for job in _completed_jobs(limit, workflow_type):
        eval_record = _score_job(job)
        records.append(eval_record)
        if dry_run:
            continue
        with evaluation_span(
            "rag_groundedness",
            job_id=job.job_id,
            workflow_type=job.workflow_type,
            config_context=config_context,
            attributes={
                "groundedness": eval_record["scores"]["groundedness"],
                "hallucination_risk": eval_record["scores"]["hallucination_risk"],
                "rag_relevance": eval_record["scores"]["rag_relevance"],
            },
        ):
            set_span_attributes(
                {
                    **eval_record["metrics"],
                    "has_grounding_evidence": eval_record["flags"]["has_grounding_evidence"],
                    "cache_hit": eval_record["flags"]["cache_hit"],
                },
                config_context=config_context,
            )
        _log_mlflow(eval_record, config_context)

    flush_observability()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline observability evaluations for completed workflow jobs.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum completed jobs to evaluate.")
    parser.add_argument("--workflow-type", default=None, help="Optional workflow_type filter.")
    parser.add_argument("--dry-run", action="store_true", help="Print records without sending telemetry or MLflow logs.")
    args = parser.parse_args()
    records = run_evaluations(args.limit, args.workflow_type, args.dry_run)
    print(json.dumps(_json_safe(records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
