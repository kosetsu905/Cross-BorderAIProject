from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

SUPPORT_AGENT_PROMPT_FIELDS = ("role", "goal", "backstory")
SUPPORT_TASK_PROMPT_FIELDS = ("description", "expected_output")
SUPPORT_GUIDELINES = (
    "The response must be accurate, empathetic, and written in the customer's language. "
    "It must not invent order status, product facts, delivery dates, refund approval, return labels, "
    "tracking numbers, or policy exceptions. It must preserve any required human handoff."
)
TRACE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SINGLE_BRACE_VARIABLE_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_PROMPT_CACHE_LOCK = RLock()


class PromptLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt_name: str = Field(..., min_length=1)
    alias: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)
    source: str = Field(..., pattern=r"^(mlflow|cache)$")


class LoadedSupportPrompts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    agents: dict[str, dict[str, Any]]
    tasks: dict[str, dict[str, Any]]
    lineage: dict[str, PromptLineage] = Field(default_factory=dict)


def configure_mlflow(config_context: dict[str, Any]) -> tuple[Any, str]:
    """Configure the official MLflow client and return its experiment ID."""
    import mlflow

    if os.getenv("MLFLOW_ADMIN_USERNAME") and not os.getenv("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["MLFLOW_ADMIN_USERNAME"]
    if os.getenv("MLFLOW_ADMIN_PASSWORD") and not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["MLFLOW_ADMIN_PASSWORD"]
    tracking_uri = str(config_context.get("mlflow_tracking_uri") or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = str(config_context.get("mlflow_experiment_name") or "cross-border-ai")
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    experiment_id = str(experiment.experiment_id)
    from mlflow.entities.trace_location import MlflowExperimentLocation

    mlflow.tracing.set_destination(
        MlflowExperimentLocation(experiment_id=experiment_id)
    )
    return mlflow, experiment_id


def load_support_prompts(
    agents: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    config_context: dict[str, Any],
) -> LoadedSupportPrompts:
    """Load support prompt fields from MLflow production aliases with safe fallbacks."""
    if not bool(config_context.get("mlflow_prompt_registry_enabled")):
        return LoadedSupportPrompts(agents=agents, tasks=tasks)

    loaded_agents: dict[str, dict[str, Any]] = {}
    loaded_tasks: dict[str, dict[str, Any]] = {}
    lineage: dict[str, PromptLineage] = {}
    for name, config in agents.items():
        loaded, prompt_lineage = _load_prompt_config(
            prompt_kind="agent",
            config_name=name,
            local_config=config,
            prompt_fields=SUPPORT_AGENT_PROMPT_FIELDS,
            config_context=config_context,
        )
        loaded_agents[name] = loaded
        lineage[prompt_lineage.prompt_name] = prompt_lineage
    for name, config in tasks.items():
        loaded, prompt_lineage = _load_prompt_config(
            prompt_kind="task",
            config_name=name,
            local_config=config,
            prompt_fields=SUPPORT_TASK_PROMPT_FIELDS,
            config_context=config_context,
        )
        loaded_tasks[name] = loaded
        lineage[prompt_lineage.prompt_name] = prompt_lineage

    _record_prompt_lineage(lineage, config_context)
    return LoadedSupportPrompts(agents=loaded_agents, tasks=loaded_tasks, lineage=lineage)


def bootstrap_support_governance(
    *,
    agents: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    dataset_records: list[dict[str, Any]],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    """Seed official MLflow prompts, dataset, and built-in scorers idempotently."""
    mlflow, experiment_id = configure_mlflow(config_context)
    alias = str(config_context.get("mlflow_support_prompt_alias") or "production")
    prompt_report: list[dict[str, Any]] = []
    for name, config in agents.items():
        prompt_report.append(
            _register_prompt_if_missing(
                mlflow=mlflow,
                prompt_kind="agent",
                config_name=name,
                local_config=config,
                prompt_fields=SUPPORT_AGENT_PROMPT_FIELDS,
                alias=alias,
            )
        )
    for name, config in tasks.items():
        prompt_report.append(
            _register_prompt_if_missing(
                mlflow=mlflow,
                prompt_kind="task",
                config_name=name,
                local_config=config,
                prompt_fields=SUPPORT_TASK_PROMPT_FIELDS,
                alias=alias,
            )
        )

    dataset_name = str(
        config_context.get("mlflow_support_evaluation_dataset_name") or "support-governance"
    )
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=dataset_name)
        dataset_status = "existing"
    except Exception:
        dataset = mlflow.genai.datasets.create_dataset(
            name=dataset_name,
            experiment_id=[experiment_id],
            tags={"workflow": "support", "purpose": "governance"},
        )
        dataset_status = "created"
    dataset.merge_records(dataset_records)

    scorer_report = register_official_support_scorers(config_context, experiment_id)
    return {
        "experiment_id": experiment_id,
        "prompts": prompt_report,
        "dataset": {
            "name": dataset_name,
            "dataset_id": dataset.dataset_id,
            "status": dataset_status,
            "record_count": len(dataset_records),
        },
        "scorers": scorer_report,
    }


def build_official_support_scorers(model: str) -> list[Any]:
    """Build only MLflow-provided support quality scorers."""
    from mlflow.genai.scorers import (
        Completeness,
        ConversationalRoleAdherence,
        Guidelines,
        PIIDetection,
        RelevanceToQuery,
        Safety,
        UserFrustration,
    )

    return [
        RelevanceToQuery(model=model),
        Completeness(model=model),
        Safety(model=model),
        PIIDetection(),
        ConversationalRoleAdherence(model=model),
        UserFrustration(model=model),
        Guidelines(name="support_guidelines", guidelines=SUPPORT_GUIDELINES, model=model),
    ]


def register_official_support_scorers(
    config_context: dict[str, Any],
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Register official scorers and optionally activate official online judges."""
    from mlflow.genai.scorers import ScorerSamplingConfig, get_scorer

    model = str(
        config_context.get("mlflow_genai_judge_default_model")
        or "openrouter:/qwen/qwen3.7-plus"
    )
    automatic_enabled = bool(config_context.get("mlflow_automatic_evaluation_enabled"))
    if automatic_enabled and not model.startswith("gateway:/"):
        raise ValueError(
            "MLflow Automatic Evaluation requires an AI Gateway judge model such as "
            "gateway:/support-judge. Disable MLFLOW_AUTOMATIC_EVALUATION_ENABLED until a "
            "Gateway endpoint is configured."
        )

    report: list[dict[str, Any]] = []
    for scorer in build_official_support_scorers(model):
        try:
            registered = get_scorer(name=scorer.name, experiment_id=experiment_id)
            status = "existing"
        except Exception:
            registered = scorer.register(name=scorer.name, experiment_id=experiment_id)
            status = "registered"

        automatic_status = "disabled"
        if automatic_enabled and scorer.name != "pii_detection":
            registered.start(
                experiment_id=experiment_id,
                sampling_config=ScorerSamplingConfig(
                    sample_rate=1.0,
                    filter_string="metadata.workflow_type = 'support'",
                ),
            )
            automatic_status = "active"
        report.append(
            {
                "name": scorer.name,
                "registration": status,
                "automatic_evaluation": automatic_status,
            }
        )
    return report


def log_support_review(
    *,
    job_id: str,
    decision: str,
    reviewer: str | None,
    rationale: str | None,
    approved_response: str | None,
    original_response: str | None,
    config_context: dict[str, Any],
) -> str | None:
    """Attach official human feedback and edited expectations to a support trace."""
    if not bool(config_context.get("mlflow_tracing_enabled")):
        return None
    if not TRACE_IDENTIFIER_RE.fullmatch(job_id):
        logger.warning("Skipping MLflow support feedback for invalid job identifier")
        return None
    try:
        mlflow, experiment_id = configure_mlflow(config_context)
        traces = mlflow.search_traces(
            locations=[experiment_id],
            filter_string=f"metadata.job_id = '{job_id}'",
            max_results=1,
            order_by=["timestamp_ms DESC"],
            return_type="list",
            flush=True,
        )
        if not traces:
            logger.warning("No MLflow trace found for support job %s", job_id)
            return None
        trace_id = str(traces[0].info.trace_id)
        from mlflow.entities import AssessmentSource

        source = AssessmentSource(
            source_type="HUMAN",
            source_id=(reviewer or "support-reviewer")[:255],
        )
        mlflow.log_feedback(
            trace_id=trace_id,
            name="support_review_decision",
            value=decision,
            source=source,
            rationale=rationale,
            metadata={"job_id": job_id, "workflow_type": "support"},
        )
        if approved_response and approved_response != original_response:
            from utils.observability import redact_observability_payload

            redacted_response = redact_observability_payload(
                approved_response,
                capture_raw=False,
            )
            mlflow.log_expectation(
                trace_id=trace_id,
                name="approved_support_response",
                value=redacted_response,
                source=source,
                metadata={"job_id": job_id, "workflow_type": "support"},
            )
        return trace_id
    except Exception:
        logger.warning("Failed to log MLflow support review for job %s", job_id, exc_info=True)
        return None


def _load_prompt_config(
    *,
    prompt_kind: str,
    config_name: str,
    local_config: dict[str, Any],
    prompt_fields: tuple[str, ...],
    config_context: dict[str, Any],
) -> tuple[dict[str, Any], PromptLineage]:
    prompt_name = _prompt_name(prompt_kind, config_name)
    alias = str(config_context.get("mlflow_support_prompt_alias") or "production")
    cache_path = _prompt_cache_path(prompt_name, config_context)
    registry_error: Exception | None = None
    try:
        mlflow, _ = configure_mlflow(config_context)
        prompt = mlflow.genai.load_prompt(
            f"prompts:/{prompt_name}@{alias}",
            cache_ttl_seconds=60,
        )
        prompt_values = _parse_prompt_template(prompt.to_single_brace_format(), prompt_fields)
        merged = {**local_config, **prompt_values}
        lineage = PromptLineage(
            prompt_name=prompt_name,
            alias=alias,
            version=int(prompt.version),
            source="mlflow",
        )
        _write_prompt_cache(cache_path, merged, lineage)
        return merged, lineage
    except Exception as exc:
        registry_error = exc
        logger.warning(
            "MLflow prompt %s@%s is unavailable; using the last known prompt cache.",
            prompt_name,
            alias,
            exc_info=True,
        )

    cached = _read_prompt_cache(cache_path, prompt_fields)
    if cached is not None:
        cached_config, cached_lineage = cached
        return {**local_config, **cached_config}, cached_lineage.model_copy(update={"source": "cache"})
    raise RuntimeError(
        f"MLflow prompt {prompt_name}@{alias} is unavailable and no governed cache exists"
    ) from registry_error


def _register_prompt_if_missing(
    *,
    mlflow: Any,
    prompt_kind: str,
    config_name: str,
    local_config: dict[str, Any],
    prompt_fields: tuple[str, ...],
    alias: str,
) -> dict[str, Any]:
    prompt_name = _prompt_name(prompt_kind, config_name)
    try:
        prompt = mlflow.genai.load_prompt(prompt_name)
        status = "existing"
    except Exception:
        prompt_values = {field: local_config[field] for field in prompt_fields}
        prompt = mlflow.genai.register_prompt(
            name=prompt_name,
            template=_serialize_prompt_template(prompt_values),
            commit_message="Initial support governance import from project YAML",
            tags={
                "workflow": "support",
                "prompt_kind": prompt_kind,
                "config_name": config_name,
            },
        )
        status = "created"

    try:
        mlflow.genai.load_prompt(f"prompts:/{prompt_name}@{alias}")
    except Exception:
        mlflow.genai.set_prompt_alias(prompt_name, alias, int(prompt.version))
    return {
        "name": prompt_name,
        "version": int(prompt.version),
        "alias": alias,
        "status": status,
    }


def _record_prompt_lineage(
    lineage: dict[str, PromptLineage],
    config_context: dict[str, Any],
) -> None:
    try:
        mlflow, _ = configure_mlflow(config_context)
        span = mlflow.get_current_active_span()
        if span is None:
            return
        compact = {
            name: {
                "version": item.version,
                "alias": item.alias,
                "source": item.source,
            }
            for name, item in lineage.items()
        }
        mlflow.update_current_trace(
            metadata={"support_prompt_versions": json.dumps(compact, sort_keys=True)}
        )
        span.set_attribute("support_prompt_versions", json.dumps(compact, sort_keys=True))
    except Exception:
        logger.debug("Failed to attach MLflow prompt lineage to the current trace", exc_info=True)


def _prompt_name(prompt_kind: str, config_name: str) -> str:
    return f"support-{prompt_kind}-{config_name.replace('_', '-')}"


def _serialize_prompt_template(prompt_values: dict[str, Any]) -> str:
    serialized = json.dumps(prompt_values, ensure_ascii=False, indent=2, sort_keys=True)
    return SINGLE_BRACE_VARIABLE_RE.sub(r"{{\1}}", serialized)


def _parse_prompt_template(template: Any, prompt_fields: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(template, str):
        raise ValueError("Support prompt registry entries must use a text template")
    parsed = json.loads(template)
    if not isinstance(parsed, dict):
        raise ValueError("Support prompt registry entry must contain a JSON object")
    values: dict[str, str] = {}
    for field in prompt_fields:
        value = parsed.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Support prompt registry entry is missing string field '{field}'")
        values[field] = value
    return values


def _prompt_cache_path(prompt_name: str, config_context: dict[str, Any]) -> Path:
    cache_dir = Path(
        str(config_context.get("mlflow_prompt_cache_dir") or "artifacts/mlflow_prompt_cache")
    )
    return cache_dir / f"{prompt_name}.json"


def _write_prompt_cache(path: Path, config: dict[str, Any], lineage: PromptLineage) -> None:
    payload = {
        "config": config,
        "lineage": lineage.model_dump(),
    }
    with _PROMPT_CACHE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(path)


def _read_prompt_cache(
    path: Path,
    prompt_fields: tuple[str, ...],
) -> tuple[dict[str, str], PromptLineage] | None:
    try:
        with _PROMPT_CACHE_LOCK:
            payload = json.loads(path.read_text(encoding="utf-8"))
        config = _parse_prompt_template(json.dumps(payload["config"]), prompt_fields)
        lineage = PromptLineage.model_validate(payload["lineage"])
        return config, lineage
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
