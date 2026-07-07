from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from job_store import JobStore
from models import JobStatus, WORKFLOW_INPUT_MODELS, WorkflowType
from runtime_config import RuntimeConfig, resolve_workflow_runtime_context
from services.workflow_guardrails import (
    GuardrailAction,
    WorkflowGuardrailService,
    decision_event_payload,
)
from utils.result_cache import build_workflow_cache_key, cache_enabled, cache_ttl_seconds


CrewFunction = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredWorkflow:
    workflow_type: WorkflowType
    crew_function: CrewFunction


@dataclass(frozen=True)
class PreparedWorkflowJob:
    job_id: str
    config_context: dict[str, Any]
    cache_hit: bool = False
    inputs: dict[str, Any] | None = None
    skip_execution: bool = False


class WorkflowRegistry:
    """Registry boundary between workflow engine code and CrewAI worker adapters."""

    def __init__(self) -> None:
        self._workflows: dict[WorkflowType, RegisteredWorkflow] = {}

    def register(self, workflow_type: WorkflowType, crew_function: CrewFunction) -> None:
        self._workflows[workflow_type] = RegisteredWorkflow(
            workflow_type=workflow_type,
            crew_function=crew_function,
        )

    def get(self, workflow_type: WorkflowType) -> RegisteredWorkflow:
        workflow = self._workflows.get(workflow_type)
        if workflow is None:
            raise ValueError(f"Workflow '{workflow_type.value}' is not registered.")
        return workflow

    def list_types(self) -> list[WorkflowType]:
        return list(self._workflows.keys())

    def has(self, workflow_type: WorkflowType) -> bool:
        return workflow_type in self._workflows


def validate_workflow_inputs(
    workflow_type: WorkflowType,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    input_model = WORKFLOW_INPUT_MODELS[workflow_type]
    return input_model.model_validate(inputs).model_dump()


class WorkflowExecutionEngine:
    """Common workflow submission boundary shared by local and Celery orchestrators."""

    def __init__(self, job_store: JobStore, runtime_config: RuntimeConfig) -> None:
        self._job_store = job_store
        self._runtime_config = runtime_config
        self._registry = WorkflowRegistry()

    @property
    def registry(self) -> WorkflowRegistry:
        return self._registry

    def register_crew(self, workflow_type: WorkflowType, crew_function: CrewFunction) -> None:
        self._registry.register(workflow_type, crew_function)

    def registered_workflows(self) -> list[WorkflowType]:
        return self._registry.list_types()

    def has_workflow(self, workflow_type: WorkflowType) -> bool:
        return self._registry.has(workflow_type)

    def crew_function(self, workflow_type: WorkflowType) -> CrewFunction:
        return self._registry.get(workflow_type).crew_function

    def prepare_job(
        self,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
        provider_credentials: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        backend: str,
        queued_message: str,
        queued_payload: dict[str, Any] | None = None,
    ) -> PreparedWorkflowJob:
        config_context = resolve_workflow_runtime_context(
            self._runtime_config,
            workflow_type,
            provider_credentials,
        )
        job_id = str(uuid.uuid4())
        guardrail = WorkflowGuardrailService()
        config_context["guardrails_config_version"] = guardrail.config.get("config_version")
        input_decision = guardrail.evaluate_input(
            workflow_type,
            inputs,
            context={**config_context, "job_id": job_id, "metadata": metadata or {}},
        )
        sanitized_inputs = input_decision.sanitized_payload if isinstance(input_decision.sanitized_payload, dict) else inputs
        if input_decision.action == GuardrailAction.BLOCK:
            self._job_store.create_job(job_id, workflow_type, sanitized_inputs)
            self._job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                result=None,
                error="Input blocked by workflow guardrail.",
            )
            self._job_store.log_event(
                job_id,
                "guardrail_input_evaluated",
                "Input guardrail blocked workflow submission.",
                decision_event_payload(input_decision),
            )
            return PreparedWorkflowJob(job_id, config_context, inputs=sanitized_inputs, skip_execution=True)

        inputs = sanitized_inputs
        cached_job_id = create_cached_workflow_job(
            self._job_store,
            workflow_type,
            inputs,
            config_context,
            metadata,
            backend,
        )
        if cached_job_id:
            return PreparedWorkflowJob(cached_job_id, config_context, cache_hit=True, inputs=inputs)

        cache_key = build_workflow_cache_key(workflow_type, inputs, config_context)
        self._job_store.create_job(job_id, workflow_type, inputs, cache_key=cache_key)
        self._job_store.log_event(
            job_id,
            "guardrail_input_evaluated",
            "Input guardrail evaluated.",
            decision_event_payload(input_decision),
        )
        payload = {
            "workflow_type": workflow_type.value,
            "backend": backend,
            **(queued_payload or {}),
        }
        self._job_store.log_event(job_id, "queued", queued_message, payload)
        return PreparedWorkflowJob(job_id, config_context, cache_hit=False, inputs=inputs)


def cache_hit_usage_summary(source_job: dict[str, Any]) -> dict[str, Any]:
    return {
        "usage_metrics": {
            "cache_hit": True,
            "source_job_id": source_job.get("job_id"),
            "source_total_tokens": source_job.get("total_tokens"),
            "source_cost_usd": source_job.get("cost_usd"),
        },
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
    }


def create_cached_workflow_job(
    job_store: JobStore,
    workflow_type: WorkflowType,
    inputs: dict[str, Any],
    config_context: dict[str, Any],
    metadata: dict[str, Any] | None,
    backend: str,
) -> str | None:
    cache_key = build_workflow_cache_key(workflow_type, inputs, config_context)
    if not cache_enabled(config_context, metadata):
        return None

    cached_job = job_store.find_cached_job(cache_key, cache_ttl_seconds(config_context))
    if cached_job is None:
        return None

    job_id = str(uuid.uuid4())
    job_store.create_job(job_id, workflow_type, inputs, cache_key=cache_key)
    job_store.update_job(
        job_id,
        status=JobStatus.COMPLETED,
        result=cached_job["result"],
        cache_hit=True,
        source_job_id=cached_job["job_id"],
        **cache_hit_usage_summary(cached_job),
        error=None,
    )
    job_store.log_event(
        job_id,
        "cache_hit",
        "Workflow result served from PostgreSQL cache.",
        {
            "workflow_type": workflow_type.value,
            "backend": backend,
            "source_job_id": cached_job["job_id"],
        },
    )
    logger.info(
        "Served workflow %s from cached job %s as job %s",
        workflow_type.value,
        cached_job["job_id"],
        job_id,
    )
    return job_id
