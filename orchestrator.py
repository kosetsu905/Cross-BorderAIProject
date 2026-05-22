import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from celery.result import AsyncResult

from celery_worker.celery_app import celery_app
from job_store import InMemoryJobStore, JobStore
from models import JobStatus, WorkflowType
from runtime_config import RuntimeConfig, apply_runtime_environment, merge_runtime_context
from utils.result_cache import build_workflow_cache_key, cache_enabled, cache_ttl_seconds
from utils.usage_tracking import build_usage_summary, monotonic_time, pop_usage_metrics
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, WorkflowProgressRecorder

logger = logging.getLogger(__name__)

CrewFunction = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


USAGE_FIELD_NAMES = {
    "usage_metrics",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "duration_seconds",
}


def _split_task_result(task_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        isinstance(task_result, dict)
        and isinstance(task_result.get("data"), dict)
        and isinstance(task_result.get("meta"), dict)
    ):
        usage_fields = {
            key: value
            for key, value in task_result["meta"].items()
            if key in USAGE_FIELD_NAMES
        }
        return task_result["data"], usage_fields

    result_payload = task_result if isinstance(task_result, dict) else {"raw": str(task_result)}
    return result_payload, {}


def _cache_hit_usage_summary(source_job: dict[str, Any]) -> dict[str, Any]:
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


def _maybe_create_cached_job(
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
    usage_summary = _cache_hit_usage_summary(cached_job)
    job_store.update_job(
        job_id,
        status=JobStatus.COMPLETED,
        result=cached_job["result"],
        cache_hit=True,
        source_job_id=cached_job["job_id"],
        **usage_summary,
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


class MasterOrchestrator:
    """Central router for local CrewAI execution with persistent job state."""

    def __init__(
        self,
        job_store: JobStore | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self._crews: dict[WorkflowType, CrewFunction] = {}
        self._job_store = job_store or InMemoryJobStore()
        self._runtime_config = runtime_config or RuntimeConfig()
        logger.info("Master orchestrator initialized with %s job store.", self.job_store_name)

    @property
    def job_store_name(self) -> str:
        return self._job_store.name

    @property
    def registered_workflows(self) -> list[WorkflowType]:
        return list(self._crews.keys())

    def register_crew(self, workflow_type: WorkflowType, crew_function: CrewFunction) -> None:
        self._crews[workflow_type] = crew_function
        logger.info("Registered crew: %s", workflow_type.value)

    async def submit_job(
        self,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
        provider_credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if workflow_type not in self._crews:
            raise ValueError(f"Workflow '{workflow_type.value}' is not registered.")

        config_context = merge_runtime_context(self._runtime_config, provider_credentials)
        cached_job_id = _maybe_create_cached_job(
            self._job_store,
            workflow_type,
            inputs,
            config_context,
            metadata,
            "local",
        )
        if cached_job_id:
            return cached_job_id

        job_id = str(uuid.uuid4())
        cache_key = build_workflow_cache_key(workflow_type, inputs, config_context)
        self._job_store.create_job(job_id, workflow_type, inputs, cache_key=cache_key)
        self._job_store.log_event(
            job_id,
            "queued",
            "Local background execution scheduled.",
            {"workflow_type": workflow_type.value, "backend": "local"},
        )
        asyncio.create_task(self._run_job(job_id, workflow_type, inputs, config_context))
        logger.info("Submitted job %s for workflow %s", job_id, workflow_type.value)
        return job_id

    async def _run_job(
        self,
        job_id: str,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
        config_context: dict[str, Any],
    ) -> None:
        try:
            self._job_store.update_job(job_id, status=JobStatus.RUNNING)
            self._job_store.log_event(
                job_id,
                "running",
                "Workflow execution started.",
                {"workflow_type": workflow_type.value, "backend": "local"},
            )
            crew_function = self._crews[workflow_type]
            config_context[PROGRESS_CONTEXT_KEY] = WorkflowProgressRecorder(
                job_id=job_id,
                workflow_type=workflow_type.value,
                job_store=self._job_store,
                backend="local",
            )
            apply_runtime_environment(config_context)
            started_at = monotonic_time()
            result = await asyncio.to_thread(crew_function, inputs, config_context)
            clean_result, usage_metrics = pop_usage_metrics(result)
            usage_summary = build_usage_summary(
                usage_metrics,
                monotonic_time() - started_at,
                config_context,
            )
            self._job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=clean_result if isinstance(clean_result, dict) else {"raw": str(clean_result)},
                **usage_summary,
                error=None,
            )
            self._job_store.log_event(
                job_id,
                "completed",
                "Workflow execution completed.",
                {
                    "workflow_type": workflow_type.value,
                    "backend": "local",
                    "total_tokens": usage_summary.get("total_tokens"),
                    "cost_usd": usage_summary.get("cost_usd"),
                    "duration_seconds": usage_summary.get("duration_seconds"),
                },
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                result=None,
                duration_seconds=None,
                error=str(exc),
            )
            self._job_store.log_event(
                job_id,
                "failed",
                "Workflow execution failed.",
                {"workflow_type": workflow_type.value, "backend": "local", "error": str(exc)},
            )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self._job_store.get_job(job_id) or {
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "result": None,
            "error": "Job not found",
        }

    def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
        return self._job_store.get_job_events(job_id)


class CeleryOrchestrator:
    """Celery-backed workflow router for Redis/message-broker deployments."""

    TASK_MAP: dict[WorkflowType, str] = {
        WorkflowType.MARKETING: "workflow.marketing",
        WorkflowType.CONTENT: "workflow.content",
        WorkflowType.SUPPORT: "workflow.support",
        WorkflowType.ANALYTICS: "workflow.analytics",
        WorkflowType.SALES_IMPROVEMENT: "workflow.sales_improvement",
        WorkflowType.BIZDEV: "workflow.bizdev",
        WorkflowType.SCHEDULER: "workflow.scheduler",
    }

    def __init__(self, job_store: JobStore, runtime_config: RuntimeConfig | None = None) -> None:
        self._job_store = job_store
        self._runtime_config = runtime_config or RuntimeConfig()
        logger.info("Celery orchestrator initialized with %s job store.", self.job_store_name)

    @property
    def job_store_name(self) -> str:
        return self._job_store.name

    @property
    def registered_workflows(self) -> list[WorkflowType]:
        return list(self.TASK_MAP.keys())

    async def submit_job(
        self,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
        provider_credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        task_name = self.TASK_MAP.get(workflow_type)
        if task_name is None:
            raise ValueError(f"Workflow '{workflow_type.value}' is not registered.")

        config_context = merge_runtime_context(self._runtime_config, provider_credentials)
        cached_job_id = _maybe_create_cached_job(
            self._job_store,
            workflow_type,
            inputs,
            config_context,
            metadata,
            "celery",
        )
        if cached_job_id:
            return cached_job_id

        job_id = str(uuid.uuid4())
        cache_key = build_workflow_cache_key(workflow_type, inputs, config_context)
        self._job_store.create_job(job_id, workflow_type, inputs, cache_key=cache_key)
        self._job_store.log_event(
            job_id,
            "queued",
            "Celery task queued.",
            {"workflow_type": workflow_type.value, "backend": "celery", "task_name": task_name},
        )
        task = celery_app.send_task(
            task_name,
            args=[inputs],
            kwargs={"config_context": config_context},
            task_id=job_id,
        )
        logger.info("Queued Celery job %s for workflow %s", task.id, workflow_type.value)
        return job_id

    def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
        return self._job_store.get_job_events(job_id)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        result = AsyncResult(job_id, app=celery_app)

        if result.state == "PENDING":
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "result": None,
                "error": None,
            }

        if result.state in {"STARTED", "PROGRESS", "RETRY"}:
            progress = result.info if isinstance(result.info, dict) else None
            self._job_store.update_job(
                job_id,
                status=JobStatus.RUNNING,
                result=progress,
                error=None,
            )
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.RUNNING,
                "result": progress,
                "error": None,
            }

        if result.state == "SUCCESS":
            task_result = result.result
            result_payload, usage_fields = _split_task_result(task_result)
            self._job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=result_payload,
                **usage_fields,
                error=None,
            )
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.COMPLETED,
                "result": result_payload,
                **usage_fields,
                "error": None,
            }

        if result.state == "FAILURE":
            self._job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                result=None,
                error=str(result.result),
            )
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.FAILED,
                "result": None,
                "error": str(result.result),
            }

        self._job_store.update_job(
            job_id,
            status=JobStatus.RUNNING,
            result={"celery_state": result.state},
            error=None,
        )
        return {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "result": {"celery_state": result.state},
            "error": None,
        }
