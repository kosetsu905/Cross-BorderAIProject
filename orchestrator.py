import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from celery.result import AsyncResult

from celery_worker.celery_app import celery_app
from job_store import InMemoryJobStore, JobStore
from models import JobStatus, WorkflowType

logger = logging.getLogger(__name__)

CrewFunction = Callable[[dict[str, Any]], dict[str, Any]]


class MasterOrchestrator:
    """Central router for local CrewAI execution with persistent job state."""

    def __init__(self, job_store: JobStore | None = None) -> None:
        self._crews: dict[WorkflowType, CrewFunction] = {}
        self._job_store = job_store or InMemoryJobStore()
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

    async def submit_job(self, workflow_type: WorkflowType, inputs: dict[str, Any]) -> str:
        if workflow_type not in self._crews:
            raise ValueError(f"Workflow '{workflow_type.value}' is not registered.")

        job_id = str(uuid.uuid4())
        self._job_store.create_job(job_id, workflow_type, inputs)
        asyncio.create_task(self._run_job(job_id, workflow_type, inputs))
        logger.info("Submitted job %s for workflow %s", job_id, workflow_type.value)
        return job_id

    async def _run_job(
        self,
        job_id: str,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
    ) -> None:
        try:
            self._job_store.update_job(job_id, status=JobStatus.RUNNING)
            crew_function = self._crews[workflow_type]
            result = await asyncio.to_thread(crew_function, inputs)
            self._job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=result if isinstance(result, dict) else {"raw": str(result)},
                error=None,
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                result=None,
                error=str(exc),
            )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self._job_store.get_job(job_id) or {
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "result": None,
            "error": "Job not found",
        }


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

    def __init__(self, job_store: JobStore) -> None:
        self._job_store = job_store
        logger.info("Celery orchestrator initialized with %s job store.", self.job_store_name)

    @property
    def job_store_name(self) -> str:
        return self._job_store.name

    @property
    def registered_workflows(self) -> list[WorkflowType]:
        return list(self.TASK_MAP.keys())

    async def submit_job(self, workflow_type: WorkflowType, inputs: dict[str, Any]) -> str:
        task_name = self.TASK_MAP.get(workflow_type)
        if task_name is None:
            raise ValueError(f"Workflow '{workflow_type.value}' is not registered.")

        job_id = str(uuid.uuid4())
        self._job_store.create_job(job_id, workflow_type, inputs)
        task = celery_app.send_task(task_name, args=[inputs], task_id=job_id)
        logger.info("Queued Celery job %s for workflow %s", task.id, workflow_type.value)
        return job_id

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
            self._job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=task_result if isinstance(task_result, dict) else {"raw": str(task_result)},
                error=None,
            )
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.COMPLETED,
                "result": task_result if isinstance(task_result, dict) else {"raw": str(task_result)},
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
