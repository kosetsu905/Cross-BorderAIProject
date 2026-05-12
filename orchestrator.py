import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from models import JobStatus, WorkflowType

logger = logging.getLogger(__name__)

CrewFunction = Callable[[dict[str, Any]], dict[str, Any]]


class MasterOrchestrator:
    """Central router and in-memory job tracker for CrewAI workflows."""

    def __init__(self) -> None:
        self._crews: dict[WorkflowType, CrewFunction] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        logger.info("Master orchestrator initialized.")

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
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "result": None,
            "error": None,
        }
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
            self._jobs[job_id]["status"] = JobStatus.RUNNING
            crew_function = self._crews[workflow_type]
            result = await asyncio.to_thread(crew_function, inputs)
            self._jobs[job_id]["status"] = JobStatus.COMPLETED
            self._jobs[job_id]["result"] = result if isinstance(result, dict) else {"raw": str(result)}
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._jobs[job_id]["status"] = JobStatus.FAILED
            self._jobs[job_id]["error"] = str(exc)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self._jobs.get(
            job_id,
            {
                "job_id": job_id,
                "status": JobStatus.FAILED,
                "result": None,
                "error": "Job not found",
            },
        )
