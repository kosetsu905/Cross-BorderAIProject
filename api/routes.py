from fastapi import APIRouter, HTTPException

from models import JobResponse, JobStatus, WorkflowRequest
from orchestrator import MasterOrchestrator


def create_router(orchestrator: MasterOrchestrator) -> APIRouter:
    router = APIRouter()

    @router.post("/api/v1/workflow", response_model=JobResponse)
    async def trigger_workflow(req: WorkflowRequest) -> JobResponse:
        try:
            job_id = await orchestrator.submit_job(req.workflow_type, req.inputs)
            return JobResponse(job_id=job_id, status=JobStatus.PENDING)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/v1/workflow/{job_id}", response_model=JobResponse)
    async def get_workflow_status(job_id: str) -> JobResponse:
        job_data = orchestrator.get_job_status(job_id)
        if job_data.get("status") == JobStatus.FAILED and job_data.get("error"):
            raise HTTPException(status_code=404, detail=job_data["error"])
        return JobResponse(**job_data)

    @router.get("/health")
    async def health_check() -> dict[str, object]:
        return {
            "status": "healthy",
            "registered_workflows": [
                workflow.value for workflow in orchestrator.registered_workflows
            ],
        }

    return router
