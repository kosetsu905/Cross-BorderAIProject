from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.auth import verify_bearer_token
from models import JobEventResponse, JobResponse, JobStatus, WorkflowRequest

AuthDependency = Annotated[None, Depends(verify_bearer_token)]


def create_router(orchestrator: object) -> APIRouter:
    router = APIRouter()

    @router.post("/api/v1/workflow", response_model=JobResponse)
    async def trigger_workflow(req: WorkflowRequest, _: AuthDependency) -> JobResponse:
        try:
            provider_credentials = (
                req.provider_credentials.model_dump(exclude_none=True)
                if req.provider_credentials
                else None
            )
            job_id = await orchestrator.submit_job(
                req.workflow_type,
                req.inputs,
                provider_credentials=provider_credentials,
                metadata=req.metadata,
            )
            return JobResponse(**orchestrator.get_job_status(job_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/v1/workflow/{job_id}", response_model=JobResponse)
    async def get_workflow_status(job_id: str, _: AuthDependency) -> JobResponse:
        job_data = orchestrator.get_job_status(job_id)
        if job_data.get("status") == JobStatus.FAILED and job_data.get("error"):
            raise HTTPException(status_code=404, detail=job_data["error"])
        return JobResponse(**job_data)

    @router.get("/api/v1/workflow/{job_id}/events", response_model=list[JobEventResponse])
    async def get_workflow_events(job_id: str, _: AuthDependency) -> list[JobEventResponse]:
        return [
            JobEventResponse(**event)
            for event in orchestrator.get_job_events(job_id)
        ]

    @router.get("/health")
    async def health_check() -> dict[str, object]:
        health: dict[str, object] = {
            "status": "healthy",
            "workflow_backend": orchestrator.__class__.__name__,
            "registered_workflows": [
                workflow.value for workflow in orchestrator.registered_workflows
            ],
        }
        job_store_name = getattr(orchestrator, "job_store_name", None)
        if job_store_name:
            health["job_store"] = job_store_name
        return health

    return router
