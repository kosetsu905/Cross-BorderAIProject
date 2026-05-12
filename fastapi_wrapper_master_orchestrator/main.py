import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from business_development.crews.bizdev_crew import run_bizdev_crew
from fastapi_wrapper_master_orchestrator.models import (
    JobResponse,
    JobStatus,
    WorkflowRequest,
    WorkflowType,
)
from fastapi_wrapper_master_orchestrator.orchestrator import MasterOrchestrator

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Cross-Border E-Commerce AI Suite", version="0.1.0")
orchestrator = MasterOrchestrator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator.register_crew(WorkflowType.BIZDEV, run_bizdev_crew)


@app.post("/api/v1/workflow", response_model=JobResponse)
async def trigger_workflow(req: WorkflowRequest) -> JobResponse:
    try:
        job_id = await orchestrator.submit_job(req.workflow_type, req.inputs)
        return JobResponse(job_id=job_id, status=JobStatus.PENDING)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/workflow/{job_id}", response_model=JobResponse)
async def get_workflow_status(job_id: str) -> JobResponse:
    job_data = orchestrator.get_job_status(job_id)
    if job_data.get("status") == JobStatus.FAILED and job_data.get("error"):
        raise HTTPException(status_code=404, detail=job_data["error"])
    return JobResponse(**job_data)


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {
        "status": "healthy",
        "registered_workflows": [workflow.value for workflow in orchestrator.registered_workflows],
    }
