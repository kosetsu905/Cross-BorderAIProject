from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowType(str, Enum):
    MARKETING = "marketing"
    CONTENT = "content"
    SUPPORT = "support"
    ANALYTICS = "analytics"
    BIZDEV = "bizdev"
    SCHEDULER = "scheduler"
    SALES_IMPROVEMENT = "sales_improvement"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowRequest(BaseModel):
    workflow_type: WorkflowType
    inputs: dict[str, Any] = Field(..., description="Workflow-specific input parameters")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional tracing or request context metadata",
    )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None
