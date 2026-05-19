from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


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


class StrictInputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketingInputs(StrictInputModel):
    product_category: str = Field(..., min_length=1)
    product_usp: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    budget: str = Field(..., min_length=1)


class ContentInputs(StrictInputModel):
    subject: str = Field(..., min_length=1)
    product_category: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    target_languages: list[str] = Field(..., min_length=1)
    platforms: list[str] = Field(..., min_length=1)


class SupportInputs(StrictInputModel):
    customer: str = Field(..., min_length=1)
    person: str = Field(..., min_length=1)
    inquiry: str = Field(..., min_length=1)


class AnalyticsInputs(StrictInputModel):
    product_category: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    date_range: str = Field(..., min_length=1)
    currency: str = Field(..., min_length=1)


class BizDevInputs(StrictInputModel):
    product_category: str = Field(..., min_length=1)
    partnership_type: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    target_languages: list[str] = Field(..., min_length=1)
    key_decision_maker_roles: str = Field(..., min_length=1)


class SchedulerInputs(StrictInputModel):
    event_type: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    event_list: str = Field(..., min_length=1)
    preferred_launch_window: str = Field(..., min_length=1)


class SalesImprovementInputs(StrictInputModel):
    product_category: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    current_avg_conversion: str = Field(..., min_length=1)
    target_conversion: str = Field(..., min_length=1)
    date_range: str = Field(..., min_length=1)


WORKFLOW_INPUT_MODELS: dict[WorkflowType, type[StrictInputModel]] = {
    WorkflowType.MARKETING: MarketingInputs,
    WorkflowType.CONTENT: ContentInputs,
    WorkflowType.SUPPORT: SupportInputs,
    WorkflowType.ANALYTICS: AnalyticsInputs,
    WorkflowType.BIZDEV: BizDevInputs,
    WorkflowType.SCHEDULER: SchedulerInputs,
    WorkflowType.SALES_IMPROVEMENT: SalesImprovementInputs,
}


class WorkflowRequest(BaseModel):
    workflow_type: WorkflowType
    inputs: dict[str, Any] = Field(..., description="Workflow-specific input parameters")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional tracing or request context metadata",
    )

    @model_validator(mode="after")
    def validate_workflow_inputs(self) -> "WorkflowRequest":
        input_model = WORKFLOW_INPUT_MODELS[self.workflow_type]
        try:
            validated_inputs = input_model.model_validate(self.inputs)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid inputs for workflow '{self.workflow_type.value}': {exc}"
            ) from exc

        self.inputs = validated_inputs.model_dump()
        return self


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None
