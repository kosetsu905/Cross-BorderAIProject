from enum import Enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class WorkflowType(str, Enum):
    MARKETING = "marketing"
    CONTENT = "content"
    SUPPORT = "support"
    ANALYTICS = "analytics"
    BIZDEV = "bizdev"
    SCHEDULER = "scheduler"
    SALES_IMPROVEMENT = "sales_improvement"


WORKFLOW_ROUTE_TYPE = "workflow_route"


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
    target_languages: list[str] | None = Field(
        None,
        description=(
            "Optional target language codes or names. If omitted, Marketing should use "
            "the primary local language for each target market."
        ),
    )


class ContentInputs(StrictInputModel):
    subject: str = Field(..., min_length=1)
    product_category: str = Field(..., min_length=1)
    product_features: str | None = Field(
        None,
        description=(
            "Optional product features or selling points to ground generated content "
            "in the user's actual product instead of broad category-level discussion."
        ),
    )
    target_markets: str = Field(..., min_length=1)
    target_languages: list[str] = Field(..., min_length=1)
    platforms: list[str] = Field(..., min_length=1)
    brand_voice: str | None = Field(
        None,
        description="Optional brand voice guidance for localized content and visual assets.",
    )
    brand_name: str | None = Field(
        None,
        description="Optional brand or product entity name for transparent Reddit GEO content.",
    )
    product_url: str | None = Field(
        None,
        description="Optional http/https product URL used as a single contextual Reddit reference.",
    )
    primary_keywords: list[str] | None = Field(
        None,
        description="Optional seed SEO keywords for multi-engine metadata generation.",
    )
    generate_reddit_geo: bool = Field(
        False,
        description=(
            "When true, Content Creation returns a Reddit-ready GEO post package for "
            "manual review and publication."
        ),
    )
    generate_visual_assets: bool = Field(
        False,
        description="When true, Content Creation may call the configured OpenAI Image API.",
    )
    image_generation_count: int = Field(
        1,
        ge=1,
        le=4,
        description="Number of images to generate per language/market when visual generation is enabled.",
    )
    image_quality: str = Field(
        "low",
        min_length=1,
        description="OpenAI image generation quality setting, such as auto, low, medium, or high.",
    )
    image_size: str = Field(
        "1024x1024",
        min_length=1,
        description="OpenAI image generation size setting, such as 1024x1024.",
    )

    @field_validator("brand_name", "product_url", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("product_url")
    @classmethod
    def _validate_product_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.lower()
        if not (normalized.startswith("http://") or normalized.startswith("https://")):
            raise ValueError("product_url must start with http:// or https://")
        return value


class SupportInputs(StrictInputModel):
    customer: str = Field(..., min_length=1)
    person: str = Field(..., min_length=1)
    inquiry: str = Field(..., min_length=1)
    ticket_id: str | None = None
    customer_email: str | None = None
    phone_number: str | None = None
    inquiry_text: str | None = None
    order_id: str | None = None
    item_sku: str | None = None
    return_reason: str | None = None
    order_history: dict[str, Any] | None = None
    customer_tier: str | None = None
    product_category: str | None = None
    product_category_hint: str | None = None
    use_case: str | None = None
    use_case_if_provided: str | None = None
    order_id_if_provided: str | None = None
    region: str | None = None
    detected_language: str | None = None
    language_plan: str | None = None
    channel: str | None = None
    session_id: str | None = None
    channel_thread_id: str | None = None
    channel_message_id: str | None = None
    sender_profile: dict[str, Any] | None = None
    attachments: list[dict[str, Any]] | None = None
    conversation_history: list[dict[str, Any]] | None = None


class AnalyticsInputs(StrictInputModel):
    product_category: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    date_range: str = Field(..., min_length=1)
    currency: str | None = Field(
        None,
        min_length=1,
        description="ISO 4217 reporting currency. Defaults from base_currency or USD.",
    )
    base_currency: str | None = Field(
        None,
        min_length=1,
        description="Analytics 1.1 alias for currency; normalized into currency.",
    )
    chatbi_query: str | None = None
    historical_metrics: dict[str, Any] | list[dict[str, Any]] | str | None = None
    channel_metrics: dict[str, Any] | list[dict[str, Any]] | str | None = None
    sku: str | None = None
    campaign_id: str | None = None
    forecasted_demand: int | None = Field(None, ge=0)
    price_adjustment: str | None = None
    alert_message: str | None = None
    low_stock_forecast: bool = False
    conversion_anomaly: bool = False
    macro_risk: bool = False
    critical_alert: bool = False

    @field_validator(
        "currency",
        "base_currency",
        "chatbi_query",
        "sku",
        "campaign_id",
        "price_adjustment",
        "alert_message",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def normalize_currency_alias(self) -> "AnalyticsInputs":
        reporting_currency = (self.currency or self.base_currency or "USD").strip().upper()
        self.currency = reporting_currency
        self.base_currency = (self.base_currency or reporting_currency).strip().upper()
        return self


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


class ProviderCredentials(StrictInputModel):
    llm_provider: str | None = None
    llm_profile: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str | None = None
    llm_base_url: str | None = None
    llm_disable_reasoning: bool | None = None
    crewai_memory_enabled: bool | None = None
    crewai_memory_workflows: str | None = None
    workflow_context_max_chars: int | None = Field(None, ge=1000, le=50000)
    task_context_max_chars: int | None = Field(None, ge=500, le=20000)
    workflow_model_tiering_enabled: bool | None = None
    workflow_worker_llm_profile: str | None = None
    workflow_reviewer_llm_profile: str | None = None
    tool_cache_enabled: bool | None = None
    tool_cache_ttl_seconds: int | None = Field(None, ge=0, le=604800)
    tool_execution_async_enabled: bool | None = None
    content_image_model: str | None = None
    content_image_scoring_model: str | None = None
    content_image_artifact_dir: str | None = None
    serper_api_key: str | None = None
    support_serper_pre_sales_enabled: bool | None = None
    support_serper_order_fulfillment_enabled: bool | None = None
    support_serper_post_sales_enabled: bool | None = None
    shopify_store_domain: str | None = None
    shopify_admin_access_token: str | None = None
    shopify_api_version: str | None = None
    amazon_sp_api_endpoint: str | None = None
    amazon_sp_api_access_token: str | None = None
    amazon_marketplace_ids: str | None = None
    support_knowledge_dir: str | None = None
    google_ads_developer_token: str | None = None
    google_ads_access_token: str | None = None
    google_ads_customer_id: str | None = None
    gmail_access_token: str | None = None
    gmail_client_id: str | None = None
    gmail_client_secret: str | None = None
    gmail_refresh_token: str | None = None
    gmail_sender_email: str | None = None
    gmail_send_enabled: bool | None = None
    gmail_watch_topic_name: str | None = None
    gmail_watch_label_ids: str | None = None
    gmail_sync_enabled: bool | None = None
    whatsapp_access_token: str | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_business_account_id: str | None = None
    whatsapp_verify_token: str | None = None
    whatsapp_app_secret: str | None = None
    whatsapp_send_enabled: bool | None = None
    whatsapp_graph_api_version: str | None = None
    whatsapp_provider: str | None = None
    ycloud_api_key: str | None = None
    ycloud_whatsapp_from: str | None = None
    ycloud_waba_id: str | None = None
    ycloud_base_url: str | None = None
    ycloud_webhook_secret: str | None = None
    meta_access_token: str | None = None
    meta_ad_account_id: str | None = None
    meta_page_id: str | None = None
    tiktok_access_token: str | None = None
    tiktok_advertiser_id: str | None = None
    workflow_async_execution_enabled: bool | None = None
    workflow_router_enabled: bool | None = None
    workflow_router_llm_fallback_enabled: bool | None = None
    workflow_router_confidence_threshold: float | None = Field(None, ge=0.0, le=1.0)
    workflow_router_llm_profile: str | None = None


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
    provider_credentials: ProviderCredentials | None = Field(
        None,
        description=(
            "Optional request-scoped provider credentials. These are passed to the worker "
            "for this job and are not stored in the job input history."
        ),
    )
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


class WorkflowGroupItem(StrictInputModel):
    name: str | None = Field(
        None,
        min_length=1,
        max_length=64,
        description="Optional unique name for this child workflow in the group result.",
    )
    workflow_type: WorkflowType
    inputs: dict[str, Any] = Field(..., description="Workflow-specific input parameters")
    provider_credentials: ProviderCredentials | None = Field(
        None,
        description="Optional request-scoped provider credentials for this child workflow.",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional tracing or request context metadata for this child workflow.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def validate_workflow_inputs(self) -> "WorkflowGroupItem":
        input_model = WORKFLOW_INPUT_MODELS[self.workflow_type]
        try:
            validated_inputs = input_model.model_validate(self.inputs)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid inputs for workflow '{self.workflow_type.value}': {exc}"
            ) from exc

        self.inputs = validated_inputs.model_dump()
        return self


class WorkflowGroupRequest(StrictInputModel):
    workflows: list[WorkflowGroupItem] = Field(..., min_length=2, max_length=7)
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional tracing or request context metadata for the workflow group.",
    )

    @model_validator(mode="after")
    def validate_unique_child_names(self) -> "WorkflowGroupRequest":
        names: set[str] = set()
        for item in self.workflows:
            resolved_name = item.name or item.workflow_type.value
            if resolved_name in names:
                raise ValueError(
                    f"Workflow group child names must be unique; duplicate name '{resolved_name}'."
                )
            names.add(resolved_name)
        return self


class WorkflowRouteNode(StrictInputModel):
    name: str = Field(..., min_length=1, max_length=64)
    workflow_type: WorkflowType
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("depends_on", mode="before")
    @classmethod
    def _normalize_dependencies(cls, value: Any) -> Any:
        if value in (None, ""):
            return []
        return value

    @field_validator("depends_on")
    @classmethod
    def _strip_dependencies(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class WorkflowRoutePlan(StrictInputModel):
    goal: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_review: bool
    missing_inputs: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)
    nodes: list[WorkflowRouteNode] = Field(default_factory=list, max_length=7)
    waves: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_node_graph(self) -> "WorkflowRoutePlan":
        names: set[str] = set()
        for node in self.nodes:
            if node.name in names:
                raise ValueError(f"Workflow route node names must be unique; duplicate name '{node.name}'.")
            names.add(node.name)
        for node in self.nodes:
            unknown = [dependency for dependency in node.depends_on if dependency not in names]
            if unknown:
                raise ValueError(f"Workflow route node '{node.name}' has unknown dependencies: {unknown}.")
            if node.name in node.depends_on:
                raise ValueError(f"Workflow route node '{node.name}' cannot depend on itself.")
        self.waves = self.waves or _route_waves(self.nodes)
        return self


class WorkflowRouteRequest(StrictInputModel):
    goal: str = Field(..., min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    preferred_workflows: list[WorkflowType] | None = None
    excluded_workflows: list[WorkflowType] | None = None
    provider_credentials: ProviderCredentials | None = Field(
        None,
        description="Optional request-scoped provider credentials used for planning and child workflows.",
    )
    metadata: dict[str, Any] | None = Field(None, description="Optional tracing or request context metadata.")

    @field_validator("preferred_workflows", "excluded_workflows", mode="before")
    @classmethod
    def _empty_workflow_lists_to_none(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return value

    @model_validator(mode="after")
    def validate_workflow_filters(self) -> "WorkflowRouteRequest":
        preferred = set(self.preferred_workflows or [])
        excluded = set(self.excluded_workflows or [])
        overlap = preferred & excluded
        if overlap:
            names = ", ".join(sorted(item.value for item in overlap))
            raise ValueError(f"preferred_workflows and excluded_workflows overlap: {names}")
        return self


def _route_waves(nodes: list[WorkflowRouteNode]) -> list[list[str]]:
    remaining = {node.name: set(node.depends_on) for node in nodes}
    waves: list[list[str]] = []
    completed: set[str] = set()
    while remaining:
        ready = sorted(name for name, dependencies in remaining.items() if dependencies <= completed)
        if not ready:
            raise ValueError("Workflow route dependencies contain a cycle.")
        waves.append(ready)
        completed.update(ready)
        for name in ready:
            remaining.pop(name, None)
    return waves


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    cache_hit: bool | None = None
    source_job_id: str | None = None
    usage_metrics: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None
    error: str | None = None


class JobEventResponse(BaseModel):
    event_id: int
    job_id: str
    event_type: str
    message: str
    payload: dict[str, Any] | None = None
    created_at: datetime | None = None
