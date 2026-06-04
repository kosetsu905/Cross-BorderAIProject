from enum import Enum
from datetime import datetime
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
    primary_keywords: list[str] | None = Field(
        None,
        description="Optional seed SEO keywords for multi-engine metadata generation.",
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
        "auto",
        min_length=1,
        description="OpenAI image generation quality setting, such as auto, low, medium, or high.",
    )
    image_size: str = Field(
        "1024x1024",
        min_length=1,
        description="OpenAI image generation size setting, such as 1024x1024.",
    )


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


class ProviderCredentials(StrictInputModel):
    llm_provider: str | None = None
    llm_profile: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str | None = None
    llm_base_url: str | None = None
    llm_disable_reasoning: bool | None = None
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
