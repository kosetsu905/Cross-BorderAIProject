import json
import os
import re
from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


RUNTIME_CONFIG_KEYS = {
    "llm_provider",
    "llm_profile",
    "llm_api_key",
    "llm_model_name",
    "llm_base_url",
    "llm_disable_reasoning",
    "llm_profiles",
    "content_image_model",
    "content_image_scoring_model",
    "content_image_artifact_dir",
    "support_qa_mode",
    "support_llm_profile",
    "openai_api_key",
    "openai_model_name",
    "crewai_memory_enabled",
    "serper_api_key",
    "crunchbase_api_key",
    "apollo_api_key",
    "ecom_api_token",
    "crm_api_token",
    "shopify_store_domain",
    "shopify_admin_access_token",
    "shopify_api_version",
    "amazon_sp_api_endpoint",
    "amazon_sp_api_access_token",
    "amazon_marketplace_ids",
    "support_knowledge_dir",
    "support_handoff_webhook_url",
    "support_session_redis_url",
    "support_session_ttl_seconds",
    "support_session_history_limit",
    "support_serper_pre_sales_enabled",
    "support_serper_order_fulfillment_enabled",
    "support_serper_post_sales_enabled",
    "holiday_api_key",
    "google_ads_developer_token",
    "google_ads_access_token",
    "google_ads_customer_id",
    "gmail_access_token",
    "gmail_client_id",
    "gmail_client_secret",
    "gmail_refresh_token",
    "gmail_sender_email",
    "gmail_send_enabled",
    "gmail_watch_topic_name",
    "gmail_watch_label_ids",
    "gmail_sync_enabled",
    "whatsapp_access_token",
    "whatsapp_phone_number_id",
    "whatsapp_business_account_id",
    "whatsapp_verify_token",
    "whatsapp_app_secret",
    "whatsapp_send_enabled",
    "whatsapp_graph_api_version",
    "whatsapp_provider",
    "ycloud_api_key",
    "ycloud_whatsapp_from",
    "ycloud_waba_id",
    "ycloud_base_url",
    "ycloud_webhook_secret",
    "pim_backend",
    "pim_akeneo_base_url",
    "pim_akeneo_api_key",
    "pim_plytix_base_url",
    "pim_plytix_api_key",
    "pim_custom_base_url",
    "pim_custom_api_key",
    "intent_classifier_enabled",
    "intent_classifier_model_path",
    "intent_router_llm_fallback_enabled",
    "intent_router_confidence_threshold",
    "meta_access_token",
    "meta_ad_account_id",
    "meta_page_id",
    "tiktok_access_token",
    "tiktok_advertiser_id",
    "openai_input_cost_per_1m_tokens",
    "openai_output_cost_per_1m_tokens",
    "workflow_result_cache_enabled",
    "workflow_result_cache_ttl_seconds",
    "workflow_async_execution_enabled",
    "content_language_concurrency",
    "marketing_market_concurrency",
    "serper_deep_read_enabled",
    "serper_deep_read_max_pages",
    "serper_deep_read_concurrency",
    "serper_deep_read_timeout_seconds",
    "serper_deep_read_max_chars",
}


class LLMProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    llm_provider: str = Field(..., min_length=1)
    llm_model_name: str = Field(..., min_length=1)
    llm_base_url: str | None = Field(default=None, min_length=1)
    llm_api_key_env: str | None = Field(default=None, min_length=1)
    llm_disable_reasoning: bool | None = None

    @field_validator("llm_provider", "llm_model_name", "llm_base_url", "llm_api_key_env", mode="before")
    @classmethod
    def _strip_string(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("llm_provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return value.lower()

    @field_validator("llm_base_url")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else value

    @field_validator("llm_api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str | None) -> str | None:
        if value and not ENV_NAME_RE.fullmatch(value):
            raise ValueError("llm_api_key_env must be a valid environment variable name")
        return value


@dataclass(frozen=True)
class RuntimeConfig:
    llm_provider: str = "openai"
    llm_profile: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_disable_reasoning: bool = False
    llm_profiles: dict[str, LLMProfileConfig] = dataclass_field(default_factory=dict)
    content_image_model: str = "gpt-image-2"
    content_image_scoring_model: str = "gpt-4o-mini"
    content_image_artifact_dir: str = "artifacts/content_creation"
    support_qa_mode: str = "full_llm"
    support_llm_profile: str | None = None
    openai_api_key: str | None = None
    openai_model_name: str = "gpt-4o-mini"
    crewai_memory_enabled: bool = False
    serper_api_key: str | None = None
    crunchbase_api_key: str | None = None
    apollo_api_key: str | None = None
    ecom_api_token: str | None = None
    crm_api_token: str | None = None
    shopify_store_domain: str | None = None
    shopify_admin_access_token: str | None = None
    shopify_api_version: str = "2025-07"
    amazon_sp_api_endpoint: str | None = None
    amazon_sp_api_access_token: str | None = None
    amazon_marketplace_ids: str | None = None
    support_knowledge_dir: str | None = None
    support_handoff_webhook_url: str | None = None
    support_session_redis_url: str | None = None
    support_session_ttl_seconds: int = 86400
    support_session_history_limit: int = 20
    support_serper_pre_sales_enabled: bool = False
    support_serper_order_fulfillment_enabled: bool = False
    support_serper_post_sales_enabled: bool = False
    holiday_api_key: str | None = None
    google_ads_developer_token: str | None = None
    google_ads_access_token: str | None = None
    google_ads_customer_id: str | None = None
    gmail_access_token: str | None = None
    gmail_client_id: str | None = None
    gmail_client_secret: str | None = None
    gmail_refresh_token: str | None = None
    gmail_sender_email: str | None = None
    gmail_send_enabled: bool = False
    gmail_watch_topic_name: str | None = None
    gmail_watch_label_ids: str = "INBOX"
    gmail_sync_enabled: bool = False
    whatsapp_access_token: str | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_business_account_id: str | None = None
    whatsapp_verify_token: str | None = None
    whatsapp_app_secret: str | None = None
    whatsapp_send_enabled: bool = False
    whatsapp_graph_api_version: str = "v23.0"
    whatsapp_provider: str = "ycloud"
    ycloud_api_key: str | None = None
    ycloud_whatsapp_from: str | None = None
    ycloud_waba_id: str | None = None
    ycloud_base_url: str = "https://api.ycloud.com/v2"
    ycloud_webhook_secret: str | None = None
    pim_backend: str = "akeneo"
    pim_akeneo_base_url: str | None = None
    pim_akeneo_api_key: str | None = None
    pim_plytix_base_url: str | None = None
    pim_plytix_api_key: str | None = None
    pim_custom_base_url: str | None = None
    pim_custom_api_key: str | None = None
    intent_classifier_enabled: bool = False
    intent_classifier_model_path: str | None = None
    intent_router_llm_fallback_enabled: bool = True
    intent_router_confidence_threshold: float = 0.75
    meta_access_token: str | None = None
    meta_ad_account_id: str | None = None
    meta_page_id: str | None = None
    tiktok_access_token: str | None = None
    tiktok_advertiser_id: str | None = None
    openai_input_cost_per_1m_tokens: float = 0.0
    openai_output_cost_per_1m_tokens: float = 0.0
    workflow_result_cache_enabled: bool = True
    workflow_result_cache_ttl_seconds: int = 3600
    workflow_async_execution_enabled: bool = True
    content_language_concurrency: int = 4
    marketing_market_concurrency: int = 4
    serper_deep_read_enabled: bool = False
    serper_deep_read_max_pages: int = 3
    serper_deep_read_concurrency: int = 5
    serper_deep_read_timeout_seconds: int = 10
    serper_deep_read_max_chars: int = 4000

    def as_context(self) -> dict[str, Any]:
        context = asdict(self)
        context["llm_profiles"] = {
            name: profile.model_dump(exclude_none=True)
            if isinstance(profile, LLMProfileConfig)
            else dict(profile)
            for name, profile in self.llm_profiles.items()
        }
        return context


def _normalize_profile_name(profile_name: str) -> str:
    normalized = profile_name.strip()
    if not LLM_PROFILE_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "LLM profile names must be 1-64 characters and contain only letters, numbers, underscores, or hyphens."
        )
    return normalized


def parse_llm_profiles(raw_json: str | None) -> dict[str, LLMProfileConfig]:
    if not raw_json or not raw_json.strip():
        return {}
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LLM_PROFILES_JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM_PROFILES_JSON must be a JSON object keyed by profile name.")

    profiles: dict[str, LLMProfileConfig] = {}
    for raw_name, raw_profile in parsed.items():
        if not isinstance(raw_name, str):
            raise ValueError("LLM profile names must be strings.")
        profile_name = _normalize_profile_name(raw_name)
        if profile_name in profiles:
            raise ValueError(f"Duplicate LLM profile name after normalization: {profile_name}")
        if not isinstance(raw_profile, dict):
            raise ValueError(f"LLM profile '{profile_name}' must be a JSON object.")
        profiles[profile_name] = LLMProfileConfig.model_validate(raw_profile)
    return profiles


def _profile_from_context(context: dict[str, Any], profile_name: str) -> LLMProfileConfig:
    profiles = context.get("llm_profiles") or {}
    if profile_name not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        raise ValueError(f"Unknown LLM profile '{profile_name}'. Available profiles: {available}.")
    profile = profiles[profile_name]
    if isinstance(profile, LLMProfileConfig):
        return profile
    if isinstance(profile, dict):
        return LLMProfileConfig.model_validate(profile)
    raise ValueError(f"LLM profile '{profile_name}' has an invalid configuration.")


def _resolved_profile_api_key(profile_name: str, profile: LLMProfileConfig) -> str | None:
    if not profile.llm_api_key_env:
        return None
    api_key = os.getenv(profile.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"LLM profile '{profile_name}' references missing or empty environment variable "
            f"'{profile.llm_api_key_env}'."
        )
    return api_key


def apply_llm_profile_context(
    config_context: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    normalized_name = _normalize_profile_name(profile_name)
    profile = _profile_from_context(config_context, normalized_name)
    context = dict(config_context)
    context["llm_profile"] = normalized_name
    context["llm_provider"] = profile.llm_provider
    context["llm_model_name"] = profile.llm_model_name
    context["llm_base_url"] = (
        profile.llm_base_url
        if profile.llm_base_url
        else DEFAULT_OPENROUTER_BASE_URL if profile.llm_provider == "openrouter" else None
    )
    api_key = _resolved_profile_api_key(normalized_name, profile)
    if api_key:
        context["llm_api_key"] = api_key
    if profile.llm_disable_reasoning is not None:
        context["llm_disable_reasoning"] = profile.llm_disable_reasoning
    return context


def resolve_workflow_runtime_context(
    base_context: RuntimeConfig | dict[str, Any],
    workflow_type: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = base_context.as_context() if isinstance(base_context, RuntimeConfig) else dict(base_context)
    workflow_value = workflow_type.value if hasattr(workflow_type, "value") else str(workflow_type)

    if workflow_value == "support" and context.get("support_llm_profile"):
        context = apply_llm_profile_context(context, str(context["support_llm_profile"]))

    request_profile = (overrides or {}).get("llm_profile")
    direct_overrides = {
        key: value
        for key, value in (overrides or {}).items()
        if key != "llm_profile"
    }
    context = merge_runtime_context(context, direct_overrides)

    if request_profile not in (None, ""):
        context = apply_llm_profile_context(context, str(request_profile))
    return context


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    if fallback:
        fallback_value = os.getenv(fallback)
        if fallback_value not in (None, ""):
            return fallback_value
    return None


def load_runtime_config() -> RuntimeConfig:
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_api_key = _env("LLM_API_KEY", "OPENAI_API_KEY")
    llm_model_name = os.getenv("LLM_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    llm_base_url = os.getenv("LLM_BASE_URL")
    if not llm_base_url and llm_provider.lower() == "openrouter":
        llm_base_url = DEFAULT_OPENROUTER_BASE_URL
    llm_profiles = parse_llm_profiles(os.getenv("LLM_PROFILES_JSON"))
    support_llm_profile = os.getenv("SUPPORT_LLM_PROFILE")
    support_llm_profile = _normalize_profile_name(support_llm_profile) if support_llm_profile else None
    if support_llm_profile:
        apply_llm_profile_context(
            {
                "llm_profiles": {
                    name: profile.model_dump(exclude_none=True)
                    for name, profile in llm_profiles.items()
                }
            },
            support_llm_profile,
        )
    return RuntimeConfig(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        llm_disable_reasoning=_bool_env("LLM_DISABLE_REASONING", False),
        llm_profiles=llm_profiles,
        content_image_model=os.getenv("CONTENT_IMAGE_MODEL", "gpt-image-2"),
        content_image_scoring_model=os.getenv("CONTENT_IMAGE_SCORING_MODEL", "gpt-4o-mini"),
        content_image_artifact_dir=os.getenv(
            "CONTENT_IMAGE_ARTIFACT_DIR",
            "artifacts/content_creation",
        ),
        support_qa_mode=_support_qa_mode_env(),
        support_llm_profile=support_llm_profile,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        crewai_memory_enabled=_bool_env("CREWAI_MEMORY_ENABLED", False),
        serper_api_key=os.getenv("SERPER_API_KEY"),
        crunchbase_api_key=os.getenv("CRUNCHBASE_API_KEY"),
        apollo_api_key=os.getenv("APOLLO_API_KEY"),
        ecom_api_token=os.getenv("ECOM_API_TOKEN"),
        crm_api_token=os.getenv("CRM_API_TOKEN"),
        shopify_store_domain=os.getenv("SHOPIFY_STORE_DOMAIN"),
        shopify_admin_access_token=os.getenv("SHOPIFY_ADMIN_ACCESS_TOKEN"),
        shopify_api_version=os.getenv("SHOPIFY_API_VERSION", "2025-07"),
        amazon_sp_api_endpoint=os.getenv("AMAZON_SP_API_ENDPOINT"),
        amazon_sp_api_access_token=os.getenv("AMAZON_SP_API_ACCESS_TOKEN"),
        amazon_marketplace_ids=os.getenv("AMAZON_MARKETPLACE_IDS"),
        support_knowledge_dir=os.getenv("SUPPORT_KNOWLEDGE_DIR"),
        support_handoff_webhook_url=_env("SUPPORT_HANDOFF_WEBHOOK_URL", "SLACK_WEBHOOK_URL"),
        support_session_redis_url=_env("SUPPORT_SESSION_REDIS_URL", "CELERY_BROKER_URL"),
        support_session_ttl_seconds=_int_env("SUPPORT_SESSION_TTL_SECONDS", 86400),
        support_session_history_limit=_int_env("SUPPORT_SESSION_HISTORY_LIMIT", 20),
        support_serper_pre_sales_enabled=_bool_env("SUPPORT_SERPER_PRE_SALES_ENABLED", False),
        support_serper_order_fulfillment_enabled=_bool_env("SUPPORT_SERPER_ORDER_FULFILLMENT_ENABLED", False),
        support_serper_post_sales_enabled=_bool_env("SUPPORT_SERPER_POST_SALES_ENABLED", False),
        holiday_api_key=os.getenv("HOLIDAY_API_KEY"),
        google_ads_developer_token=os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
        google_ads_access_token=os.getenv("GOOGLE_ADS_ACCESS_TOKEN"),
        google_ads_customer_id=os.getenv("GOOGLE_ADS_CUSTOMER_ID"),
        gmail_access_token=os.getenv("GMAIL_ACCESS_TOKEN"),
        gmail_client_id=os.getenv("GMAIL_CLIENT_ID"),
        gmail_client_secret=os.getenv("GMAIL_CLIENT_SECRET"),
        gmail_refresh_token=os.getenv("GMAIL_REFRESH_TOKEN"),
        gmail_sender_email=os.getenv("GMAIL_SENDER_EMAIL"),
        gmail_send_enabled=_bool_env("GMAIL_SEND_ENABLED", False),
        gmail_watch_topic_name=os.getenv("GMAIL_WATCH_TOPIC_NAME"),
        gmail_watch_label_ids=os.getenv("GMAIL_WATCH_LABEL_IDS", "INBOX"),
        gmail_sync_enabled=_bool_env("GMAIL_SYNC_ENABLED", False),
        whatsapp_access_token=_env("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_TOKEN"),
        whatsapp_phone_number_id=_env("WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_PHONE_ID"),
        whatsapp_business_account_id=os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID"),
        whatsapp_verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN"),
        whatsapp_app_secret=os.getenv("WHATSAPP_APP_SECRET"),
        whatsapp_send_enabled=_bool_env("WHATSAPP_SEND_ENABLED", False),
        whatsapp_graph_api_version=os.getenv("WHATSAPP_GRAPH_API_VERSION", "v23.0"),
        whatsapp_provider=os.getenv("WHATSAPP_PROVIDER", "ycloud"),
        ycloud_api_key=os.getenv("YCLOUD_API_KEY"),
        ycloud_whatsapp_from=os.getenv("YCLOUD_WHATSAPP_FROM"),
        ycloud_waba_id=os.getenv("YCLOUD_WABA_ID"),
        ycloud_base_url=os.getenv("YCLOUD_BASE_URL", "https://api.ycloud.com/v2"),
        ycloud_webhook_secret=os.getenv("YCLOUD_WEBHOOK_SECRET"),
        pim_backend=os.getenv("PIM_BACKEND", "akeneo"),
        pim_akeneo_base_url=os.getenv("PIM_AKENEO_BASE_URL"),
        pim_akeneo_api_key=os.getenv("PIM_AKENEO_API_KEY"),
        pim_plytix_base_url=os.getenv("PIM_PLYTIX_BASE_URL"),
        pim_plytix_api_key=os.getenv("PIM_PLYTIX_API_KEY"),
        pim_custom_base_url=os.getenv("PIM_CUSTOM_BASE_URL"),
        pim_custom_api_key=os.getenv("PIM_CUSTOM_API_KEY"),
        intent_classifier_enabled=_bool_env("INTENT_CLASSIFIER_ENABLED", False),
        intent_classifier_model_path=os.getenv("INTENT_CLASSIFIER_MODEL_PATH"),
        intent_router_llm_fallback_enabled=_bool_env("INTENT_ROUTER_LLM_FALLBACK_ENABLED", True),
        intent_router_confidence_threshold=_float_env_with_default("INTENT_ROUTER_CONFIDENCE_THRESHOLD", 0.75),
        meta_access_token=os.getenv("META_ACCESS_TOKEN"),
        meta_ad_account_id=os.getenv("META_AD_ACCOUNT_ID"),
        meta_page_id=os.getenv("META_PAGE_ID"),
        tiktok_access_token=os.getenv("TIKTOK_ACCESS_TOKEN"),
        tiktok_advertiser_id=os.getenv("TIKTOK_ADVERTISER_ID"),
        openai_input_cost_per_1m_tokens=_float_env("OPENAI_INPUT_COST_PER_1M_TOKENS"),
        openai_output_cost_per_1m_tokens=_float_env("OPENAI_OUTPUT_COST_PER_1M_TOKENS"),
        workflow_result_cache_enabled=_bool_env("WORKFLOW_RESULT_CACHE_ENABLED", True),
        workflow_result_cache_ttl_seconds=_int_env("WORKFLOW_RESULT_CACHE_TTL_SECONDS", 3600),
        workflow_async_execution_enabled=_bool_env("WORKFLOW_ASYNC_EXECUTION_ENABLED", True),
        content_language_concurrency=_int_env("CONTENT_LANGUAGE_CONCURRENCY", 4),
        marketing_market_concurrency=_int_env("MARKETING_MARKET_CONCURRENCY", 4),
        serper_deep_read_enabled=_bool_env("SERPER_DEEP_READ_ENABLED", False),
        serper_deep_read_max_pages=_int_env("SERPER_DEEP_READ_MAX_PAGES", 3),
        serper_deep_read_concurrency=_int_env("SERPER_DEEP_READ_CONCURRENCY", 5),
        serper_deep_read_timeout_seconds=_int_env("SERPER_DEEP_READ_TIMEOUT_SECONDS", 10),
        serper_deep_read_max_chars=_int_env("SERPER_DEEP_READ_MAX_CHARS", 4000),
    )


def _float_env(name: str) -> float:
    try:
        return float(os.getenv(name, "0") or 0)
    except ValueError:
        return 0.0


def _float_env_with_default(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def _support_qa_mode_env() -> str:
    value = os.getenv("SUPPORT_QA_MODE", "full_llm").strip().lower()
    if value not in {"full_llm", "adaptive_fast"}:
        raise ValueError("SUPPORT_QA_MODE must be 'full_llm' or 'adaptive_fast'.")
    return value


def apply_runtime_environment(config: RuntimeConfig | dict[str, Any]) -> None:
    context = config.as_context() if isinstance(config, RuntimeConfig) else config
    llm_api_key = context.get("llm_api_key") or context.get("openai_api_key")
    llm_model_name = context.get("llm_model_name") or context.get("openai_model_name")
    llm_base_url = context.get("llm_base_url")
    env_map = {
        "OPENAI_API_KEY": llm_api_key,
        "OPENAI_MODEL_NAME": llm_model_name,
        "SERPER_API_KEY": context.get("serper_api_key"),
    }
    for env_name, value in env_map.items():
        if value:
            os.environ[env_name] = str(value)
    for env_name in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
        if llm_base_url:
            os.environ[env_name] = str(llm_base_url)
        else:
            os.environ.pop(env_name, None)


def merge_runtime_context(
    base_context: RuntimeConfig | dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = base_context.as_context() if isinstance(base_context, RuntimeConfig) else dict(base_context)
    for key, value in (overrides or {}).items():
        if key in RUNTIME_CONFIG_KEYS and value not in (None, ""):
            context[key] = value
    return context
