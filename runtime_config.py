import os
from dataclasses import asdict, dataclass
from typing import Any


RUNTIME_CONFIG_KEYS = {
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
    "meta_access_token",
    "meta_ad_account_id",
    "meta_page_id",
    "tiktok_access_token",
    "tiktok_advertiser_id",
    "openai_input_cost_per_1m_tokens",
    "openai_output_cost_per_1m_tokens",
    "workflow_result_cache_enabled",
    "workflow_result_cache_ttl_seconds",
    "content_language_concurrency",
    "marketing_market_concurrency",
    "serper_deep_read_enabled",
    "serper_deep_read_max_pages",
    "serper_deep_read_concurrency",
    "serper_deep_read_timeout_seconds",
    "serper_deep_read_max_chars",
}


@dataclass(frozen=True)
class RuntimeConfig:
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
    meta_access_token: str | None = None
    meta_ad_account_id: str | None = None
    meta_page_id: str | None = None
    tiktok_access_token: str | None = None
    tiktok_advertiser_id: str | None = None
    openai_input_cost_per_1m_tokens: float = 0.0
    openai_output_cost_per_1m_tokens: float = 0.0
    workflow_result_cache_enabled: bool = True
    workflow_result_cache_ttl_seconds: int = 3600
    content_language_concurrency: int = 4
    marketing_market_concurrency: int = 4
    serper_deep_read_enabled: bool = False
    serper_deep_read_max_pages: int = 3
    serper_deep_read_concurrency: int = 5
    serper_deep_read_timeout_seconds: int = 10
    serper_deep_read_max_chars: int = 4000

    def as_context(self) -> dict[str, Any]:
        return asdict(self)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
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
        whatsapp_access_token=os.getenv("WHATSAPP_ACCESS_TOKEN"),
        whatsapp_phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID"),
        whatsapp_business_account_id=os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID"),
        whatsapp_verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN"),
        whatsapp_app_secret=os.getenv("WHATSAPP_APP_SECRET"),
        whatsapp_send_enabled=_bool_env("WHATSAPP_SEND_ENABLED", False),
        whatsapp_graph_api_version=os.getenv("WHATSAPP_GRAPH_API_VERSION", "v23.0"),
        meta_access_token=os.getenv("META_ACCESS_TOKEN"),
        meta_ad_account_id=os.getenv("META_AD_ACCOUNT_ID"),
        meta_page_id=os.getenv("META_PAGE_ID"),
        tiktok_access_token=os.getenv("TIKTOK_ACCESS_TOKEN"),
        tiktok_advertiser_id=os.getenv("TIKTOK_ADVERTISER_ID"),
        openai_input_cost_per_1m_tokens=_float_env("OPENAI_INPUT_COST_PER_1M_TOKENS"),
        openai_output_cost_per_1m_tokens=_float_env("OPENAI_OUTPUT_COST_PER_1M_TOKENS"),
        workflow_result_cache_enabled=_bool_env("WORKFLOW_RESULT_CACHE_ENABLED", True),
        workflow_result_cache_ttl_seconds=_int_env("WORKFLOW_RESULT_CACHE_TTL_SECONDS", 3600),
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


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def apply_runtime_environment(config: RuntimeConfig | dict[str, Any]) -> None:
    context = config.as_context() if isinstance(config, RuntimeConfig) else config
    env_map = {
        "OPENAI_API_KEY": context.get("openai_api_key"),
        "OPENAI_MODEL_NAME": context.get("openai_model_name"),
        "SERPER_API_KEY": context.get("serper_api_key"),
    }
    for env_name, value in env_map.items():
        if value:
            os.environ[env_name] = str(value)


def merge_runtime_context(
    base_context: RuntimeConfig | dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = base_context.as_context() if isinstance(base_context, RuntimeConfig) else dict(base_context)
    for key, value in (overrides or {}).items():
        if key in RUNTIME_CONFIG_KEYS and value not in (None, ""):
            context[key] = value
    return context
