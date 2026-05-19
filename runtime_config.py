import os
from dataclasses import asdict, dataclass
from typing import Any


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
    holiday_api_key: str | None = None
    google_ads_developer_token: str | None = None
    google_ads_access_token: str | None = None
    google_ads_customer_id: str | None = None
    meta_access_token: str | None = None
    meta_ad_account_id: str | None = None
    meta_page_id: str | None = None
    tiktok_access_token: str | None = None
    tiktok_advertiser_id: str | None = None

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
        holiday_api_key=os.getenv("HOLIDAY_API_KEY"),
        google_ads_developer_token=os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
        google_ads_access_token=os.getenv("GOOGLE_ADS_ACCESS_TOKEN"),
        google_ads_customer_id=os.getenv("GOOGLE_ADS_CUSTOMER_ID"),
        meta_access_token=os.getenv("META_ACCESS_TOKEN"),
        meta_ad_account_id=os.getenv("META_AD_ACCOUNT_ID"),
        meta_page_id=os.getenv("META_PAGE_ID"),
        tiktok_access_token=os.getenv("TIKTOK_ACCESS_TOKEN"),
        tiktok_advertiser_id=os.getenv("TIKTOK_ADVERTISER_ID"),
    )


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
