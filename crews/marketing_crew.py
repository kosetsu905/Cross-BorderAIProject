from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.marketing_tools import (
    ComplianceCheckerTool,
    KeywordResearchTool,
    PlatformAdSpecsTool,
)
from tools.integrations.cross_platform_ads_tools import (
    GoogleAdsKeywordTool,
    MetaAdsTool,
    TikTokAdsTool,
)
from utils.crew_result import serialize_crew_result

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "marketing"


class CampaignAdVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(..., description="Target ad platform")
    region: str = Field(..., description="Target market or region")
    headline: str = Field(..., description="Ad headline")
    body_text: str = Field(..., description="Primary ad copy")
    cta: str = Field(..., description="Call to action")


class ComplianceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Target market or region")
    platform: str = Field(..., description="Ad platform")
    approved: bool = Field(..., description="Whether the campaign item is approved")
    notes: str = Field(..., description="Compliance notes and required edits")


class FinalCampaignOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source: str = Field(
        ...,
        description=(
            "Provider status for ad platform integrations, such as "
            "development_fallback, mixed, or live_provider."
        ),
    )
    confidence_level: str = Field(
        ...,
        description="Confidence level based on whether live ad platform data was available",
    )
    assumptions: list[str] = Field(
        ..., description="Important caveats about fallback ad platform data or inferred benchmarks"
    )
    strategy_summary: str = Field(..., description="Concise campaign strategy overview")
    ad_variants: list[CampaignAdVariant] = Field(
        ..., description="Platform and region specific ad copy variants"
    )
    compliance_status: list[ComplianceStatus] = Field(
        ..., description="Approval status and key compliance notes per region/platform"
    )
    launch_checklist: list[str] = Field(
        ..., description="Step-by-step pre-launch checklist"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _google_ads_tool(config_context: dict[str, Any]) -> GoogleAdsKeywordTool:
    return GoogleAdsKeywordTool(
        google_ads_access_token=config_context.get("google_ads_access_token"),
        google_ads_customer_id=config_context.get("google_ads_customer_id"),
        google_ads_developer_token=config_context.get("google_ads_developer_token"),
    )


def _meta_ads_tool(config_context: dict[str, Any]) -> MetaAdsTool:
    return MetaAdsTool(
        meta_access_token=config_context.get("meta_access_token"),
        meta_ad_account_id=config_context.get("meta_ad_account_id"),
        meta_page_id=config_context.get("meta_page_id"),
    )


def _tiktok_ads_tool(config_context: dict[str, Any]) -> TikTokAdsTool:
    return TikTokAdsTool(
        tiktok_access_token=config_context.get("tiktok_access_token"),
        tiktok_advertiser_id=config_context.get("tiktok_advertiser_id"),
    )


def _build_research_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool(), _google_ads_tool(config_context)]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


def _memory_enabled(config_context: dict[str, Any]) -> bool:
    return bool(config_context.get("crewai_memory_enabled"))


def _has_all_config(config_context: dict[str, Any], names: tuple[str, ...]) -> bool:
    return all(bool(config_context.get(name)) for name in names)


def _platform_provider_status(config_context: dict[str, Any]) -> dict[str, Any]:
    provider_groups = {
        "Google Ads": (
            "google_ads_access_token",
            "google_ads_customer_id",
            "google_ads_developer_token",
        ),
        "Meta Ads": (
            "meta_access_token",
            "meta_ad_account_id",
        ),
        "TikTok Ads": (
            "tiktok_access_token",
            "tiktok_advertiser_id",
        ),
    }
    live_platforms = [
        platform
        for platform, required_env in provider_groups.items()
        if _has_all_config(config_context, required_env)
    ]
    fallback_platforms = [
        platform
        for platform, required_env in provider_groups.items()
        if not _has_all_config(config_context, required_env)
    ]

    if fallback_platforms and live_platforms:
        data_source = "mixed"
        confidence_level = "Medium"
    elif fallback_platforms:
        data_source = "development_fallback"
        confidence_level = "Illustrative"
    else:
        data_source = "live_provider"
        confidence_level = "High"

    assumptions: list[str] = []
    if fallback_platforms:
        assumptions.append(
            "The following ad platform integrations used development fallback data "
            f"because credentials are missing or incomplete: {', '.join(fallback_platforms)}."
        )
        assumptions.append(
            "Keyword ideas, CPC estimates, audience estimates, trend benchmarks, and "
            "platform compliance checks should be validated with live ad platform APIs "
            "before production launch."
        )

    assumptions.append(
        "Amazon Ads is not connected to a live integration in this project yet; any "
        "Amazon-specific recommendations are generated from static specs or model inference."
    )

    return {
        "data_source": data_source,
        "confidence_level": confidence_level,
        "assumptions": assumptions,
    }


def _apply_provider_status(result: dict[str, Any], config_context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.update(_platform_provider_status(config_context))
    return normalized


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_marketing_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    strategist = Agent(
        config=agents_config["campaign_strategist"],
        tools=_build_research_tools(config_context),
    )
    copywriter = Agent(
        config=agents_config["ad_copywriter"],
        tools=[KeywordResearchTool(), _google_ads_tool(config_context)],
    )
    optimizer = Agent(
        config=agents_config["channel_optimizer"],
        tools=[PlatformAdSpecsTool(), _meta_ads_tool(config_context), _tiktok_ads_tool(config_context)],
    )
    qa_agent = Agent(
        config=agents_config["compliance_qa_specialist"],
        tools=[ComplianceCheckerTool(), _meta_ads_tool(config_context), _tiktok_ads_tool(config_context)],
        allow_delegation=True,
    )

    research_task = Task(config=tasks_config["market_research"], agent=strategist)
    strategy_task = Task(
        config=tasks_config["campaign_strategy"],
        agent=strategist,
        context=[research_task],
    )
    copy_task = Task(
        config=tasks_config["ad_copy_generation"],
        agent=copywriter,
        context=[strategy_task],
    )
    qa_task = Task(
        config=tasks_config["compliance_qa_review"],
        agent=qa_agent,
        context=[copy_task, strategy_task],
        output_pydantic=FinalCampaignOutput,
    )

    marketing_crew = Crew(
        agents=[strategist, copywriter, optimizer, qa_agent],
        tasks=[research_task, strategy_task, copy_task, qa_task],
        verbose=False,
        memory=_memory_enabled(config_context),
    )

    result = _serialize_crew_result(marketing_crew.kickoff(inputs=inputs))
    return _apply_provider_status(result, config_context)
