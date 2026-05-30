from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.analytics_tools import CompetitorBenchmarkTool, EcomPlatformMetricsTool
from utils.crew_result import serialize_crew_result
from utils.workflow_progress import attach_task_progress
from utils.project_intelligence import augment_agents_config

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "analytics"

class RegionalKPI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Target market or region")
    sales_volume: str = Field(..., description="Total sales or revenue")
    conversion_rate: str = Field(..., description="Conversion rate percentage")
    roas: str = Field(..., description="Return on ad spend")


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(..., description="Target market or region for the claim")
    claim: str = Field(..., description="Source-backed competitive or market claim")
    evidence_summary: str = Field(..., description="Short summary of the supporting evidence")
    sources: list[str] = Field(..., description="Source URLs used to support the claim")
    confidence: str = Field(..., description="Evidence confidence level for this claim")


class MarketIntelligence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Target market or region")
    competitor_signals: list[str] = Field(
        ..., description="Competitor names, positioning, product, or messaging signals found in sources"
    )
    pricing_signals: list[str] = Field(
        ..., description="Source-backed pricing, discount, affordability, or price-positioning observations"
    )
    demand_signals: list[str] = Field(
        ..., description="Source-backed demand, consumer preference, market trend, or adoption observations"
    )
    channel_signals: list[str] = Field(
        ..., description="Source-backed channel, marketplace, retail, platform, or go-to-market observations"
    )
    source_urls: list[str] = Field(..., description="URLs supporting this region's market intelligence")
    confidence: str = Field(..., description="Confidence level for this region's market intelligence")


class EvidenceSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strongest_supported_findings: list[str] = Field(
        ..., description="Findings with the clearest source support"
    )
    weak_or_unverified_findings: list[str] = Field(
        ..., description="Claims that need more source validation or live provider data"
    )
    source_coverage_notes: list[str] = Field(
        ..., description="Notes about source coverage, gaps, duplicates, or market-specific limitations"
    )


class AnalyticsReportOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source: str = Field(
        ...,
        description=(
            "Source of the underlying data, such as external_platform_api, "
            "provider_ready_stub, or development_fallback_sample"
        ),
    )
    confidence_level: str = Field(
        ..., description="Confidence level for the analytics report"
    )
    assumptions: list[str] = Field(
        ..., description="Assumptions and validation caveats for the analysis"
    )
    executive_summary: str = Field(
        ..., description="High-level overview of performance and key findings"
    )
    regional_kpis: list[RegionalKPI] = Field(
        ..., description="Key performance indicators broken down by region"
    )
    competitive_insights: str = Field(
        ..., description="Competitor benchmarking and market positioning analysis"
    )
    market_intelligence_by_region: list[MarketIntelligence] = Field(
        ...,
        description=(
            "Detailed per-region market intelligence so search and deep-read evidence "
            "is preserved instead of compressed into the executive summary"
        ),
    )
    source_evidence: list[SourceEvidence] = Field(
        ...,
        description=(
            "Source-backed evidence items for competitive insights, including "
            "claim, market, URLs, and claim-level confidence"
        ),
    )
    evidence_synthesis: EvidenceSynthesis = Field(
        ..., description="Cross-source synthesis of what is well supported and what remains weak"
    )
    actionable_recommendations: list[str] = Field(
        ..., description="Prioritized next steps for optimization"
    )
    risk_alerts: list[str] = Field(
        ..., description="Potential risks, inventory issues, or policy changes"
    )
    data_quality_notes: list[str] = Field(
        ..., description="Data quality caveats separating fallback metrics from live search evidence"
    )
    recommended_next_research: list[str] = Field(
        ..., description="Follow-up research or integrations needed to improve confidence"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_analysis_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = []
    if config_context.get("serper_api_key"):
        tools.append(SerperDevTool())
    return tools


def _build_research_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [
        CompetitorBenchmarkTool(
            serper_api_key=config_context.get("serper_api_key"),
            deep_read_enabled=bool(config_context.get("serper_deep_read_enabled")),
            deep_read_max_pages=int(config_context.get("serper_deep_read_max_pages") or 3),
            deep_read_concurrency=int(config_context.get("serper_deep_read_concurrency") or 5),
            deep_read_timeout_seconds=int(config_context.get("serper_deep_read_timeout_seconds") or 10),
            deep_read_max_chars=int(config_context.get("serper_deep_read_max_chars") or 4000),
        ),
        ScrapeWebsiteTool(),
    ]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


def _memory_enabled(config_context: dict[str, Any]) -> bool:
    return bool(config_context.get("crewai_memory_enabled"))


def _provider_status(config_context: dict[str, Any]) -> dict[str, Any]:
    has_shopify = bool(
        config_context.get("shopify_store_domain")
        and (config_context.get("shopify_admin_access_token") or config_context.get("ecom_api_token"))
    )
    has_amazon = bool(
        config_context.get("amazon_sp_api_endpoint")
        and (config_context.get("amazon_sp_api_access_token") or config_context.get("ecom_api_token"))
        and config_context.get("amazon_marketplace_ids")
    )
    has_ecom_token = has_shopify or has_amazon
    has_serper_token = bool(config_context.get("serper_api_key"))

    if not has_ecom_token and not has_serper_token:
        return {
            "data_source": "development_fallback",
            "confidence_level": "Illustrative",
            "assumptions": [
                "Analytics used development fallback metrics because ECOM_API_TOKEN is not configured.",
                "Competitive benchmarking also used fallback data because SERPER_API_KEY is not configured.",
                "Sales, conversion, ROAS, market, and competitor insights are sample values until validated with live platform data.",
            ],
        }

    if not has_ecom_token:
        return {
            "data_source": "mixed",
            "confidence_level": "Low",
            "assumptions": [
                "Analytics platform metrics used development fallback data because ECOM_API_TOKEN is not configured.",
                "Competitive research uses live Serper search snippets and optional source-page deep reads when SERPER_API_KEY is configured, but regional KPIs remain illustrative until connected to a live commerce platform.",
            ],
        }

    return {
        "data_source": "external_commerce_api",
        "confidence_level": "Medium",
        "assumptions": [
            "E-commerce metrics are fetched from configured Shopify Admin API or Amazon SP-API order endpoints when available.",
            "Order APIs do not provide full-funnel conversion, CPC, ROAS, or inventory health without additional analytics, ad, and inventory integrations.",
            "Competitor intelligence is not connected to a dedicated market-data provider; validate market claims before using them operationally.",
        ],
    }


def _apply_provider_status(result: dict[str, Any], config_context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.update(_provider_status(config_context))
    return normalized


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_analytics_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='analytics')
    tasks_config = _load_yaml_config("tasks.yaml")

    collector = Agent(
        config=agents_config["data_collector"],
        tools=[
            EcomPlatformMetricsTool(
                ecom_api_token=config_context.get("ecom_api_token"),
                shopify_store_domain=config_context.get("shopify_store_domain"),
                shopify_admin_access_token=config_context.get("shopify_admin_access_token"),
                shopify_api_version=config_context.get("shopify_api_version") or "2025-07",
                amazon_sp_api_endpoint=config_context.get("amazon_sp_api_endpoint"),
                amazon_sp_api_access_token=config_context.get("amazon_sp_api_access_token"),
                amazon_marketplace_ids=config_context.get("amazon_marketplace_ids"),
            )
        ],
    )
    analyst = Agent(
        config=agents_config["data_analyst"],
        tools=_build_analysis_tools(config_context),
    )
    researcher = Agent(
        config=agents_config["market_researcher"],
        tools=_build_research_tools(config_context),
    )
    reporter = Agent(
        config=agents_config["report_generator"],
        tools=_build_analysis_tools(config_context),
    )

    collect_task = Task(config=tasks_config["data_collection"], agent=collector)
    analyze_task = Task(
        config=tasks_config["performance_analysis"],
        agent=analyst,
        context=[collect_task],
    )
    research_task = Task(
        config=tasks_config["market_research"],
        agent=researcher,
        context=[collect_task],
    )
    report_task = Task(
        config=tasks_config["insight_report"],
        agent=reporter,
        context=[analyze_task, research_task],
        output_pydantic=AnalyticsReportOutput,
    )
    tasks = [collect_task, analyze_task, research_task, report_task]
    attach_task_progress(config_context, "analytics", tasks, list(tasks_config.keys()))

    analytics_crew = Crew(
        agents=[collector, analyst, researcher, reporter],
        tasks=tasks,
        verbose=False,
        memory=_memory_enabled(config_context),
    )

    result = _serialize_crew_result(analytics_crew.kickoff(inputs=inputs))
    return _apply_provider_status(result, config_context)
