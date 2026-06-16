from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from models import AnalyticsInputs
from tools.custom.analytics_tools import (
    AdvancedAttributionTool,
    ChatBISQLPreviewTool,
    ClosedLoopAutomationPlanTool,
    CompetitorBenchmarkTool,
    EcomPlatformMetricsTool,
    GlobalMacroRiskTool,
    PredictiveAnomalyTool,
)
from utils.crew_result import serialize_crew_result
from utils.llm_config import build_llm
from utils.project_intelligence import augment_agents_config
from utils.workflow_progress import attach_task_progress

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "analytics"

class RegionalKPI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Target market or region")
    sales_volume: str = Field(
        ...,
        description="Total sales or revenue with an ISO 4217 currency code, never a bare currency symbol",
    )
    currency_code: str = Field(..., description="ISO 4217 currency code for sales_volume")
    conversion_rate: str = Field(..., description="Conversion rate percentage")
    roas: str = Field(..., description="Return on ad spend")


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(..., description="Target market or region for the claim")
    claim: str = Field(..., description="Source-backed competitive or market claim")
    evidence_summary: str = Field(..., description="Short summary of the supporting evidence")
    sources: list[str] = Field(..., description="Source URLs used to support the claim")
    confidence: str = Field(..., description="Evidence confidence level for this claim")
    currency_context: str = Field(
        ...,
        description="ISO 4217 currency context for monetary claims, or not_applicable/source_unspecified",
    )
    market_alignment: str = Field(
        ...,
        description="same_market, mixed_market, cross_market_reference, or unknown_market",
    )
    source_ids: list[str] = Field(
        ..., description="Source bibliography IDs supporting this evidence item"
    )


class MarketVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(..., description="Target market or region")
    performance_verdict: str = Field(
        ..., description="Answer-first market verdict based on available platform and public evidence"
    )
    opportunity_assessment: str = Field(..., description="Main opportunity or upside in this market")
    risk_assessment: str = Field(..., description="Main risk, constraint, or uncertainty in this market")
    key_findings: list[str] = Field(
        ..., description="Short market-specific findings, separating facts from inferences"
    )
    evidence_ids: list[str] = Field(..., description="Source bibliography IDs used for this verdict")
    confidence: str = Field(..., description="Confidence level for this market verdict")


class PublicMarketFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(..., description="Target market or region for the public-source fact")
    fact_type: str = Field(..., description="Type such as sales, pricing, availability, policy, demand, or competitor")
    statement: str = Field(..., description="Source-backed factual statement")
    value: str = Field(..., description="Specific value if present, otherwise not_applicable")
    time_period: str = Field(..., description="Actual time window from the source, or source_unspecified")
    source_ids: list[str] = Field(..., description="Source bibliography IDs supporting this fact")
    confidence: str = Field(..., description="Confidence level for this fact")


class SourceBibliographyItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., description="Stable source identifier such as S1")
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    domain: str = Field(..., description="Source domain")
    market: str = Field(..., description="Assigned or most relevant market for this source")
    source_type: str = Field(..., description="Source type such as organic, shopping, or page_excerpt")
    read_status: str = Field(..., description="Deep-read status such as ok, failed, skipped, or snippet_only")
    market_alignment: str = Field(..., description="same_market, mixed_market, cross_market_reference, or unknown_market")
    currency_context: str = Field(..., description="ISO currency context, not_applicable, or source_unspecified")
    key_points: list[str] = Field(..., description="Short source-backed points useful for the report")
    reliability: str = Field(..., description="Reliability assessment for this source")


class MarketIntelligence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Target market or region")
    market_currency_code: str = Field(..., description="Default ISO 4217 currency code for this market")
    product_availability: str = Field(
        ...,
        description="Whether the product/category is available, unavailable, or unverified in this market",
    )
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


class AttributionChannelContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str = Field(..., description="Marketing, marketplace, or organic channel")
    contribution_pct: float = Field(..., description="Observed contribution share as a percentage")
    basis: str = Field(..., description="Metric used for the contribution calculation")


class AttributionBudgetRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str = Field(..., description="Channel receiving the recommendation")
    recommendation: str = Field(..., description="Budget recommendation or testing posture")
    rationale: str = Field(..., description="Reason grounded in provided metrics")


class AdvancedAttributionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="computed, insufficient_data, or skipped")
    confidence_level: str = Field(..., description="Confidence level for attribution")
    method: str = Field(..., description="Attribution method used or missing-data reason")
    channel_contributions: list[AttributionChannelContribution] = Field(
        ..., description="Observed channel contribution shares"
    )
    did_incremental_lift_pct: float | None = Field(
        None, description="Incremental lift percentage when treatment/control data is provided"
    )
    true_roi: float | None = Field(
        None, description="Incremental ROI when revenue and cost are provided"
    )
    budget_recommendations: list[AttributionBudgetRecommendation] = Field(
        ..., description="Conservative budget recommendations"
    )
    data_quality_notes: list[str] = Field(..., description="Attribution caveats")


class MacroFxRate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair: str = Field(..., description="Currency pair")
    rate: float | None = Field(None, description="Provided FX rate")
    source: str = Field(..., description="Source of the FX rate")


class MacroTariffAlert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Market or region")
    policy_change: str = Field(..., description="Policy or tariff change")
    effective: str = Field(..., description="Effective date or source-unspecified marker")
    source: str = Field(..., description="Source of the alert")


class MacroRiskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="computed, insufficient_data, or skipped")
    confidence_level: str = Field(..., description="Confidence level for macro risk")
    base_currency: str = Field(..., description="ISO 4217 base currency")
    target_markets: str = Field(..., description="Requested target markets")
    fx_rates: list[MacroFxRate] = Field(..., description="Provided FX rates")
    tariff_alerts: list[MacroTariffAlert] = Field(..., description="Provided tariff or policy alerts")
    margin_impact_pct: float | None = Field(None, description="Provided or derived margin impact percentage")
    risk_level: str = Field(..., description="Macro risk level")
    strategic_recommendations: list[str] = Field(..., description="Macro-risk mitigations")
    data_quality_notes: list[str] = Field(..., description="Macro data caveats")


class PredictiveForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: str = Field(..., description="Forecast horizon period")
    predicted_value: float | None = Field(None, description="Forecast value from provided history")
    basis: str = Field(..., description="Forecast basis")


class PredictiveAnomaly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: str = Field(..., description="Anomaly period")
    metric: str = Field(..., description="Anomalous metric")
    severity: str = Field(..., description="Severity label")
    description: str = Field(..., description="Reason or description")


class PredictiveInsightsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="computed, insufficient_data, or skipped")
    confidence_level: str = Field(..., description="Confidence level for predictive insights")
    product_category: str = Field(..., description="Analyzed product category")
    forecast_14d: list[PredictiveForecastPoint] = Field(..., description="14-period forecast")
    anomalies: list[PredictiveAnomaly] = Field(..., description="Detected or provided anomalies")
    model_confidence: str = Field(..., description="Model confidence explanation")
    data_quality_notes: list[str] = Field(..., description="Predictive analytics caveats")


class ChatBIResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="preview_only, skipped, or error")
    intent: str = Field(..., description="Classified query intent")
    generated_sql: str = Field(..., description="Safe SQL preview, never executed")
    business_insight: str = Field(..., description="Business interpretation of the preview")
    safety_notes: list[str] = Field(..., description="Safety and execution caveats")


class AutomationPlanAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(..., description="Planned automation action type")
    platform: str = Field(..., description="Target platform")
    target: str = Field(..., description="SKU, campaign, channel, or other target")
    status: str = Field(..., description="planned, skipped, or insufficient_data")
    execution_mode: str = Field(..., description="Always dry_run for analytics 1.1")
    reason: str = Field(..., description="Reason the action was planned or skipped")
    required_credentials: list[str] = Field(..., description="Credentials required for future live execution")


class AutomationPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="planned, skipped, or insufficient_data")
    execution_mode: str = Field(..., description="Always dry_run")
    actions: list[AutomationPlanAction] = Field(..., description="Dry-run automation actions")
    data_quality_notes: list[str] = Field(..., description="Automation safety caveats")


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
    reporting_currency: str = Field(
        ...,
        description="ISO 4217 currency used for platform KPI reporting, usually the request currency",
    )
    executive_summary: str = Field(
        ..., description="High-level overview of performance and key findings"
    )
    answer_first_summary: list[str] = Field(
        ..., description="Three to five direct conclusions before detailed market analysis"
    )
    regional_kpis: list[RegionalKPI] = Field(
        ..., description="Key performance indicators broken down by region"
    )
    market_verdicts: list[MarketVerdict] = Field(
        ..., description="One answer-first verdict per requested market with evidence IDs"
    )
    public_market_facts: list[PublicMarketFact] = Field(
        ..., description="Public web-source facts such as sales, pricing, availability, policy, demand, or competitor facts"
    )
    source_bibliography: list[SourceBibliographyItem] = Field(
        ..., description="Numbered public-source bibliography used by facts and verdicts"
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
    advanced_attribution: AdvancedAttributionOutput = Field(
        ..., description="Analytics 1.1 attribution and causal inference summary"
    )
    macro_risk: MacroRiskOutput = Field(
        ..., description="Analytics 1.1 global macro and risk fusion summary"
    )
    predictive_insights: PredictiveInsightsOutput = Field(
        ..., description="Analytics 1.1 forecast and anomaly summary"
    )
    chatbi_response: ChatBIResponseOutput = Field(
        ..., description="Analytics 1.1 ChatBI SQL preview response"
    )
    automation_plan: AutomationPlanOutput = Field(
        ..., description="Analytics 1.1 closed-loop automation dry-run plan"
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
    tools: list[Any] = [
        AdvancedAttributionTool(),
        ChatBISQLPreviewTool(),
    ]
    if config_context.get("serper_api_key"):
        tools.append(SerperDevTool())
    return tools


def _build_collector_tools(
    platform_metrics_tool: EcomPlatformMetricsTool,
) -> list[Any]:
    return [
        platform_metrics_tool,
        PredictiveAnomalyTool(),
    ]


def _build_research_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [
        GlobalMacroRiskTool(),
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


def _workflow_async_enabled(config_context: dict[str, Any]) -> bool:
    value = config_context.get("workflow_async_execution_enabled", True)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _has_configured_commerce_provider(config_context: dict[str, Any]) -> bool:
    has_shopify = bool(
        config_context.get("shopify_store_domain")
        and (config_context.get("shopify_admin_access_token") or config_context.get("ecom_api_token"))
    )
    has_amazon = bool(
        config_context.get("amazon_sp_api_endpoint")
        and (config_context.get("amazon_sp_api_access_token") or config_context.get("ecom_api_token"))
        and config_context.get("amazon_marketplace_ids")
    )
    return has_shopify or has_amazon


def _provider_status(
    config_context: dict[str, Any],
    has_live_platform_metrics: bool,
) -> dict[str, Any]:
    has_ecom_token = _has_configured_commerce_provider(config_context)
    has_serper_token = bool(config_context.get("serper_api_key"))

    if not has_live_platform_metrics and not has_serper_token:
        return {
            "data_source": "platform_metrics_unavailable",
            "confidence_level": "Low",
            "assumptions": [
                "Platform KPIs are unavailable because no successful Shopify or Amazon commerce metrics call completed.",
                "regional_kpis is intentionally empty; sales volume, conversion rate, ROAS, CPC, inventory, and SKU metrics are not estimated.",
                "Competitive benchmarking also used fallback data because SERPER_API_KEY is not configured.",
            ],
        }

    if not has_live_platform_metrics:
        platform_reason = (
            "Configured commerce provider did not return live metrics."
            if has_ecom_token
            else "Shopify or Amazon commerce provider credentials are not configured."
        )
        return {
            "data_source": "mixed",
            "confidence_level": "Low",
            "assumptions": [
                f"Platform KPIs are unavailable: {platform_reason}",
                "regional_kpis is intentionally empty; sales volume, conversion rate, ROAS, CPC, inventory, and SKU metrics are not estimated.",
                "Competitive research uses live Serper search snippets and optional source-page deep reads when SERPER_API_KEY is configured, but those market signals are not platform KPIs.",
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


def _requested_currency(inputs: dict[str, Any]) -> str:
    value = str(inputs.get("currency") or "USD").strip().upper()
    return value if value else "USD"


def _sales_volume_with_currency(value: Any, currency_code: str) -> str:
    text = str(value or "").strip()
    if not text:
        return f"{currency_code} 0"
    upper_text = text.upper()
    if upper_text.startswith(f"{currency_code} "):
        return text
    if text.startswith("$"):
        return f"{currency_code} {text[1:].strip()}"
    return text


def _apply_currency_context(result: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    reporting_currency = _requested_currency(inputs)
    normalized["reporting_currency"] = str(
        normalized.get("reporting_currency") or reporting_currency
    ).strip().upper()

    regional_kpis = normalized.get("regional_kpis")
    if isinstance(regional_kpis, list):
        normalized_kpis: list[Any] = []
        for item in regional_kpis:
            if not isinstance(item, dict):
                normalized_kpis.append(item)
                continue
            kpi = dict(item)
            currency_code = str(kpi.get("currency_code") or normalized["reporting_currency"]).strip().upper()
            kpi["currency_code"] = currency_code
            kpi["sales_volume"] = _sales_volume_with_currency(kpi.get("sales_volume"), currency_code)
            normalized_kpis.append(kpi)
        normalized["regional_kpis"] = normalized_kpis

    source_evidence = normalized.get("source_evidence")
    if isinstance(source_evidence, list):
        normalized_evidence: list[Any] = []
        for item in source_evidence:
            if not isinstance(item, dict):
                normalized_evidence.append(item)
                continue
            evidence = dict(item)
            evidence.setdefault("currency_context", "source_unspecified")
            evidence.setdefault("market_alignment", "unknown_market")
            evidence.setdefault("source_ids", [])
            normalized_evidence.append(evidence)
        normalized["source_evidence"] = normalized_evidence

    return normalized


def _append_unique_text(values: Any, text: str) -> list[str]:
    items = [str(item) for item in values] if isinstance(values, list) else []
    if text not in items:
        items.append(text)
    return items


def _apply_platform_kpi_guardrail(
    result: dict[str, Any],
    has_live_platform_metrics: bool,
) -> dict[str, Any]:
    normalized = dict(result)
    if has_live_platform_metrics:
        return normalized

    normalized["regional_kpis"] = []
    normalized["data_quality_notes"] = _append_unique_text(
        normalized.get("data_quality_notes"),
        (
            "Platform KPI metrics are unavailable from live commerce providers; "
            "regional_kpis is intentionally empty rather than populated with fallback values."
        ),
    )
    return normalized


def _apply_provider_status(
    result: dict[str, Any],
    config_context: dict[str, Any],
    inputs: dict[str, Any],
    has_live_platform_metrics: bool,
) -> dict[str, Any]:
    normalized = dict(result)
    normalized.update(_provider_status(config_context, has_live_platform_metrics))
    normalized = _apply_platform_kpi_guardrail(normalized, has_live_platform_metrics)
    return _apply_currency_context(normalized, inputs)


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_analytics_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}
    normalized_inputs = AnalyticsInputs.model_validate(inputs).model_dump()

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='analytics')
    tasks_config = _load_yaml_config("tasks.yaml")
    llm = build_llm(config_context)

    platform_metrics_tool = EcomPlatformMetricsTool(
        ecom_api_token=config_context.get("ecom_api_token"),
        shopify_store_domain=config_context.get("shopify_store_domain"),
        shopify_admin_access_token=config_context.get("shopify_admin_access_token"),
        shopify_api_version=config_context.get("shopify_api_version") or "2025-07",
        amazon_sp_api_endpoint=config_context.get("amazon_sp_api_endpoint"),
        amazon_sp_api_access_token=config_context.get("amazon_sp_api_access_token"),
        amazon_marketplace_ids=config_context.get("amazon_marketplace_ids"),
    )
    async_enabled = _workflow_async_enabled(config_context)

    collector = Agent(
        config=agents_config["data_collector"],
        llm=llm,
        tools=_build_collector_tools(platform_metrics_tool),
    )
    analyst = Agent(
        config=agents_config["data_analyst"],
        llm=llm,
        tools=_build_analysis_tools(config_context),
    )
    researcher = Agent(
        config=agents_config["market_researcher"],
        llm=llm,
        tools=_build_research_tools(config_context),
    )
    reporter = Agent(
        config=agents_config["report_generator"],
        llm=llm,
        tools=_build_analysis_tools(config_context),
    )
    automation_planner = Agent(
        config=agents_config["automation_planner"],
        llm=llm,
        tools=[ClosedLoopAutomationPlanTool()],
    )

    collect_task = Task(config=tasks_config["data_collection"], agent=collector)
    analyze_task = Task(
        config=tasks_config["performance_analysis"],
        agent=analyst,
        context=[collect_task],
        async_execution=async_enabled,
    )
    research_task = Task(
        config=tasks_config["market_research"],
        agent=researcher,
        context=[collect_task],
        async_execution=async_enabled,
    )
    automation_task = Task(
        config=tasks_config["automation_planning"],
        agent=automation_planner,
        context=[collect_task, analyze_task, research_task],
    )
    report_task = Task(
        config=tasks_config["insight_report"],
        agent=reporter,
        context=[collect_task, analyze_task, research_task, automation_task],
        output_pydantic=AnalyticsReportOutput,
    )
    tasks = [collect_task, analyze_task, research_task, automation_task, report_task]
    attach_task_progress(config_context, "analytics", tasks, list(tasks_config.keys()))

    analytics_crew = Crew(
        agents=[collector, analyst, researcher, automation_planner, reporter],
        tasks=tasks,
        verbose=False,
        memory=_memory_enabled(config_context),
    )

    result = _serialize_crew_result(analytics_crew.kickoff(inputs=normalized_inputs))
    return _apply_provider_status(
        result,
        config_context,
        normalized_inputs,
        platform_metrics_tool.successful_live_fetch_count > 0,
    )

