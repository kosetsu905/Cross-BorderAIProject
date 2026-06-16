from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.sales_tools import CRMFunnelTool, CROHeuristicsTool, PricingIntelTool
from utils.crew_memory import build_crew_memory
from utils.crew_result import serialize_crew_result
from utils.model_tiering import ModelTierRouter
from utils.project_intelligence import augment_agents_config
from utils.tool_cache import build_cached_scrape_tool, build_cached_serper_tool
from utils.workflow_progress import attach_task_progress

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "sales_improvement"


class CROTestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(..., description="A/B test hypothesis")
    expected_impact: str = Field(..., description="Expected impact or uplift")
    implementation_steps: str = Field(..., description="Implementation guidance")
    measurement_kpi: str = Field(..., description="Primary KPI to measure")
    priority: str = Field(..., description="Priority level")


class PricingRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Target region")
    recommendation: str = Field(..., description="Pricing or discount recommendation")
    margin_impact: str = Field(..., description="Expected margin impact")
    rationale: str = Field(..., description="Reason for the recommendation")


class SalesFunnelContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., description="Compact funnel analysis summary")
    bottlenecks: list[str] = Field(default_factory=list, description="Observed funnel bottlenecks")
    regional_gaps: list[str] = Field(default_factory=list, description="Regional performance gaps")
    assumptions: list[str] = Field(default_factory=list, description="Funnel data assumptions")
    data_quality_notes: list[str] = Field(default_factory=list, description="Funnel data caveats")


class SalesCROContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., description="Compact CRO recommendation summary")
    cro_test_matrix: list[CROTestItem] = Field(default_factory=list, description="Prioritized CRO tests")
    data_quality_notes: list[str] = Field(default_factory=list, description="CRO caveats")


class SalesPricingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., description="Compact pricing strategy summary")
    pricing_recommendations: list[PricingRecommendation] = Field(
        default_factory=list,
        description="Region-specific pricing recommendations",
    )
    data_quality_notes: list[str] = Field(default_factory=list, description="Pricing caveats")


class SalesImprovementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source: str = Field(
        ...,
        description=(
            "Source of the underlying data, such as external_crm_api or "
            "development_fallback_sample"
        ),
    )
    confidence_level: str = Field(
        ..., description="Confidence level for the recommendations"
    )
    assumptions: list[str] = Field(
        ..., description="Assumptions and validation caveats for the analysis"
    )
    funnel_analysis_summary: str = Field(
        ..., description="Executive summary of funnel bottlenecks and regional gaps"
    )
    cro_test_matrix: list[CROTestItem] = Field(
        ..., description="Prioritized A/B tests with hypotheses, impact, and KPIs"
    )
    pricing_recommendations: list[PricingRecommendation] = Field(
        ..., description="Region-specific pricing and discount recommendations"
    )
    implementation_roadmap: list[str] = Field(
        ..., description="30/60/90 day action plan with responsibilities"
    )
    risk_mitigation: list[str] = Field(
        ..., description="Implementation risks and recommended countermeasures"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_research_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [build_cached_scrape_tool(config_context, purpose="sales_improvement_research")]
    if config_context.get("serper_api_key"):
        tools.insert(0, build_cached_serper_tool(config_context, purpose="sales_improvement_research"))
    return tools


def _crew_memory(config_context: dict[str, Any]) -> Any:
    return build_crew_memory(config_context, workflow="sales_improvement")


def _workflow_async_enabled(config_context: dict[str, Any]) -> bool:
    value = config_context.get("workflow_async_execution_enabled", True)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _provider_status(config_context: dict[str, Any]) -> dict[str, Any]:
    has_shopify = bool(
        config_context.get("shopify_store_domain")
        and (config_context.get("shopify_admin_access_token") or config_context.get("crm_api_token"))
    )
    has_amazon = bool(
        config_context.get("amazon_sp_api_endpoint")
        and (config_context.get("amazon_sp_api_access_token") or config_context.get("crm_api_token"))
        and config_context.get("amazon_marketplace_ids")
    )
    if not (has_shopify or has_amazon):
        return {
            "data_source": "development_fallback",
            "confidence_level": "Illustrative",
            "assumptions": [
                "Sales Improvement used development fallback funnel data because Shopify or Amazon commerce API credentials are not configured.",
                "Funnel bottlenecks, regional gaps, pricing guidance, and expected uplifts are illustrative until validated with real CRM or commerce analytics.",
            ],
        }

    return {
        "data_source": "external_commerce_api",
        "confidence_level": "Medium",
        "assumptions": [
            "Funnel analysis uses configured Shopify Admin API or Amazon SP-API order data when available.",
            "Orders data can show revenue, order volume, cancellations, and regional gaps, but full product-page/cart/payment-step funnel analysis still requires analytics or CRM event data.",
            "CRO and pricing tools include heuristic or fallback guidance and should be validated against real funnel, margin, and competitor data before execution.",
        ],
    }


def _apply_provider_status(result: dict[str, Any], config_context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.update(_provider_status(config_context))
    return normalized


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_sales_improvement_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='sales_improvement')
    tasks_config = _load_yaml_config("tasks.yaml")
    llm_router = ModelTierRouter(config_context)
    async_enabled = _workflow_async_enabled(config_context)

    funnel_analyst = Agent(
        config=agents_config["funnel_analyst"],
        llm=llm_router.llm_for_agent(agents_config["funnel_analyst"]),
        tools=[
            CRMFunnelTool(
                crm_api_token=config_context.get("crm_api_token"),
                shopify_store_domain=config_context.get("shopify_store_domain"),
                shopify_admin_access_token=config_context.get("shopify_admin_access_token"),
                shopify_api_version=config_context.get("shopify_api_version") or "2025-07",
                amazon_sp_api_endpoint=config_context.get("amazon_sp_api_endpoint"),
                amazon_sp_api_access_token=config_context.get("amazon_sp_api_access_token"),
                amazon_marketplace_ids=config_context.get("amazon_marketplace_ids"),
                tool_cache_context=config_context,
            )
        ],
    )
    cro_specialist = Agent(
        config=agents_config["cro_specialist"],
        llm=llm_router.llm_for_agent(agents_config["cro_specialist"]),
        tools=[CROHeuristicsTool(), *_build_research_tools(config_context)],
    )
    pricing_strategist = Agent(
        config=agents_config["pricing_strategist"],
        llm=llm_router.llm_for_agent(agents_config["pricing_strategist"]),
        tools=[PricingIntelTool()],
    )
    playbook_coach = Agent(
        config=agents_config["playbook_coach"],
        llm=llm_router.llm_for_agent(agents_config["playbook_coach"]),
    )

    funnel_task = Task(
        config=tasks_config["funnel_analysis"],
        agent=funnel_analyst,
        output_pydantic=SalesFunnelContext,
    )
    cro_task = Task(
        config=tasks_config["cro_recommendations"],
        agent=cro_specialist,
        context=[funnel_task],
        async_execution=async_enabled,
        output_pydantic=SalesCROContext,
    )
    pricing_task = Task(
        config=tasks_config["pricing_optimization"],
        agent=pricing_strategist,
        context=[funnel_task],
        async_execution=async_enabled,
        output_pydantic=SalesPricingContext,
    )
    playbook_task = Task(
        config=tasks_config["playbook_generation"],
        agent=playbook_coach,
        context=[cro_task, pricing_task],
        output_pydantic=SalesImprovementOutput,
    )
    tasks = [funnel_task, cro_task, pricing_task, playbook_task]
    attach_task_progress(config_context, "sales_improvement", tasks, list(tasks_config.keys()))

    sales_improvement_crew = Crew(
        agents=[funnel_analyst, cro_specialist, pricing_strategist, playbook_coach],
        tasks=tasks,
        verbose=False,
        cache=True,
        memory=_crew_memory(config_context),
    )

    result = _serialize_crew_result(sales_improvement_crew.kickoff(inputs=inputs))
    return _apply_provider_status(result, config_context)
