import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.sales_tools import CRMFunnelTool, CROHeuristicsTool, PricingIntelTool

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "sales_improvement"
DEFAULT_MODEL = "gpt-4o-mini"


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


def _build_research_tools() -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool()]
    if os.getenv("SERPER_API_KEY"):
        tools.insert(0, SerperDevTool())
    return tools


def _memory_enabled() -> bool:
    return os.getenv("CREWAI_MEMORY_ENABLED", "false").lower() in {"1", "true", "yes"}


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    pydantic_result = getattr(result, "pydantic", None)
    if pydantic_result is not None:
        if hasattr(pydantic_result, "model_dump"):
            return pydantic_result.model_dump()
        if hasattr(pydantic_result, "dict"):
            return pydantic_result.dict()

    json_dict = getattr(result, "json_dict", None)
    if isinstance(json_dict, dict):
        return json_dict

    raw = getattr(result, "raw", None)
    if raw is not None:
        return {"raw": raw}

    if isinstance(result, dict):
        return result

    return {"raw": str(result)}


def run_sales_improvement_crew(inputs: dict[str, Any]) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    funnel_analyst = Agent(
        config=agents_config["funnel_analyst"],
        tools=[CRMFunnelTool()],
    )
    cro_specialist = Agent(
        config=agents_config["cro_specialist"],
        tools=[CROHeuristicsTool(), *_build_research_tools()],
    )
    pricing_strategist = Agent(
        config=agents_config["pricing_strategist"],
        tools=[PricingIntelTool()],
    )
    playbook_coach = Agent(
        config=agents_config["playbook_coach"],
    )

    funnel_task = Task(config=tasks_config["funnel_analysis"], agent=funnel_analyst)
    cro_task = Task(
        config=tasks_config["cro_recommendations"],
        agent=cro_specialist,
        context=[funnel_task],
    )
    pricing_task = Task(
        config=tasks_config["pricing_optimization"],
        agent=pricing_strategist,
        context=[funnel_task],
    )
    playbook_task = Task(
        config=tasks_config["playbook_generation"],
        agent=playbook_coach,
        context=[cro_task, pricing_task],
        output_pydantic=SalesImprovementOutput,
    )

    sales_improvement_crew = Crew(
        agents=[funnel_analyst, cro_specialist, pricing_strategist, playbook_coach],
        tasks=[funnel_task, cro_task, pricing_task, playbook_task],
        verbose=False,
        memory=_memory_enabled(),
    )

    return _serialize_crew_result(sales_improvement_crew.kickoff(inputs=inputs))
