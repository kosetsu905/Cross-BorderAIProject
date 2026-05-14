import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.analytics_tools import CompetitorBenchmarkTool, EcomPlatformMetricsTool

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "analytics"
DEFAULT_MODEL = "gpt-4o-mini"


class RegionalKPI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="Target market or region")
    sales_volume: str = Field(..., description="Total sales or revenue")
    conversion_rate: str = Field(..., description="Conversion rate percentage")
    roas: str = Field(..., description="Return on ad spend")


class AnalyticsReportOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(
        ..., description="High-level overview of performance and key findings"
    )
    regional_kpis: list[RegionalKPI] = Field(
        ..., description="Key performance indicators broken down by region"
    )
    competitive_insights: str = Field(
        ..., description="Competitor benchmarking and market positioning analysis"
    )
    actionable_recommendations: list[str] = Field(
        ..., description="Prioritized next steps for optimization"
    )
    risk_alerts: list[str] = Field(
        ..., description="Potential risks, inventory issues, or policy changes"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_analysis_tools() -> list[Any]:
    tools: list[Any] = []
    if os.getenv("SERPER_API_KEY"):
        tools.append(SerperDevTool())
    return tools


def _build_research_tools() -> list[Any]:
    tools: list[Any] = [CompetitorBenchmarkTool(), ScrapeWebsiteTool()]
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


def run_analytics_crew(inputs: dict[str, Any]) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    collector = Agent(
        config=agents_config["data_collector"],
        tools=[EcomPlatformMetricsTool()],
    )
    analyst = Agent(
        config=agents_config["data_analyst"],
        tools=_build_analysis_tools(),
    )
    researcher = Agent(
        config=agents_config["market_researcher"],
        tools=_build_research_tools(),
    )
    reporter = Agent(
        config=agents_config["report_generator"],
        tools=_build_analysis_tools(),
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

    analytics_crew = Crew(
        agents=[collector, analyst, researcher, reporter],
        tasks=[collect_task, analyze_task, research_task, report_task],
        verbose=False,
        memory=_memory_enabled(),
    )

    return _serialize_crew_result(analytics_crew.kickoff(inputs=inputs))
