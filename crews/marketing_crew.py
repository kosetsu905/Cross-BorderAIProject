import os
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

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "marketing"
DEFAULT_MODEL = "gpt-4o-mini"


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


def run_marketing_crew(inputs: dict[str, Any]) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    strategist = Agent(
        config=agents_config["campaign_strategist"],
        tools=_build_research_tools(),
    )
    copywriter = Agent(
        config=agents_config["ad_copywriter"],
        tools=[KeywordResearchTool()],
    )
    optimizer = Agent(
        config=agents_config["channel_optimizer"],
        tools=[PlatformAdSpecsTool()],
    )
    qa_agent = Agent(
        config=agents_config["compliance_qa_specialist"],
        tools=[ComplianceCheckerTool()],
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
        memory=_memory_enabled(),
    )

    return _serialize_crew_result(marketing_crew.kickoff(inputs=inputs))
