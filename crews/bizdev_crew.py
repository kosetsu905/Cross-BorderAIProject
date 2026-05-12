import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "business_development"
DEFAULT_MODEL = "gpt-4o-mini"

from tools.custom.bizdev_tools import (
    B2BLeadLookupTool,
    CRMFormatterTool,
    OutreachToneValidator,
)


class LeadProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str = Field(..., description="Target company name")
    region: str = Field(..., description="Target market or region")
    decision_maker_role: str = Field(..., description="Key contact title or role")
    partnership_fit: str = Field(..., description="Strategic alignment rationale")


class OutreachStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    touch_number: int = Field(..., description="Sequence step number")
    platform: str = Field(..., description="Channel such as Email or LinkedIn")
    subject_line: str = Field(..., description="Message subject or opening line")
    body: str = Field(..., description="Core outreach copy")
    cta: str = Field(..., description="Clear call to action")


class CRMLeadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str = Field(..., description="Lead company name")
    industry: str = Field("", description="Lead industry or category")
    region: str = Field(..., description="Lead market or region")
    status: str = Field(..., description="CRM lead status")
    source: str = Field(..., description="Lead source label")


class CRMActivityLogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    touch: int = Field(..., description="Touchpoint number")
    channel: str = Field(..., description="Outreach channel")
    content_preview: str = Field(..., description="Short preview of the outreach content")


class CRMPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crm_system: str = Field(..., description="Target-compatible CRM system family")
    lead_record: CRMLeadRecord = Field(..., description="Structured lead record")
    activity_log: list[CRMActivityLogItem] = Field(
        ..., description="Outreach activity log entries"
    )
    next_action_date: str = Field(..., description="Next recommended follow-up timing")
    status: str = Field(..., description="Formatter status")


class BizDevOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_leads: list[LeadProfile] = Field(
        ..., description="Curated list of high-potential partnership leads"
    )
    value_proposition: str = Field(
        ..., description="Tailored partnership value proposition summary"
    )
    outreach_sequences: list[OutreachStep] = Field(
        ..., description="Multi-touch, localized outreach sequence"
    )
    follow_up_cadence: list[str] = Field(
        ..., description="30-day follow-up timeline and triggers"
    )
    crm_payload: CRMPayload = Field(
        ..., description="Structured CRM-ready JSON payload"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_research_tools() -> list[Any]:
    tools: list[Any] = [B2BLeadLookupTool()]
    if os.getenv("SERPER_API_KEY"):
        tools.insert(0, SerperDevTool())
    return tools


def _build_strategy_tools() -> list[Any]:
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


def run_bizdev_crew(inputs: dict[str, Any]) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    prospector = Agent(
        config=agents_config["lead_prospector"],
        tools=_build_research_tools(),
    )
    strategist = Agent(
        config=agents_config["partnership_strategist"],
        tools=_build_strategy_tools(),
    )
    outreach_agent = Agent(
        config=agents_config["outreach_specialist"],
        tools=[OutreachToneValidator()],
    )
    pipeline_agent = Agent(
        config=agents_config["pipeline_manager"],
        tools=[CRMFormatterTool()],
    )

    research_task = Task(config=tasks_config["lead_research"], agent=prospector)
    strategy_task = Task(
        config=tasks_config["strategy_mapping"],
        agent=strategist,
        context=[research_task],
    )
    outreach_task = Task(
        config=tasks_config["outreach_creation"],
        agent=outreach_agent,
        context=[strategy_task, research_task],
    )
    sync_task = Task(
        config=tasks_config["pipeline_sync"],
        agent=pipeline_agent,
        context=[outreach_task, research_task],
        output_pydantic=BizDevOutput,
    )

    bizdev_crew = Crew(
        agents=[prospector, strategist, outreach_agent, pipeline_agent],
        tasks=[research_task, strategy_task, outreach_task, sync_task],
        verbose=False,
        memory=_memory_enabled(),
    )

    return _serialize_crew_result(bizdev_crew.kickoff(inputs=inputs))
