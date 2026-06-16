from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field
from utils.crew_memory import build_crew_memory
from utils.crew_result import serialize_crew_result
from utils.llm_config import build_llm
from utils.project_intelligence import augment_agents_config
from utils.workflow_progress import attach_task_progress

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "business_development"
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

    data_source: str = Field(
        ...,
        description=(
            "Provider status for lead enrichment, such as development_fallback, "
            "mixed, or live_provider."
        ),
    )
    confidence_level: str = Field(
        ...,
        description="Confidence level based on whether real B2B provider data was available",
    )
    assumptions: list[str] = Field(
        ..., description="Important caveats about fallback data, inferred leads, or placeholders"
    )
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


def _build_research_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [
        B2BLeadLookupTool(
            crunchbase_api_key=config_context.get("crunchbase_api_key"),
            apollo_api_key=config_context.get("apollo_api_key"),
        )
    ]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


def _build_strategy_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool()]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


def _crew_memory(config_context: dict[str, Any]) -> Any:
    return build_crew_memory(config_context, workflow="bizdev")


def _provider_status(config_context: dict[str, Any]) -> dict[str, Any]:
    has_b2b_provider_key = bool(
        config_context.get("crunchbase_api_key") or config_context.get("apollo_api_key")
    )
    if not has_b2b_provider_key:
        return {
            "data_source": "development_fallback",
            "confidence_level": "Illustrative",
            "assumptions": [
                "Business Development lead enrichment used development fallback data because CRUNCHBASE_API_KEY and APOLLO_API_KEY are not configured.",
                "Company contacts, milestones, and partnership fit scores are placeholders until validated with a live B2B data provider.",
            ],
        }

    return {
        "data_source": "provider_ready_stub",
        "confidence_level": "Low",
        "assumptions": [
            "A B2B provider key is configured, but the current B2B lead lookup tool still uses a provider-ready placeholder endpoint.",
            "Lead records should be validated after connecting a real Apollo, Crunchbase, or compliant B2B data provider implementation.",
        ],
    }


def _apply_provider_status(result: dict[str, Any], config_context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.update(_provider_status(config_context))
    return normalized


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_bizdev_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='bizdev')
    tasks_config = _load_yaml_config("tasks.yaml")
    llm = build_llm(config_context)

    prospector = Agent(
        config=agents_config["lead_prospector"],
        llm=llm,
        tools=_build_research_tools(config_context),
    )
    strategist = Agent(
        config=agents_config["partnership_strategist"],
        llm=llm,
        tools=_build_strategy_tools(config_context),
    )
    outreach_agent = Agent(
        config=agents_config["outreach_specialist"],
        llm=llm,
        tools=[OutreachToneValidator()],
    )
    pipeline_agent = Agent(
        config=agents_config["pipeline_manager"],
        llm=llm,
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
    tasks = [research_task, strategy_task, outreach_task, sync_task]
    attach_task_progress(config_context, "bizdev", tasks, list(tasks_config.keys()))

    bizdev_crew = Crew(
        agents=[prospector, strategist, outreach_agent, pipeline_agent],
        tasks=tasks,
        verbose=False,
        memory=_crew_memory(config_context),
    )

    result = _serialize_crew_result(bizdev_crew.kickoff(inputs=inputs))
    return _apply_provider_status(result, config_context)
