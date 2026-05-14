import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "support"
DEFAULT_MODEL = "gpt-4o-mini"


class SupportTicketOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_summary: str = Field(..., description="Brief summary of the customer inquiry")
    resolution_steps: list[str] = Field(
        ..., description="Step-by-step resolution or troubleshooting guide"
    )
    drafted_response: str = Field(
        ..., description="Final, polished response ready to send to the customer"
    )
    qa_notes: str = Field(..., description="Quality assurance feedback and compliance checks")
    recommended_follow_up: str = Field(
        ..., description="Suggested next steps or follow-up timing"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_support_tools() -> list[Any]:
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


def run_support_crew(inputs: dict[str, Any]) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    support_agent = Agent(
        config=agents_config["senior_support_agent"],
        tools=_build_support_tools(),
        allow_delegation=True,
    )
    qa_agent = Agent(
        config=agents_config["support_qa_specialist"],
        allow_delegation=False,
    )

    resolution_task = Task(
        config=tasks_config["inquiry_resolution"],
        agent=support_agent,
    )
    qa_task = Task(
        config=tasks_config["quality_assurance_review"],
        agent=qa_agent,
        context=[resolution_task],
        output_pydantic=SupportTicketOutput,
    )

    support_crew = Crew(
        agents=[support_agent, qa_agent],
        tasks=[resolution_task, qa_task],
        verbose=False,
        memory=_memory_enabled(),
    )

    return _serialize_crew_result(support_crew.kickoff(inputs=inputs))
