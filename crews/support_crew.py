from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.support_rag_tools import SupportKnowledgeSearchTool
from utils.crew_result import serialize_crew_result

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "support"


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


def _build_support_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [
        SupportKnowledgeSearchTool(
            knowledge_dir=config_context.get("support_knowledge_dir")
            or str(BASE_DIR / "docs" / "knowledge_base")
        ),
        ScrapeWebsiteTool(),
    ]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


def _memory_enabled(config_context: dict[str, Any]) -> bool:
    return bool(config_context.get("crewai_memory_enabled"))


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_support_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    support_agent = Agent(
        config=agents_config["senior_support_agent"],
        tools=_build_support_tools(config_context),
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
        memory=_memory_enabled(config_context),
    )

    return _serialize_crew_result(support_crew.kickoff(inputs=inputs))
