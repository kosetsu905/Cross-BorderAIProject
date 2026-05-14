import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "content"
DEFAULT_MODEL = "gpt-4o-mini"


class SocialMediaPost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(..., description="Target platform")
    language: str = Field(..., description="Language code such as en, de, or ja")
    content: str = Field(..., description="Optimized post content with hashtags or mentions")


class ContentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article: str = Field(..., description="Full blog or article in markdown format")
    social_media_posts: list[SocialMediaPost] = Field(
        ..., description="Platform and language specific social posts"
    )
    seo_keywords: list[str] = Field(
        ..., description="Primary and secondary SEO keywords used"
    )
    compliance_notes: str = Field(..., description="Quality and compliance review notes")


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_search_tools() -> list[Any]:
    tools: list[Any] = []
    if os.getenv("SERPER_API_KEY"):
        tools.append(SerperDevTool())
    return tools


def _build_creation_tools() -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool()]
    if os.getenv("SERPER_API_KEY"):
        tools.append(SerperDevTool())
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


def run_content_crew(inputs: dict[str, Any]) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    trend_monitor = Agent(
        config=agents_config["trend_monitor"],
        tools=_build_search_tools(),
    )
    content_strategist = Agent(
        config=agents_config["content_strategist"],
        tools=_build_search_tools(),
    )
    multilingual_creator = Agent(
        config=agents_config["multilingual_creator"],
        tools=_build_creation_tools(),
    )
    qa_specialist = Agent(
        config=agents_config["qa_specialist"],
    )

    research_task = Task(config=tasks_config["trend_research"], agent=trend_monitor)
    strategy_task = Task(
        config=tasks_config["content_strategy"],
        agent=content_strategist,
        context=[research_task],
    )
    creation_task = Task(
        config=tasks_config["content_creation"],
        agent=multilingual_creator,
        context=[strategy_task, research_task],
    )
    qa_task = Task(
        config=tasks_config["quality_assurance"],
        agent=qa_specialist,
        context=[creation_task],
        output_pydantic=ContentOutput,
    )

    content_crew = Crew(
        agents=[trend_monitor, content_strategist, multilingual_creator, qa_specialist],
        tasks=[research_task, strategy_task, creation_task, qa_task],
        verbose=False,
        memory=_memory_enabled(),
    )

    return _serialize_crew_result(content_crew.kickoff(inputs=inputs))
