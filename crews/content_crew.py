from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field
from utils.crew_result import serialize_crew_result

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "content"


class SocialMediaPost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(..., description="Target platform")
    language: str = Field(..., description="Language code such as en, de, or ja")
    content: str = Field(..., description="Optimized post content with hashtags or mentions")


class LocalizedArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="Language code such as en, de, ja, or zh-CN")
    title: str = Field(..., description="Localized article title")
    article: str = Field(..., description="Full localized blog or article in markdown format")


class ContentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    localized_articles: list[LocalizedArticle] = Field(
        ..., description="One localized article for each requested target language"
    )
    social_media_posts: list[SocialMediaPost] = Field(
        ..., description="Social posts covering requested languages and platforms"
    )
    seo_keywords: list[str] = Field(
        ..., description="Primary and secondary SEO keywords used"
    )
    compliance_notes: str = Field(..., description="Quality and compliance review notes")


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_search_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = []
    if config_context.get("serper_api_key"):
        tools.append(SerperDevTool())
    return tools


def _build_creation_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool()]
    if config_context.get("serper_api_key"):
        tools.append(SerperDevTool())
    return tools


def _memory_enabled(config_context: dict[str, Any]) -> bool:
    return bool(config_context.get("crewai_memory_enabled"))


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def _normalize_language(language: str) -> str:
    aliases = {
        "cn": "zh-CN",
        "zh": "zh-CN",
        "zh_cn": "zh-CN",
        "zh-cn": "zh-CN",
        "chinese": "zh-CN",
    }
    key = language.strip().lower().replace(" ", "")
    return aliases.get(key, language.strip())


def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    normalized["target_languages"] = [
        _normalize_language(str(language))
        for language in normalized.get("target_languages", [])
    ]
    return normalized


def _annotate_localized_output(result: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(result)
    expected_languages = {
        _normalize_language(str(language))
        for language in inputs.get("target_languages", [])
    }
    expected_platforms = {
        str(platform).strip()
        for platform in inputs.get("platforms", [])
    }
    article_languages = {
        _normalize_language(str(article.get("language", "")))
        for article in result.get("localized_articles", [])
    }
    post_languages = {
        _normalize_language(str(post.get("language", "")))
        for post in result.get("social_media_posts", [])
    }
    post_platforms = {
        str(post.get("platform", "")).strip()
        for post in result.get("social_media_posts", [])
    }

    missing_articles = sorted(expected_languages - article_languages)
    missing_post_languages = sorted(expected_languages - post_languages)
    missing_post_platforms = sorted(expected_platforms - post_platforms)

    if missing_articles or missing_post_languages or missing_post_platforms:
        notes = str(annotated.get("compliance_notes", "")).strip()
        coverage_note = (
            " Coverage notice: output did not include every requested item. "
            f"Missing localized_articles={missing_articles}; "
            f"missing social media languages={missing_post_languages}; "
            f"missing social media platforms={missing_post_platforms}."
        )
        annotated["compliance_notes"] = f"{notes}{coverage_note}".strip()

    return annotated


def run_content_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}
    normalized_inputs = _normalize_inputs(inputs)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    trend_monitor = Agent(
        config=agents_config["trend_monitor"],
        tools=_build_search_tools(config_context),
    )
    content_strategist = Agent(
        config=agents_config["content_strategist"],
        tools=_build_search_tools(config_context),
    )
    multilingual_creator = Agent(
        config=agents_config["multilingual_creator"],
        tools=_build_creation_tools(config_context),
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
        memory=_memory_enabled(config_context),
    )

    result = _serialize_crew_result(content_crew.kickoff(inputs=normalized_inputs))
    return _annotate_localized_output(result, normalized_inputs)
