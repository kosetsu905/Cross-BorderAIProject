from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field
from utils.crew_result import serialize_crew_result
from utils.usage_tracking import INTERNAL_USAGE_KEY
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, PROGRESS_SPAN, PROGRESS_START

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "content"


class SocialMediaPost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(..., description="Target platform")
    language: str = Field(..., description="Requested language code")
    content: str = Field(..., description="Optimized post content with hashtags or mentions")


class LocalizedArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="Requested language code")
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


class PerLanguageContentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    localized_article: LocalizedArticle = Field(
        ..., description="One localized article for the requested target language"
    )
    social_media_posts: list[SocialMediaPost] = Field(
        ..., description="Social posts for the requested language and platforms"
    )
    seo_keywords: list[str] = Field(
        ..., description="Primary and secondary SEO keywords used for this language"
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


def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    normalized["product_features_summary"] = _summarize_product_features(
        normalized.get("product_features")
    )
    return normalized


def _summarize_product_features(features: Any) -> str:
    if not features:
        return "No product-specific features were provided. Keep claims broad and avoid inventing product details."

    text = str(features).strip()
    if not text:
        return "No usable product-specific features were provided. Keep claims broad and avoid inventing product details."

    return text


def _split_markets(value: Any) -> list[str]:
    return [
        market.strip()
        for market in str(value or "").split(",")
        if market.strip()
    ]


def _target_market_for_language(inputs: dict[str, Any], language_index: int) -> str:
    markets = _split_markets(inputs.get("target_markets"))
    languages = [
        str(language).strip()
        for language in inputs.get("target_languages", [])
        if str(language).strip()
    ]
    if len(markets) == len(languages):
        return markets[language_index]
    return str(inputs.get("target_markets", ""))


def _content_language_concurrency(config_context: dict[str, Any]) -> int:
    try:
        value = int(config_context.get("content_language_concurrency") or 4)
    except (TypeError, ValueError):
        value = 4
    return max(1, min(value, 16))


def _progress_recorder(config_context: dict[str, Any]) -> Any | None:
    recorder = config_context.get(PROGRESS_CONTEXT_KEY)
    if all(hasattr(recorder, name) for name in ("emit_plan", "task_started", "task_completed")):
        return recorder
    return None


def _merge_language_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    articles: list[dict[str, Any]] = []
    social_posts: list[dict[str, Any]] = []
    seo_keywords: list[str] = []
    compliance_notes: list[str] = []

    for output in outputs:
        article = output.get("localized_article")
        if isinstance(article, dict):
            articles.append(article)
        social_posts.extend(output.get("social_media_posts") or [])
        for keyword in output.get("seo_keywords") or []:
            if keyword not in seo_keywords:
                seo_keywords.append(keyword)
        note = str(output.get("compliance_notes", "")).strip()
        if note:
            note_language = _output_language(output)
            prefix = f"[{note_language}] " if note_language else ""
            compliance_notes.append(f"{prefix}{note}")

    return {
        "localized_articles": articles,
        "social_media_posts": social_posts,
        "seo_keywords": seo_keywords,
        "compliance_notes": " ".join(compliance_notes),
    }


def _output_language(output: dict[str, Any]) -> str:
    article = output.get("localized_article")
    if isinstance(article, dict) and article.get("language"):
        return str(article["language"]).strip()

    posts = output.get("social_media_posts") or []
    for post in posts:
        if isinstance(post, dict) and post.get("language"):
            return str(post["language"]).strip()

    return ""


def _merge_usage_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for result in results:
        usage = result.get(INTERNAL_USAGE_KEY)
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            try:
                numeric_value = int(value)
            except (TypeError, ValueError):
                continue
            merged[key] = int(merged.get(key, 0)) + numeric_value
    return merged


def _annotate_localized_output(result: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(result)
    expected_languages = {
        str(language).strip()
        for language in inputs.get("target_languages", [])
    }
    expected_platforms = {
        str(platform).strip()
        for platform in inputs.get("platforms", [])
    }
    article_languages = {
        str(article.get("language", "")).strip()
        for article in result.get("localized_articles", [])
    }
    post_languages = {
        str(post.get("language", "")).strip()
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


def _run_research_strategy(
    inputs: dict[str, Any],
    agents_config: dict[str, Any],
    tasks_config: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    research_strategy_lead = Agent(
        config=agents_config["research_strategy_lead"],
        tools=_build_search_tools(config_context),
    )
    research_strategy_task = Task(
        config=tasks_config["research_and_strategy"],
        agent=research_strategy_lead,
    )
    content_crew = Crew(
        agents=[research_strategy_lead],
        tasks=[research_strategy_task],
        verbose=False,
        memory=_memory_enabled(config_context),
    )
    return _serialize_crew_result(content_crew.kickoff(inputs=inputs))


def _run_language_generation(
    language: str,
    language_index: int,
    inputs: dict[str, Any],
    strategy_context: dict[str, Any],
    agents_config: dict[str, Any],
    tasks_config: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    multilingual_editor = Agent(
        config=agents_config["multilingual_editor"],
        tools=_build_creation_tools(config_context),
    )
    creation_qa_task = Task(
        config=tasks_config["content_creation_and_qa"],
        agent=multilingual_editor,
        output_pydantic=PerLanguageContentOutput,
    )
    content_crew = Crew(
        agents=[multilingual_editor],
        tasks=[creation_qa_task],
        verbose=False,
        memory=_memory_enabled(config_context),
    )
    language_inputs = {
        **inputs,
        "target_language": language,
        "target_languages": [language],
        "target_market": _target_market_for_language(inputs, language_index),
        "target_markets": _target_market_for_language(inputs, language_index),
        "strategy_context": _serialize_strategy_context(strategy_context),
    }
    return _serialize_crew_result(content_crew.kickoff(inputs=language_inputs))


def _serialize_strategy_context(strategy_context: dict[str, Any]) -> str:
    clean_context = {
        key: value
        for key, value in strategy_context.items()
        if key != INTERNAL_USAGE_KEY
    }
    return yaml.safe_dump(clean_context, allow_unicode=True, sort_keys=False)


def run_content_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}
    normalized_inputs = _normalize_inputs(inputs)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")
    languages = [str(language).strip() for language in normalized_inputs.get("target_languages", []) if str(language).strip()]
    if not languages:
        raise ValueError("Content workflow requires at least one target language.")

    recorder = _progress_recorder(config_context)
    task_names = ["research_and_strategy", *[f"content_creation_and_qa:{language}" for language in languages]]
    if recorder:
        recorder.emit_plan(task_names)
        recorder.task_started(0, len(task_names), "research_and_strategy", agents_config["research_strategy_lead"]["role"])

    strategy_context = _run_research_strategy(
        normalized_inputs,
        agents_config,
        tasks_config,
        config_context,
    )

    if recorder:
        recorder.task_completed(0, len(task_names), "research_and_strategy", agents_config["research_strategy_lead"]["role"])

    outputs_by_language: dict[str, dict[str, Any]] = {}
    max_workers = min(_content_language_concurrency(config_context), len(languages))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_language = {}
        completed_language_count = 0
        language_agent_role = agents_config["multilingual_editor"]["role"]
        current_progress = PROGRESS_START + (PROGRESS_SPAN / len(task_names))
        for index, language in enumerate(languages, start=1):
            if recorder:
                recorder.emit_progress(
                    "task_started",
                    f"Task {index + 1}/{len(task_names)} started: content_creation_and_qa:{language}",
                    current_progress,
                    {
                        "task_index": index + 1,
                        "total_tasks": len(task_names),
                        "task_name": f"content_creation_and_qa:{language}",
                        "agent_role": language_agent_role,
                    },
                )
            future = executor.submit(
                _run_language_generation,
                language,
                index - 1,
                normalized_inputs,
                strategy_context,
                agents_config,
                tasks_config,
                dict(config_context),
            )
            future_to_language[future] = (index, language)

        for future in as_completed(future_to_language):
            index, language = future_to_language[future]
            outputs_by_language[language] = future.result()
            completed_language_count += 1
            if recorder:
                progress = PROGRESS_START + (
                    PROGRESS_SPAN * (1 + completed_language_count) / len(task_names)
                )
                recorder.emit_progress(
                    "task_completed",
                    f"Task {index + 1}/{len(task_names)} completed: content_creation_and_qa:{language}",
                    progress,
                    {
                        "task_index": index + 1,
                        "total_tasks": len(task_names),
                        "task_name": f"content_creation_and_qa:{language}",
                        "agent_role": language_agent_role,
                    },
                )

    language_outputs = [outputs_by_language[language] for language in languages]
    result = _merge_language_outputs(language_outputs)
    usage_metrics = _merge_usage_metrics([strategy_context, *language_outputs])
    if usage_metrics:
        result[INTERNAL_USAGE_KEY] = usage_metrics
    return _annotate_localized_output(result, normalized_inputs)
