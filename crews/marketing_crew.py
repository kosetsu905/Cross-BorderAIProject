from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.marketing_tools import ComplianceCheckerTool, KeywordResearchTool, PlatformAdSpecsTool
from tools.integrations.cross_platform_ads_tools import (
    GoogleAdsKeywordTool,
    MetaAdsTool,
    TikTokAdsTool,
)
from utils.crew_result import serialize_crew_result
from utils.usage_tracking import INTERNAL_USAGE_KEY
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, PROGRESS_SPAN, PROGRESS_START
from utils.project_intelligence import augment_agents_config

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "marketing"
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

    data_source: str = Field(
        ...,
        description=(
            "Provider status for ad platform integrations, such as "
            "development_fallback, mixed, or live_provider."
        ),
    )
    confidence_level: str = Field(
        ...,
        description="Confidence level based on whether live ad platform data was available",
    )
    assumptions: list[str] = Field(
        ..., description="Important caveats about fallback ad platform data or inferred benchmarks"
    )
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


class PerMarketCampaignOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_summary: str = Field(..., description="Market-specific campaign summary")
    ad_variants: list[CampaignAdVariant] = Field(
        ..., description="Platform-specific ad copy variants for this market"
    )
    compliance_status: list[ComplianceStatus] = Field(
        ..., description="Approval status and notes for this market"
    )
    launch_checklist: list[str] = Field(
        ..., description="Market-specific launch checklist"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _google_ads_tool(config_context: dict[str, Any]) -> GoogleAdsKeywordTool:
    return GoogleAdsKeywordTool(
        google_ads_access_token=config_context.get("google_ads_access_token"),
        google_ads_customer_id=config_context.get("google_ads_customer_id"),
        google_ads_developer_token=config_context.get("google_ads_developer_token"),
    )


def _meta_ads_tool(config_context: dict[str, Any]) -> MetaAdsTool:
    return MetaAdsTool(
        meta_access_token=config_context.get("meta_access_token"),
        meta_ad_account_id=config_context.get("meta_ad_account_id"),
        meta_page_id=config_context.get("meta_page_id"),
    )


def _tiktok_ads_tool(config_context: dict[str, Any]) -> TikTokAdsTool:
    return TikTokAdsTool(
        tiktok_access_token=config_context.get("tiktok_access_token"),
        tiktok_advertiser_id=config_context.get("tiktok_advertiser_id"),
    )


def _build_research_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool(), _google_ads_tool(config_context)]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


def _memory_enabled(config_context: dict[str, Any]) -> bool:
    return bool(config_context.get("crewai_memory_enabled"))


def _has_all_config(config_context: dict[str, Any], names: tuple[str, ...]) -> bool:
    return all(bool(config_context.get(name)) for name in names)


def _platform_provider_status(config_context: dict[str, Any]) -> dict[str, Any]:
    provider_groups = {
        "Google Ads": (
            "google_ads_access_token",
            "google_ads_customer_id",
            "google_ads_developer_token",
        ),
        "Meta Ads": (
            "meta_access_token",
            "meta_ad_account_id",
        ),
        "TikTok Ads": (
            "tiktok_access_token",
            "tiktok_advertiser_id",
        ),
    }
    live_platforms = [
        platform
        for platform, required_env in provider_groups.items()
        if _has_all_config(config_context, required_env)
    ]
    fallback_platforms = [
        platform
        for platform, required_env in provider_groups.items()
        if not _has_all_config(config_context, required_env)
    ]

    if fallback_platforms and live_platforms:
        data_source = "mixed"
        confidence_level = "Medium"
    elif fallback_platforms:
        data_source = "development_fallback"
        confidence_level = "Illustrative"
    else:
        data_source = "live_provider"
        confidence_level = "High"

    assumptions: list[str] = []
    if fallback_platforms:
        assumptions.append(
            "The following ad platform integrations used development fallback data "
            f"because credentials are missing or incomplete: {', '.join(fallback_platforms)}."
        )
        assumptions.append(
            "Keyword ideas, CPC estimates, audience estimates, trend benchmarks, and "
            "platform compliance checks should be validated with live ad platform APIs "
            "before production launch."
        )

    assumptions.append(
        "Amazon Ads is not connected to a live integration in this project yet; any "
        "Amazon-specific recommendations are generated from static specs or model inference."
    )

    return {
        "data_source": data_source,
        "confidence_level": confidence_level,
        "assumptions": assumptions,
    }


def _apply_provider_status(result: dict[str, Any], config_context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.update(_platform_provider_status(config_context))
    return _apply_marketing_quality_checks(normalized)


def _apply_marketing_quality_checks(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    variants = normalized.get("ad_variants") or []
    compliance_items = normalized.get("compliance_status") or []
    compliance_by_key = {
        (
            str(item.get("region", "")).strip().casefold(),
            str(item.get("platform", "")).strip().casefold(),
        ): item
        for item in compliance_items
        if isinstance(item, dict)
    }

    for variant in variants:
        if not isinstance(variant, dict):
            continue
        key = (
            str(variant.get("region", "")).strip().casefold(),
            str(variant.get("platform", "")).strip().casefold(),
        )
        compliance = compliance_by_key.get(key)
        if compliance is None:
            compliance = {
                "region": variant.get("region", ""),
                "platform": variant.get("platform", ""),
                "approved": True,
                "notes": "",
            }
            compliance_items.append(compliance)
            compliance_by_key[key] = compliance

        variant_text = " ".join(
            str(variant.get(field, ""))
            for field in ("headline", "body_text", "cta")
        )
        notes = str(compliance.get("notes") or "").strip()
        if not notes:
            notes = (
                "No obvious restricted claim detected in this draft; validate local law, platform policy, "
                "privacy wording, and substantiation before launch."
            )
        compliance["notes"] = notes

    normalized["compliance_status"] = compliance_items
    return normalized


def _append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing} {note}"


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def _split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _language_plan(inputs: dict[str, Any]) -> str:
    markets = _split_csv(inputs.get("target_markets"))
    languages = [
        str(language).strip()
        for language in inputs.get("target_languages", []) or []
        if str(language).strip()
    ]

    if languages and len(languages) == len(markets):
        pairs = [
            f"{market}: use {language}"
            for market, language in zip(markets, languages, strict=False)
        ]
        return "; ".join(pairs)

    if languages:
        return (
            f"Use these requested target languages where appropriate: {', '.join(languages)}. "
            "Map them to the target markets in order when possible; otherwise use the primary local language."
        )

    return (
        "No explicit target_languages were provided. For each target market, write ad copy in "
        "the primary local language normally used by consumers in that market. For example, Korea should "
        "use Korean, Russia should use Russian, China should use Simplified Chinese, Japan should use Japanese, "
        "Germany should use German, and English-speaking markets should use local English."
    )


def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    normalized["language_plan"] = _language_plan(normalized)
    return normalized


def _marketing_market_concurrency(config_context: dict[str, Any]) -> int:
    try:
        value = int(config_context.get("marketing_market_concurrency") or 4)
    except (TypeError, ValueError):
        value = 4
    return max(1, min(value, 16))


def _progress_recorder(config_context: dict[str, Any]) -> Any | None:
    recorder = config_context.get(PROGRESS_CONTEXT_KEY)
    if all(hasattr(recorder, name) for name in ("emit_plan", "task_started", "task_completed")):
        return recorder
    return None


def _strategy_context_text(strategy_context: dict[str, Any]) -> str:
    clean_context = {
        key: value
        for key, value in strategy_context.items()
        if key != INTERNAL_USAGE_KEY
    }
    return yaml.safe_dump(clean_context, allow_unicode=True, sort_keys=False)


def _language_for_market(inputs: dict[str, Any], market_index: int) -> str:
    languages = [
        str(language).strip()
        for language in inputs.get("target_languages", []) or []
        if str(language).strip()
    ]
    markets = _split_csv(inputs.get("target_markets"))
    if len(languages) == len(markets):
        return languages[market_index]
    return ""


def _language_plan_for_market(inputs: dict[str, Any], market: str, market_index: int) -> str:
    language = _language_for_market(inputs, market_index)
    if language:
        return f"{market}: use {language}"
    return (
        f"{market}: use the primary local language normally used by consumers in this market. "
        "If the market name is ambiguous, note the ambiguity in assumptions."
    )


def _product_category_for_market(inputs: dict[str, Any], market: str) -> str:
    product_category = str(inputs.get("product_category", "")).strip()
    return (
        f"{product_category}. Treat this as a descriptive source-language product category; "
        f"translate or localize it for {market} instead of copying the source phrase verbatim."
    )


def _run_strategy_channel_plan(
    inputs: dict[str, Any],
    agents_config: dict[str, Any],
    tasks_config: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    strategy_channel_planner = Agent(
        config=agents_config["strategy_channel_planner"],
        tools=[
            *_build_research_tools(config_context),
            KeywordResearchTool(),
            PlatformAdSpecsTool(),
            _meta_ads_tool(config_context),
            _tiktok_ads_tool(config_context),
        ],
    )
    strategy_task = Task(
        config=tasks_config["strategy_channel_planning"],
        agent=strategy_channel_planner,
    )
    marketing_crew = Crew(
        agents=[strategy_channel_planner],
        tasks=[strategy_task],
        verbose=False,
        memory=_memory_enabled(config_context),
    )
    return _serialize_crew_result(marketing_crew.kickoff(inputs=inputs))


def _run_market_creative_package(
    market: str,
    market_index: int,
    inputs: dict[str, Any],
    strategy_context: dict[str, Any],
    agents_config: dict[str, Any],
    tasks_config: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    creative_compliance_specialist = Agent(
        config=agents_config["creative_compliance_specialist"],
        tools=[
            KeywordResearchTool(),
            _google_ads_tool(config_context),
            ComplianceCheckerTool(),
            _meta_ads_tool(config_context),
            _tiktok_ads_tool(config_context),
        ],
        allow_delegation=True,
    )
    creative_compliance_task = Task(
        config=tasks_config["creative_compliance_package"],
        agent=creative_compliance_specialist,
        output_pydantic=PerMarketCampaignOutput,
    )
    marketing_crew = Crew(
        agents=[creative_compliance_specialist],
        tasks=[creative_compliance_task],
        verbose=False,
        memory=_memory_enabled(config_context),
    )
    market_inputs = {
        **inputs,
        "source_product_category": inputs.get("product_category", ""),
        "product_category": _product_category_for_market(inputs, market),
        "target_market": market,
        "target_markets": market,
        "target_languages": [_language_for_market(inputs, market_index)] if _language_for_market(inputs, market_index) else [],
        "language_plan": _language_plan_for_market(inputs, market, market_index),
        "strategy_context": _strategy_context_text(strategy_context),
    }
    return _serialize_crew_result(marketing_crew.kickoff(inputs=market_inputs))


def _merge_market_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_summaries: list[str] = []
    ad_variants: list[dict[str, Any]] = []
    compliance_status: list[dict[str, Any]] = []
    launch_checklist: list[str] = []

    for output in outputs:
        summary = str(output.get("strategy_summary", "")).strip()
        if summary:
            strategy_summaries.append(summary)
        ad_variants.extend(output.get("ad_variants") or [])
        compliance_status.extend(output.get("compliance_status") or [])
        for item in output.get("launch_checklist") or []:
            if item not in launch_checklist:
                launch_checklist.append(item)

    return {
        "strategy_summary": " ".join(strategy_summaries),
        "ad_variants": ad_variants,
        "compliance_status": compliance_status,
        "launch_checklist": launch_checklist,
    }


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


def run_marketing_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}
    normalized_inputs = _normalize_inputs(inputs)

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='marketing')
    tasks_config = _load_yaml_config("tasks.yaml")
    markets = _split_csv(normalized_inputs.get("target_markets"))
    if not markets:
        raise ValueError("Marketing workflow requires at least one target market.")

    recorder = _progress_recorder(config_context)
    task_names = ["strategy_channel_planning", *[f"creative_compliance_package:{market}" for market in markets]]
    if recorder:
        recorder.emit_plan(task_names)
        recorder.task_started(
            0,
            len(task_names),
            "strategy_channel_planning",
            agents_config["strategy_channel_planner"]["role"],
        )

    strategy_context = _run_strategy_channel_plan(
        normalized_inputs,
        agents_config,
        tasks_config,
        config_context,
    )
    if recorder:
        recorder.task_completed(
            0,
            len(task_names),
            "strategy_channel_planning",
            agents_config["strategy_channel_planner"]["role"],
        )

    outputs_by_market: dict[str, dict[str, Any]] = {}
    max_workers = min(_marketing_market_concurrency(config_context), len(markets))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_market = {}
        completed_market_count = 0
        creative_role = agents_config["creative_compliance_specialist"]["role"]
        current_progress = PROGRESS_START + (PROGRESS_SPAN / len(task_names))
        for index, market in enumerate(markets, start=1):
            if recorder:
                recorder.emit_progress(
                    "task_started",
                    f"Task {index + 1}/{len(task_names)} started: creative_compliance_package:{market}",
                    current_progress,
                    {
                        "task_index": index + 1,
                        "total_tasks": len(task_names),
                        "task_name": f"creative_compliance_package:{market}",
                        "agent_role": creative_role,
                    },
                )
            future = executor.submit(
                _run_market_creative_package,
                market,
                index - 1,
                normalized_inputs,
                strategy_context,
                agents_config,
                tasks_config,
                dict(config_context),
            )
            future_to_market[future] = (index, market)

        for future in as_completed(future_to_market):
            index, market = future_to_market[future]
            outputs_by_market[market] = future.result()
            completed_market_count += 1
            if recorder:
                progress = PROGRESS_START + (
                    PROGRESS_SPAN * (1 + completed_market_count) / len(task_names)
                )
                recorder.emit_progress(
                    "task_completed",
                    f"Task {index + 1}/{len(task_names)} completed: creative_compliance_package:{market}",
                    progress,
                    {
                        "task_index": index + 1,
                        "total_tasks": len(task_names),
                        "task_name": f"creative_compliance_package:{market}",
                        "agent_role": creative_role,
                    },
                )

    market_outputs = [outputs_by_market[market] for market in markets]
    result = _merge_market_outputs(market_outputs)
    usage_metrics = _merge_usage_metrics([strategy_context, *market_outputs])
    if usage_metrics:
        result[INTERNAL_USAGE_KEY] = usage_metrics
    return _apply_provider_status(result, config_context)
