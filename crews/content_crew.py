import contextvars
import re
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from pydantic import BaseModel, ConfigDict, Field
from tools.custom.content_tools import (
    CulturalComplianceCheckerTool,
    MultiEngineSEOOptimizerTool,
    MultimodalLocalizationTool,
    OpenAIImageGenerationTool,
    RedditGeoSearchTool,
    VisualAssetScoringTool,
)
from utils.crew_memory import build_crew_memory
from utils.crew_result import serialize_crew_result
from utils.model_tiering import ModelTierRouter
from utils.observability import agent_span, set_span_attributes, stage_span
from utils.project_intelligence import augment_agents_config
from utils.tool_cache import build_cached_scrape_tool, build_cached_serper_tool
from utils.usage_tracking import INTERNAL_USAGE_KEY
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, PROGRESS_SPAN, PROGRESS_START

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "content"
CONTENT_PROGRESS_TASK_INDEX_KEY = "_content_progress_task_index"
CONTENT_PROGRESS_TOTAL_TASKS_KEY = "_content_progress_total_tasks"
CONTENT_STAGE_PROGRESS_OFFSETS = {
    "content_generation": 0.12,
    "visual_localization": 0.35,
    "seo_metadata": 0.48,
    "cultural_compliance": 0.62,
    "image_generation": 0.76,
    "visual_scoring": 0.9,
    "content_assembly": 0.98,
}
CONTENT_TRACE_TASK_NAME_KEY = "_content_trace_task_name"
CONTENT_TRACE_AGENT_ROLE_KEY = "_content_trace_agent_role"
CONTENT_PARTIAL_SECRET_KEYS = {
    "api_key",
    "authorization",
    "b64_json",
    "client_secret",
    "credential",
    "openai_api_key",
    "refresh_token",
    "secret",
    "token",
}
CONTENT_PARTIAL_MAX_TEXT_LENGTH = 20000
CONTENT_ARTICLE_MAX_CHARS = 3500
CONTENT_TITLE_MAX_CHARS = 160
CONTENT_SOCIAL_POST_MAX_CHARS = 600
CONTENT_COMPLIANCE_NOTES_MAX_CHARS = 1000
CONTENT_SEO_KEYWORDS_MAX_COUNT = 12
CONTENT_GENERATION_MAX_TOKENS = 4096
CONTENT_GENERATION_AGENT_MAX_RETRY_LIMIT = 1
CONTENT_GENERATION_AGENT_MAX_ITER = 3
CONTENT_GENERATION_AGENT_MAX_EXECUTION_SECONDS = 240


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


class VisualAdaptationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_guide: str = Field(..., description="Market-specific visual style guidance")
    color_palette: str = Field(..., description="Recommended color palette")
    model_demographics: str = Field(..., description="Respectful casting and styling guidance")
    background_scene: str = Field(..., description="Culturally appropriate scene guidance")
    cultural_notes: list[str] = Field(..., description="Visual cultural do and do-not notes")
    ai_image_prompt: str = Field(..., description="Ready-to-use AI image prompt")


class VideoScriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_number: int = Field(..., ge=1)
    duration_sec: int = Field(..., ge=1)
    visual_description: str = Field(..., description="Shot or motion description")
    voiceover_script: str = Field(..., description="Localized or translatable voiceover guidance")
    on_screen_text: str = Field(..., description="Suggested on-screen text")
    background_music_mood: str = Field(..., description="Music and sound mood")
    cultural_adaptation_note: str = Field(..., description="Market adaptation note")


class LocalizedAltText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="Language code or name")
    alt_text: str = Field(..., description="Localized image alt text")


class SEOEngineStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str = Field(..., description="Search engine or regional search surface")
    title_template: str = Field(..., description="Localized SEO title template")
    meta_description_template: str = Field(..., description="Localized meta description template")
    keyword_focus: list[str] = Field(..., description="Primary and secondary keyword targets")
    structural_requirements: list[str] = Field(..., description="Technical SEO requirements")
    content_tone_guidance: str = Field(..., description="Search-engine-specific content tone")
    regional_boost_factors: list[str] = Field(..., description="Regional ranking signal reminders")


class MultiEngineSEOMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_url_slug: str = Field(..., description="Canonical URL slug for this localized content")
    engine_specific_metadata: list[SEOEngineStrategy] = Field(
        ..., description="Search engine specific metadata strategies"
    )
    schema_markup_jsonld: str = Field(..., description="Product JSON-LD structured data")
    alt_text_variants: list[LocalizedAltText] = Field(..., description="Localized alt text variants")
    hreflang_tags: list[str] = Field(..., description="hreflang tags for localized pages")


class CulturalRiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(..., description="Target market")
    language: str = Field(..., description="Target language")
    risk_flags: list[str] = Field(..., description="Detected risk flags")
    compliance_checklist: list[str] = Field(..., description="Compliance review checklist")
    recommended_actions: list[str] = Field(..., description="Recommended mitigation actions")


class VisualAssetGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Generation status")
    asset_path: str | None = Field(None, description="Local generated asset path")
    asset_url: str | None = Field(None, description="Remote asset URL if local bytes were not returned")
    model: str = Field(..., description="Image generation model")
    prompt: str = Field(..., description="Image prompt used")
    revised_prompt: str | None = Field(None, description="Provider-revised prompt if available")
    content_type: str | None = Field(None, description="Generated asset content type")
    duration_seconds: float | None = Field(None, ge=0, description="Image generation duration")
    attempts: int | None = Field(None, ge=0, description="Image API attempts made")
    last_status_code: int | None = Field(None, ge=100, le=599, description="Last image API HTTP status")
    retryable_status: bool | None = Field(None, description="Whether the last image API status is retryable")
    error: str | None = Field(None, description="Generation error or skip reason")


class VisualAssetScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Scoring status")
    asset_path: str | None = Field(None, description="Local asset path that was scored")
    prompt_alignment_score: float = Field(..., ge=0, le=100)
    cultural_fit_score: float = Field(..., ge=0, le=100)
    brand_voice_score: float = Field(..., ge=0, le=100)
    publish_readiness_score: float = Field(..., ge=0, le=100)
    duration_seconds: float | None = Field(None, ge=0, description="Vision scoring duration")
    notes: str = Field(..., description="Vision scoring notes")
    error: str | None = Field(None, description="Scoring error or skip reason")


class RedditGeoSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., description="Stable source identifier used by Reddit GEO posts")
    target_market: str = Field(..., description="Market associated with the Reddit search query")
    target_language: str = Field(default="", description="Target language associated with the source")
    subreddit: str = Field(..., description="Subreddit extracted from the source URL")
    title: str = Field(..., description="Reddit thread or result title")
    snippet: str = Field(..., description="Search snippet used for context")
    url: str = Field(..., description="Reddit source URL")
    query_type: str = Field(..., description="Search query category that found this source")
    query: str = Field(..., description="Serper search query that found this source")
    data_source: str = Field(..., description="Provider status such as serper_search or serper_error")
    confidence_level: str = Field(..., description="Confidence level for this Reddit context source")


class RedditGeoPost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_market: str = Field(..., description="Target market for the Reddit post")
    language: str = Field(..., description="Language code or name for the Reddit post")
    recommended_subreddit: str = Field(..., description="Recommended subreddit or manual review placeholder")
    title_options: list[str] = Field(..., description="Reddit-ready title options")
    body: str = Field(..., description="Transparent weak-brand Reddit post body with at most one contextual link")
    body_without_link: str = Field(..., description="Alternative Reddit post body without product links")
    disclosure_note: str = Field(..., description="Transparent affiliation or relationship disclosure")
    ai_search_entity_signals: list[str] = Field(
        ..., description="Product, brand, category, market, and use-case entity signals"
    )
    source_ids: list[str] = Field(..., description="Reddit GEO source IDs used to ground this post")
    moderation_notes: list[str] = Field(..., description="Manual subreddit rule and tone checks before posting")
    data_source: str = Field(..., description="Provider status used to build the Reddit post")
    confidence_level: str = Field(..., description="Confidence level based on source availability")


class LocalizedContentEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="Requested target language code or name")
    target_market: str = Field(..., description="Target market for the localized entities")
    subject: str = Field(..., min_length=1, description="Localized content subject")
    product_category: str = Field(..., min_length=1, description="Localized product category")
    brand_name: str = Field(
        "",
        description="Localized brand or entity name; empty when no source brand name was provided",
    )
    brand_voice: str = Field(..., min_length=1, description="Localized brand voice guidance")
    primary_keywords: list[str] = Field(
        ..., min_length=1, description="Localized SEO keyword seeds"
    )


class MarketMultimodalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_market: str = Field(..., description="Target market")
    language_code: str = Field(..., description="Target language")
    visual_spec: VisualAdaptationSpec = Field(..., description="Visual adaptation spec")
    video_script: list[VideoScriptSegment] = Field(..., description="Localized video storyboard")
    image_text_consistency_check: str = Field(..., description="Image/copy consistency guidance")
    recommended_platforms: list[str] = Field(..., description="Recommended distribution platforms")


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
    localized_entities: list[LocalizedContentEntities] = Field(
        default_factory=list,
        description="Localized product, brand, and SEO entities by language and market",
    )
    multimodal_outputs: list[MarketMultimodalOutput] = Field(
        default_factory=list,
        description="Market and language-specific visual localization outputs",
    )
    seo_outputs: list[MultiEngineSEOMetadata] = Field(
        default_factory=list,
        description="Market and language-specific multi-engine SEO metadata",
    )
    cultural_risk_assessments: list[CulturalRiskAssessment] = Field(
        default_factory=list,
        description="Cultural and compliance assessments by language/market",
    )
    visual_assets: list[VisualAssetGenerationResult] = Field(
        default_factory=list,
        description="Optional generated visual assets",
    )
    visual_asset_scores: list[VisualAssetScore] = Field(
        default_factory=list,
        description="Optional generated visual asset scores",
    )
    reddit_geo_posts: list[RedditGeoPost] = Field(
        default_factory=list,
        description="Optional Reddit-ready GEO posts for manual review and publication",
    )
    reddit_geo_sources: list[RedditGeoSource] = Field(
        default_factory=list,
        description="Reddit search sources used to ground the Reddit GEO posts",
    )


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
    localized_entities: LocalizedContentEntities = Field(
        ..., description="Localized product, brand, and SEO entities for downstream tools"
    )
    multimodal_output: MarketMultimodalOutput | None = Field(
        None,
        description="Market-specific multimodal localization output",
    )
    seo_metadata: MultiEngineSEOMetadata | None = Field(
        None,
        description="Market-specific multi-engine SEO metadata",
    )
    cultural_risk_assessment: CulturalRiskAssessment | None = Field(
        None,
        description="Market-specific cultural risk assessment",
    )
    visual_assets: list[VisualAssetGenerationResult] = Field(
        default_factory=list,
        description="Optional generated visual assets for this language",
    )
    visual_asset_scores: list[VisualAssetScore] = Field(
        default_factory=list,
        description="Optional vision scores for generated visual assets",
    )
    reddit_geo_posts: list[RedditGeoPost] = Field(
        default_factory=list,
        description="Optional Reddit-ready GEO posts for this language and market",
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_search_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = []
    if config_context.get("serper_api_key"):
        tools.append(build_cached_serper_tool(config_context, purpose="content_strategy_search"))
    return tools


def _build_content_intelligence_tools() -> list[Any]:
    return [
        MultimodalLocalizationTool(),
        MultiEngineSEOOptimizerTool(),
        CulturalComplianceCheckerTool(),
    ]


def _build_creation_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [
        build_cached_scrape_tool(config_context, purpose="content_creation_scrape"),
        *_build_content_intelligence_tools(),
    ]
    if config_context.get("serper_api_key"):
        tools.append(build_cached_serper_tool(config_context, purpose="content_creation_search"))
    return tools


def _crew_memory(config_context: dict[str, Any]) -> Any:
    return build_crew_memory(config_context, workflow="content")


def _positive_int_config(
    config_context: dict[str, Any],
    key: str,
    default: int,
) -> int:
    value = config_context.get(key)
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _content_generation_llm_context(config_context: dict[str, Any]) -> dict[str, Any]:
    context = dict(config_context)
    has_explicit_limit = any(
        context.get(key) not in (None, "")
        for key in (
            "llm_max_tokens",
            "max_tokens",
            "llm_max_completion_tokens",
            "max_completion_tokens",
        )
    )
    if not has_explicit_limit:
        context["llm_max_tokens"] = _positive_int_config(
            config_context,
            "content_generation_max_tokens",
            CONTENT_GENERATION_MAX_TOKENS,
        )
    return context


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def _normalize_per_language_content_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for list_field in ("visual_assets", "visual_asset_scores", "reddit_geo_posts"):
        if normalized.get(list_field) is None:
            normalized[list_field] = []

    article = normalized.get("localized_article")
    if not isinstance(article, dict):
        return normalized

    article_payload = dict(article)
    nested_compliance_notes = article_payload.pop("compliance_notes", None)
    normalized["localized_article"] = article_payload
    if "compliance_notes" not in normalized and nested_compliance_notes is not None:
        normalized["compliance_notes"] = nested_compliance_notes
    return normalized


def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    normalized["product_features_summary"] = _summarize_product_features(
        normalized.get("product_features")
    )
    normalized["brand_voice_summary"] = _brand_voice(normalized)
    normalized["brand_name_summary"] = _brand_name_summary(normalized)
    normalized["product_url_summary"] = _product_url_summary(normalized)
    normalized["primary_keywords_summary"] = ", ".join(_primary_keywords(normalized))
    normalized["reddit_geo_enabled"] = _reddit_geo_enabled(normalized)
    normalized["reddit_geo_context_summary"] = "Reddit GEO generation is disabled for this request."
    normalized["content_article_max_chars"] = CONTENT_ARTICLE_MAX_CHARS
    normalized["content_title_max_chars"] = CONTENT_TITLE_MAX_CHARS
    normalized["content_social_post_max_chars"] = CONTENT_SOCIAL_POST_MAX_CHARS
    normalized["content_compliance_notes_max_chars"] = CONTENT_COMPLIANCE_NOTES_MAX_CHARS
    normalized["content_seo_keywords_max_count"] = CONTENT_SEO_KEYWORDS_MAX_COUNT
    return normalized


def _summarize_product_features(features: Any) -> str:
    if not features:
        return "No product-specific features were provided. Keep claims broad and avoid inventing product details."

    text = str(features).strip()
    if not text:
        return "No usable product-specific features were provided. Keep claims broad and avoid inventing product details."

    return text


def _brand_voice(inputs: dict[str, Any]) -> str:
    value = str(inputs.get("brand_voice") or "").strip()
    return value or "Premium, trustworthy, practical, and culturally respectful"


def _primary_keywords(inputs: dict[str, Any]) -> list[str]:
    raw_keywords = inputs.get("primary_keywords")
    if isinstance(raw_keywords, list):
        keywords = [
            str(keyword).strip()
            for keyword in raw_keywords
            if str(keyword).strip()
        ]
        if keywords:
            return keywords
    subject = str(inputs.get("subject") or "").strip()
    product_category = str(inputs.get("product_category") or "").strip()
    seed = subject or product_category or "cross-border product"
    return [f"{seed} buying guide", f"{seed} benefits", f"{seed} review"]


def _brand_name_summary(inputs: dict[str, Any]) -> str:
    value = str(inputs.get("brand_name") or "").strip()
    return value or "No brand name was provided; avoid inventing a named brand."


def _product_url_summary(inputs: dict[str, Any]) -> str:
    value = str(inputs.get("product_url") or "").strip()
    return value or "No product URL was provided; produce a no-link Reddit variant."


def _reddit_geo_enabled(inputs: dict[str, Any]) -> bool:
    if bool(inputs.get("generate_reddit_geo")):
        return True
    return any(
        str(platform).strip().casefold() == "reddit"
        for platform in inputs.get("platforms", [])
    )


def _product_name(inputs: dict[str, Any]) -> str:
    return str(inputs.get("subject") or inputs.get("product_category") or "Product").strip()


def _asset_output_slug(inputs: dict[str, Any], language: str, target_market: str) -> str:
    value = f"{_product_name(inputs)}-{target_market}-{language}"
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part) or "content-visual"


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


def _localized_inputs_for_language(
    output: dict[str, Any],
    inputs: dict[str, Any],
    language: str,
    target_market: str,
) -> dict[str, Any]:
    return _localized_inputs_from_entities(
        inputs,
        _localized_entities_payload(output),
        language,
        target_market,
    )


def _localized_inputs_from_entities(
    inputs: dict[str, Any],
    entities: dict[str, Any] | None,
    language: str,
    target_market: str,
) -> dict[str, Any]:
    localized = {
        **inputs,
        "target_language": language,
        "target_languages": [language],
        "target_market": target_market,
        "target_markets": target_market,
    }
    if not entities:
        return localized

    field_map = {
        "subject": "subject",
        "product_category": "product_category",
        "brand_voice": "brand_voice",
    }
    for entity_key, input_key in field_map.items():
        value = str(entities.get(entity_key) or "").strip()
        if value:
            localized[input_key] = value

    brand_name = str(entities.get("brand_name") or "").strip()
    if brand_name and not _contains_forbidden_source_term(brand_name, inputs, language):
        localized["brand_name"] = brand_name
    elif inputs.get("brand_name"):
        localized.pop("brand_name", None)

    forbidden_source_terms = _forbidden_source_terms_for_language(inputs, language)
    primary_keywords = [
        str(keyword).strip()
        for keyword in entities.get("primary_keywords") or []
        if str(keyword).strip()
        and not any(term in str(keyword).strip() for term in forbidden_source_terms)
    ]
    if primary_keywords:
        localized["primary_keywords"] = primary_keywords

    return localized


def _localized_entities_payload(output: dict[str, Any]) -> dict[str, Any] | None:
    entities = output.get("localized_entities")
    return entities if isinstance(entities, dict) else None


def _source_language_terms(inputs: dict[str, Any]) -> list[str]:
    raw_terms: list[str] = []
    for key in ("subject", "product_category", "brand_name", "brand_voice"):
        value = str(inputs.get(key) or "").strip()
        if value:
            raw_terms.append(value)
    raw_terms.extend(
        str(keyword).strip()
        for keyword in inputs.get("primary_keywords") or []
        if str(keyword).strip()
    )

    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return sorted(terms, key=len, reverse=True)


def _has_cjk(value: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in value)


def _is_short_cjk_term(value: str) -> bool:
    return 0 < len(value) <= 2 and all(
        "\u4e00" <= character <= "\u9fff" for character in value
    )


def _source_term_allowed_in_language(term: str, language: str) -> bool:
    normalized_language = language.casefold()
    if normalized_language.startswith("zh") and _has_cjk(term):
        return True
    return normalized_language.startswith("ja") and _is_short_cjk_term(term)


def _forbidden_source_terms_for_language(
    inputs: dict[str, Any],
    language: str,
) -> list[str]:
    return [
        term
        for term in _source_language_terms(inputs)
        if not _source_term_allowed_in_language(term, language)
    ]


def _contains_forbidden_source_term(
    value: str,
    inputs: dict[str, Any],
    language: str,
) -> bool:
    return any(term in value for term in _forbidden_source_terms_for_language(inputs, language))


def _source_term_replacements(
    inputs: dict[str, Any],
    localized_inputs: dict[str, Any],
    language: str,
) -> dict[str, str]:
    fallback_keywords = [
        keyword
        for keyword in _primary_keywords(localized_inputs)
        if not _contains_forbidden_source_term(keyword, inputs, language)
    ]
    fallback = (
        fallback_keywords[0]
        if fallback_keywords
        else str(
            localized_inputs.get("subject")
            or localized_inputs.get("product_category")
            or ""
        ).strip()
    )
    candidates = {
        str(inputs.get("subject") or "").strip(): str(localized_inputs.get("subject") or "").strip(),
        str(inputs.get("product_category") or "").strip(): str(localized_inputs.get("product_category") or "").strip(),
        str(inputs.get("brand_name") or "").strip(): str(localized_inputs.get("brand_name") or fallback).strip(),
        str(inputs.get("brand_voice") or "").strip(): str(localized_inputs.get("brand_voice") or "").strip(),
    }
    for keyword in inputs.get("primary_keywords") or []:
        candidates[str(keyword).strip()] = fallback

    replacements: dict[str, str] = {}
    for source, replacement in candidates.items():
        if (
            source
            and replacement
            and source != replacement
            and not _source_term_allowed_in_language(source, language)
        ):
            replacements[source] = replacement
    return dict(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))


def _replace_source_terms(
    value: str,
    inputs: dict[str, Any],
    localized_inputs: dict[str, Any],
    language: str,
) -> str:
    replaced = value
    for source, replacement in _source_term_replacements(inputs, localized_inputs, language).items():
        replaced = replaced.replace(source, replacement)
    return replaced


def _replace_source_terms_in_value(
    value: Any,
    inputs: dict[str, Any],
    localized_inputs: dict[str, Any],
    language: str,
) -> Any:
    if isinstance(value, str):
        return _replace_source_terms(value, inputs, localized_inputs, language)
    if isinstance(value, list):
        return [
            _replace_source_terms_in_value(item, inputs, localized_inputs, language)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_source_terms_in_value(item, inputs, localized_inputs, language)
            for key, item in value.items()
        }
    return value


def _build_reddit_geo_context(
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    if not _reddit_geo_enabled(inputs):
        return {
            "enabled": False,
            "status": "disabled",
            "data_source": "not_requested",
            "confidence_level": "none",
            "assumption_notice": "Reddit GEO generation was not requested.",
            "sources": [],
            "query_pack": [],
            "provider_error": None,
        }

    tool = RedditGeoSearchTool(
        serper_api_key=config_context.get("serper_api_key"),
        tool_cache_context=config_context,
    )
    context = tool._run(
        subject=str(inputs.get("subject") or ""),
        product_category=str(inputs.get("product_category") or ""),
        target_markets=str(inputs.get("target_markets") or ""),
        primary_keywords=_primary_keywords(inputs),
        target_languages=[
            str(language).strip()
            for language in inputs.get("target_languages", [])
            if str(language).strip()
        ],
        brand_name=str(inputs.get("brand_name") or "").strip() or None,
    )
    context["enabled"] = True
    return context


def _reddit_geo_context_summary(context: dict[str, Any]) -> str:
    if not context.get("enabled"):
        return "Reddit GEO generation is disabled for this request."

    lines = [
        (
            "Reddit GEO generation is enabled. "
            f"Status: {context.get('status')}; "
            f"data_source: {context.get('data_source')}; "
            f"confidence_level: {context.get('confidence_level')}."
        ),
        str(context.get("assumption_notice") or "").strip(),
    ]
    provider_error = str(context.get("provider_error") or "").strip()
    if provider_error:
        lines.append(f"Provider error summary: {provider_error}")

    sources = [
        source
        for source in context.get("sources", [])
        if isinstance(source, dict)
    ]
    if not sources:
        lines.append(
            "No live Reddit source snippets are available; generate a cautious draft "
            "and mark subreddit fit for human validation."
        )
        return "\n".join(line for line in lines if line).strip()

    lines.append("Use these Reddit snippets as context only; do not claim Reddit consensus:")
    for source in sources[:10]:
        lines.append(
            "- "
            f"{source.get('source_id')}: r/{source.get('subreddit')} "
            f"({source.get('target_market')}) - {source.get('title')} "
            f"| snippet: {source.get('snippet')}"
        )
    return "\n".join(line for line in lines if line).strip()[:4000]


def _reddit_geo_sources_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    data_source = str(context.get("data_source") or "not_requested")
    confidence_level = str(context.get("confidence_level") or "none")
    return [
        {
            "source_id": str(source.get("source_id") or ""),
            "target_market": str(source.get("target_market") or ""),
            "target_language": str(source.get("target_language") or ""),
            "subreddit": str(source.get("subreddit") or ""),
            "title": str(source.get("title") or ""),
            "snippet": str(source.get("snippet") or ""),
            "url": str(source.get("url") or ""),
            "query_type": str(source.get("query_type") or ""),
            "query": str(source.get("query") or ""),
            "data_source": data_source,
            "confidence_level": confidence_level,
        }
        for source in context.get("sources", [])
        if isinstance(source, dict)
    ]


def _openai_api_key_for_images(config_context: dict[str, Any]) -> str | None:
    if config_context.get("openai_api_key"):
        return str(config_context["openai_api_key"])
    if str(config_context.get("llm_provider") or "").lower() == "openai":
        api_key = config_context.get("llm_api_key")
        return str(api_key) if api_key else None
    return None


def _enrich_language_output(
    output: dict[str, Any],
    inputs: dict[str, Any],
    language: str,
    target_market: str,
    config_context: dict[str, Any],
) -> dict[str, Any]:
    tool_inputs = _localized_inputs_for_language(output, inputs, language, target_market)
    product = _product_name(tool_inputs)
    brand_voice = _brand_voice(tool_inputs)
    primary_keywords = _primary_keywords(tool_inputs)
    brand_name = str(tool_inputs.get("brand_name") or "").strip() or "YourBrand"
    platforms = [
        str(platform).strip()
        for platform in tool_inputs.get("platforms", [])
        if str(platform).strip()
    ]

    stage_started_at = time.monotonic()
    _emit_content_stage(
        config_context,
        "visual_localization",
        "started",
        f"Building visual localization spec for {language}.",
        language=language,
        target_market=target_market,
    )
    try:
        with _content_stage_trace(
            config_context,
            "visual_localization",
            language=language,
            target_market=target_market,
        ):
            localization_output = MultimodalLocalizationTool()._run(
                product=product,
                target_market=target_market,
                language=language,
                brand_voice=brand_voice,
                platforms=platforms,
            )
    except Exception as exc:
        _emit_content_stage(
            config_context,
            "visual_localization",
            "failed",
            f"Visual localization failed for {language}.",
            language=language,
            target_market=target_market,
            duration_seconds=time.monotonic() - stage_started_at,
            error_summary=_error_summary(exc),
        )
        raise
    _emit_content_stage(
        config_context,
        "visual_localization",
        "completed",
        f"Visual localization spec ready for {language}.",
        language=language,
        target_market=target_market,
        duration_seconds=time.monotonic() - stage_started_at,
    )
    _emit_content_partial(
        config_context,
        "visual_localization",
        "visual_brief",
        _visual_localization_preview(localization_output),
        language=language,
        target_market=target_market,
    )

    stage_started_at = time.monotonic()
    _emit_content_stage(
        config_context,
        "seo_metadata",
        "started",
        f"Building SEO metadata for {language}.",
        language=language,
        target_market=target_market,
    )
    try:
        with _content_stage_trace(
            config_context,
            "seo_metadata",
            language=language,
            target_market=target_market,
        ):
            seo_output = MultiEngineSEOOptimizerTool()._run(
                product=product,
                target_market=target_market,
                language=language,
                primary_keywords=primary_keywords,
                brand_name=brand_name,
            )
    except Exception as exc:
        _emit_content_stage(
            config_context,
            "seo_metadata",
            "failed",
            f"SEO metadata failed for {language}.",
            language=language,
            target_market=target_market,
            duration_seconds=time.monotonic() - stage_started_at,
            error_summary=_error_summary(exc),
        )
        raise
    _emit_content_stage(
        config_context,
        "seo_metadata",
        "completed",
        f"SEO metadata ready for {language}.",
        language=language,
        target_market=target_market,
        duration_seconds=time.monotonic() - stage_started_at,
    )
    _emit_content_partial(
        config_context,
        "seo_metadata",
        "seo_metadata",
        seo_output,
        language=language,
        target_market=target_market,
    )

    stage_started_at = time.monotonic()
    _emit_content_stage(
        config_context,
        "cultural_compliance",
        "started",
        f"Checking cultural compliance for {language}.",
        language=language,
        target_market=target_market,
    )
    try:
        with _content_stage_trace(
            config_context,
            "cultural_compliance",
            language=language,
            target_market=target_market,
        ):
            compliance_output = CulturalComplianceCheckerTool()._run(
                content_summary=_content_summary(output, localization_output, seo_output),
                target_market=target_market,
                language=language,
            )
    except Exception as exc:
        _emit_content_stage(
            config_context,
            "cultural_compliance",
            "failed",
            f"Cultural compliance failed for {language}.",
            language=language,
            target_market=target_market,
            duration_seconds=time.monotonic() - stage_started_at,
            error_summary=_error_summary(exc),
        )
        raise
    _emit_content_stage(
        config_context,
        "cultural_compliance",
        "completed",
        f"Cultural compliance check ready for {language}.",
        language=language,
        target_market=target_market,
        duration_seconds=time.monotonic() - stage_started_at,
    )
    _emit_content_partial(
        config_context,
        "cultural_compliance",
        "compliance",
        compliance_output,
        language=language,
        target_market=target_market,
    )

    visual_assets: list[dict[str, Any]] = []
    visual_asset_scores: list[dict[str, Any]] = []
    if bool(tool_inputs.get("generate_visual_assets")):
        image_tool = OpenAIImageGenerationTool(
            openai_api_key=_openai_api_key_for_images(config_context),
            image_model=str(config_context.get("content_image_model") or "gpt-image-2"),
            artifact_dir=str(
                config_context.get("content_image_artifact_dir")
                or "artifacts/content_creation"
            ),
            tool_execution_context=config_context,
        )
        image_generation_lock = config_context.get("content_image_generation_lock")
        stage_started_at = time.monotonic()
        _emit_content_stage(
            config_context,
            "image_generation",
            "started",
            f"Generating visual assets for {language}.",
            language=language,
            target_market=target_market,
        )
        try:
            with _content_stage_trace(
                config_context,
                "image_generation",
                language=language,
                target_market=target_market,
            ) as mark_stage_status:
                if image_generation_lock is not None:
                    with image_generation_lock:
                        image_result = image_tool._run(
                            prompt=localization_output["visual_spec"]["ai_image_prompt"],
                            output_slug=_asset_output_slug(tool_inputs, language, target_market),
                            image_generation_count=int(tool_inputs.get("image_generation_count") or 1),
                            image_quality=str(tool_inputs.get("image_quality") or "auto"),
                            image_size=str(tool_inputs.get("image_size") or "1024x1024"),
                        )
                else:
                    image_result = image_tool._run(
                        prompt=localization_output["visual_spec"]["ai_image_prompt"],
                        output_slug=_asset_output_slug(tool_inputs, language, target_market),
                        image_generation_count=int(tool_inputs.get("image_generation_count") or 1),
                        image_quality=str(tool_inputs.get("image_quality") or "auto"),
                        image_size=str(tool_inputs.get("image_size") or "1024x1024"),
                    )
                visual_assets = _visual_assets_from_generation_result(image_result)
                image_status = str(image_result.get("status") or "completed")
                stage_status = "completed"
                if image_status.startswith("skipped"):
                    stage_status = "skipped"
                elif image_status == "failed":
                    stage_status = "failed"
                mark_stage_status(stage_status, asset_count=len(visual_assets))
        except Exception as exc:
            _emit_content_stage(
                config_context,
                "image_generation",
                "failed",
                f"Image generation failed for {language}.",
                language=language,
                target_market=target_market,
                duration_seconds=time.monotonic() - stage_started_at,
                asset_count=0,
                error_summary=_error_summary(exc),
            )
            raise
        _emit_content_stage(
            config_context,
            "image_generation",
            stage_status,
            f"Image generation {stage_status} for {language}.",
            language=language,
            target_market=target_market,
            duration_seconds=time.monotonic() - stage_started_at,
            asset_count=len(visual_assets),
            error_summary=_error_summary(image_result.get("error")),
        )
        _emit_content_partial(
            config_context,
            "image_generation",
            "images",
            _image_generation_preview(image_result, visual_assets),
            language=language,
            target_market=target_market,
        )
        scoring_tool = VisualAssetScoringTool(
            openai_api_key=_openai_api_key_for_images(config_context),
            scoring_model=str(
                config_context.get("content_image_scoring_model")
                or config_context.get("openai_model_name")
                or "gpt-4o-mini"
            ),
            tool_execution_context=config_context,
        )
        stage_started_at = time.monotonic()
        _emit_content_stage(
            config_context,
            "visual_scoring",
            "started",
            f"Scoring visual assets for {language}.",
            language=language,
            target_market=target_market,
            asset_count=len(visual_assets),
        )
        try:
            with _content_stage_trace(
                config_context,
                "visual_scoring",
                language=language,
                target_market=target_market,
            ) as mark_stage_status:
                visual_asset_scores = _score_visual_assets(
                    scoring_tool,
                    visual_assets,
                    localization_output["visual_spec"]["ai_image_prompt"],
                    target_market,
                    language,
                    brand_voice,
                )
                score_statuses = {
                    str(score.get("status") or "")
                    for score in visual_asset_scores
                    if isinstance(score, dict)
                }
                score_stage_status = "completed"
                if score_statuses and all(status.startswith("skipped") for status in score_statuses):
                    score_stage_status = "skipped"
                elif any(status == "failed" for status in score_statuses):
                    score_stage_status = "failed"
                mark_stage_status(
                    score_stage_status,
                    asset_count=len(visual_assets),
                    score_count=len(visual_asset_scores),
                )
        except Exception as exc:
            _emit_content_stage(
                config_context,
                "visual_scoring",
                "failed",
                f"Visual scoring failed for {language}.",
                language=language,
                target_market=target_market,
                duration_seconds=time.monotonic() - stage_started_at,
                asset_count=len(visual_assets),
                score_count=0,
                error_summary=_error_summary(exc),
            )
            raise
        score_error = next(
            (
                str(score.get("error"))
                for score in visual_asset_scores
                if isinstance(score, dict) and score.get("error")
            ),
            None,
        )
        _emit_content_stage(
            config_context,
            "visual_scoring",
            score_stage_status,
            f"Visual scoring {score_stage_status} for {language}.",
            language=language,
            target_market=target_market,
            duration_seconds=time.monotonic() - stage_started_at,
            asset_count=len(visual_assets),
            score_count=len(visual_asset_scores),
            error_summary=_error_summary(score_error),
        )
    else:
        with _content_stage_trace(
            config_context,
            "image_generation",
            language=language,
            target_market=target_market,
            success_status="skipped",
        ) as mark_stage_status:
            mark_stage_status("skipped", asset_count=0)
        _emit_content_stage(
            config_context,
            "image_generation",
            "skipped",
            f"Visual asset generation disabled for {language}.",
            language=language,
            target_market=target_market,
            asset_count=0,
        )
        _emit_content_partial(
            config_context,
            "image_generation",
            "images",
            {"status": "skipped", "assets": [], "error_summary": None},
            language=language,
            target_market=target_market,
        )
        with _content_stage_trace(
            config_context,
            "visual_scoring",
            language=language,
            target_market=target_market,
            success_status="skipped",
        ) as mark_stage_status:
            mark_stage_status("skipped", asset_count=0, score_count=0)
        _emit_content_stage(
            config_context,
            "visual_scoring",
            "skipped",
            f"Visual scoring skipped for {language}.",
            language=language,
            target_market=target_market,
            asset_count=0,
            score_count=0,
        )

    enriched = dict(output)
    enriched["multimodal_output"] = localization_output
    enriched["seo_metadata"] = seo_output
    enriched["cultural_risk_assessment"] = compliance_output
    enriched["visual_assets"] = visual_assets
    enriched["visual_asset_scores"] = visual_asset_scores
    return enriched


def _content_summary(
    output: dict[str, Any],
    localization_output: dict[str, Any],
    seo_output: dict[str, Any],
) -> str:
    article = output.get("localized_article")
    article_text = ""
    if isinstance(article, dict):
        article_text = f"{article.get('title', '')}\n{article.get('article', '')}"
    social_text = " ".join(
        str(post.get("content", ""))
        for post in output.get("social_media_posts", [])
        if isinstance(post, dict)
    )
    return "\n".join(
        [
            article_text[:2000],
            social_text[:1000],
            str(localization_output)[:1000],
            str(seo_output)[:1000],
        ]
    )


def _content_generation_preview(output: dict[str, Any]) -> dict[str, Any]:
    article = output.get("localized_article")
    social_posts = output.get("social_media_posts")
    return {
        "status": str(output.get("status") or "completed"),
        "localized_article": article if isinstance(article, dict) else None,
        "social_media_posts": social_posts if isinstance(social_posts, list) else [],
        "seo_keywords": output.get("seo_keywords") if isinstance(output.get("seo_keywords"), list) else [],
        "localized_entities": (
            output.get("localized_entities")
            if isinstance(output.get("localized_entities"), dict)
            else None
        ),
        "compliance_notes": str(output.get("compliance_notes") or "").strip(),
    }


def _failed_content_generation_preview(
    language: str,
    target_market: str,
    error: Exception | str | None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "language": language,
        "target_market": target_market,
        "error_summary": _error_summary(error) or "Content generation failed.",
        "retry_available": True,
    }


def _failed_language_output(
    language: str,
    target_market: str,
    error: Exception | str | None,
) -> dict[str, Any]:
    error_summary = _error_summary(error) or "Content generation failed."
    return {
        "status": "failed",
        "language": language,
        "target_market": target_market,
        "localized_article": None,
        "social_media_posts": [],
        "seo_keywords": [],
        "compliance_notes": f"Generation failed for {language} ({target_market}): {error_summary}",
        "visual_assets": [],
        "visual_asset_scores": [],
        "reddit_geo_posts": [],
        "error_summary": error_summary,
    }


def _visual_localization_preview(localization_output: dict[str, Any]) -> dict[str, Any]:
    visual_spec = localization_output.get("visual_spec")
    return {
        "visual_spec": visual_spec if isinstance(visual_spec, dict) else {},
        "image_text_consistency_check": localization_output.get("image_text_consistency_check"),
        "recommended_platforms": localization_output.get("recommended_platforms") or [],
    }


def _image_generation_preview(
    image_result: dict[str, Any],
    visual_assets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": image_result.get("status"),
        "assets": [
            {
                "status": asset.get("status"),
                "asset_path": asset.get("asset_path"),
                "asset_url": asset.get("asset_url"),
                "model": asset.get("model"),
                "content_type": asset.get("content_type"),
                "duration_seconds": asset.get("duration_seconds"),
                "attempts": asset.get("attempts"),
                "last_status_code": asset.get("last_status_code"),
                "retryable_status": asset.get("retryable_status"),
                "error_summary": _error_summary(asset.get("error")),
            }
            for asset in visual_assets
            if isinstance(asset, dict)
        ],
        "duration_seconds": image_result.get("duration_seconds"),
        "attempts": image_result.get("attempts"),
        "last_status_code": image_result.get("last_status_code"),
        "retryable_status": image_result.get("retryable_status"),
        "error_summary": _error_summary(image_result.get("error")),
    }


def _visual_assets_from_generation_result(image_result: dict[str, Any]) -> list[dict[str, Any]]:
    assets = image_result.get("assets")
    if isinstance(assets, list) and assets:
        return [
            {
                "status": str(asset.get("status") or image_result.get("status") or "unknown"),
                "asset_path": asset.get("asset_path"),
                "asset_url": asset.get("asset_url"),
                "model": str(asset.get("model") or image_result.get("model") or ""),
                "prompt": str(asset.get("prompt") or image_result.get("prompt") or ""),
                "revised_prompt": asset.get("revised_prompt"),
                "content_type": asset.get("content_type"),
                "duration_seconds": asset.get("duration_seconds") or image_result.get("duration_seconds"),
                "attempts": asset.get("attempts") or image_result.get("attempts"),
                "last_status_code": asset.get("last_status_code") or image_result.get("last_status_code"),
                "retryable_status": (
                    asset.get("retryable_status")
                    if asset.get("retryable_status") is not None
                    else image_result.get("retryable_status")
                ),
                "error": asset.get("error"),
            }
            for asset in assets
            if isinstance(asset, dict)
        ]
    return [
        {
            "status": str(image_result.get("status") or "skipped"),
            "asset_path": None,
            "asset_url": None,
            "model": str(image_result.get("model") or ""),
            "prompt": str(image_result.get("prompt") or ""),
            "revised_prompt": None,
            "content_type": None,
            "duration_seconds": image_result.get("duration_seconds"),
            "attempts": image_result.get("attempts"),
            "last_status_code": image_result.get("last_status_code"),
            "retryable_status": image_result.get("retryable_status"),
            "error": image_result.get("error"),
        }
    ]


def _score_visual_assets(
    scoring_tool: VisualAssetScoringTool,
    visual_assets: list[dict[str, Any]],
    prompt: str,
    target_market: str,
    language: str,
    brand_voice: str,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for asset in visual_assets:
        asset_path = asset.get("asset_path")
        if not asset_path:
            scores.append(
                {
                    "status": "skipped_missing_local_asset",
                    "asset_path": None,
                    "prompt_alignment_score": 0.0,
                    "cultural_fit_score": 0.0,
                    "brand_voice_score": 0.0,
                    "publish_readiness_score": 0.0,
                    "duration_seconds": 0.0,
                    "notes": "No local asset path was available for vision scoring.",
                    "error": asset.get("error"),
                }
            )
            continue
        scores.append(
            scoring_tool._run(
                asset_path=str(asset_path),
                prompt=prompt,
                target_market=target_market,
                language=language,
                brand_voice=brand_voice,
            )
        )
    return scores


def _content_language_concurrency(config_context: dict[str, Any]) -> int:
    try:
        value = int(config_context.get("content_language_concurrency") or 4)
    except (TypeError, ValueError):
        value = 4
    return max(1, min(value, 16))


def _progress_recorder(config_context: dict[str, Any]) -> Any | None:
    recorder = config_context.get(PROGRESS_CONTEXT_KEY)
    if all(
        hasattr(recorder, name)
        for name in ("emit_plan", "task_started", "task_completed", "emit_progress")
    ):
        return recorder
    return None


def _content_trace_identity(config_context: dict[str, Any]) -> dict[str, str | None]:
    recorder = _progress_recorder(config_context)
    job_id = getattr(recorder, "job_id", None) or config_context.get("job_id") or "content-local"
    workflow_type = (
        getattr(recorder, "workflow_type", None)
        or config_context.get("workflow_type")
        or "content"
    )
    backend = getattr(recorder, "backend", None) or config_context.get("backend")
    return {
        "job_id": str(job_id),
        "workflow_type": str(workflow_type),
        "backend": str(backend) if backend not in (None, "") else None,
    }


def _safe_content_stage_trace_update(attributes: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"asset_count", "score_count", "status"}
    return {
        key: value
        for key, value in attributes.items()
        if key in allowed_keys and value not in (None, "", [])
    }


@contextmanager
def _content_stage_trace(
    config_context: dict[str, Any],
    stage: str,
    *,
    language: str | None = None,
    target_market: str | None = None,
    task_name: str | None = None,
    agent_role: str | None = None,
    success_status: str = "completed",
) -> Iterator[Callable[..., None]]:
    trace_identity = _content_trace_identity(config_context)
    stage_task_name = str(task_name or config_context.get(CONTENT_TRACE_TASK_NAME_KEY) or "") or None
    stage_agent_role = str(agent_role or config_context.get(CONTENT_TRACE_AGENT_ROLE_KEY) or "") or None
    final_attributes: dict[str, Any] = {"status": success_status}

    def mark_status(status: str, **attributes: Any) -> None:
        final_attributes.update(_safe_content_stage_trace_update({"status": status, **attributes}))
        set_span_attributes(final_attributes, config_context=config_context)

    with stage_span(
        stage,
        job_id=trace_identity["job_id"],
        workflow_type=trace_identity["workflow_type"],
        task_name=stage_task_name,
        agent_role=stage_agent_role,
        language=language,
        target_market=target_market,
        status="running",
        backend=trace_identity["backend"],
        config_context=config_context,
    ):
        try:
            yield mark_status
        except Exception:
            mark_status("failed")
            raise
        else:
            mark_status(str(final_attributes.get("status") or success_status))


def _content_stage_progress(config_context: dict[str, Any], stage: str, status: str) -> float:
    total_tasks = config_context.get(CONTENT_PROGRESS_TOTAL_TASKS_KEY)
    task_index = config_context.get(CONTENT_PROGRESS_TASK_INDEX_KEY)
    try:
        total = max(int(total_tasks), 1)
        index = max(int(task_index), 0)
    except (TypeError, ValueError):
        if stage == "content_assembly":
            return PROGRESS_START + PROGRESS_SPAN
        return PROGRESS_START

    stage_offset = CONTENT_STAGE_PROGRESS_OFFSETS.get(stage, 0.5)
    if status == "started":
        stage_offset = max(stage_offset - 0.08, 0.0)
    task_progress = PROGRESS_START + (PROGRESS_SPAN * index / total)
    task_span = PROGRESS_SPAN / total
    return min(PROGRESS_START + PROGRESS_SPAN, task_progress + (task_span * stage_offset))


def _error_summary(error: Exception | str | None) -> str | None:
    if error is None:
        return None
    summary = str(error).strip()
    if not summary:
        return None
    summary = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", summary)
    summary = re.sub(
        r"(?i)(api[_\s-]?key|authorization|token|secret)(\s*[:=]\s*)\S+",
        r"\1\2***",
        summary,
    )
    lowered = summary.lower()
    if "length limit was reached" in lowered or "completion_tokens=16384" in lowered:
        return "Output exceeded generation limit; content was truncated before valid JSON could be parsed."
    if "<!doctype html" in lowered or "<html" in lowered:
        if "520" in summary:
            return "HTTP 520: upstream image API returned an HTML error page."
        return "Upstream image API returned an HTML error page."
    return summary[:240]


def _utc_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sensitive_preview_key(key: Any) -> bool:
    normalized_key = str(key or "").lower()
    return any(secret_key in normalized_key for secret_key in CONTENT_PARTIAL_SECRET_KEYS)


def _safe_preview_content(value: Any) -> Any:
    if isinstance(value, dict):
        safe_payload: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_preview_key(key):
                continue
            safe_payload[str(key)] = _safe_preview_content(item)
        return safe_payload
    if isinstance(value, list):
        return [_safe_preview_content(item) for item in value]
    if isinstance(value, str):
        cleaned = value.strip()
        if "<!doctype html" in cleaned.lower() or "<html" in cleaned.lower():
            return _error_summary(cleaned)
        if len(cleaned) > CONTENT_PARTIAL_MAX_TEXT_LENGTH:
            return f"{cleaned[:CONTENT_PARTIAL_MAX_TEXT_LENGTH]}..."
        return cleaned
    return value


def _emit_content_partial(
    config_context: dict[str, Any],
    stage: str,
    preview_type: str,
    content: dict[str, Any],
    *,
    language: str | None = None,
    target_market: str | None = None,
) -> None:
    recorder = _progress_recorder(config_context)
    if not recorder or not hasattr(recorder, "job_store") or not hasattr(recorder, "job_id"):
        return

    payload: dict[str, Any] = {
        "scope": "content",
        "stage": stage,
        "language": language,
        "target_market": target_market,
        "preview_type": preview_type,
        "content": _safe_preview_content(content),
        "created_at": _utc_iso_timestamp(),
    }
    recorder.job_store.log_event(
        recorder.job_id,
        "content_partial",
        f"Content preview updated: {preview_type}.",
        payload,
    )


def _emit_content_stage(
    config_context: dict[str, Any],
    stage: str,
    status: str,
    message: str,
    *,
    language: str | None = None,
    target_market: str | None = None,
    duration_seconds: float | None = None,
    asset_count: int | None = None,
    score_count: int | None = None,
    error_summary: str | None = None,
) -> None:
    recorder = _progress_recorder(config_context)
    if not recorder:
        return

    payload: dict[str, Any] = {
        "scope": "content",
        "stage": stage,
        "status": status,
    }
    if language:
        payload["language"] = language
    if target_market:
        payload["target_market"] = target_market
    if duration_seconds is not None:
        payload["duration_seconds"] = round(float(duration_seconds), 3)
    if asset_count is not None:
        payload["asset_count"] = asset_count
    if score_count is not None:
        payload["score_count"] = score_count
    if error_summary:
        payload["error_summary"] = error_summary

    recorder.emit_progress(
        "content_stage",
        message,
        _content_stage_progress(config_context, stage, status),
        payload,
    )


def _merge_language_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    articles: list[dict[str, Any]] = []
    social_posts: list[dict[str, Any]] = []
    seo_keywords: list[str] = []
    compliance_notes: list[str] = []
    localized_entities: list[dict[str, Any]] = []
    multimodal_outputs: list[dict[str, Any]] = []
    seo_outputs: list[dict[str, Any]] = []
    cultural_risk_assessments: list[dict[str, Any]] = []
    visual_assets: list[dict[str, Any]] = []
    visual_asset_scores: list[dict[str, Any]] = []
    reddit_geo_posts: list[dict[str, Any]] = []

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
        entities = _localized_entities_payload(output)
        if entities:
            localized_entities.append(entities)
        multimodal_output = output.get("multimodal_output")
        if isinstance(multimodal_output, dict):
            multimodal_outputs.append(multimodal_output)
        seo_metadata = output.get("seo_metadata")
        if isinstance(seo_metadata, dict):
            seo_outputs.append(seo_metadata)
        cultural_risk_assessment = output.get("cultural_risk_assessment")
        if isinstance(cultural_risk_assessment, dict):
            cultural_risk_assessments.append(cultural_risk_assessment)
        visual_assets.extend(
            asset
            for asset in output.get("visual_assets", [])
            if isinstance(asset, dict)
        )
        visual_asset_scores.extend(
            score
            for score in output.get("visual_asset_scores", [])
            if isinstance(score, dict)
        )
        reddit_geo_posts.extend(
            post
            for post in output.get("reddit_geo_posts", [])
            if isinstance(post, dict)
        )

    return {
        "localized_articles": articles,
        "social_media_posts": social_posts,
        "seo_keywords": seo_keywords,
        "compliance_notes": " ".join(compliance_notes),
        "localized_entities": localized_entities,
        "multimodal_outputs": multimodal_outputs,
        "seo_outputs": seo_outputs,
        "cultural_risk_assessments": cultural_risk_assessments,
        "visual_assets": visual_assets,
        "visual_asset_scores": visual_asset_scores,
        "reddit_geo_posts": reddit_geo_posts,
        "reddit_geo_sources": [],
        "production_ready_assets": _production_ready_assets(
            articles,
            social_posts,
            seo_outputs,
            visual_assets,
            reddit_geo_posts,
        ),
    }


def _production_ready_assets(
    articles: list[dict[str, Any]],
    social_posts: list[dict[str, Any]],
    seo_outputs: list[dict[str, Any]],
    visual_assets: list[dict[str, Any]],
    reddit_geo_posts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for article in articles:
        assets.append(
            {
                "asset_type": "localized_article",
                "language": article.get("language"),
                "title": article.get("title"),
                "status": "ready_for_editorial_review",
            }
        )
    for post in social_posts:
        assets.append(
            {
                "asset_type": "social_media_post",
                "language": post.get("language"),
                "platform": post.get("platform"),
                "status": "ready_for_scheduler_review",
            }
        )
    for seo_output in seo_outputs:
        assets.append(
            {
                "asset_type": "seo_metadata",
                "canonical_url_slug": seo_output.get("canonical_url_slug"),
                "status": "ready_for_cms_mapping",
            }
        )
    for visual_asset in visual_assets:
        assets.append(
            {
                "asset_type": "visual_asset",
                "asset_path": visual_asset.get("asset_path"),
                "asset_url": visual_asset.get("asset_url"),
                "duration_seconds": visual_asset.get("duration_seconds"),
                "status": visual_asset.get("status"),
            }
        )
    for post in reddit_geo_posts:
        assets.append(
            {
                "asset_type": "reddit_geo_post",
                "language": post.get("language"),
                "target_market": post.get("target_market"),
                "recommended_subreddit": post.get("recommended_subreddit"),
                "status": "ready_for_manual_reddit_review",
            }
        )
    return assets


def _apply_reddit_geo_context(
    result: dict[str, Any],
    inputs: dict[str, Any],
    reddit_geo_context: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(result)
    if not _reddit_geo_enabled(inputs):
        normalized["reddit_geo_posts"] = []
        normalized["reddit_geo_sources"] = []
        normalized["production_ready_assets"] = _production_ready_assets(
            normalized.get("localized_articles", []),
            normalized.get("social_media_posts", []),
            normalized.get("seo_outputs", []),
            normalized.get("visual_assets", []),
            [],
        )
        return normalized

    normalized["reddit_geo_sources"] = _reddit_geo_sources_from_context(reddit_geo_context)
    reddit_geo_posts = [
        post
        for post in normalized.get("reddit_geo_posts", [])
        if isinstance(post, dict)
    ]
    if not reddit_geo_posts:
        reddit_geo_posts = _fallback_reddit_geo_posts(normalized, inputs, reddit_geo_context)
    reddit_geo_posts = _sanitize_reddit_geo_posts(reddit_geo_posts, inputs, normalized)
    normalized["reddit_geo_posts"] = reddit_geo_posts
    normalized["production_ready_assets"] = _production_ready_assets(
        normalized.get("localized_articles", []),
        normalized.get("social_media_posts", []),
        normalized.get("seo_outputs", []),
        normalized.get("visual_assets", []),
        reddit_geo_posts,
    )
    return normalized


def _fallback_reddit_geo_posts(
    result: dict[str, Any],
    inputs: dict[str, Any],
    reddit_geo_context: dict[str, Any],
) -> list[dict[str, Any]]:
    articles = [
        article
        for article in result.get("localized_articles", [])
        if isinstance(article, dict)
    ]
    posts: list[dict[str, Any]] = []
    for index, article in enumerate(articles):
        language = str(article.get("language") or "").strip()
        target_market = _target_market_for_language(inputs, index)
        localized_inputs = _localized_inputs_from_entities(
            inputs,
            _localized_entities_for_article(result, article, index),
            language,
            target_market,
        )
        sources = _reddit_sources_for_market(reddit_geo_context, target_market)
        source_ids = [str(source.get("source_id")) for source in sources[:3] if source.get("source_id")]
        recommended_subreddit = _recommended_subreddit(sources)
        title = str(article.get("title") or localized_inputs.get("subject") or "Product discussion").strip()
        body_without_link = _fallback_reddit_body_without_link(article, localized_inputs, target_market)
        product_url = str(inputs.get("product_url") or "").strip()
        body = (
            f"{body_without_link}\n\nProduct reference for context: {product_url}"
            if product_url
            else body_without_link
        )
        posts.append(
            {
                "target_market": target_market,
                "language": language or "unspecified",
                "recommended_subreddit": recommended_subreddit,
                "title_options": [
                    title[:180],
                    (
                        "What should buyers in "
                        f"{target_market} check before choosing "
                        f"{localized_inputs.get('subject') or localized_inputs.get('product_category')}?"
                    )[:180],
                ],
                "body": body,
                "body_without_link": body_without_link,
                "disclosure_note": _reddit_disclosure_note(localized_inputs),
                "ai_search_entity_signals": _reddit_entity_signals(localized_inputs, target_market, language),
                "source_ids": source_ids,
                "moderation_notes": [
                    "Manual review required before posting; verify subreddit rules, flair, and self-promotion limits.",
                    "Use transparent affiliation disclosure and avoid presenting the post as neutral user consensus.",
                    "If the subreddit discourages links, use body_without_link.",
                ],
                "data_source": str(reddit_geo_context.get("data_source") or "serper_unavailable"),
                "confidence_level": str(reddit_geo_context.get("confidence_level") or "low"),
            }
        )
    return posts


def _localized_entities_for_article(
    result: dict[str, Any],
    article: dict[str, Any],
    article_index: int,
) -> dict[str, Any] | None:
    entities_list = [
        item
        for item in result.get("localized_entities", [])
        if isinstance(item, dict)
    ]
    language = str(article.get("language") or "").strip()
    for entities in entities_list:
        if str(entities.get("language") or "").strip() == language:
            return entities
    if 0 <= article_index < len(entities_list):
        return entities_list[article_index]
    return None


def _sanitize_reddit_geo_posts(
    reddit_geo_posts: list[dict[str, Any]],
    inputs: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _sanitize_reddit_geo_post(
            post,
            inputs,
            _localized_inputs_for_reddit_post(inputs, result, post, index),
        )
        for index, post in enumerate(reddit_geo_posts)
        if isinstance(post, dict)
    ]


def _localized_inputs_for_reddit_post(
    inputs: dict[str, Any],
    result: dict[str, Any],
    post: dict[str, Any],
    post_index: int,
) -> dict[str, Any]:
    language = str(post.get("language") or "").strip()
    target_market = str(post.get("target_market") or "").strip()
    entities_list = [
        item
        for item in result.get("localized_entities", [])
        if isinstance(item, dict)
    ]
    entities = next(
        (
            item
            for item in entities_list
            if str(item.get("language") or "").strip() == language
        ),
        None,
    )
    if entities is None and 0 <= post_index < len(entities_list):
        entities = entities_list[post_index]
    if not language and entities:
        language = str(entities.get("language") or "").strip()
    if not target_market and entities:
        target_market = str(entities.get("target_market") or "").strip()
    return _localized_inputs_from_entities(inputs, entities, language, target_market)


def _sanitize_reddit_geo_post(
    post: dict[str, Any],
    inputs: dict[str, Any],
    localized_inputs: dict[str, Any],
) -> dict[str, Any]:
    product_url = str(inputs.get("product_url") or "").strip()
    sanitized = dict(post)
    language = str(
        sanitized.get("language") or localized_inputs.get("target_language") or ""
    ).strip()
    body_without_link = _remove_product_links_and_placeholders(
        str(sanitized.get("body_without_link") or sanitized.get("body") or ""),
        product_url,
    )
    body = str(sanitized.get("body") or body_without_link)
    if product_url:
        body = _replace_product_link_placeholders(body, product_url)
    else:
        body = body_without_link
    sanitized["body"] = _replace_source_terms(body, inputs, localized_inputs, language)
    sanitized["body_without_link"] = _replace_source_terms(
        body_without_link,
        inputs,
        localized_inputs,
        language,
    )
    sanitized["title_options"] = _replace_source_terms_in_value(
        sanitized.get("title_options"),
        inputs,
        localized_inputs,
        language,
    )
    sanitized["disclosure_note"] = _replace_source_terms(
        str(sanitized.get("disclosure_note") or ""),
        inputs,
        localized_inputs,
        language,
    )
    sanitized["ai_search_entity_signals"] = _replace_source_terms_in_value(
        sanitized.get("ai_search_entity_signals"),
        inputs,
        localized_inputs,
        language,
    )
    sanitized["moderation_notes"] = _ensure_reddit_moderation_notes(
        _replace_source_terms_in_value(
            sanitized.get("moderation_notes"),
            inputs,
            localized_inputs,
            language,
        )
    )
    return sanitized


def _replace_product_link_placeholders(text: str, product_url: str) -> str:
    value = text
    for placeholder in (
        "[your product link]",
        "[product link]",
        "{product_url}",
        "<product_url>",
        "your product link",
        "product link",
    ):
        value = value.replace(placeholder, product_url)
    return value


def _remove_product_links_and_placeholders(text: str, product_url: str) -> str:
    value = text
    if product_url:
        value = value.replace(product_url, "")
    value = re.sub(
        r"\[[^\]]*(?:product link|your product link)[^\]]*\]\([^)]+\)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    for placeholder in (
        "[your product link]",
        "[product link]",
        "{product_url}",
        "<product_url>",
        "your product link",
        "product link",
    ):
        value = re.sub(re.escape(placeholder), "", value, flags=re.IGNORECASE)
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _ensure_reddit_moderation_notes(value: Any) -> list[str]:
    notes = [
        str(item).strip()
        for item in value
        if str(item).strip()
    ] if isinstance(value, list) else []
    required_note = "Use body_without_link if the subreddit discourages promotional links."
    if required_note not in notes:
        notes.append(required_note)
    return notes


def _reddit_sources_for_market(
    reddit_geo_context: dict[str, Any],
    target_market: str,
) -> list[dict[str, Any]]:
    sources = [
        source
        for source in reddit_geo_context.get("sources", [])
        if isinstance(source, dict)
    ]
    matching = [
        source
        for source in sources
        if str(source.get("target_market") or "").casefold() == target_market.casefold()
    ]
    return matching or sources


def _recommended_subreddit(sources: list[dict[str, Any]]) -> str:
    for source in sources:
        subreddit = str(source.get("subreddit") or "").strip()
        if subreddit:
            return f"r/{subreddit}"
    return "manual_subreddit_review_required"


def _fallback_reddit_body_without_link(
    article: dict[str, Any],
    inputs: dict[str, Any],
    target_market: str,
) -> str:
    excerpt = _compact_reddit_excerpt(str(article.get("article") or ""))
    subject = str(inputs.get("subject") or inputs.get("product_category") or "this product").strip()
    return (
        f"{_reddit_disclosure_note(inputs)}\n\n"
        f"I am preparing a practical buyer-focused post about {subject} for {target_market}. "
        "The angle I would like feedback on is below:\n\n"
        f"{excerpt}\n\n"
        "What would you want clarified before trusting or buying something like this?"
    )


def _compact_reddit_excerpt(value: str) -> str:
    text = re.sub(r"(?m)^#+\s*", "", value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1200] or "Draft content is available, but the article body was empty."


def _reddit_disclosure_note(inputs: dict[str, Any]) -> str:
    brand_name = str(inputs.get("brand_name") or "").strip()
    if brand_name:
        return f"Disclosure: I am affiliated with {brand_name}."
    return "Disclosure: I am affiliated with the product team behind this topic."


def _reddit_entity_signals(inputs: dict[str, Any], target_market: str, language: str) -> list[str]:
    values = [
        str(inputs.get("brand_name") or "").strip(),
        str(inputs.get("subject") or "").strip(),
        str(inputs.get("product_category") or "").strip(),
        target_market,
        language,
        str(inputs.get("product_url") or "").strip(),
    ]
    return [value for value in values if value]


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
    if _reddit_geo_enabled(inputs):
        expected_platforms = {
            platform
            for platform in expected_platforms
            if platform.casefold() != "reddit"
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
    llm_router = ModelTierRouter(config_context)
    research_strategy_lead = Agent(
        config=agents_config["research_strategy_lead"],
        llm=llm_router.llm_for_agent(agents_config["research_strategy_lead"]),
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
        cache=True,
        memory=_crew_memory(config_context),
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
    target_market = _target_market_for_language(inputs, language_index)
    language_task_name = f"content_creation_and_qa:{language}"
    language_agent_role = str(
        agents_config["multilingual_editor"].get("role")
        or "Multilingual Content Creator & Quality Editor"
    )
    traced_config_context = dict(config_context)
    traced_config_context[CONTENT_TRACE_TASK_NAME_KEY] = language_task_name
    traced_config_context[CONTENT_TRACE_AGENT_ROLE_KEY] = language_agent_role
    trace_identity = _content_trace_identity(traced_config_context)
    with agent_span(
        job_id=str(trace_identity["job_id"]),
        workflow_type=str(trace_identity["workflow_type"]),
        task_name=language_task_name,
        agent_role=language_agent_role,
        backend=trace_identity["backend"],
        config_context=traced_config_context,
    ):
        llm_router = ModelTierRouter(_content_generation_llm_context(traced_config_context))
        multilingual_editor = Agent(
            config=agents_config["multilingual_editor"],
            llm=llm_router.llm_for_agent(agents_config["multilingual_editor"]),
            tools=_build_creation_tools(traced_config_context),
            max_retry_limit=CONTENT_GENERATION_AGENT_MAX_RETRY_LIMIT,
            max_iter=CONTENT_GENERATION_AGENT_MAX_ITER,
            max_execution_time=CONTENT_GENERATION_AGENT_MAX_EXECUTION_SECONDS,
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
            cache=True,
            memory=_crew_memory(traced_config_context),
        )
        language_inputs = {
            **inputs,
            "target_language": language,
            "target_languages": [language],
            "target_market": target_market,
            "target_markets": target_market,
            "strategy_context": _serialize_strategy_context(strategy_context),
        }
        stage_started_at = time.monotonic()
        _emit_content_stage(
            traced_config_context,
            "content_generation",
            "started",
            f"Generating localized content for {language}.",
            language=language,
            target_market=target_market,
        )
        try:
            with _content_stage_trace(
                traced_config_context,
                "content_generation",
                language=language,
                target_market=target_market,
            ):
                raw_result = _serialize_crew_result(content_crew.kickoff(inputs=language_inputs))
                usage_metrics = raw_result.get(INTERNAL_USAGE_KEY)
                validation_payload = {
                    key: value
                    for key, value in raw_result.items()
                    if key != INTERNAL_USAGE_KEY
                }
                validation_payload = _normalize_per_language_content_payload(
                    validation_payload
                )
                result = PerLanguageContentOutput.model_validate(
                    validation_payload
                ).model_dump()
                if isinstance(usage_metrics, dict):
                    result[INTERNAL_USAGE_KEY] = usage_metrics
        except Exception as exc:
            error_summary = _error_summary(exc)
            _emit_content_stage(
                traced_config_context,
                "content_generation",
                "failed",
                f"Localized content generation failed for {language}.",
                language=language,
                target_market=target_market,
                duration_seconds=time.monotonic() - stage_started_at,
                error_summary=error_summary,
            )
            _emit_content_partial(
                traced_config_context,
                "content_generation",
                "content_package",
                _failed_content_generation_preview(language, target_market, exc),
                language=language,
                target_market=target_market,
            )
            raise
        _emit_content_stage(
            traced_config_context,
            "content_generation",
            "completed",
            f"Localized content ready for {language}.",
            language=language,
            target_market=target_market,
            duration_seconds=time.monotonic() - stage_started_at,
        )
        _emit_content_partial(
            traced_config_context,
            "content_generation",
            "content_package",
            _content_generation_preview(result),
            language=language,
            target_market=target_market,
        )
        localized_inputs = _localized_inputs_for_language(
            result,
            language_inputs,
            language,
            target_market,
        )
        return _enrich_language_output(
            result,
            localized_inputs,
            language,
            target_market,
            traced_config_context,
        )


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
    reddit_geo_context = _build_reddit_geo_context(normalized_inputs, config_context)
    normalized_inputs["reddit_geo_context_summary"] = _reddit_geo_context_summary(reddit_geo_context)

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='content')
    tasks_config = _load_yaml_config("tasks.yaml")
    languages = [
        str(language).strip()
        for language in normalized_inputs.get("target_languages", [])
        if str(language).strip()
    ]
    if not languages:
        raise ValueError("Content workflow requires at least one target language.")

    recorder = _progress_recorder(config_context)
    task_names = ["research_and_strategy", *[f"content_creation_and_qa:{language}" for language in languages]]
    if recorder:
        recorder.emit_plan(task_names)
        recorder.task_started(
            0,
            len(task_names),
            "research_and_strategy",
            agents_config["research_strategy_lead"]["role"],
        )

    strategy_context = _run_research_strategy(
        normalized_inputs,
        agents_config,
        tasks_config,
        config_context,
    )

    if recorder:
        recorder.task_completed(
            0,
            len(task_names),
            "research_and_strategy",
            agents_config["research_strategy_lead"]["role"],
        )

    worker_config_context = dict(config_context)
    if bool(normalized_inputs.get("generate_visual_assets")):
        worker_config_context["content_image_generation_lock"] = Lock()

    outputs_by_language: dict[str, dict[str, Any]] = {}
    max_workers = min(_content_language_concurrency(config_context), len(languages))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_language = {}
        completed_language_count = 0
        language_agent_role = agents_config["multilingual_editor"]["role"]
        current_progress = PROGRESS_START + (PROGRESS_SPAN / len(task_names))
        for index, language in enumerate(languages, start=1):
            target_market = _target_market_for_language(normalized_inputs, index - 1)
            if recorder:
                recorder.emit_progress(
                    "task_started",
                    f"Task {index + 1}/{len(task_names)} started: content_creation_and_qa:{language}",
                    current_progress,
                    {
                        "scope": "content",
                        "stage": "content_generation",
                        "status": "started",
                        "task_index": index + 1,
                        "total_tasks": len(task_names),
                        "task_name": f"content_creation_and_qa:{language}",
                        "agent_role": language_agent_role,
                        "language": language,
                        "target_market": target_market,
                    },
                )
            language_config_context = dict(worker_config_context)
            language_config_context[CONTENT_PROGRESS_TASK_INDEX_KEY] = index
            language_config_context[CONTENT_PROGRESS_TOTAL_TASKS_KEY] = len(task_names)
            thread_context = contextvars.copy_context()
            future = executor.submit(
                thread_context.run,
                _run_language_generation,
                language,
                index - 1,
                normalized_inputs,
                strategy_context,
                agents_config,
                tasks_config,
                language_config_context,
            )
            future_to_language[future] = (index, language)

        for future in as_completed(future_to_language):
            index, language = future_to_language[future]
            target_market = _target_market_for_language(normalized_inputs, index - 1)
            try:
                outputs_by_language[language] = future.result()
                language_status = "completed"
                language_message = (
                    f"Task {index + 1}/{len(task_names)} completed: "
                    f"content_creation_and_qa:{language}"
                )
                error_summary = None
            except Exception as exc:
                error_summary = _error_summary(exc)
                outputs_by_language[language] = _failed_language_output(
                    language,
                    target_market,
                    exc,
                )
                _emit_content_partial(
                    config_context,
                    "content_generation",
                    "content_package",
                    _failed_content_generation_preview(language, target_market, exc),
                    language=language,
                    target_market=target_market,
                )
                language_status = "failed"
                language_message = (
                    f"Task {index + 1}/{len(task_names)} failed: "
                    f"content_creation_and_qa:{language}"
                )
            completed_language_count += 1
            if recorder:
                progress = PROGRESS_START + (
                    PROGRESS_SPAN * (1 + completed_language_count) / len(task_names)
                )
                payload = {
                    "scope": "content",
                    "stage": "content_generation",
                    "status": language_status,
                    "task_index": index + 1,
                    "total_tasks": len(task_names),
                    "task_name": f"content_creation_and_qa:{language}",
                    "agent_role": language_agent_role,
                    "language": language,
                    "target_market": target_market,
                }
                if error_summary:
                    payload["error_summary"] = error_summary
                recorder.emit_progress(
                    "task_completed" if language_status == "completed" else "task_failed",
                    language_message,
                    progress,
                    payload,
                )

    language_outputs = [outputs_by_language[language] for language in languages]
    assembly_context = dict(config_context)
    assembly_context[CONTENT_PROGRESS_TASK_INDEX_KEY] = len(task_names) - 1
    assembly_context[CONTENT_PROGRESS_TOTAL_TASKS_KEY] = len(task_names)
    assembly_started_at = time.monotonic()
    _emit_content_stage(
        assembly_context,
        "content_assembly",
        "started",
        "Assembling final content package.",
    )
    try:
        with _content_stage_trace(assembly_context, "content_assembly") as mark_stage_status:
            result = _merge_language_outputs(language_outputs)
            usage_metrics = _merge_usage_metrics([strategy_context, *language_outputs])
            if usage_metrics:
                result[INTERNAL_USAGE_KEY] = usage_metrics
            annotated_result = _annotate_localized_output(result, normalized_inputs)
            annotated_result = _apply_reddit_geo_context(
                annotated_result,
                normalized_inputs,
                reddit_geo_context,
            )
            mark_stage_status(
                "completed",
                asset_count=len(annotated_result.get("visual_assets", [])),
                score_count=len(annotated_result.get("visual_asset_scores", [])),
            )
    except Exception as exc:
        _emit_content_stage(
            assembly_context,
            "content_assembly",
            "failed",
            "Final content package assembly failed.",
            duration_seconds=time.monotonic() - assembly_started_at,
            error_summary=_error_summary(exc),
        )
        raise
    _emit_content_stage(
        assembly_context,
        "content_assembly",
        "completed",
        "Final content package assembled.",
        duration_seconds=time.monotonic() - assembly_started_at,
        asset_count=len(annotated_result.get("visual_assets", [])),
        score_count=len(annotated_result.get("visual_asset_scores", [])),
    )
    return annotated_result
