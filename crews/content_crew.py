from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field
from tools.custom.content_tools import (
    CulturalComplianceCheckerTool,
    MultiEngineSEOOptimizerTool,
    MultimodalLocalizationTool,
    OpenAIImageGenerationTool,
    VisualAssetScoringTool,
)
from utils.crew_result import serialize_crew_result
from utils.llm_config import build_llm
from utils.project_intelligence import augment_agents_config
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
    error: str | None = Field(None, description="Generation error or skip reason")


class VisualAssetScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Scoring status")
    asset_path: str | None = Field(None, description="Local asset path that was scored")
    prompt_alignment_score: float = Field(..., ge=0, le=100)
    cultural_fit_score: float = Field(..., ge=0, le=100)
    brand_voice_score: float = Field(..., ge=0, le=100)
    publish_readiness_score: float = Field(..., ge=0, le=100)
    notes: str = Field(..., description="Vision scoring notes")
    error: str | None = Field(None, description="Scoring error or skip reason")


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


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_search_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = []
    if config_context.get("serper_api_key"):
        tools.append(SerperDevTool())
    return tools


def _build_content_intelligence_tools() -> list[Any]:
    return [
        MultimodalLocalizationTool(),
        MultiEngineSEOOptimizerTool(),
        CulturalComplianceCheckerTool(),
    ]


def _build_creation_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool(), *_build_content_intelligence_tools()]
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
    normalized["brand_voice_summary"] = _brand_voice(normalized)
    normalized["primary_keywords_summary"] = ", ".join(_primary_keywords(normalized))
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
    product = _product_name(inputs)
    brand_voice = _brand_voice(inputs)
    primary_keywords = _primary_keywords(inputs)
    platforms = [
        str(platform).strip()
        for platform in inputs.get("platforms", [])
        if str(platform).strip()
    ]

    localization_output = MultimodalLocalizationTool()._run(
        product=product,
        target_market=target_market,
        language=language,
        brand_voice=brand_voice,
        platforms=platforms,
    )
    seo_output = MultiEngineSEOOptimizerTool()._run(
        product=product,
        target_market=target_market,
        language=language,
        primary_keywords=primary_keywords,
    )
    compliance_output = CulturalComplianceCheckerTool()._run(
        content_summary=_content_summary(output, localization_output, seo_output),
        target_market=target_market,
        language=language,
    )

    visual_assets: list[dict[str, Any]] = []
    visual_asset_scores: list[dict[str, Any]] = []
    if bool(inputs.get("generate_visual_assets")):
        image_tool = OpenAIImageGenerationTool(
            openai_api_key=_openai_api_key_for_images(config_context),
            image_model=str(config_context.get("content_image_model") or "gpt-image-2"),
            artifact_dir=str(
                config_context.get("content_image_artifact_dir")
                or "artifacts/content_creation"
            ),
        )
        image_result = image_tool._run(
            prompt=localization_output["visual_spec"]["ai_image_prompt"],
            output_slug=_asset_output_slug(inputs, language, target_market),
            image_generation_count=int(inputs.get("image_generation_count") or 1),
            image_quality=str(inputs.get("image_quality") or "auto"),
            image_size=str(inputs.get("image_size") or "1024x1024"),
        )
        visual_assets = _visual_assets_from_generation_result(image_result)
        scoring_tool = VisualAssetScoringTool(
            openai_api_key=_openai_api_key_for_images(config_context),
            scoring_model=str(
                config_context.get("content_image_scoring_model")
                or config_context.get("openai_model_name")
                or "gpt-4o-mini"
            ),
        )
        visual_asset_scores = _score_visual_assets(
            scoring_tool,
            visual_assets,
            localization_output["visual_spec"]["ai_image_prompt"],
            target_market,
            language,
            brand_voice,
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
    if all(hasattr(recorder, name) for name in ("emit_plan", "task_started", "task_completed")):
        return recorder
    return None


def _merge_language_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    articles: list[dict[str, Any]] = []
    social_posts: list[dict[str, Any]] = []
    seo_keywords: list[str] = []
    compliance_notes: list[str] = []
    multimodal_outputs: list[dict[str, Any]] = []
    seo_outputs: list[dict[str, Any]] = []
    cultural_risk_assessments: list[dict[str, Any]] = []
    visual_assets: list[dict[str, Any]] = []
    visual_asset_scores: list[dict[str, Any]] = []

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

    return {
        "localized_articles": articles,
        "social_media_posts": social_posts,
        "seo_keywords": seo_keywords,
        "compliance_notes": " ".join(compliance_notes),
        "multimodal_outputs": multimodal_outputs,
        "seo_outputs": seo_outputs,
        "cultural_risk_assessments": cultural_risk_assessments,
        "visual_assets": visual_assets,
        "visual_asset_scores": visual_asset_scores,
        "production_ready_assets": _production_ready_assets(
            articles,
            social_posts,
            seo_outputs,
            visual_assets,
        ),
    }


def _production_ready_assets(
    articles: list[dict[str, Any]],
    social_posts: list[dict[str, Any]],
    seo_outputs: list[dict[str, Any]],
    visual_assets: list[dict[str, Any]],
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
                "status": visual_asset.get("status"),
            }
        )
    return assets


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
    llm = build_llm(config_context)
    research_strategy_lead = Agent(
        config=agents_config["research_strategy_lead"],
        llm=llm,
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
    llm = build_llm(config_context)
    multilingual_editor = Agent(
        config=agents_config["multilingual_editor"],
        llm=llm,
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
    result = _serialize_crew_result(content_crew.kickoff(inputs=language_inputs))
    return _enrich_language_output(
        result,
        inputs,
        language,
        language_inputs["target_market"],
        config_context,
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

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='content')
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
