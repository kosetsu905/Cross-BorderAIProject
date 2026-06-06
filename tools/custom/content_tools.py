import base64
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

logger = logging.getLogger(__name__)

OPENAI_IMAGE_GENERATION_URL = "https://api.openai.com/v1/images/generations"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
SERPER_SEARCH_URL = "https://google.serper.dev/search"
DEFAULT_CONTENT_ARTIFACT_DIR = "artifacts/content_creation"
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


class MultimodalLocalizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str = Field(..., min_length=1)
    target_market: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    brand_voice: str = Field(default="Premium, trustworthy, and culturally respectful")
    platforms: list[str] | None = None


class MultiEngineSEOInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str = Field(..., min_length=1)
    target_market: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    primary_keywords: list[str] | None = None
    brand_name: str = Field(default="YourBrand")
    base_url: str = Field(default="https://example.com")


class CulturalComplianceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_summary: str = Field(..., min_length=1)
    target_market: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)


class OpenAIImageGenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)
    output_slug: str = Field(default="content-visual")
    image_generation_count: int = Field(default=1, ge=1, le=4)
    image_quality: str = Field(default="auto")
    image_size: str = Field(default="1024x1024")


class VisualAssetScoringInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_path: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    target_market: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    brand_voice: str = Field(default="Premium, trustworthy, and culturally respectful")


class RedditGeoSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., min_length=1)
    product_category: str = Field(..., min_length=1)
    target_markets: str = Field(..., min_length=1)
    primary_keywords: list[str] | None = None
    target_languages: list[str] | None = None
    brand_name: str | None = None
    max_results_per_query: int = Field(default=5, ge=1, le=10)


class RedditGeoQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_market: str = Field(..., min_length=1)
    target_language: str = Field(default="", description="Language assigned to this target market")
    query_type: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class RedditGeoSourceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    target_market: str = Field(..., min_length=1)
    target_language: str = Field(default="", description="Language assigned to this Reddit source")
    subreddit: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    snippet: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    query_type: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class RedditGeoSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., min_length=1)
    data_source: str = Field(..., min_length=1)
    confidence_level: str = Field(..., min_length=1)
    assumption_notice: str = Field(..., min_length=1)
    query_pack: list[RedditGeoQueryResult] = Field(default_factory=list)
    sources: list[RedditGeoSourceResult] = Field(default_factory=list)
    duration_seconds: float = Field(..., ge=0)
    provider_error: str | None = None


class MultimodalLocalizationTool(BaseTool):
    name: str = "Multimodal Localization Engine"
    description: str = (
        "Generates market-aware visual specs, AI image prompts, short video "
        "storyboards, and image-text consistency guidance."
    )
    args_schema: type[BaseModel] = MultimodalLocalizationInput

    def _run(
        self,
        product: str,
        target_market: str,
        language: str,
        brand_voice: str = "Premium, trustworthy, and culturally respectful",
        platforms: list[str] | None = None,
    ) -> dict[str, Any]:
        market_profile = _market_profile(target_market)
        selected_platforms = platforms or market_profile["platforms"]
        visual_text_language = _visual_text_language(language)
        image_prompt = (
            f"Premium e-commerce product photography of {product}. "
            f"Market: {target_market}. Visual style: {market_profile['style']}. "
            f"Palette: {market_profile['colors']}. Scene: {market_profile['background']}. "
            f"Brand voice: {brand_voice}. Use clear product detail, natural lighting, "
            "culturally respectful props, no prohibited symbols, no unsupported claims. "
            "If any readable text, signage, packaging copy, CTA, or label appears in the image, "
            f"it must be only {visual_text_language}; do not mix another language except brand "
            "names or product names, and avoid gibberish or pseudo-text."
        )
        return {
            "target_market": target_market,
            "language_code": language,
            "visual_spec": {
                "style_guide": market_profile["style"],
                "color_palette": market_profile["colors"],
                "model_demographics": market_profile["model_guidance"],
                "background_scene": market_profile["background"],
                "cultural_notes": market_profile["cultural_notes"],
                "ai_image_prompt": image_prompt,
            },
            "video_script": _video_script(product, target_market, brand_voice),
            "image_text_consistency_check": (
                "Check that the hero benefit, props, color mood, on-screen text, "
                f"and CTA all support the {language} copy. Any visible text must be only "
                f"{visual_text_language}, without adding claims that are absent from "
                "first-party product inputs."
            ),
            "recommended_platforms": selected_platforms,
        }


class MultiEngineSEOOptimizerTool(BaseTool):
    name: str = "Multi-Search Engine SEO Optimizer"
    description: str = (
        "Generates engine-specific SEO metadata, JSON-LD, alt text variants, "
        "and hreflang tags for cross-border e-commerce content."
    )
    args_schema: type[BaseModel] = MultiEngineSEOInput

    def _run(
        self,
        product: str,
        target_market: str,
        language: str,
        primary_keywords: list[str] | None = None,
        brand_name: str = "YourBrand",
        base_url: str = "https://example.com",
    ) -> dict[str, Any]:
        keywords = _clean_keywords(primary_keywords, product)
        slug = _slugify(f"{product}-{language}")
        canonical_url_slug = f"/products/{slug}"
        url = f"{base_url.rstrip('/')}{canonical_url_slug}"
        schema_ld = {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": product,
            "description": (
                f"{product} content package localized for {target_market}. "
                "Validate final product claims against first-party source material."
            ),
            "brand": {"@type": "Brand", "name": brand_name},
            "offers": {
                "@type": "Offer",
                "url": url,
                "priceCurrency": _currency_for_market(target_market),
                "availability": "https://schema.org/InStock",
            },
        }
        return {
            "canonical_url_slug": canonical_url_slug,
            "engine_specific_metadata": [
                _engine_strategy("Google", product, target_market, keywords),
                _engine_strategy("Baidu", product, target_market, keywords),
                _engine_strategy("Yandex", product, target_market, keywords),
                _engine_strategy("Naver", product, target_market, keywords),
                _engine_strategy("Yahoo_Japan", product, target_market, keywords),
                _engine_strategy("Google_Saudi", product, target_market, keywords),
            ],
            "schema_markup_jsonld": json.dumps(schema_ld, ensure_ascii=False, indent=2),
            "alt_text_variants": [
                {
                    "language": language,
                    "alt_text": f"{product} shown for {target_market} with {keywords[0]} focus",
                },
                {
                    "language": "en",
                    "alt_text": f"Localized product image of {product} emphasizing {keywords[0]}",
                },
            ],
            "hreflang_tags": [
                f'<link rel="alternate" hreflang="{language}" href="{url}" />',
                f"<link rel=\"alternate\" hreflang=\"x-default\" href=\"{base_url.rstrip('/')}/products/{_slugify(product)}\" />",
            ],
        }


class CulturalComplianceCheckerTool(BaseTool):
    name: str = "Cultural Compliance & Risk Auditor"
    description: str = (
        "Scans localized content summaries for cultural sensitivity, brand safety, "
        "and regional e-commerce compliance reminders."
    )
    args_schema: type[BaseModel] = CulturalComplianceInput

    def _run(self, content_summary: str, target_market: str, language: str) -> dict[str, Any]:
        profile = _market_profile(target_market)
        normalized_summary = content_summary.lower()
        flags = [
            risk["message"]
            for risk in profile["risk_terms"]
            if risk["term"] in normalized_summary
        ]
        return {
            "market": target_market,
            "language": language,
            "risk_flags": flags or ["No critical cultural risks detected by rule scan."],
            "compliance_checklist": [
                f"Language output targets {language}.",
                "Product claims should be traceable to user-provided features or verified sources.",
                "Visual prompts should avoid prohibited products, symbols, and unsafe gestures.",
                *profile["regulatory_notes"],
            ],
            "recommended_actions": [
                "Run final native-speaker review before launch.",
                "Validate platform ad policy for each channel before paid distribution.",
                "A/B test imagery and CTA tone with local audiences.",
            ],
        }


class RedditGeoSearchTool(BaseTool):
    name: str = "Reddit GEO Search Tool"
    description: str = (
        "Finds Reddit discussion context by target market through Serper site:reddit.com "
        "searches for manual Reddit GEO content planning."
    )
    args_schema: type[BaseModel] = RedditGeoSearchInput
    serper_api_key: str | None = None

    def _run(
        self,
        subject: str,
        product_category: str,
        target_markets: str,
        primary_keywords: list[str] | None = None,
        target_languages: list[str] | None = None,
        brand_name: str | None = None,
        max_results_per_query: int = 5,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        query_pack = _reddit_geo_query_pack(
            subject,
            product_category,
            target_markets,
            primary_keywords,
            target_languages,
            brand_name,
        )
        if not self.serper_api_key:
            return _reddit_geo_output(
                status="skipped_missing_credentials",
                data_source="serper_unavailable",
                confidence_level="low",
                assumption_notice=(
                    "SERPER_API_KEY is not configured, so Reddit GEO context is limited "
                    "to generic manual-review guidance."
                ),
                query_pack=query_pack,
                sources=[],
                started_at=started_at,
            )

        try:
            sources: list[dict[str, str]] = []
            for query_def in query_pack:
                payload = _post_serper_search(
                    self.serper_api_key,
                    query_def["query"],
                    max_results_per_query,
                )
                sources.extend(_reddit_sources_from_serper_payload(payload, query_def))

            sources = _dedupe_reddit_sources(sources)
            for index, source in enumerate(sources, start=1):
                source["source_id"] = f"R{index}"

            return _reddit_geo_output(
                status="live_search",
                data_source="serper_search",
                confidence_level="medium" if sources else "low",
                assumption_notice=(
                    "Reddit GEO context is based on live Serper search snippets from Reddit. "
                    "Review each subreddit rule page and source thread before posting."
                ),
                query_pack=query_pack,
                sources=sources,
                started_at=started_at,
            )
        except Exception as exc:
            logger.warning("Reddit GEO Serper lookup failed: %s", exc)
            return _reddit_geo_output(
                status="fallback_after_provider_error",
                data_source="serper_error",
                confidence_level="low",
                assumption_notice=(
                    "Reddit GEO search failed. Generated Reddit posts should be treated "
                    "as draft templates until a human validates subreddit fit."
                ),
                query_pack=query_pack,
                sources=[],
                started_at=started_at,
                provider_error=_safe_error(exc),
            )


class OpenAIImageGenerationTool(BaseTool):
    name: str = "OpenAI Image Generation Tool"
    description: str = (
        "Optionally generates image assets using the OpenAI Image API and stores "
        "local artifact paths instead of returning base64 payloads."
    )
    args_schema: type[BaseModel] = OpenAIImageGenerationInput
    openai_api_key: str | None = None
    image_model: str = "gpt-image-2"
    artifact_dir: str = DEFAULT_CONTENT_ARTIFACT_DIR

    def _run(
        self,
        prompt: str,
        output_slug: str = "content-visual",
        image_generation_count: int = 1,
        image_quality: str = "auto",
        image_size: str = "1024x1024",
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        attempt_metadata: dict[str, int | None] = {"attempts": 0, "last_status_code": None}
        if not self.openai_api_key:
            return {
                "status": "skipped_missing_credentials",
                "model": self.image_model,
                "prompt": prompt,
                "assets": [],
                "duration_seconds": 0.0,
                "attempts": 0,
                "last_status_code": None,
                "retryable_status": False,
                "error": "OPENAI_API_KEY is not configured for content image generation.",
            }

        payload = {
            "model": self.image_model,
            "prompt": prompt,
            "n": max(1, min(int(image_generation_count), 4)),
            "quality": image_quality,
            "size": image_size,
        }
        try:
            response_payload = _post_openai_json(
                OPENAI_IMAGE_GENERATION_URL,
                self.openai_api_key,
                payload,
                timeout_seconds=120,
                attempt_recorder=lambda status_code: _record_http_attempt(
                    attempt_metadata,
                    status_code,
                ),
            )
            assets = _save_image_generation_payload(
                response_payload,
                self.artifact_dir,
                output_slug,
                self.image_model,
                prompt,
                time.monotonic() - started_at,
            )
            return {
                "status": "completed" if assets else "completed_no_local_asset",
                "model": self.image_model,
                "prompt": prompt,
                "assets": assets,
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "attempts": attempt_metadata["attempts"],
                "last_status_code": attempt_metadata["last_status_code"],
                "retryable_status": _is_retryable_status_code(
                    attempt_metadata["last_status_code"]
                ),
                "error": None,
            }
        except Exception as exc:
            if _is_image_generation_moderation_block(exc):
                fallback_prompt = _safe_image_generation_fallback_prompt()
                fallback_payload = {
                    **payload,
                    "prompt": fallback_prompt,
                }
                try:
                    response_payload = _post_openai_json(
                        OPENAI_IMAGE_GENERATION_URL,
                        self.openai_api_key,
                        fallback_payload,
                        timeout_seconds=120,
                        attempt_recorder=lambda status_code: _record_http_attempt(
                            attempt_metadata,
                            status_code,
                        ),
                    )
                    assets = _save_image_generation_payload(
                        response_payload,
                        self.artifact_dir,
                        output_slug,
                        self.image_model,
                        fallback_prompt,
                        time.monotonic() - started_at,
                    )
                    return {
                        "status": "completed" if assets else "completed_no_local_asset",
                        "model": self.image_model,
                        "prompt": fallback_prompt,
                        "assets": assets,
                        "duration_seconds": round(time.monotonic() - started_at, 3),
                        "attempts": attempt_metadata["attempts"],
                        "last_status_code": attempt_metadata["last_status_code"],
                        "retryable_status": _is_retryable_status_code(
                            attempt_metadata["last_status_code"]
                        ),
                        "error": None,
                    }
                except Exception as fallback_exc:
                    logger.warning("OpenAI safe fallback image generation failed: %s", fallback_exc)
                    last_status_code = _last_status_code(attempt_metadata, fallback_exc)
                    return {
                        "status": "failed",
                        "model": self.image_model,
                        "prompt": fallback_prompt,
                        "assets": [],
                        "duration_seconds": round(time.monotonic() - started_at, 3),
                        "attempts": attempt_metadata["attempts"],
                        "last_status_code": last_status_code,
                        "retryable_status": _is_retryable_status_code(last_status_code),
                        "error": (
                            "Initial image prompt was blocked by moderation and the safe "
                            f"fallback prompt also failed: {_safe_error(fallback_exc)}"
                        ),
                    }
            logger.warning("OpenAI image generation failed: %s", exc)
            last_status_code = _last_status_code(attempt_metadata, exc)
            return {
                "status": "failed",
                "model": self.image_model,
                "prompt": prompt,
                "assets": [],
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "attempts": attempt_metadata["attempts"],
                "last_status_code": last_status_code,
                "retryable_status": _is_retryable_status_code(last_status_code),
                "error": _safe_error(exc),
            }


class VisualAssetScoringTool(BaseTool):
    name: str = "Visual Asset Scoring Tool"
    description: str = (
        "Optionally scores generated assets with an OpenAI vision-capable model "
        "for prompt alignment, cultural fit, brand voice, and publish readiness."
    )
    args_schema: type[BaseModel] = VisualAssetScoringInput
    openai_api_key: str | None = None
    scoring_model: str = "gpt-4o-mini"

    def _run(
        self,
        asset_path: str,
        prompt: str,
        target_market: str,
        language: str,
        brand_voice: str = "Premium, trustworthy, and culturally respectful",
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        if not self.openai_api_key:
            return {
                "status": "skipped_missing_credentials",
                "asset_path": asset_path,
                "prompt_alignment_score": 0.0,
                "cultural_fit_score": 0.0,
                "brand_voice_score": 0.0,
                "publish_readiness_score": 0.0,
                "duration_seconds": 0.0,
                "notes": "OPENAI_API_KEY is not configured for visual asset scoring.",
                "error": None,
            }

        path = Path(asset_path)
        if not path.is_file():
            return {
                "status": "skipped_missing_asset",
                "asset_path": asset_path,
                "prompt_alignment_score": 0.0,
                "cultural_fit_score": 0.0,
                "brand_voice_score": 0.0,
                "publish_readiness_score": 0.0,
                "duration_seconds": 0.0,
                "notes": "Asset path does not exist.",
                "error": None,
            }

        try:
            data_url = _image_data_url(path)
            response_payload = _post_openai_json(
                OPENAI_RESPONSES_URL,
                self.openai_api_key,
                _visual_score_payload(
                    self.scoring_model,
                    data_url,
                    prompt,
                    target_market,
                    language,
                    brand_voice,
                ),
                timeout_seconds=90,
            )
            parsed = _parse_visual_score_response(response_payload)
            return {
                "status": "completed",
                "asset_path": str(path.resolve()),
                **parsed,
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "error": None,
            }
        except Exception as exc:
            logger.warning("OpenAI visual asset scoring failed: %s", exc)
            return {
                "status": "failed",
                "asset_path": asset_path,
                "prompt_alignment_score": 0.0,
                "cultural_fit_score": 0.0,
                "brand_voice_score": 0.0,
                "publish_readiness_score": 0.0,
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "notes": "Vision scoring failed; route asset to manual review.",
                "error": _safe_error(exc),
            }


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.TransportError,
            httpx.WriteError,
            httpx.WriteTimeout,
        ),
    )


def _record_http_attempt(attempt_metadata: dict[str, int | None], status_code: int | None) -> None:
    if status_code is None:
        attempt_metadata["attempts"] = int(attempt_metadata.get("attempts") or 0) + 1
        return
    attempt_metadata["last_status_code"] = status_code


def _last_status_code(
    attempt_metadata: dict[str, int | None],
    exc: Exception,
) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return attempt_metadata.get("last_status_code")


def _is_retryable_status_code(status_code: int | None) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def _is_image_generation_moderation_block(exc: Exception) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code != 400:
        return False
    try:
        payload = exc.response.json()
    except ValueError:
        payload = {}
    error_payload = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error_payload, dict):
        code = str(error_payload.get("code") or "").lower()
        error_type = str(error_payload.get("type") or "").lower()
        if code == "moderation_blocked" or error_type == "image_generation_user_error":
            return True
    text = exc.response.text.lower()
    return "moderation_blocked" in text or "image_generation_user_error" in text


def _safe_image_generation_fallback_prompt() -> str:
    return (
        "Premium e-commerce studio product display of generic sports collectible "
        "trading card packs and protective card sleeves on a clean neutral table. "
        "No real athletes, celebrities, teams, leagues, flags, medals, official event "
        "marks, logos, brand names, jersey designs, investment claims, pricing claims, "
        "readable text, labels, signage, watermarks, or unsupported product claims. "
        "Use neutral lighting, clear product detail, soft depth of field, and safe "
        "commercial catalog styling."
    )


@retry(
    retry=retry_if_exception(_is_retryable_exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.25, min=0, max=2),
    reraise=True,
)
def _post_openai_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    attempt_recorder: Callable[[int | None], None] | None = None,
) -> dict[str, Any]:
    if attempt_recorder:
        attempt_recorder(None)
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    if attempt_recorder:
        attempt_recorder(response.status_code)
    response.raise_for_status()
    return response.json()


@retry(
    retry=retry_if_exception(_is_retryable_exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.25, min=0, max=2),
    reraise=True,
)
def _post_serper_search(
    serper_api_key: str,
    query: str,
    max_results: int,
) -> dict[str, Any]:
    response = httpx.post(
        SERPER_SEARCH_URL,
        headers={
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": max_results},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _reddit_geo_output(
    *,
    status: str,
    data_source: str,
    confidence_level: str,
    assumption_notice: str,
    query_pack: list[dict[str, str]],
    sources: list[dict[str, str]],
    started_at: float,
    provider_error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "data_source": data_source,
        "confidence_level": confidence_level,
        "assumption_notice": assumption_notice,
        "query_pack": query_pack,
        "sources": sources,
        "duration_seconds": round(time.monotonic() - started_at, 3),
        "provider_error": provider_error,
    }
    return RedditGeoSearchOutput.model_validate(payload).model_dump()


def _reddit_geo_query_pack(
    subject: str,
    product_category: str,
    target_markets: str,
    primary_keywords: list[str] | None,
    target_languages: list[str] | None,
    brand_name: str | None,
) -> list[dict[str, str]]:
    markets = _split_target_markets(target_markets)
    keyword = _first_keyword(primary_keywords, subject)
    brand_term = f' "{brand_name.strip()}"' if isinstance(brand_name, str) and brand_name.strip() else ""
    query_pack: list[dict[str, str]] = []
    for index, market in enumerate(markets):
        target_language = _target_language_for_market_index(target_languages, index, len(markets))
        query_pack.extend(
            [
                {
                    "target_market": market,
                    "target_language": target_language,
                    "query_type": "reddit_need_discussion",
                    "query": (
                        f'site:reddit.com/r/ "{subject}" "{market}" '
                        f'"{keyword}" advice OR recommendations{brand_term}'
                    ),
                },
                {
                    "target_market": market,
                    "target_language": target_language,
                    "query_type": "reddit_category_questions",
                    "query": (
                        f'site:reddit.com/r/ "{product_category}" "{market}" '
                        '"what should I buy" OR "is it worth it"'
                    ),
                },
            ]
        )
    return query_pack


def _target_language_for_market_index(
    target_languages: list[str] | None,
    market_index: int,
    market_count: int,
) -> str:
    languages = [
        str(language).strip()
        for language in target_languages or []
        if str(language).strip()
    ]
    if len(languages) == market_count:
        return languages[market_index]
    if len(languages) == 1:
        return languages[0]
    return ""


def _split_target_markets(value: str) -> list[str]:
    markets = [
        market.strip()
        for market in str(value or "").split(",")
        if market.strip()
    ]
    return markets or ["Global"]


def _first_keyword(primary_keywords: list[str] | None, subject: str) -> str:
    for keyword in primary_keywords or []:
        if isinstance(keyword, str) and keyword.strip():
            return keyword.strip()
    return subject.strip()


def _reddit_sources_from_serper_payload(
    payload: dict[str, Any],
    query_def: dict[str, str],
) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for item in payload.get("organic") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("link") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        url = _normalized_reddit_url_for_language(
            url,
            query_def.get("target_language", ""),
        )
        if url is None:
            continue
        if _snippet_is_obvious_translation_mismatch(
            snippet,
            query_def.get("target_language", ""),
        ):
            continue
        subreddit = _subreddit_from_url(url)
        if not url or not title or not snippet or not subreddit:
            continue
        sources.append(
            {
                "source_id": "",
                "target_market": query_def["target_market"],
                "target_language": query_def.get("target_language", ""),
                "subreddit": subreddit,
                "title": title,
                "snippet": snippet,
                "url": url,
                "query_type": query_def["query_type"],
                "query": query_def["query"],
            }
        )
    return sources


def _normalized_reddit_url_for_language(url: str, target_language: str) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    translation_languages = [
        value
        for key, value in query_items
        if key.lower() == "tl" and value
    ]
    for translation_language in translation_languages:
        if not _reddit_translation_matches_target(translation_language, target_language):
            return None
    filtered_query_items = [
        (key, value)
        for key, value in query_items
        if key.lower() != "tl"
    ]
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(filtered_query_items, doseq=True),
            parsed.fragment,
        )
    )


def _reddit_translation_matches_target(translation_language: str, target_language: str) -> bool:
    if not target_language:
        return False
    translation = translation_language.strip().lower()
    target = target_language.strip().lower().replace("_", "-")
    if not translation or not target:
        return False
    if target.startswith("zh"):
        return translation.startswith("zh")
    target_base = target.split("-", 1)[0]
    translation_base = translation.split("-", 1)[0]
    return target_base == translation_base


def _snippet_is_obvious_translation_mismatch(snippet: str, target_language: str) -> bool:
    target = target_language.strip().lower()
    if target.startswith(("zh", "ja", "ko")):
        return False
    cjk_count = sum(1 for character in snippet if "\u4e00" <= character <= "\u9fff")
    return cjk_count >= 8


def _subreddit_from_url(url: str) -> str:
    match = re.search(r"reddit\.com/r/([^/?#]+)", url, flags=re.IGNORECASE)
    if not match:
        return ""
    subreddit = match.group(1).strip()
    if not subreddit or subreddit.lower() in {"all", "popular"}:
        return ""
    return subreddit


def _reddit_source_key(source: dict[str, str]) -> str:
    url = source.get("url", "").strip()
    if url:
        return re.sub(r"[?#].*$", "", url).rstrip("/").lower()
    return "|".join(
        [
            source.get("subreddit", "").lower(),
            source.get("title", "").strip().lower(),
            source.get("snippet", "").strip().lower(),
        ]
    )


def _dedupe_reddit_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique_sources: list[dict[str, str]] = []
    for source in sources:
        key = _reddit_source_key(source)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_sources.append(source)
    return unique_sources


def _save_image_generation_payload(
    response_payload: dict[str, Any],
    artifact_dir: str,
    output_slug: str,
    model: str,
    prompt: str,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []
    for index, item in enumerate(response_payload.get("data") or [], start=1):
        revised_prompt = item.get("revised_prompt")
        if item.get("b64_json"):
            image_bytes = base64.b64decode(str(item["b64_json"]))
            file_name = f"{_slugify(output_slug)}-{uuid.uuid4().hex[:8]}-{index}.png"
            path = output_dir / file_name
            path.write_bytes(image_bytes)
            assets.append(
                {
                    "status": "completed",
                    "asset_path": str(path.resolve()),
                    "asset_url": None,
                    "model": model,
                    "prompt": prompt,
                    "revised_prompt": revised_prompt,
                    "content_type": "image/png",
                    "duration_seconds": round(duration_seconds, 3),
                }
            )
        elif item.get("url"):
            assets.append(
                {
                    "status": "completed_remote_url",
                    "asset_path": None,
                    "asset_url": str(item["url"]),
                    "model": model,
                    "prompt": prompt,
                    "revised_prompt": revised_prompt,
                    "content_type": "image/remote",
                    "duration_seconds": round(duration_seconds, 3),
                }
            )
    return assets


def _visual_score_payload(
    model: str,
    data_url: str,
    prompt: str,
    target_market: str,
    language: str,
    brand_voice: str,
) -> dict[str, Any]:
    visual_text_language = _visual_text_language(language)
    rubric = (
        "Score this e-commerce visual asset as JSON only with keys "
        "prompt_alignment_score, cultural_fit_score, brand_voice_score, "
        "publish_readiness_score, notes. Scores must be numbers from 0 to 100. "
        f"Prompt: {prompt}. Market: {target_market}. Language: {language}. "
        f"Brand voice: {brand_voice}. Check for unsupported product claims, "
        "culturally risky props, text legibility, and production readiness. "
        "Inspect every visible word, label, sign, package, and CTA. If readable "
        f"text appears, it must be only {visual_text_language}; if another language, "
        "mixed-language copy, gibberish, or pseudo-text appears, set publish_readiness_score "
        "to 60 or lower and explain the language issue in notes."
    )
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": rubric},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ],
    }


def _parse_visual_score_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    text = _extract_response_text(response_payload)
    json_text = _extract_json_object_text(text)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        parsed = {"notes": text}
    return {
        "prompt_alignment_score": _score(parsed.get("prompt_alignment_score")),
        "cultural_fit_score": _score(parsed.get("cultural_fit_score")),
        "brand_voice_score": _score(parsed.get("brand_voice_score")),
        "publish_readiness_score": _score(parsed.get("publish_readiness_score")),
        "notes": str(parsed.get("notes") or "No scoring notes returned.").strip(),
    }


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    fenced_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        stripped,
        flags=re.IGNORECASE,
    )
    if fenced_match:
        stripped = fenced_match.group(1).strip()

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return stripped[object_start : object_end + 1].strip()
    return stripped


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    if isinstance(response_payload.get("output_text"), str):
        return str(response_payload["output_text"]).strip()
    chunks: list[str] = []
    for output in response_payload.get("output") or []:
        for content in output.get("content") or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _market_profile(target_market: str) -> dict[str, Any]:
    key = _market_key(target_market)
    profiles: dict[str, dict[str, Any]] = {
        "japan": {
            "style": "Minimalist clean, precise detail shots, quiet premium mood",
            "colors": "Pastel natural neutrals with soft seasonal accent colors",
            "model_guidance": "Use modest styling, natural expressions, and local lifestyle context.",
            "background": "Modern compact home, wooden table, soft daylight, seasonal botanical accents",
            "platforms": ["Instagram", "LINE", "TikTok", "YouTube Shorts"],
            "cultural_notes": [
                "Avoid overly aggressive CTAs.",
                "Emphasize craftsmanship, reliability, and after-sales support.",
                "Use close-ups to show detail instead of exaggerated claims.",
            ],
            "regulatory_notes": [
                "Reference clear return and seller information for Japanese commerce pages.",
            ],
            "risk_terms": [
                {"term": "guaranteed cure", "message": "Avoid unverified health claims in Japan."},
                {"term": "best ever", "message": "Replace absolute superiority claims with qualified wording."},
            ],
        },
        "saudi_arabia": {
            "style": "Warm hospitality, family sharing, rich but restrained premium mood",
            "colors": "Warm gold, deep green, cream, and soft neutral accents",
            "model_guidance": "Use modest wardrobe guidance and respectful family or home context.",
            "background": "Modern home gathering, coffee service, dates, warm evening light",
            "platforms": ["Instagram", "Snapchat", "TikTok", "YouTube Shorts"],
            "cultural_notes": [
                "Avoid alcohol, pork, dating themes, or disrespectful religious references.",
                "Use modest styling and family-oriented scenes.",
                "Check Arabic text direction and typography before launch.",
            ],
            "regulatory_notes": [
                "Confirm Arabic customer support, VAT display, and local e-commerce requirements.",
            ],
            "risk_terms": [
                {"term": "alcohol", "message": "Remove alcohol imagery or references for Saudi Arabia."},
                {"term": "dating", "message": "Avoid dating or romance framing for this market."},
                {"term": "pork", "message": "Remove pork imagery or references for this market."},
            ],
        },
        "china": {
            "style": "Benefit-forward product clarity with trust badges and mobile commerce context",
            "colors": "Clean white base with restrained red or gold promotional accents",
            "model_guidance": "Use local shopping context and clear product-in-use framing.",
            "background": "Mobile-first product detail page scene, home use, clean product close-ups",
            "platforms": ["Douyin", "WeChat", "Xiaohongshu", "Tmall"],
            "cultural_notes": [
                "Avoid unsupported official authorization or certification claims.",
                "Keep mobile product detail hierarchy clear.",
            ],
            "regulatory_notes": [
                "Verify local platform rules, consumer disclosures, and claim substantiation.",
            ],
            "risk_terms": [
                {"term": "officially certified", "message": "Verify certification before using this claim."},
                {"term": "medical", "message": "Avoid medical claims unless legally substantiated."},
            ],
        },
    }
    return profiles.get(
        key,
        {
            "style": "Locally adapted premium e-commerce photography with clear product detail",
            "colors": "Balanced brand colors with local seasonal accent options",
            "model_guidance": "Use inclusive, respectful lifestyle context and avoid stereotypes.",
            "background": "Clean lifestyle setting that matches local usage occasions",
            "platforms": ["Instagram", "TikTok", "YouTube Shorts"],
            "cultural_notes": [
                "Avoid stereotypes and unsupported cultural claims.",
                "Validate final copy with an in-market reviewer.",
            ],
            "regulatory_notes": [
                "Confirm local consumer, advertising, and platform policy requirements.",
            ],
            "risk_terms": [
                {"term": "guaranteed", "message": "Avoid absolute guarantees without proof."},
                {"term": "#1", "message": "Avoid ranking claims without substantiation."},
            ],
        },
    )


def _video_script(product: str, target_market: str, brand_voice: str) -> list[dict[str, Any]]:
    profile = _market_profile(target_market)
    return [
        {
            "scene_number": 1,
            "duration_sec": 3,
            "visual_description": f"Close-up hero shot of {product} in the localized scene.",
            "voiceover_script": "Introduce the primary customer need in a calm, locally natural tone.",
            "on_screen_text": "Core benefit",
            "background_music_mood": "Subtle, premium, and regionally appropriate.",
            "cultural_adaptation_note": profile["cultural_notes"][0],
        },
        {
            "scene_number": 2,
            "duration_sec": 5,
            "visual_description": "Show product-in-use with practical detail and no exaggerated claims.",
            "voiceover_script": f"Connect the product benefit to {brand_voice} brand positioning.",
            "on_screen_text": "Useful detail",
            "background_music_mood": "Warm but restrained.",
            "cultural_adaptation_note": profile["cultural_notes"][1],
        },
        {
            "scene_number": 3,
            "duration_sec": 4,
            "visual_description": "End on product packshot with localized CTA and support cue.",
            "voiceover_script": "Invite the customer to learn more or shop with confidence.",
            "on_screen_text": "Shop now",
            "background_music_mood": "Clean resolution with light uplift.",
            "cultural_adaptation_note": "Keep CTA compliant with local platform and ad policies.",
        },
    ]


def _engine_strategy(
    engine: str,
    product: str,
    target_market: str,
    keywords: list[str],
) -> dict[str, Any]:
    strategy_map = {
        "Google": {
            "requirements": ["Product JSON-LD", "FAQPage opportunities", "E-E-A-T proof points"],
            "tone": "Authoritative, helpful, and evidence-aware.",
            "boosts": ["Localize H1 and first paragraph", "Add expert or support proof where verified"],
        },
        "Baidu": {
            "requirements": ["Mobile speed", "Exact keyword placement", "Chinese tokenization review"],
            "tone": "Direct, trust-building, and benefit-forward.",
            "boosts": ["Local customer service signals", "Platform-compatible metadata"],
        },
        "Yandex": {
            "requirements": ["Geo relevance", "Dwell-time friendly intro", "Clear metadata"],
            "tone": "Practical and regionally contextual.",
            "boosts": ["City or region terms where relevant", "Low-bounce content structure"],
        },
        "Naver": {
            "requirements": ["Blog-style depth", "Community Q&A angle", "Open Graph tags"],
            "tone": "Experiential, detailed, and conversational.",
            "boosts": ["KakaoTalk sharing CTA", "Community discussion prompts"],
        },
        "Yahoo_Japan": {
            "requirements": ["Yahoo Shopping compatibility", "Mobile-first metadata", "Support detail"],
            "tone": "Polite, detailed, and trust-building.",
            "boosts": ["Seasonal shopping calendar", "LINE account integration cue"],
        },
        "Google_Saudi": {
            "requirements": ["RTL-ready HTML", "Arabic alt text review", "Product and Offer schema"],
            "tone": "Warm, family-oriented, and respectful.",
            "boosts": ["VAT display reminder", "Local delivery and support cues"],
        },
    }
    strategy = strategy_map[engine]
    return {
        "engine": engine,
        "title_template": f"{product} | {keywords[0]} for {target_market}",
        "meta_description_template": (
            f"Explore {product} with {keywords[0]} benefits for {target_market}. "
            "Verify final claims before publication."
        ),
        "keyword_focus": keywords,
        "structural_requirements": strategy["requirements"],
        "content_tone_guidance": strategy["tone"],
        "regional_boost_factors": strategy["boosts"],
    }


def _clean_keywords(primary_keywords: list[str] | None, product: str) -> list[str]:
    keywords = [
        keyword.strip()
        for keyword in (primary_keywords or [])
        if isinstance(keyword, str) and keyword.strip()
    ]
    if not keywords:
        keywords = [f"{product} benefits", f"{product} buying guide", f"{product} review"]
    return keywords[:8]


def _market_key(target_market: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", target_market.strip().lower()).strip("_")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "content-asset"


def _currency_for_market(target_market: str) -> str:
    market = _market_key(target_market)
    if "japan" in market:
        return "JPY"
    if "saudi" in market:
        return "SAR"
    if "uk" in market or "united_kingdom" in market:
        return "GBP"
    if "germany" in market or "france" in market or "eu" in market:
        return "EUR"
    return "USD"


def _visual_text_language(language: str) -> str:
    key = _market_key(language)
    language_map = {
        "ja": "Japanese",
        "jp": "Japanese",
        "ja_jp": "Japanese",
        "japanese": "Japanese",
        "zh": "Simplified Chinese suitable for Singapore",
        "zh_cn": "Simplified Chinese suitable for Singapore",
        "zh_hans": "Simplified Chinese suitable for Singapore",
        "zh_sg": "Simplified Chinese suitable for Singapore",
        "zh_hans_sg": "Simplified Chinese suitable for Singapore",
        "chinese": "Simplified Chinese suitable for Singapore",
        "en": "English",
        "en_us": "English",
        "en_gb": "English",
        "en_au": "English",
        "english": "English",
    }
    return language_map.get(key, language.strip() or "the target language")


def _score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 100.0))


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
    return str(exc)
