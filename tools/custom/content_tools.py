import base64
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
