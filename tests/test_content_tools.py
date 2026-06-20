import base64
import contextvars
import json
import os
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from admin_dashboard import (
    ARTIFACT_ROOT,
    _content_inputs_from_form_values,
    _content_live_preview_groups,
    _content_result_payload,
    _content_timeline_entries,
    _content_reddit_geo_review_assets,
    _normalized_visual_score,
    _reddit_geo_display_sections,
    _safe_artifact_path,
    _should_show_content_image_placeholder,
)
from crews.content_crew import (
    CONTENT_ARTICLE_MAX_CHARS,
    CONTENT_GENERATION_MAX_TOKENS,
    CONTENT_SEO_KEYWORDS_MAX_COUNT,
    PerLanguageContentOutput,
    _apply_reddit_geo_context,
    _content_generation_llm_context,
    _enrich_language_output,
    _error_summary,
    _normalize_inputs,
    _normalize_per_language_content_payload,
    run_content_crew,
)
from job_store import InMemoryJobStore
from models import ContentInputs, WorkflowType
from tools.custom.content_tools import (
    MultimodalLocalizationTool,
    OpenAIImageGenerationTool,
    RedditGeoSearchTool,
    VisualAssetScoringTool,
)
from utils.observability import workflow_span
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, WorkflowProgressRecorder


class ContentTraceObservation:
    def __init__(
        self,
        name: str,
        as_type: str,
        metadata: dict[str, object] | None,
        parent_name: str | None,
    ) -> None:
        self.name = name
        self.as_type = as_type
        self.metadata = metadata or {}
        self.parent_name = parent_name
        self.ended = False


class ContentTraceContext:
    def __init__(self, client: "ContentTraceLangfuseClient", observation: ContentTraceObservation) -> None:
        self.client = client
        self.observation = observation
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> ContentTraceObservation:
        self._token = self.client.current_observation.set(self.observation.name)
        return self.observation

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.observation.ended = True
        if self._token is not None:
            self.client.current_observation.reset(self._token)


class ContentTraceLangfuseClient:
    def __init__(self) -> None:
        self.current_observation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "content_trace_current_observation",
            default=None,
        )
        self.observations: list[ContentTraceObservation] = []
        self.current_span_updates: list[dict[str, object]] = []

    def start_as_current_observation(self, **kwargs: object) -> ContentTraceContext:
        observation = ContentTraceObservation(
            name=str(kwargs["name"]),
            as_type=str(kwargs.get("as_type") or "span"),
            metadata=kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None,
            parent_name=self.current_observation.get(),
        )
        self.observations.append(observation)
        return ContentTraceContext(self, observation)

    def update_current_span(self, **kwargs: object) -> None:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            self.current_span_updates.append(
                {
                    "name": self.current_observation.get(),
                    "metadata": metadata,
                }
            )


def localized_entities(
    language: str = "ja",
    target_market: str = "Japan",
    subject: str = "Smart Thermos",
    product_category: str = "Drinkware",
    brand_name: str = "ThermoCo",
    brand_voice: str = "Premium",
    primary_keywords: list[str] | None = None,
) -> dict[str, object]:
    return {
        "language": language,
        "target_market": target_market,
        "subject": subject,
        "product_category": product_category,
        "brand_name": brand_name,
        "brand_voice": brand_voice,
        "primary_keywords": primary_keywords or ["smart thermos"],
    }


class ContentToolTests(unittest.TestCase):
    def test_reddit_geo_search_without_serper_key_returns_fallback(self) -> None:
        tool = RedditGeoSearchTool(serper_api_key=None)

        result = tool._run(
            subject="Smart Thermos",
            product_category="Drinkware",
            target_markets="Japan",
            primary_keywords=["insulated bottle"],
            brand_name="ThermoCo",
        )

        self.assertEqual(result["status"], "skipped_missing_credentials")
        self.assertEqual(result["data_source"], "serper_unavailable")
        self.assertEqual(result["confidence_level"], "low")
        self.assertEqual(result["sources"], [])
        self.assertGreaterEqual(len(result["query_pack"]), 1)

    def test_reddit_geo_search_extracts_subreddit_sources(self) -> None:
        request = httpx.Request("POST", "https://google.serper.dev/search")
        response = httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Best thermos for winter hikes?",
                        "link": "https://www.reddit.com/r/BuyItForLife/comments/abc123/best_thermos/",
                        "snippet": "People discuss durable insulated bottles for cold weather.",
                    },
                    {
                        "title": "Best thermos for winter hikes?",
                        "link": "https://www.reddit.com/r/BuyItForLife/comments/abc123/best_thermos/?utm_source=x",
                        "snippet": "Duplicate URL should be removed.",
                    },
                ]
            },
            request=request,
        )
        tool = RedditGeoSearchTool(serper_api_key="serper-test-key")

        with patch("tools.custom.content_tools.httpx.post", return_value=response) as post:
            result = tool._run(
                subject="Smart Thermos",
                product_category="Drinkware",
                target_markets="Japan",
                primary_keywords=["insulated bottle"],
                brand_name="ThermoCo",
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["status"], "live_search")
        self.assertEqual(result["data_source"], "serper_search")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["source_id"], "R1")
        self.assertEqual(result["sources"][0]["subreddit"], "BuyItForLife")

    def test_reddit_geo_search_filters_non_target_translation_urls(self) -> None:
        request = httpx.Request("POST", "https://google.serper.dev/search")
        response = httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Translated Reddit page",
                        "link": "https://www.reddit.com/r/soccer/comments/abc123/cards/?tl=zh-hans",
                        "snippet": "这是中文翻译内容，不应该进入德语GEO上下文。",
                    },
                    {
                        "title": "Soccer card collectors in Germany",
                        "link": "https://www.reddit.com/r/SoccerCards/comments/def456/germany_cards/?utm_source=search",
                        "snippet": "Collectors discuss German market availability and card grading.",
                    },
                ]
            },
            request=request,
        )
        tool = RedditGeoSearchTool(serper_api_key="serper-test-key")

        with patch("tools.custom.content_tools.httpx.post", return_value=response):
            result = tool._run(
                subject="World Cup Soccer Cards",
                product_category="Sports Collectibles",
                target_markets="Germany",
                target_languages=["de"],
            )

        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["target_language"], "de")
        self.assertEqual(result["sources"][0]["subreddit"], "SoccerCards")
        self.assertNotIn("tl=zh", result["sources"][0]["url"])
        self.assertNotIn("中文", result["sources"][0]["snippet"])

    def test_reddit_geo_search_retries_retryable_status(self) -> None:
        request = httpx.Request("POST", "https://google.serper.dev/search")
        retry_response = httpx.Response(520, text="temporary error", request=request)
        ok_response = httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Thermos discussion",
                        "link": "https://www.reddit.com/r/CampingGear/comments/abc123/thermos/",
                        "snippet": "Campers compare insulated bottles.",
                    }
                ]
            },
            request=request,
        )
        tool = RedditGeoSearchTool(serper_api_key="serper-test-key")

        with patch(
            "tools.custom.content_tools.httpx.post",
            side_effect=[retry_response, ok_response, ok_response],
        ) as post:
            result = tool._run(
                subject="Smart Thermos",
                product_category="Drinkware",
                target_markets="Japan",
            )

        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["status"], "live_search")
        self.assertEqual(result["sources"][0]["subreddit"], "CampingGear")

    def test_reddit_geo_search_provider_error_returns_fallback(self) -> None:
        tool = RedditGeoSearchTool(serper_api_key="serper-test-key")

        with patch(
            "tools.custom.content_tools.httpx.post",
            side_effect=httpx.ConnectError("network unavailable"),
        ) as post:
            result = tool._run(
                subject="Smart Thermos",
                product_category="Drinkware",
                target_markets="Japan",
            )

        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["status"], "fallback_after_provider_error")
        self.assertEqual(result["data_source"], "serper_error")
        self.assertEqual(result["confidence_level"], "low")
        self.assertIn("network unavailable", result["provider_error"])

    def test_image_generation_without_api_key_is_skipped(self) -> None:
        tool = OpenAIImageGenerationTool(openai_api_key=None)

        result = tool._run(prompt="Product photo prompt")

        self.assertEqual(result["status"], "skipped_missing_credentials")
        self.assertEqual(result["assets"], [])
        self.assertIn("OPENAI_API_KEY", result["error"])

    def test_image_generation_retries_520_and_saves_asset(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
        image_payload = base64.b64encode(b"fake image bytes").decode("ascii")
        responses = [
            httpx.Response(520, text="temporary Cloudflare error", request=request),
            httpx.Response(200, json={"data": [{"b64_json": image_payload}]}, request=request),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            tool = OpenAIImageGenerationTool(
                openai_api_key="test-key",
                artifact_dir=temp_dir,
            )
            with patch("tools.custom.content_tools.httpx.post", side_effect=responses) as post:
                result = tool._run(prompt="Product photo prompt", output_slug="test-image")

            self.assertEqual(post.call_count, 2)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["last_status_code"], 200)
            self.assertFalse(result["retryable_status"])
            self.assertGreaterEqual(result["duration_seconds"], 0)
            self.assertEqual(len(result["assets"]), 1)
            saved_path = Path(result["assets"][0]["asset_path"])
            self.assertTrue(saved_path.is_file())
            self.assertEqual(saved_path.read_bytes(), b"fake image bytes")
            self.assertGreaterEqual(result["assets"][0]["duration_seconds"], 0)

    def test_image_generation_moderation_block_uses_safe_fallback_prompt(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
        image_payload = base64.b64encode(b"safe fallback image bytes").decode("ascii")
        responses = [
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Your request was rejected by the safety system.",
                        "type": "image_generation_user_error",
                        "code": "moderation_blocked",
                    }
                },
                request=request,
            ),
            httpx.Response(200, json={"data": [{"b64_json": image_payload}]}, request=request),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            tool = OpenAIImageGenerationTool(
                openai_api_key="test-key",
                artifact_dir=temp_dir,
            )
            with patch("tools.custom.content_tools.httpx.post", side_effect=responses) as post:
                result = tool._run(prompt="World Cup athlete investment card prompt", output_slug="safe-image")

            self.assertEqual(post.call_count, 2)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["last_status_code"], 200)
            self.assertIn("generic sports collectible", result["prompt"])
            self.assertNotIn("World Cup athlete", result["prompt"])
            saved_path = Path(result["assets"][0]["asset_path"])
            self.assertEqual(saved_path.read_bytes(), b"safe fallback image bytes")

    def test_image_generation_moderation_block_reports_failed_fallback(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
        blocked_response = httpx.Response(
            400,
            json={
                "error": {
                    "message": "Your request was rejected by the safety system.",
                    "type": "image_generation_user_error",
                    "code": "moderation_blocked",
                }
            },
            request=request,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            tool = OpenAIImageGenerationTool(
                openai_api_key="test-key",
                artifact_dir=temp_dir,
            )
            with patch(
                "tools.custom.content_tools.httpx.post",
                side_effect=[blocked_response, blocked_response],
            ) as post:
                result = tool._run(prompt="World Cup athlete investment card prompt")

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["last_status_code"], 400)
        self.assertIn("safe fallback prompt also failed", result["error"])

    def test_image_generation_reports_520_after_retry_exhaustion(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
        responses = [
            httpx.Response(520, text="temporary Cloudflare error", request=request)
            for _ in range(3)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            tool = OpenAIImageGenerationTool(
                openai_api_key="test-key",
                artifact_dir=temp_dir,
            )
            with patch("tools.custom.content_tools.httpx.post", side_effect=responses) as post:
                result = tool._run(prompt="Product photo prompt", output_slug="test-image")

        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["last_status_code"], 520)
        self.assertTrue(result["retryable_status"])
        self.assertIn("HTTP 520", result["error"])

    def test_multimodal_prompt_requires_singapore_simplified_chinese_text(self) -> None:
        result = MultimodalLocalizationTool()._run(
            product="Sustainable Activewear",
            target_market="Singapore",
            language="zh-SG",
        )

        prompt = result["visual_spec"]["ai_image_prompt"]

        self.assertIn("only Simplified Chinese suitable for Singapore", prompt)
        self.assertIn("do not mix another language", prompt)

    def test_multimodal_prompt_requires_japanese_text(self) -> None:
        result = MultimodalLocalizationTool()._run(
            product="Sustainable Activewear",
            target_market="Japan",
            language="ja",
        )

        prompt = result["visual_spec"]["ai_image_prompt"]

        self.assertIn("only Japanese", prompt)
        self.assertIn("do not mix another language", prompt)

    def test_visual_asset_scoring_parses_response_output_text(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(
            200,
            json={
                "output_text": (
                    "```json\n"
                    '{"prompt_alignment_score": 92, "cultural_fit_score": 88, '
                    '"brand_voice_score": 90, "publish_readiness_score": 86, '
                    '"notes": "Ready after minor crop review."}'
                    "\n```"
                )
            },
            request=request,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            asset_path = Path(temp_dir) / "asset.png"
            asset_path.write_bytes(b"fake image bytes")
            tool = VisualAssetScoringTool(openai_api_key="test-key")
            with patch("tools.custom.content_tools.httpx.post", return_value=response):
                result = tool._run(
                    asset_path=str(asset_path),
                    prompt="Product photo prompt",
                    target_market="Japan",
                    language="ja",
                    brand_voice="Premium",
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["prompt_alignment_score"], 92.0)
        self.assertEqual(result["cultural_fit_score"], 88.0)
        self.assertGreaterEqual(result["duration_seconds"], 0)
        self.assertIn("minor crop", result["notes"])

    def test_content_image_generation_lock_serializes_parallel_enrichment(self) -> None:
        output = {
            "localized_article": {
                "language": "ja",
                "title": "Title",
                "article": "Article body",
            },
            "social_media_posts": [],
            "seo_keywords": ["keyword"],
            "compliance_notes": "Reviewed.",
        }
        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan, Australia",
            "target_languages": ["ja", "en"],
            "platforms": ["Instagram"],
            "generate_visual_assets": True,
            "image_generation_count": 1,
            "image_quality": "auto",
            "image_size": "1024x1024",
        }
        image_generation_lock = Lock()
        state_lock = Lock()
        active_calls = 0
        max_active_calls = 0

        def fake_image_generation_run(
            tool: OpenAIImageGenerationTool,
            prompt: str,
            output_slug: str = "content-visual",
            image_generation_count: int = 1,
            image_quality: str = "auto",
            image_size: str = "1024x1024",
        ) -> dict[str, object]:
            nonlocal active_calls, max_active_calls
            with state_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            time.sleep(0.02)
            with state_lock:
                active_calls -= 1
            return {
                "status": "completed_no_local_asset",
                "model": tool.image_model,
                "prompt": prompt,
                "assets": [],
                "duration_seconds": 0.02,
                "attempts": 1,
                "last_status_code": 200,
                "retryable_status": False,
                "error": None,
            }

        with patch.object(OpenAIImageGenerationTool, "_run", fake_image_generation_run):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        _enrich_language_output,
                        dict(output),
                        inputs,
                        "ja",
                        "Japan",
                        {"content_image_generation_lock": image_generation_lock},
                    ),
                    executor.submit(
                        _enrich_language_output,
                        dict(output),
                        inputs,
                        "en",
                        "Australia",
                        {"content_image_generation_lock": image_generation_lock},
                    ),
                ]
                for future in futures:
                    future.result()

        self.assertEqual(max_active_calls, 1)

    def test_dashboard_maps_container_artifact_path_to_local_artifacts(self) -> None:
        artifact_dir = ARTIFACT_ROOT / "content_creation"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        asset_path = artifact_dir / f"test-{uuid.uuid4().hex}.png"
        asset_path.write_bytes(b"fake image bytes")

        try:
            resolved_path, diagnostics = _safe_artifact_path(
                f"/app/artifacts/content_creation/{asset_path.name}"
            )
        finally:
            asset_path.unlink(missing_ok=True)

        self.assertEqual(resolved_path, asset_path.resolve())
        self.assertIn("mapped_from_container", "\n".join(diagnostics))

    def test_dashboard_normalizes_fenced_visual_score_notes(self) -> None:
        score = {
            "status": "completed",
            "publish_readiness_score": 0,
            "notes": (
                "```json\n"
                '{"prompt_alignment_score": 90, "cultural_fit_score": 85, '
                '"brand_voice_score": 90, "publish_readiness_score": 88, '
                '"notes": "Ready for publishing."}'
                "\n```"
            ),
        }

        normalized = _normalized_visual_score(score)

        self.assertEqual(normalized["prompt_alignment_score"], 90.0)
        self.assertEqual(normalized["cultural_fit_score"], 85.0)
        self.assertEqual(normalized["brand_voice_score"], 90.0)
        self.assertEqual(normalized["publish_readiness_score"], 88.0)
        self.assertEqual(normalized["notes"], "Ready for publishing.")

    def test_content_form_values_build_valid_payload_lists(self) -> None:
        payload = _content_inputs_from_form_values(
            subject=" Sustainable Activewear ",
            product_category="Winter Sportswear",
            product_features="Recycled shell",
            target_markets="Germany, Japan, Canada",
            target_languages="de, ja, en",
            platforms="Instagram, LinkedIn, X",
            brand_voice="Premium",
            brand_name="NorthPeak Layers",
            product_url="https://example.com/products/sustainable-activewear",
            primary_keywords="thermal activewear, recycled sportswear",
            generate_reddit_geo=True,
            generate_visual_assets=True,
            image_generation_count=2,
            image_quality="high",
            image_size="1024x1024",
        )

        self.assertEqual(payload["subject"], "Sustainable Activewear")
        self.assertEqual(payload["target_languages"], ["de", "ja", "en"])
        self.assertEqual(payload["platforms"], ["Instagram", "LinkedIn", "X"])
        self.assertEqual(
            payload["primary_keywords"],
            ["thermal activewear", "recycled sportswear"],
        )
        self.assertEqual(payload["brand_name"], "NorthPeak Layers")
        self.assertEqual(payload["product_url"], "https://example.com/products/sustainable-activewear")
        self.assertTrue(payload["generate_reddit_geo"])
        self.assertTrue(payload["generate_visual_assets"])
        self.assertEqual(payload["image_generation_count"], 2)
        self.assertEqual(ContentInputs.model_validate(payload).image_quality, "high")

        payload_without_quality = dict(payload)
        payload_without_quality.pop("image_quality")
        self.assertEqual(
            ContentInputs.model_validate(payload_without_quality).image_quality,
            "low",
        )

    def test_content_inputs_accept_reddit_geo_and_reject_invalid_product_url(self) -> None:
        payload = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan",
            "target_languages": ["ja"],
            "platforms": ["Reddit"],
            "brand_name": "ThermoCo",
            "product_url": "https://example.com/products/smart-thermos",
            "generate_reddit_geo": True,
        }

        validated = ContentInputs.model_validate(payload)

        self.assertTrue(validated.generate_reddit_geo)
        self.assertEqual(validated.product_url, "https://example.com/products/smart-thermos")

        invalid_payload = dict(payload)
        invalid_payload["product_url"] = "ftp://example.com/products/smart-thermos"
        with self.assertRaises(ValidationError):
            ContentInputs.model_validate(invalid_payload)

        extra_payload = dict(payload)
        extra_payload["reddit_api_token"] = "secret"
        with self.assertRaises(ValidationError):
            ContentInputs.model_validate(extra_payload)

    def test_content_result_payload_detects_reddit_geo_only_result(self) -> None:
        latest_job = {
            "result": {
                "reddit_geo_posts": [],
                "reddit_geo_sources": [],
            }
        }

        self.assertIs(_content_result_payload(latest_job), latest_job["result"])

    def test_reddit_geo_display_sections_hide_translated_sources(self) -> None:
        content_result = {
            "reddit_geo_posts": [
                {
                    "language": "de",
                    "target_market": "Germany",
                    "recommended_subreddit": "r/SoccerCards",
                    "title_options": ["Welche Karten sind sinnvoll?"],
                    "body": "Disclosure: Wir sind mit der Produktseite verbunden.",
                    "body_without_link": "Disclosure: Wir sind mit der Produktseite verbunden.",
                    "disclosure_note": "Disclosure: Wir sind mit der Produktseite verbunden.",
                    "ai_search_entity_signals": ["World Cup Soccer Cards", "Germany"],
                    "source_ids": ["R1", "R2"],
                    "moderation_notes": ["Regeln vor dem Posten prüfen."],
                    "data_source": "serper_search",
                    "confidence_level": "medium",
                }
            ],
            "reddit_geo_sources": [
                {
                    "source_id": "R1",
                    "target_market": "Germany",
                    "target_language": "de",
                    "subreddit": "SoccerCards",
                    "title": "German collectors",
                    "snippet": "Collectors discuss grading and availability.",
                    "url": "https://www.reddit.com/r/SoccerCards/comments/abc/cards/",
                    "query_type": "reddit_need_discussion",
                    "query": "site:reddit.com/r cards",
                    "data_source": "serper_search",
                    "confidence_level": "medium",
                },
                {
                    "source_id": "R2",
                    "target_market": "Germany",
                    "target_language": "de",
                    "subreddit": "soccer",
                    "title": "Translated page",
                    "snippet": "这是中文翻译内容。",
                    "url": "https://www.reddit.com/r/soccer/comments/abc/cards/?tl=zh-hans",
                    "query_type": "reddit_need_discussion",
                    "query": "site:reddit.com/r cards",
                    "data_source": "serper_search",
                    "confidence_level": "medium",
                },
            ],
        }

        sections = _reddit_geo_display_sections(content_result)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["recommended_subreddit"], "r/SoccerCards")
        self.assertEqual([source["source_id"] for source in sections[0]["sources"]], ["R1"])

    def test_reddit_geo_review_assets_include_production_ready_assets(self) -> None:
        latest_job = {
            "result": {
                "reddit_geo_posts": [
                    {
                        "language": "en",
                        "target_market": "United States",
                        "recommended_subreddit": "r/SoccerCards",
                        "title_options": ["How do you compare card sleeves?"],
                        "body": "Disclosure: connected to the product page.",
                        "body_without_link": "Disclosure: connected to the product page.",
                        "disclosure_note": "Disclosure: connected to the product page.",
                        "ai_search_entity_signals": ["Soccer card sleeves"],
                        "source_ids": ["R1"],
                        "moderation_notes": ["Review subreddit rules first."],
                        "data_source": "serper_search",
                        "confidence_level": "medium",
                    }
                ],
                "reddit_geo_sources": [
                    {
                        "source_id": "R1",
                        "target_market": "United States",
                        "target_language": "en",
                        "subreddit": "SoccerCards",
                        "title": "Sleeve recommendations",
                        "snippet": "Collectors compare sleeve quality.",
                        "url": "https://www.reddit.com/r/SoccerCards/comments/abc/sleeves/",
                        "query_type": "reddit_need_discussion",
                        "query": "site:reddit.com/r/SoccerCards sleeves",
                        "data_source": "serper_search",
                        "confidence_level": "medium",
                    }
                ],
                "production_ready_assets": [
                    {
                        "asset_type": "reddit_geo_post",
                        "language": "en",
                        "target_market": "United States",
                        "recommended_subreddit": "r/SoccerCards",
                        "status": "ready_for_manual_reddit_review",
                    }
                ],
            }
        }

        review_assets = _content_reddit_geo_review_assets(latest_job)

        self.assertEqual(len(review_assets), 1)
        self.assertEqual(review_assets[0]["status"], "ready_for_manual_reddit_review")
        self.assertEqual(review_assets[0]["source_ids"], ["R1"])
        self.assertEqual(review_assets[0]["title_options"], ["How do you compare card sleeves?"])

    def test_content_generation_defaults_to_bounded_llm_context(self) -> None:
        context = _content_generation_llm_context({})

        self.assertEqual(context["llm_max_tokens"], CONTENT_GENERATION_MAX_TOKENS)
        self.assertEqual(
            _content_generation_llm_context({"content_generation_max_tokens": 2048})["llm_max_tokens"],
            2048,
        )
        self.assertEqual(
            _content_generation_llm_context({"llm_max_tokens": 8192})["llm_max_tokens"],
            8192,
        )

        normalized = _normalize_inputs(
            {
                "subject": "Sustainable Activewear",
                "product_category": "Sportswear",
                "target_markets": "Japan",
                "target_languages": ["ja"],
                "platforms": ["Instagram"],
            }
        )

        self.assertEqual(normalized["content_article_max_chars"], CONTENT_ARTICLE_MAX_CHARS)
        self.assertEqual(normalized["content_seo_keywords_max_count"], CONTENT_SEO_KEYWORDS_MAX_COUNT)

    def test_content_task_prompt_contains_length_constraints(self) -> None:
        prompt = Path("config/content/tasks.yaml").read_text(encoding="utf-8")

        self.assertIn("{content_article_max_chars}", prompt)
        self.assertIn("{content_social_post_max_chars}", prompt)
        self.assertIn("{content_seo_keywords_max_count}", prompt)
        self.assertIn("Return at most one social post per requested", prompt)
        self.assertIn("All reddit_geo_posts fields that contain prose", prompt)
        self.assertIn("must be written in", prompt)
        self.assertIn("localized_entities", prompt)
        self.assertIn("Do not preserve source-language brand names", prompt)
        self.assertIn("Return compliance_notes as a top-level field", prompt)
        self.assertIn("localized_article must contain only language, title, and article", prompt)
        self.assertIn("never return null for list fields", prompt)
        self.assertIn("Never output placeholder links", prompt)

    def test_per_language_content_output_requires_localized_entities(self) -> None:
        with self.assertRaises(ValidationError):
            PerLanguageContentOutput.model_validate(
                {
                    "localized_article": {
                        "language": "de",
                        "title": "Titel",
                        "article": "Artikel",
                    },
                    "social_media_posts": [],
                    "seo_keywords": ["Bergsteigen"],
                    "compliance_notes": "Geprüft.",
                }
            )

    def test_per_language_payload_normalizes_misnested_compliance_notes(self) -> None:
        payload = {
            "localized_article": {
                "language": "de",
                "title": "Titel",
                "article": "Artikel",
                "compliance_notes": "Reviewed.",
            },
            "social_media_posts": [],
            "seo_keywords": ["Bergsteigen"],
            "localized_entities": localized_entities(
                language="de",
                target_market="Germany",
                subject="Bergsteiger-Set",
                product_category="Alpine Ausrustung",
                brand_name="Alpen-Set",
                brand_voice="Praktisch",
                primary_keywords=["Bergsteigen"],
            ),
        }

        normalized = _normalize_per_language_content_payload(payload)
        result = PerLanguageContentOutput.model_validate(normalized)

        self.assertEqual(result.compliance_notes, "Reviewed.")
        self.assertNotIn("compliance_notes", normalized["localized_article"])

    def test_per_language_payload_normalizes_nullable_optional_lists(self) -> None:
        payload = {
            "localized_article": {
                "language": "de",
                "title": "Titel",
                "article": "Artikel",
            },
            "social_media_posts": [],
            "seo_keywords": ["Bergsteigen"],
            "compliance_notes": "Reviewed.",
            "localized_entities": localized_entities(
                language="de",
                target_market="Germany",
                subject="Bergsteiger-Set",
                product_category="Alpine Ausrustung",
                brand_name="Alpen-Set",
                brand_voice="Praktisch",
                primary_keywords=["Bergsteigen"],
            ),
            "visual_assets": None,
            "visual_asset_scores": None,
            "reddit_geo_posts": None,
        }

        normalized = _normalize_per_language_content_payload(payload)
        result = PerLanguageContentOutput.model_validate(normalized)

        self.assertEqual(result.visual_assets, [])
        self.assertEqual(result.visual_asset_scores, [])
        self.assertEqual(result.reddit_geo_posts, [])

    def test_per_language_payload_normalization_keeps_unknown_article_extras_strict(self) -> None:
        payload = {
            "localized_article": {
                "language": "de",
                "title": "Titel",
                "article": "Artikel",
                "compliance_notes": "Reviewed.",
                "foo": "bar",
            },
            "social_media_posts": [],
            "seo_keywords": ["Bergsteigen"],
            "localized_entities": localized_entities(
                language="de",
                target_market="Germany",
                subject="Bergsteiger-Set",
                product_category="Alpine Ausrustung",
                brand_name="Alpen-Set",
                brand_voice="Praktisch",
                primary_keywords=["Bergsteigen"],
            ),
        }

        normalized = _normalize_per_language_content_payload(payload)

        with self.assertRaises(ValidationError):
            PerLanguageContentOutput.model_validate(normalized)

    def test_content_enrichment_uses_localized_entities_for_seo_and_visuals(self) -> None:
        output = {
            "localized_article": {
                "language": "de",
                "title": "Bergsteiger-Set für kalte Höhenlagen",
                "article": "Ein praktischer Leitfaden für alpine Touren.",
            },
            "social_media_posts": [],
            "seo_keywords": ["Bergsteigen", "Kälteschutz"],
            "compliance_notes": "Geprüft.",
            "localized_entities": localized_entities(
                language="de",
                target_market="Germany",
                subject="Bergsteiger-Set für kalte Höhenlagen",
                product_category="Alpine Ausrüstung",
                brand_name="Alpen-Set",
                brand_voice="Praktisch und vertrauenswürdig",
                primary_keywords=["Bergsteigen", "Kälteschutz"],
            ),
        }
        inputs = {
            "subject": "寒带登山套装",
            "product_category": "登山套装",
            "target_markets": "Germany",
            "target_languages": ["de"],
            "platforms": ["Instagram"],
            "brand_voice": "寒带地区登山",
            "brand_name": "登山套装",
            "primary_keywords": ["登山"],
            "generate_visual_assets": False,
        }

        result = _enrich_language_output(output, inputs, "de", "Germany", {})

        serialized = json.dumps(
            {
                "seo_metadata": result["seo_metadata"],
                "multimodal_output": result["multimodal_output"],
            },
            ensure_ascii=False,
        )
        for source_term in ("寒带登山套装", "登山套装", "登山"):
            self.assertNotIn(source_term, serialized)
        self.assertIn("Bergsteiger-Set", serialized)
        self.assertIn("Bergsteigen", serialized)

    def test_content_enrichment_filters_source_keywords_from_localized_entities(self) -> None:
        source_subject = "\u5bd2\u5e26\u767b\u5c71\u5957\u88c5"
        source_category = "\u767b\u5c71\u5957\u88c5"
        source_voice = "\u5bd2\u5e26\u5730\u533a\u767b\u5c71"
        source_keyword = "\u767b\u5c71"
        output = {
            "localized_article": {
                "language": "de",
                "title": "Ausrustung fur Bergsteiger",
                "article": "Ein Leitfaden fur kalte Hohenlagen.",
            },
            "social_media_posts": [],
            "seo_keywords": ["Bergsteigen"],
            "compliance_notes": "Reviewed.",
            "localized_entities": localized_entities(
                language="de",
                target_market="Germany",
                subject="Ausrustung fur Bergsteiger",
                product_category="Winter-Bergsteigausrustung",
                brand_name="",
                brand_voice="Ausrustungen fur kalte Regionen",
                primary_keywords=[
                    source_keyword,
                    source_category,
                    "Bergsteigen",
                    "Hochgebirgsausrustung",
                ],
            ),
        }
        inputs = {
            "subject": source_subject,
            "product_category": source_category,
            "target_markets": "Germany",
            "target_languages": ["de"],
            "platforms": ["Instagram"],
            "brand_voice": source_voice,
            "brand_name": source_category,
            "primary_keywords": [source_keyword],
            "generate_visual_assets": False,
        }

        result = _enrich_language_output(output, inputs, "de", "Germany", {})

        serialized = json.dumps(
            {
                "seo_metadata": result["seo_metadata"],
                "multimodal_output": result["multimodal_output"],
            },
            ensure_ascii=False,
        )
        for source_term in (source_subject, source_category, source_voice, source_keyword):
            self.assertNotIn(source_term, serialized)
        self.assertIn("Bergsteigen", serialized)

    def test_reddit_geo_draft_sanitization_replaces_source_terms(self) -> None:
        source_subject = "\u5bd2\u5e26\u767b\u5c71\u5957\u88c5"
        source_category = "\u767b\u5c71\u5957\u88c5"
        source_voice = "\u5bd2\u5e26\u5730\u533a\u767b\u5c71"
        source_keyword = "\u767b\u5c71"
        result = {
            "localized_articles": [
                {
                    "language": "de",
                    "title": "Ausrustung fur Bergsteiger",
                    "article": "Artikel uber alpine Ausrustung.",
                }
            ],
            "social_media_posts": [],
            "seo_outputs": [],
            "visual_assets": [],
            "reddit_geo_posts": [
                {
                    "target_market": "Germany",
                    "language": "de",
                    "recommended_subreddit": "r/alpinism",
                    "title_options": [f"Ist {source_category} fur kalte Touren sinnvoll?"],
                    "body": f"Schaut euch die {source_category} an. #{source_keyword} https://example.com/product",
                    "body_without_link": f"Diskussion uber {source_category} und #{source_keyword}.",
                    "disclosure_note": f"Ich bin mit {source_category} verbunden.",
                    "ai_search_entity_signals": [source_category, source_keyword, "Germany"],
                    "source_ids": ["R1"],
                    "moderation_notes": [f"Prufe Regeln fur {source_keyword}."],
                    "data_source": "serper_search",
                    "confidence_level": "medium",
                }
            ],
            "localized_entities": [
                localized_entities(
                    language="de",
                    target_market="Germany",
                    subject="Ausrustung fur Bergsteiger",
                    product_category="Winter-Bergsteigausrustung",
                    brand_name="Alpen-Ausrustung",
                    brand_voice="Ausrustungen fur kalte Regionen",
                    primary_keywords=["Bergsteigen", "Hochgebirgsausrustung"],
                )
            ],
        }
        inputs = {
            "subject": source_subject,
            "product_category": source_category,
            "target_markets": "Germany",
            "target_languages": ["de"],
            "platforms": ["Reddit"],
            "brand_voice": source_voice,
            "brand_name": source_category,
            "product_url": "https://example.com/product",
            "primary_keywords": [source_keyword],
            "generate_reddit_geo": True,
        }
        reddit_context = {
            "enabled": True,
            "data_source": "serper_search",
            "confidence_level": "medium",
            "sources": [],
        }

        localized_result = _apply_reddit_geo_context(result, inputs, reddit_context)

        serialized = json.dumps(localized_result["reddit_geo_posts"], ensure_ascii=False)
        for source_term in (source_subject, source_category, source_voice, source_keyword):
            self.assertNotIn(source_term, serialized)
        self.assertIn("Alpen-Ausrustung", serialized)
        self.assertIn("Bergsteigen", serialized)

    def test_reddit_geo_fallback_uses_localized_entities(self) -> None:
        result = {
            "localized_articles": [
                {
                    "language": "de",
                    "title": "Bergsteiger-Set für kalte Höhenlagen",
                    "article": "Artikel über alpine Ausrüstung.",
                }
            ],
            "social_media_posts": [],
            "seo_outputs": [],
            "visual_assets": [],
            "reddit_geo_posts": [],
            "localized_entities": [
                localized_entities(
                    language="de",
                    target_market="Germany",
                    subject="Bergsteiger-Set für kalte Höhenlagen",
                    product_category="Alpine Ausrüstung",
                    brand_name="Alpen-Set",
                    brand_voice="Praktisch und vertrauenswürdig",
                    primary_keywords=["Bergsteigen", "Kälteschutz"],
                )
            ],
        }
        inputs = {
            "subject": "寒带登山套装",
            "product_category": "登山套装",
            "target_markets": "Germany",
            "target_languages": ["de"],
            "platforms": ["Reddit"],
            "brand_name": "登山套装",
            "product_url": "https://example.com/product",
            "primary_keywords": ["登山"],
            "generate_reddit_geo": True,
        }
        reddit_context = {
            "enabled": True,
            "data_source": "serper_unavailable",
            "confidence_level": "low",
            "sources": [],
        }

        localized_result = _apply_reddit_geo_context(result, inputs, reddit_context)

        serialized = json.dumps(localized_result["reddit_geo_posts"], ensure_ascii=False)
        for source_term in ("寒带登山套装", "登山套装", "登山"):
            self.assertNotIn(source_term, serialized)
        self.assertIn("Bergsteiger-Set", serialized)
        self.assertIn("Alpen-Set", serialized)

    def test_content_timeline_entries_normalize_task_and_stage_events(self) -> None:
        events = [
            {
                "event_id": 1,
                "event_type": "task_started",
                "message": "Task 1/2 started: research_and_strategy",
                "payload": {
                    "workflow_type": "content",
                    "task_name": "research_and_strategy",
                    "agent_role": "Strategy Lead",
                },
                "created_at": "2026-06-06T00:00:00Z",
            },
            {
                "event_id": 2,
                "event_type": "content_stage",
                "message": "Image generation completed for ja.",
                "payload": {
                    "scope": "content",
                    "stage": "image_generation",
                    "status": "completed",
                    "language": "ja",
                    "target_market": "Japan",
                    "asset_count": 1,
                },
                "created_at": "2026-06-06T00:00:01Z",
            },
        ]

        entries = _content_timeline_entries(events)

        self.assertEqual(entries[0]["stage"], "research_strategy")
        self.assertEqual(entries[0]["status"], "running")
        self.assertEqual(entries[1]["step"], "Image generation")
        self.assertEqual(entries[1]["status"], "completed")
        self.assertEqual(entries[1]["asset_count"], 1)

    def test_content_timeline_entries_show_failed_and_retrying_stages(self) -> None:
        events = [
            {
                "event_id": 1,
                "event_type": "content_stage",
                "message": "Retrying localized content for ja.",
                "payload": {
                    "scope": "content",
                    "stage": "content_generation",
                    "status": "retrying",
                    "language": "ja",
                    "target_market": "Japan",
                },
            },
            {
                "event_id": 2,
                "event_type": "task_failed",
                "message": "Task failed: content_creation_and_qa:ja",
                "payload": {
                    "scope": "content",
                    "stage": "content_generation",
                    "status": "failed",
                    "language": "ja",
                    "target_market": "Japan",
                    "error_summary": "Output exceeded generation limit.",
                },
            },
        ]

        entries = _content_timeline_entries(events)

        self.assertEqual(entries[0]["status"], "retrying")
        self.assertEqual(entries[0]["status_label"], "Retrying")
        self.assertEqual(entries[1]["status"], "failed")
        self.assertEqual(entries[1]["error_summary"], "Output exceeded generation limit.")

    def test_content_image_placeholder_condition(self) -> None:
        running_job = {"status": "running", "result": None}
        visual_inputs = {"generate_visual_assets": True}

        self.assertTrue(
            _should_show_content_image_placeholder(running_job, [], visual_inputs)
        )
        self.assertFalse(
            _should_show_content_image_placeholder(
                {"status": "completed", "result": None},
                [],
                visual_inputs,
            )
        )
        self.assertFalse(
            _should_show_content_image_placeholder(
                {
                    "status": "running",
                    "result": {"visual_assets": [{"asset_path": "artifacts/content_creation/a.png"}]},
                },
                [],
                visual_inputs,
            )
        )

    def test_content_enrichment_emits_safe_stage_events(self) -> None:
        store = InMemoryJobStore()
        store.create_job("job-1", WorkflowType.CONTENT, {})
        recorder = WorkflowProgressRecorder(
            job_id="job-1",
            workflow_type="content",
            job_store=store,
            backend="local",
        )
        output = {
            "localized_article": {
                "language": "ja",
                "title": "Title",
                "article": "Article body",
            },
            "social_media_posts": [],
            "seo_keywords": ["keyword"],
            "compliance_notes": "Reviewed.",
        }
        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan",
            "target_languages": ["ja"],
            "platforms": ["Instagram"],
            "generate_visual_assets": True,
            "image_generation_count": 1,
            "image_quality": "auto",
            "image_size": "1024x1024",
        }

        def fake_image_generation_run(
            tool: OpenAIImageGenerationTool,
            prompt: str,
            output_slug: str = "content-visual",
            image_generation_count: int = 1,
            image_quality: str = "auto",
            image_size: str = "1024x1024",
        ) -> dict[str, object]:
            return {
                "status": "completed",
                "model": tool.image_model,
                "prompt": prompt,
                "assets": [
                    {
                        "status": "completed",
                        "asset_path": "artifacts/content_creation/fake.png",
                        "model": tool.image_model,
                        "prompt": prompt,
                        "duration_seconds": 0.01,
                    }
                ],
                "duration_seconds": 0.01,
                "attempts": 1,
                "last_status_code": 200,
                "retryable_status": False,
                "error": None,
            }

        def fake_visual_score_run(
            tool: VisualAssetScoringTool,
            asset_path: str,
            prompt: str,
            target_market: str,
            language: str,
            brand_voice: str = "Premium, trustworthy, and culturally respectful",
        ) -> dict[str, object]:
            return {
                "status": "completed",
                "asset_path": asset_path,
                "prompt_alignment_score": 90.0,
                "cultural_fit_score": 90.0,
                "brand_voice_score": 90.0,
                "publish_readiness_score": 90.0,
                "duration_seconds": 0.01,
                "notes": f"Scored with {tool.scoring_model}.",
                "error": None,
            }

        with patch.object(OpenAIImageGenerationTool, "_run", fake_image_generation_run):
            with patch.object(VisualAssetScoringTool, "_run", fake_visual_score_run):
                _enrich_language_output(
                    output,
                    inputs,
                    "ja",
                    "Japan",
                    {
                        PROGRESS_CONTEXT_KEY: recorder,
                        "openai_api_key": "sk-secret-test",
                    },
                )

        stage_payloads = [
            event["payload"]
            for event in store.get_job_events("job-1")
            if event["event_type"] == "content_stage"
        ]
        partial_payloads = [
            event["payload"]
            for event in store.get_job_events("job-1")
            if event["event_type"] == "content_partial"
        ]
        stages = {payload["stage"] for payload in stage_payloads}
        partial_types = {payload["preview_type"] for payload in partial_payloads}
        serialized_payloads = json.dumps(
            [*stage_payloads, *partial_payloads],
            ensure_ascii=False,
        )

        self.assertIn("image_generation", stages)
        self.assertIn("visual_scoring", stages)
        self.assertIn("visual_brief", partial_types)
        self.assertIn("seo_metadata", partial_types)
        self.assertIn("compliance", partial_types)
        self.assertIn("images", partial_types)
        self.assertNotIn("sk-secret-test", serialized_payloads)
        self.assertNotIn("b64_json", serialized_payloads)

    def test_live_preview_groups_merge_out_of_order_partial_events(self) -> None:
        events = [
            {
                "event_id": 3,
                "event_type": "content_partial",
                "payload": {
                    "scope": "content",
                    "stage": "image_generation",
                    "language": "ja",
                    "target_market": "Japan",
                    "preview_type": "images",
                    "content": {
                        "status": "failed",
                        "assets": [],
                        "error_summary": "HTTP 520: upstream image API returned an HTML error page.",
                    },
                    "created_at": "2026-06-06T00:00:03Z",
                },
            },
            {
                "event_id": 1,
                "event_type": "content_partial",
                "payload": {
                    "scope": "content",
                    "stage": "content_generation",
                    "language": "ja",
                    "target_market": "Japan",
                    "preview_type": "content_package",
                    "content": {
                        "localized_article": {
                            "language": "ja",
                            "title": "Japanese Title",
                            "article": "Article body",
                        },
                        "social_media_posts": [],
                        "seo_keywords": ["keyword"],
                        "compliance_notes": "Reviewed.",
                    },
                    "created_at": "2026-06-06T00:00:01Z",
                },
            },
        ]

        groups = _content_live_preview_groups(events)

        self.assertEqual(len(groups), 1)
        previews = groups[0]["previews"]
        self.assertEqual(
            previews["content_package"]["localized_article"]["title"],
            "Japanese Title",
        )
        self.assertEqual(previews["images"]["status"], "failed")
        self.assertIn("HTTP 520", previews["images"]["error_summary"])

    def test_live_preview_groups_include_failed_content_package(self) -> None:
        events = [
            {
                "event_id": 1,
                "event_type": "content_partial",
                "payload": {
                    "scope": "content",
                    "stage": "content_generation",
                    "language": "ja",
                    "target_market": "Japan",
                    "preview_type": "content_package",
                    "content": {
                        "status": "failed",
                        "error_summary": "Output exceeded generation limit.",
                        "retry_available": True,
                    },
                },
            }
        ]

        groups = _content_live_preview_groups(events)

        self.assertEqual(len(groups), 1)
        content_package = groups[0]["previews"]["content_package"]
        self.assertEqual(content_package["status"], "failed")
        self.assertTrue(content_package["retry_available"])

    def test_content_partial_image_failure_sanitizes_html_error(self) -> None:
        store = InMemoryJobStore()
        store.create_job("job-1", WorkflowType.CONTENT, {})
        recorder = WorkflowProgressRecorder(
            job_id="job-1",
            workflow_type="content",
            job_store=store,
            backend="local",
        )
        output = {
            "localized_article": {
                "language": "ja",
                "title": "Title",
                "article": "Article body",
            },
            "social_media_posts": [],
            "seo_keywords": ["keyword"],
            "compliance_notes": "Reviewed.",
        }
        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan",
            "target_languages": ["ja"],
            "platforms": ["Instagram"],
            "generate_visual_assets": True,
            "image_generation_count": 1,
            "image_quality": "low",
            "image_size": "1024x1024",
        }
        html_error = (
            "HTTP 520: <!DOCTYPE html><html><head><title>"
            "api.openai.com | 520: Web server is returning an unknown error"
            "</title></head></html>"
        )

        def fake_failed_image_generation(
            tool: OpenAIImageGenerationTool,
            prompt: str,
            output_slug: str = "content-visual",
            image_generation_count: int = 1,
            image_quality: str = "auto",
            image_size: str = "1024x1024",
        ) -> dict[str, object]:
            return {
                "status": "failed",
                "model": tool.image_model,
                "prompt": prompt,
                "assets": [],
                "duration_seconds": 1.0,
                "attempts": 3,
                "last_status_code": 520,
                "retryable_status": True,
                "error": html_error,
            }

        with patch.object(OpenAIImageGenerationTool, "_run", fake_failed_image_generation):
            _enrich_language_output(
                output,
                inputs,
                "ja",
                "Japan",
                {PROGRESS_CONTEXT_KEY: recorder, "openai_api_key": "sk-secret-test"},
            )

        image_partials = [
            event["payload"]
            for event in store.get_job_events("job-1")
            if event["event_type"] == "content_partial"
            and event["payload"]["preview_type"] == "images"
        ]

        self.assertTrue(image_partials)
        serialized = json.dumps(image_partials, ensure_ascii=False)
        self.assertIn("HTTP 520", serialized)
        self.assertNotIn("<html", serialized.lower())
        self.assertNotIn("sk-secret-test", serialized)

    def test_error_summary_sanitizes_generation_limit_and_secrets(self) -> None:
        summary = _error_summary(
            "Could not parse response content as the length limit was reached - "
            "CompletionUsage(completion_tokens=16384, api_key=sk-secret-test-value)"
        )

        self.assertEqual(
            summary,
            "Output exceeded generation limit; content was truncated before valid JSON could be parsed.",
        )

    def test_content_workflow_injects_reddit_geo_context_and_merges_outputs(self) -> None:
        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan",
            "target_languages": ["ja"],
            "platforms": ["Reddit"],
            "brand_name": "ThermoCo",
            "product_url": "https://example.com/products/smart-thermos",
            "generate_reddit_geo": True,
            "generate_visual_assets": False,
        }
        reddit_context = {
            "status": "live_search",
            "data_source": "serper_search",
            "confidence_level": "medium",
            "assumption_notice": "Review subreddit rules before posting.",
            "query_pack": [],
            "sources": [
                {
                    "source_id": "R1",
                    "target_market": "Japan",
                    "subreddit": "BuyItForLife",
                    "title": "Best thermos for winter hikes?",
                    "snippet": "People compare durable insulated bottles.",
                    "url": "https://www.reddit.com/r/BuyItForLife/comments/abc123/best_thermos/",
                    "query_type": "reddit_need_discussion",
                    "query": "site:reddit.com/r/ thermos",
                }
            ],
            "duration_seconds": 0.01,
            "provider_error": None,
        }
        captured_strategy_inputs: dict[str, object] = {}

        def fake_research_strategy(
            normalized_inputs: dict[str, object],
            agents_config: dict[str, object],
            tasks_config: dict[str, object],
            config_context: dict[str, object],
        ) -> dict[str, object]:
            captured_strategy_inputs.update(normalized_inputs)
            return {"strategy": "ok"}

        def fake_language_generation(
            language: str,
            language_index: int,
            normalized_inputs: dict[str, object],
            strategy_context: dict[str, object],
            agents_config: dict[str, object],
            tasks_config: dict[str, object],
            config_context: dict[str, object],
        ) -> dict[str, object]:
            self.assertIn("R1", str(normalized_inputs["reddit_geo_context_summary"]))
            return {
                "localized_article": {
                    "language": language,
                    "title": "Smart Thermos buying guide",
                    "article": "Article body",
                },
                "social_media_posts": [],
                "seo_keywords": ["smart thermos"],
                "compliance_notes": "Reviewed.",
                "visual_assets": [],
                "visual_asset_scores": [],
                "reddit_geo_posts": [
                    {
                        "target_market": "Japan",
                        "language": language,
                        "recommended_subreddit": "r/BuyItForLife",
                        "title_options": ["Smart Thermos buying questions"],
                        "body": "Disclosure: I am affiliated with ThermoCo.\n\nUseful context: https://example.com/products/smart-thermos",
                        "body_without_link": "Disclosure: I am affiliated with ThermoCo.\n\nUseful context.",
                        "disclosure_note": "Disclosure: I am affiliated with ThermoCo.",
                        "ai_search_entity_signals": ["ThermoCo", "Smart Thermos", "Japan"],
                        "source_ids": ["R1"],
                        "moderation_notes": ["Check subreddit self-promotion rules."],
                        "data_source": "serper_search",
                        "confidence_level": "medium",
                    }
                ],
            }

        with patch("crews.content_crew.RedditGeoSearchTool._run", return_value=reddit_context):
            with patch("crews.content_crew._run_research_strategy", side_effect=fake_research_strategy):
                with patch("crews.content_crew._run_language_generation", side_effect=fake_language_generation):
                    result = run_content_crew(inputs, {"serper_api_key": "serper-test-key"})

        self.assertTrue(captured_strategy_inputs["reddit_geo_enabled"])
        self.assertIn("R1", str(captured_strategy_inputs["reddit_geo_context_summary"]))
        self.assertEqual(result["reddit_geo_sources"][0]["source_id"], "R1")
        self.assertEqual(result["reddit_geo_sources"][0]["data_source"], "serper_search")
        self.assertEqual(result["reddit_geo_posts"][0]["recommended_subreddit"], "r/BuyItForLife")
        self.assertTrue(
            any(asset["asset_type"] == "reddit_geo_post" for asset in result["production_ready_assets"])
        )

    def test_content_workflow_keeps_reddit_geo_empty_when_disabled(self) -> None:
        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan",
            "target_languages": ["ja"],
            "platforms": ["Instagram"],
            "generate_visual_assets": False,
        }

        def fake_language_generation(
            language: str,
            language_index: int,
            normalized_inputs: dict[str, object],
            strategy_context: dict[str, object],
            agents_config: dict[str, object],
            tasks_config: dict[str, object],
            config_context: dict[str, object],
        ) -> dict[str, object]:
            return {
                "localized_article": {
                    "language": language,
                    "title": "Title",
                    "article": "Article body",
                },
                "social_media_posts": [
                    {
                        "platform": "Instagram",
                        "language": language,
                        "content": "Post",
                    }
                ],
                "seo_keywords": ["keyword"],
                "compliance_notes": "Reviewed.",
                "visual_assets": [],
                "visual_asset_scores": [],
            }

        with patch("crews.content_crew.RedditGeoSearchTool._run") as search:
            with patch("crews.content_crew._run_research_strategy", return_value={"strategy": "ok"}):
                with patch("crews.content_crew._run_language_generation", side_effect=fake_language_generation):
                    result = run_content_crew(inputs, {})

        search.assert_not_called()
        self.assertEqual(result["reddit_geo_posts"], [])
        self.assertEqual(result["reddit_geo_sources"], [])

    def test_content_workflow_emits_threaded_language_agent_and_stage_spans(self) -> None:
        class FakeRouter:
            def __init__(self, config_context: dict[str, object]) -> None:
                self.config_context = config_context

            def llm_for_agent(self, agent_config: dict[str, object]) -> object:
                return object()

        class FakeAgent:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FakeTask:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FakeCrew:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def kickoff(self, inputs: dict[str, object]) -> dict[str, object]:
                language = str(inputs["target_language"])
                return {
                    "localized_article": {
                        "language": language,
                        "title": f"Localized title {language}",
                        "article": f"Localized body {language}",
                    },
                    "social_media_posts": [],
                    "seo_keywords": [f"keyword-{language}"],
                    "compliance_notes": "Reviewed.",
                    "localized_entities": localized_entities(
                        language=language,
                        target_market=str(inputs["target_market"]),
                        subject=f"Localized product {language}",
                        product_category=f"Localized category {language}",
                        brand_name=f"Localized brand {language}",
                        brand_voice=f"Localized voice {language}",
                        primary_keywords=[f"keyword-{language}"],
                    ),
                }

        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "United States, China",
            "target_languages": ["en", "zh"],
            "platforms": ["Instagram"],
            "generate_visual_assets": False,
            "content_language_concurrency": 2,
        }
        store = InMemoryJobStore()
        store.create_job("job-content-trace", WorkflowType.CONTENT, inputs)
        config_context: dict[str, object] = {
            "observability_enabled": True,
            "otel_enabled": False,
            "workflow_type": "content",
        }
        recorder = WorkflowProgressRecorder(
            job_id="job-content-trace",
            workflow_type="content",
            job_store=store,
            backend="local",
            config_context=config_context,
        )
        config_context[PROGRESS_CONTEXT_KEY] = recorder
        client = ContentTraceLangfuseClient()

        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "public", "LANGFUSE_SECRET_KEY": "secret"},
            clear=True,
        ):
            with patch("utils.observability._langfuse_client", return_value=client):
                with patch("crews.content_crew.ModelTierRouter", FakeRouter):
                    with patch("crews.content_crew.Agent", FakeAgent):
                        with patch("crews.content_crew.Task", FakeTask):
                            with patch("crews.content_crew.Crew", FakeCrew):
                                with patch("crews.content_crew._build_creation_tools", return_value=[]):
                                    with patch("crews.content_crew._crew_memory", return_value=None):
                                        with patch(
                                            "crews.content_crew._run_research_strategy",
                                            return_value={"strategy": "ok"},
                                        ):
                                            with workflow_span(
                                                "content",
                                                job_id="job-content-trace",
                                                backend="local",
                                                config_context=config_context,
                                            ):
                                                result = run_content_crew(inputs, config_context)

        observations_by_name = {observation.name: observation for observation in client.observations}
        self.assertEqual(len(result["localized_articles"]), 2)
        self.assertIn("workflow.content", observations_by_name)
        for language in ("en", "zh"):
            agent_name = f"agent.content_creation_and_qa:{language}"
            self.assertIn(agent_name, observations_by_name)
            self.assertEqual(observations_by_name[agent_name].as_type, "agent")
            self.assertEqual(observations_by_name[agent_name].parent_name, "workflow.content")
            self.assertEqual(
                observations_by_name[agent_name].metadata["agent_role"],
                "Multilingual Content Creator & Quality Editor",
            )
            for stage in (
                "content_generation",
                "visual_localization",
                "seo_metadata",
                "cultural_compliance",
                "image_generation",
                "visual_scoring",
            ):
                stage_name = f"stage.{stage}:{language}"
                self.assertIn(stage_name, observations_by_name)
                self.assertEqual(observations_by_name[stage_name].parent_name, agent_name)
                self.assertEqual(observations_by_name[stage_name].metadata["language"], language)
                self.assertEqual(observations_by_name[stage_name].metadata["stage"], stage)

        self.assertIn("stage.content_assembly", observations_by_name)
        self.assertEqual(
            observations_by_name["stage.content_assembly"].parent_name,
            "workflow.content",
        )
        status_updates = [
            update
            for update in client.current_span_updates
            if isinstance(update.get("metadata"), dict) and update["metadata"].get("status")
        ]
        self.assertTrue(
            any(
                update["name"] == "stage.image_generation:en"
                and update["metadata"].get("status") == "skipped"
                for update in status_updates
            )
        )
        self.assertTrue(
            any(
                update["name"] == "stage.visual_scoring:zh"
                and update["metadata"].get("status") == "skipped"
                for update in status_updates
            )
        )
        serialized_trace_metadata = json.dumps(
            [
                observation.metadata
                for observation in client.observations
            ],
            ensure_ascii=False,
            default=str,
        )
        self.assertNotIn("Localized body", serialized_trace_metadata)

    def test_content_workflow_merges_successful_language_when_another_fails(self) -> None:
        inputs = {
            "subject": "Sustainable Activewear",
            "product_category": "Sportswear",
            "target_markets": "Japan, China",
            "target_languages": ["ja", "zh-CN"],
            "platforms": ["Instagram"],
            "generate_visual_assets": False,
        }

        def fake_language_generation(
            language: str,
            language_index: int,
            inputs: dict[str, object],
            strategy_context: dict[str, object],
            agents_config: dict[str, object],
            tasks_config: dict[str, object],
            config_context: dict[str, object],
        ) -> dict[str, object]:
            if language == "ja":
                raise RuntimeError(
                    "Could not parse response content as the length limit was reached - "
                    "CompletionUsage(completion_tokens=16384, api_key=sk-secret-test-value)"
                )
            return {
                "localized_article": {
                    "language": language,
                    "title": "中文标题",
                    "article": "中文正文",
                },
                "social_media_posts": [
                    {
                        "platform": "Instagram",
                        "language": language,
                        "content": "中文社媒文案",
                    }
                ],
                "seo_keywords": ["可持续运动服"],
                "compliance_notes": "Reviewed.",
                "visual_assets": [],
                "visual_asset_scores": [],
            }

        with patch("crews.content_crew._run_research_strategy", return_value={"strategy": "ok"}):
            with patch("crews.content_crew._run_language_generation", side_effect=fake_language_generation):
                result = run_content_crew(inputs, {})

        self.assertEqual(len(result["localized_articles"]), 1)
        self.assertEqual(result["localized_articles"][0]["language"], "zh-CN")
        self.assertIn("Generation failed for ja", result["compliance_notes"])
        self.assertIn("Coverage notice", result["compliance_notes"])
        self.assertNotIn("sk-secret-test", result["compliance_notes"])

    def test_content_workflow_enrichment_keeps_visual_generation_disabled_by_default(self) -> None:
        output = {
            "localized_article": {
                "language": "ja",
                "title": "Title",
                "article": "Article body",
            },
            "social_media_posts": [],
            "seo_keywords": ["keyword"],
            "compliance_notes": "Reviewed.",
        }
        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan",
            "target_languages": ["ja"],
            "platforms": ["Instagram"],
            "generate_visual_assets": False,
        }

        result = _enrich_language_output(output, inputs, "ja", "Japan", {})

        self.assertIn("multimodal_output", result)
        self.assertIn("seo_metadata", result)
        self.assertIn("cultural_risk_assessment", result)
        self.assertEqual(result["visual_assets"], [])
        self.assertEqual(result["visual_asset_scores"], [])

    def test_content_workflow_enrichment_records_skipped_image_status(self) -> None:
        output = {
            "localized_article": {
                "language": "ja",
                "title": "Title",
                "article": "Article body",
            },
            "social_media_posts": [],
            "seo_keywords": ["keyword"],
            "compliance_notes": "Reviewed.",
        }
        inputs = {
            "subject": "Smart Thermos",
            "product_category": "Drinkware",
            "target_markets": "Japan",
            "target_languages": ["ja"],
            "platforms": ["Instagram"],
            "generate_visual_assets": True,
            "image_generation_count": 1,
            "image_quality": "auto",
            "image_size": "1024x1024",
        }

        result = _enrich_language_output(output, inputs, "ja", "Japan", {})

        self.assertEqual(result["visual_assets"][0]["status"], "skipped_missing_credentials")
        self.assertEqual(result["visual_assets"][0]["duration_seconds"], 0.0)
        self.assertEqual(
            result["visual_asset_scores"][0]["status"],
            "skipped_missing_local_asset",
        )


if __name__ == "__main__":
    unittest.main()
