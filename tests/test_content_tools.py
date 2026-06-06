import base64
import json
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import httpx

from admin_dashboard import (
    ARTIFACT_ROOT,
    _content_inputs_from_form_values,
    _content_live_preview_groups,
    _content_timeline_entries,
    _normalized_visual_score,
    _safe_artifact_path,
    _should_show_content_image_placeholder,
)
from crews.content_crew import (
    CONTENT_ARTICLE_MAX_CHARS,
    CONTENT_GENERATION_MAX_TOKENS,
    CONTENT_SEO_KEYWORDS_MAX_COUNT,
    _content_generation_llm_context,
    _enrich_language_output,
    _error_summary,
    _normalize_inputs,
    run_content_crew,
)
from job_store import InMemoryJobStore
from models import ContentInputs, WorkflowType
from tools.custom.content_tools import (
    MultimodalLocalizationTool,
    OpenAIImageGenerationTool,
    VisualAssetScoringTool,
)
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, WorkflowProgressRecorder


class ContentToolTests(unittest.TestCase):
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
            primary_keywords="thermal activewear, recycled sportswear",
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
        self.assertTrue(payload["generate_visual_assets"])
        self.assertEqual(payload["image_generation_count"], 2)
        self.assertEqual(ContentInputs.model_validate(payload).image_quality, "high")

        payload_without_quality = dict(payload)
        payload_without_quality.pop("image_quality")
        self.assertEqual(
            ContentInputs.model_validate(payload_without_quality).image_quality,
            "low",
        )

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
