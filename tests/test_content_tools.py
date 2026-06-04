import base64
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import httpx

from admin_dashboard import ARTIFACT_ROOT, _normalized_visual_score, _safe_artifact_path
from crews.content_crew import _enrich_language_output
from tools.custom.content_tools import (
    MultimodalLocalizationTool,
    OpenAIImageGenerationTool,
    VisualAssetScoringTool,
)


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
