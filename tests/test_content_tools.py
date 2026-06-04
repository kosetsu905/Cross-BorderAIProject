import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from crews.content_crew import _enrich_language_output
from tools.custom.content_tools import (
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

    def test_image_generation_retries_transient_status_and_saves_asset(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
        image_payload = base64.b64encode(b"fake image bytes").decode("ascii")
        responses = [
            httpx.Response(429, json={"error": "rate limited"}, request=request),
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
            self.assertEqual(len(result["assets"]), 1)
            saved_path = Path(result["assets"][0]["asset_path"])
            self.assertTrue(saved_path.is_file())
            self.assertEqual(saved_path.read_bytes(), b"fake image bytes")

    def test_visual_asset_scoring_parses_response_output_text(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(
            200,
            json={
                "output_text": (
                    '{"prompt_alignment_score": 92, "cultural_fit_score": 88, '
                    '"brand_voice_score": 90, "publish_readiness_score": 86, '
                    '"notes": "Ready after minor crop review."}'
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
        self.assertIn("minor crop", result["notes"])

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
        self.assertEqual(
            result["visual_asset_scores"][0]["status"],
            "skipped_missing_local_asset",
        )


if __name__ == "__main__":
    unittest.main()
