import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.train_intent_classifier import IntentClassifierTool, export_label_map
from services.pim_connector import PIMConnector


class CustomerServiceExpectedOutcomeTests(unittest.TestCase):
    def test_pim_integration_uses_mock_fallback_without_credentials(self) -> None:
        env = {"PIM_AKENEO_BASE_URL": "", "PIM_AKENEO_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            result = asyncio.run(PIMConnector(backend="akeneo").search_product("Smart Camera", "JP", "ja"))

        self.assertTrue(result.product_found)
        self.assertEqual(result.data_source, "mock_fallback")
        self.assertTrue(result.variants)
        self.assertTrue(result.variants[0].name.get("ja"))

    def test_multilingual_intent_classifier_examples_with_fake_pipeline(self) -> None:
        examples = [
            ("Is this waterproof?", "en", "pre_sales"),
            ("注文を追跡したい", "ja", "order_fulfillment"),
            ("商品が壊れていました", "ja", "post_sales_support"),
            ("¿Cuándo llegará mi pedido?", "es", "order_fulfillment"),
        ]

        class FakeClassifier:
            def __call__(self, text):
                if "waterproof" in text:
                    return [{"label": "LABEL_0", "score": 0.93}]
                if "追跡" in text or "pedido" in text:
                    return [{"label": "LABEL_1", "score": 0.91}]
                if "壊れて" in text:
                    return [{"label": "LABEL_2", "score": 0.94}]
                return [{"label": "LABEL_2", "score": 0.5}]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_label_map(tmpdir)
            classifier = IntentClassifierTool(tmpdir, classifier=FakeClassifier())
            for text, language, expected in examples:
                with self.subTest(text=text, language=language):
                    result = classifier.predict(text, language)
                    self.assertEqual(result["detected_intent"], expected)
                    self.assertGreaterEqual(result["confidence_score"], 0.75)
                    self.assertFalse(result["requires_human_review"])
                    self.assertEqual(result["language_detected"], language)


if __name__ == "__main__":
    unittest.main()
