import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import create_router
from crews.support_crew import (
    CustomerServiceOutput,
    _attach_catalog_knowledge_context,
    _attach_customer_service_context,
    _build_automation_context,
    _normalize_inputs,
)
from models import WorkflowType
from runtime_config import load_runtime_config
from services.pim_connector import PIMConnector, PIMQueryResult
from scripts.train_intent_classifier import (
    INTENT_LABELS,
    IntentClassifierTool,
    IntentDatasetBuilder,
    TrainingConfig,
    export_label_map,
)
from services.intent_router import classify_intent
from support_inbox import SupportInboxStore
from tests.test_whatsapp_omnichannel import FakeSupportSession
from utils.usage_tracking import INTERNAL_USAGE_KEY
from tools.custom.customer_service_tools import IntentRouterTool, OrderTrackingTool, PreSalesProductKnowledgeTool
from tools.custom.support_rag_tools import extract_catalog_product_offer, load_knowledge_chunks, search_knowledge_base


class FakePIMResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class CustomerServiceWorkflowTests(unittest.TestCase):
    def _offline_customer_service_result(self, payload: dict[str, object], final_response: str) -> dict[str, object]:
        inputs = _normalize_inputs(payload)
        router_result = IntentRouterTool()._run(
            inquiry_text=inputs["inquiry_text"],
            has_order_id=bool(inputs.get("order_id") and inputs.get("order_id") != "ORDER-NOT-PROVIDED"),
            customer_tier=inputs.get("customer_tier", "STANDARD"),
        )
        pre_sales_context = PreSalesProductKnowledgeTool()._run(
            product_category=inputs["product_category"],
            inquiry_keywords=str(inputs["inquiry_text"]).split(),
            region=inputs["region"],
        )
        order_context = OrderTrackingTool()._run(
            order_id=None if inputs["order_id"] == "ORDER-NOT-PROVIDED" else inputs["order_id"],
            customer_email=inputs.get("customer_email"),
            region=inputs["region"],
        )
        return _attach_customer_service_context(
            result={"final_response": final_response, "qa_status": "APPROVED", "escalation_needed": False},
            inputs=inputs,
            automation_context=_build_automation_context(inputs),
            router_result=router_result,
            pre_sales_context=pre_sales_context,
            order_context=order_context,
        )

    def test_all_customer_service_stages_offline(self) -> None:
        pre_result = self._offline_customer_service_result(
            {
                "customer": "Alex Kim",
                "person": "Alex Kim",
                "inquiry": "I'm deciding between the Basic and Pro camera models. Which one works better for outdoor use in rainy weather?",
                "product_category": "Smart Home Camera",
                "channel": "whatsapp",
                "language_plan": "English",
                "customer_tier": "STANDARD",
            },
            "For rainy outdoor use, the Pro model is the better fit because it includes local storage and stronger smart-home coverage.",
        )
        order_result = self._offline_customer_service_result(
            {
                "customer": "Maria Garcia",
                "person": "Maria Garcia",
                "inquiry": "Where is my order ORD-JP-2024-8842? It was supposed to arrive yesterday.",
                "order_id": "ORD-JP-2024-8842",
                "channel": "email",
                "language_plan": "English",
                "customer_tier": "STANDARD",
                "region": "JP",
            },
            "Your order has shipped with Yamato Transport. Please use the tracking link for the latest delivery scan.",
        )
        support_result = self._offline_customer_service_result(
            {
                "customer": "John Doe",
                "person": "John Doe",
                "inquiry": "The camera I received has a cracked lens. How do I return it for replacement?",
                "order_id": "ORD-US-2024-3391",
                "channel": "webchat",
                "language_plan": "English",
                "customer_tier": "STANDARD",
                "order_history": {"days_since_delivery": 5, "item_condition": "defective", "region": "US"},
            },
            "We can help replace the damaged camera. Please follow the return policy steps and use the prepaid return label.",
        )

        results = [pre_result, order_result, support_result]
        for result in results:
            CustomerServiceOutput.model_validate(result)
            self.assertTrue(result["final_response"])
            self.assertEqual(result["qa_status"], "APPROVED")
            self.assertFalse(result["escalation_needed"])

        checks = [
            pre_result["detected_intent"] == "pre_sales",
            pre_result["routing_confidence"] >= 0.75,
            order_result["detected_intent"] == "order_fulfillment",
            order_result["order_response"]["order_found"] is True,
            bool(order_result["order_response"]["tracking_info"]["tracking_url"]),
            support_result["detected_intent"] == "post_sales_support",
            "policy" in str(support_result["support_response"]).lower(),
            all(result["qa_status"] == "APPROVED" for result in results),
            not any(result["escalation_needed"] for result in results),
        ]

        self.assertTrue(all(checks), f"Customer Service offline validation failed: {checks}")

    def test_pim_mock_fallback_returns_standard_result(self) -> None:
        with patch.dict(os.environ, {"PIM_AKENEO_BASE_URL": "", "PIM_AKENEO_API_KEY": ""}, clear=False):
            result = asyncio.run(PIMConnector(backend="akeneo").search_product("Smart Home Camera", "US", "en"))

        self.assertIsInstance(result, PIMQueryResult)
        self.assertTrue(result.product_found)
        self.assertEqual(result.data_source, "mock_fallback")

    def test_pim_missing_credentials_do_not_call_http(self) -> None:
        async def run_for_backend(backend: str) -> str:
            result = await PIMConnector(backend=backend).search_product("camera", "US", "en")
            return result.data_source

        env = {
            "PIM_AKENEO_BASE_URL": "",
            "PIM_AKENEO_API_KEY": "",
            "PIM_PLYTIX_BASE_URL": "",
            "PIM_PLYTIX_API_KEY": "",
            "PIM_CUSTOM_BASE_URL": "",
            "PIM_CUSTOM_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "services.pim_connector.httpx.AsyncClient", side_effect=AssertionError("HTTP should not be called")
        ):
            self.assertEqual(asyncio.run(run_for_backend("akeneo")), "mock_fallback")
            self.assertEqual(asyncio.run(run_for_backend("plytix")), "mock_fallback")
            self.assertEqual(asyncio.run(run_for_backend("custom")), "mock_fallback")

    def test_pim_akeneo_query_uses_expected_api_shape(self) -> None:
        calls = []

        class FakeAkeneoClient:
            def __init__(self, *args, **kwargs):
                self.get_count = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, url, params=None, headers=None):
                calls.append(("GET", url, params, headers))
                self.get_count += 1
                if self.get_count == 1:
                    return FakePIMResponse({"_embedded": {"items": [{"code": "CAM-4K-PRO"}]}})
                return FakePIMResponse(
                    {
                        "code": "CAM-4K-PRO",
                        "family": "smart_home_camera",
                        "updated": "2026-01-01T00:00:00Z",
                        "values": {
                            "compatibility": [{"locale": "en", "data": "Works with HomeKit"}],
                            "resolution": [{"locale": None, "data": "4K"}],
                        },
                    }
                )

        with patch("services.pim_connector.httpx.AsyncClient", FakeAkeneoClient):
            result = asyncio.run(
                PIMConnector("akeneo", base_url="https://pim.example", api_key="token").search_product(
                    "camera", "US", "en"
                )
            )

        self.assertEqual(result.data_source, "akeneo")
        self.assertEqual(calls[0][1], "https://pim.example/api/rest/v1/products")
        self.assertEqual(calls[0][3]["Authorization"], "Bearer token")
        self.assertIn("CAM-4K-PRO", calls[1][1])

    def test_pim_plytix_and_custom_queries_map_to_standard_result(self) -> None:
        calls = []

        class FakePostClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def post(self, url, json=None, headers=None):
                calls.append(("POST", url, json, headers))
                if "/v1/entity/search" in url:
                    return FakePIMResponse(
                        {"data": [{"family_code": "camera_family", "attributes": {"compatibility": "HomeKit"}}]}
                    )
                return FakePIMResponse(
                    {
                        "found": True,
                        "family": "custom_camera",
                        "attributes": [
                            {
                                "code": "compatibility",
                                "label": {"en": "Compatibility"},
                                "value": {"en": "HomeKit"},
                                "scope": "global",
                            }
                        ],
                        "variants": [],
                        "compliance": {"US": ["FCC"]},
                        "updated_at": "2026-01-01T00:00:00Z",
                    }
                )

        with patch("services.pim_connector.httpx.AsyncClient", FakePostClient):
            plytix = asyncio.run(
                PIMConnector("plytix", base_url="https://plytix.example", api_key="key").search_product(
                    "camera", "US", "en"
                )
            )
            custom = asyncio.run(
                PIMConnector("custom", base_url="https://custom.example", api_key="key").search_product(
                    "camera", "US", "en"
                )
            )

        self.assertEqual(plytix.data_source, "plytix")
        self.assertEqual(custom.data_source, "custom_pim")
        self.assertEqual(calls[0][1], "https://plytix.example/v1/entity/search")
        self.assertEqual(calls[0][3]["X-Api-Key"], "key")
        self.assertEqual(calls[1][1], "https://custom.example/products/search")
        self.assertEqual(calls[1][3]["Authorization"], "Bearer key")

    def test_pre_sales_tool_maps_pim_result_to_context(self) -> None:
        with patch.dict(os.environ, {"PIM_AKENEO_BASE_URL": "", "PIM_AKENEO_API_KEY": ""}, clear=False):
            context = PreSalesProductKnowledgeTool()._run("Smart Home Camera", ["camera"], "JP")

        self.assertTrue(context["product_found"])
        self.assertIn("verified_features", context)
        self.assertIn("variant_options", context)
        self.assertEqual(context["data_source"], "mock_fallback")

    def test_knowledge_base_loads_pdf_catalog_chunks(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        load_knowledge_chunks.cache_clear()
        chunks = load_knowledge_chunks(str(knowledge_dir))

        pdf_chunks = [chunk for chunk in chunks if chunk.source.lower().endswith(".pdf")]
        self.assertTrue(pdf_chunks)
        self.assertTrue(any("Bluetooth" in chunk.content or "headset" in chunk.content for chunk in pdf_chunks))

    def test_knowledge_search_returns_pdf_catalog_source(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        load_knowledge_chunks.cache_clear()
        results = search_knowledge_base("Bluetooth headset", str(knowledge_dir), top_k=5)

        self.assertTrue(any(result["source"].lower().endswith(".pdf") for result in results))

    def test_catalog_product_offer_extracts_verified_price(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        load_knowledge_chunks.cache_clear()

        offer = extract_catalog_product_offer("M90 PRO wireless earphones bulk pricing", str(knowledge_dir))

        self.assertEqual(offer["status"], "found")
        self.assertEqual(offer["unit_price"], "$2.39")
        self.assertEqual(offer["carton_quantity"], "100 PCS")
        self.assertIn("M90 PRO wireless earphones", offer["product_name"])
        self.assertTrue(offer["source"].lower().endswith(".pdf"))

    def test_catalog_product_offer_extracts_wireless_headset_exact_catalog_item(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        load_knowledge_chunks.cache_clear()

        offer = extract_catalog_product_offer("Wireless Bluetooth headset product specification", str(knowledge_dir))

        self.assertEqual(offer["status"], "found")
        self.assertEqual(offer["product_name"], "Wireless Bluetooth headset")
        self.assertEqual(offer["unit_price"], "$6.50")
        self.assertEqual(offer["carton_quantity"], "40PCS")
        self.assertEqual(offer["carton_size"], "52x38x51cm")
        self.assertEqual(offer["carton_weight"], "9 kg")

    def test_catalog_product_offer_extracts_b39_specs_when_model_named(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        load_knowledge_chunks.cache_clear()

        offer = extract_catalog_product_offer("B39 wireless Bluetooth headset product specification", str(knowledge_dir))

        self.assertEqual(offer["status"], "found")
        self.assertIn("B39 wireless Bluetooth headset", offer["product_name"])
        self.assertEqual(offer["unit_price"], "$3.35")
        self.assertEqual(offer["carton_quantity"], "80PCS")
        self.assertEqual(offer["carton_size"], "76×42×50cm")
        self.assertEqual(offer["carton_weight"], "24kg")
        self.assertEqual(offer["single_product_size"], "18.5*15.8*4.6cm")
        self.assertEqual(offer["single_product_weight"], "200g")

    def test_pre_sales_context_includes_catalog_knowledge_results(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Alex",
                "inquiry": "Which Bluetooth headset has carton pricing details?",
                "product_category": "Bluetooth headset",
                "channel": "whatsapp",
            }
        )
        context = {"product_found": True, "data_source": "mock_fallback"}
        load_knowledge_chunks.cache_clear()

        _attach_catalog_knowledge_context(context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        self.assertIn(context["knowledge_data_source"], {"local_vector_knowledge_base", "local_pdf_catalog"})
        self.assertTrue(context["catalog_knowledge_results"])
        self.assertTrue(any(item["source"].lower().endswith(".pdf") for item in context["catalog_knowledge_results"]))

    def test_pre_sales_context_includes_catalog_offer_guardrails(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Alex",
                "inquiry": "Please share M90 PRO wireless earphones bulk pricing.",
                "product_category": "M90 PRO wireless earphones",
                "channel": "gmail",
            }
        )
        context = {"product_found": True, "data_source": "mock_fallback"}
        load_knowledge_chunks.cache_clear()

        _attach_catalog_knowledge_context(context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        self.assertEqual(context["catalog_product_offer"]["unit_price"], "$2.39")
        self.assertTrue(any("Do not invent discount tiers" in rule for rule in context["pricing_guardrails"]))

    def test_mock_fallback_context_does_not_expose_customer_facing_variants(self) -> None:
        inputs = _normalize_inputs(
            {
                "customer": "Alex",
                "inquiry": "Which camera should I buy?",
                "channel": "gmail",
                "session_id": "sess-mock",
            }
        )

        result = _attach_customer_service_context(
            result={"final_response": "The catalog does not list detailed variants.", "qa_status": "APPROVED"},
            inputs=inputs,
            automation_context={
                "sentiment_analysis": {"requires_human_handoff": False},
                "rma_validation": None,
                "logistics_output": None,
                "escalation_flag": False,
                "compliance_tags": [],
            },
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.92},
            pre_sales_context={
                "data_source": "mock_fallback",
                "product_found": True,
                "verified_features": ["Mock feature"],
                "compatibility_info": {"app": "Mock app"},
                "variant_options": {"Basic": {"sku": "HEADSET-BASIC", "price": {"USD": 6.30}}},
            },
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        self.assertEqual(result["pre_sales_response"]["verified_features"], [])
        self.assertEqual(result["pre_sales_response"]["compatibility_info"], [])
        self.assertEqual(result["pre_sales_response"]["variant_options"], [])

    def test_runtime_config_reads_pim_environment(self) -> None:
        env = {
            "PIM_BACKEND": "plytix",
            "PIM_AKENEO_BASE_URL": "https://akeneo.example",
            "PIM_AKENEO_API_KEY": "akeneo-key",
            "PIM_PLYTIX_BASE_URL": "https://plytix.example",
            "PIM_PLYTIX_API_KEY": "plytix-key",
            "PIM_CUSTOM_BASE_URL": "https://custom.example",
            "PIM_CUSTOM_API_KEY": "custom-key",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_runtime_config()

        self.assertEqual(config.pim_backend, "plytix")
        self.assertEqual(config.pim_akeneo_base_url, "https://akeneo.example")
        self.assertEqual(config.pim_plytix_api_key, "plytix-key")
        self.assertEqual(config.pim_custom_api_key, "custom-key")

    def test_runtime_config_reads_intent_router_environment(self) -> None:
        env = {
            "INTENT_CLASSIFIER_ENABLED": "true",
            "INTENT_CLASSIFIER_MODEL_PATH": "artifacts/intent_classifier_v1/final",
            "INTENT_ROUTER_LLM_FALLBACK_ENABLED": "false",
            "INTENT_ROUTER_CONFIDENCE_THRESHOLD": "0.8",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_runtime_config()

        self.assertTrue(config.intent_classifier_enabled)
        self.assertEqual(config.intent_classifier_model_path, "artifacts/intent_classifier_v1/final")
        self.assertFalse(config.intent_router_llm_fallback_enabled)
        self.assertEqual(config.intent_router_confidence_threshold, 0.8)

    def test_intent_training_config_defaults(self) -> None:
        config = TrainingConfig()

        self.assertEqual(config.model_name, "bert-base-multilingual-cased")
        self.assertEqual(config.num_labels, 3)
        self.assertEqual(config.output_dir, "artifacts/intent_classifier_v1")
        self.assertEqual(config.languages, ["en", "es", "fr", "de", "ja", "zh", "ar", "pt", "ko", "it"])

    def test_intent_dataset_builder_covers_languages_and_labels(self) -> None:
        config = TrainingConfig()
        samples = IntentDatasetBuilder.build_samples(config, include_augmentation=False)
        languages = {sample["language"] for sample in samples}
        intents = {sample["intent"] for sample in samples}

        self.assertEqual(languages, set(config.languages))
        self.assertEqual(intents, {"pre_sales", "order_fulfillment", "post_sales_support"})
        self.assertEqual(INTENT_LABELS, {"pre_sales": 0, "order_fulfillment": 1, "post_sales_support": 2})
        for language in config.languages:
            self.assertEqual(sum(1 for sample in samples if sample["language"] == language), 9)

    def test_intent_dataset_dry_run_needs_no_ml_dependencies(self) -> None:
        summary = IntentDatasetBuilder.dry_run_summary(TrainingConfig(languages=["en", "ja"]))

        self.assertEqual(summary["num_labels"], 3)
        self.assertEqual(summary["by_language"]["en"], 18)
        self.assertEqual(summary["by_language"]["ja"], 9)
        self.assertEqual(summary["by_intent"]["pre_sales"], 9)
        self.assertEqual(summary["label2id"]["order_fulfillment"], 1)

    def test_intent_classifier_tool_with_fake_pipeline(self) -> None:
        class FakeClassifier:
            def __call__(self, text):
                return [{"label": "LABEL_1", "score": 0.91}]

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            export_label_map(tmpdir)
            result = IntentClassifierTool(tmpdir, classifier=FakeClassifier()).predict(
                "Where is my order?", language="en"
            )

        self.assertEqual(result["detected_intent"], "order_fulfillment")
        self.assertEqual(result["confidence_score"], 0.91)
        self.assertFalse(result["requires_human_review"])
        self.assertEqual(result["language_detected"], "en")

    def test_support_yaml_contains_customer_service_agents_and_tasks(self) -> None:
        root = Path(__file__).resolve().parents[1]
        agents = yaml.safe_load((root / "config" / "support" / "agents.yaml").read_text(encoding="utf-8"))
        tasks = yaml.safe_load((root / "config" / "support" / "tasks.yaml").read_text(encoding="utf-8"))

        self.assertIn("pre_sales_specialist", agents)
        self.assertIn("order_fulfillment_specialist", agents)
        self.assertIn("senior_support_agent", agents)
        self.assertIn("support_qa_specialist", agents)
        self.assertIn("pre_sales_consultation", tasks)
        self.assertIn("order_status_handling", tasks)
        self.assertIn("inquiry_resolution", tasks)
        self.assertIn("quality_assurance_review", tasks)
        self.assertIn("{channel}", tasks["quality_assurance_review"]["description"])
        self.assertIn("{language_plan}", tasks["pre_sales_consultation"]["description"])
        self.assertIn("response_type", tasks["quality_assurance_review"]["description"])

    def test_intent_router_classifies_service_stages(self) -> None:
        router = IntentRouterTool()

        self.assertEqual(router._run("Which model is compatible with HomeKit?")["detected_intent"], "pre_sales")
        self.assertEqual(router._run("Where is my order?", has_order_id=True)["detected_intent"], "order_fulfillment")
        self.assertEqual(router._run("I need a refund for a damaged item")["detected_intent"], "post_sales_support")
        self.assertTrue(router._run("hello there")["requires_human_review"])

    def test_intent_router_routes_bulk_discount_inquiry_to_pre_sales(self) -> None:
        text = (
            "Hello,I want to buy 10 of V2G5 mechanical feel metal keyboard mouse "
            "headset three-piece set, can i get a discount?"
        )
        result = IntentRouterTool()._run(text)

        self.assertEqual(result["detected_intent"], "pre_sales")
        self.assertGreaterEqual(result["confidence_score"], 0.75)
        self.assertFalse(result["requires_human_review"])
        self.assertTrue(any(signal["name"] == "quantity_purchase_signal" for signal in result["routing_signals"]))

    def test_intent_router_routes_bar_laser_discount_inquiry_to_pre_sales(self) -> None:
        result = IntentRouterTool()._run("Hello,I want to buy 10 of Bar laser glasses, can i get a discount?")

        self.assertEqual(result["detected_intent"], "pre_sales")
        self.assertGreaterEqual(result["confidence_score"], 0.75)
        self.assertFalse(result["requires_human_review"])
        self.assertEqual(result["context_enrichment"]["product_category_hint"], "Bar Laser Glasses")

    def test_intent_router_catalog_match_boosts_pre_sales_signal(self) -> None:
        result = IntentRouterTool()._run("Can you share price and carton details for Bluetooth headset?")

        self.assertEqual(result["detected_intent"], "pre_sales")
        self.assertTrue(any(signal["name"] == "catalog_match" for signal in result["routing_signals"]))

    def test_intent_router_uses_fake_trained_classifier_signal(self) -> None:
        class FakeClassifier:
            def predict(self, text, language="en"):
                return {"detected_intent": "order_fulfillment", "confidence_score": 0.96}

        result = IntentRouterTool(classifier=FakeClassifier())._run("parcel timeline question")

        self.assertEqual(result["detected_intent"], "order_fulfillment")
        self.assertTrue(any(signal["name"] == "trained_classifier" for signal in result["routing_signals"]))

    def test_intent_router_llm_fallback_can_rescue_low_confidence(self) -> None:
        def fake_llm(text, config):
            return {"detected_intent": "pre_sales", "confidence_score": 0.88}

        result = classify_intent(
            "Do you have this in a good option for my shop?",
            config_context={"openai_api_key": "test-key"},
            llm_client=fake_llm,
        )

        self.assertEqual(result["detected_intent"], "pre_sales")
        self.assertFalse(result["requires_human_review"])
        self.assertTrue(result["llm_fallback_used"])

    def test_intent_router_low_confidence_without_llm_still_requires_review(self) -> None:
        result = classify_intent("hello there", config_context={"intent_router_llm_fallback_enabled": False})

        self.assertTrue(result["requires_human_review"])

    def test_pre_sales_and_order_tools_return_mock_data(self) -> None:
        product = PreSalesProductKnowledgeTool()._run("Smart Home Camera", ["camera"], "JP")
        order = OrderTrackingTool()._run("ORD-JP-2024-8842", None, "JP")

        self.assertTrue(product["product_found"])
        self.assertIn("verified_features", product)
        self.assertEqual(product["data_source"], "mock_fallback")
        self.assertTrue(order["order_found"])
        self.assertEqual(order["current_status"], "shipped")

    def test_customer_service_output_model_validates(self) -> None:
        parsed = CustomerServiceOutput.model_validate(
            {
                "session_id": "sess-1",
                "customer_context": {"name": "Maria", "tier": "STANDARD", "language": "English", "channel": "whatsapp"},
                "detected_intent": "pre_sales",
                "routing_confidence": 0.92,
                "pre_sales_response": {"feature_explanation": "Works with HomeKit."},
                "order_response": None,
                "support_response": None,
                "final_response": "The Pro model is the best fit.",
                "qa_status": "APPROVED",
                "compliance_flags": [],
                "recommended_follow_up": "Reply with your room size if you want a bundle recommendation.",
                "escalation_needed": False,
                "data_sources": ["mock_fallback"],
                "assumptions": [],
            }
        )

        self.assertEqual(parsed.detected_intent, "pre_sales")

    def test_customer_service_output_schema_has_no_open_objects(self) -> None:
        schema = CustomerServiceOutput.model_json_schema()
        missing_paths: list[str] = []

        def walk(node, path: str = "root") -> None:
            if isinstance(node, dict):
                if node.get("type") == "object" and node.get("additionalProperties") is not False:
                    missing_paths.append(path)
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        walk(schema)

        self.assertEqual(missing_paths, [])

    def test_attach_customer_service_context_fills_required_fields(self) -> None:
        inputs = _normalize_inputs(
            {
                "customer": "Maria",
                "person": "Maria",
                "inquiry": "Which camera should I buy?",
                "channel": "whatsapp",
                "session_id": "sess-1",
            }
        )
        result = _attach_customer_service_context(
            result={"final_response": "The Pro model is the best fit."},
            inputs=inputs,
            automation_context={
                "sentiment_analysis": {"requires_human_handoff": False},
                "rma_validation": None,
                "logistics_output": None,
                "escalation_flag": False,
                "compliance_tags": [],
            },
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.92},
            pre_sales_context={"data_source": "mock_fallback", "product_found": True},
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        self.assertEqual(result["final_response"], "The Pro model is the best fit.")
        self.assertEqual(result["detected_intent"], "pre_sales")
        self.assertEqual(result["qa_status"], "APPROVED")

    def test_attach_customer_service_context_preserves_usage_metrics_outside_schema_validation(self) -> None:
        inputs = _normalize_inputs(
            {
                "customer": "Alex",
                "person": "Alex",
                "inquiry": "I want to buy 10 of Bar laser glasses, can I get a discount?",
                "channel": "gmail",
                "session_id": "sess-usage",
            }
        )
        result = _attach_customer_service_context(
            result={
                "final_response": "We can help with a bulk quote.",
                "qa_status": "APPROVED",
                INTERNAL_USAGE_KEY: {"total_tokens": 12228, "successful_requests": 3},
            },
            inputs=inputs,
            automation_context={
                "sentiment_analysis": {"requires_human_handoff": False},
                "rma_validation": None,
                "logistics_output": None,
                "escalation_flag": False,
                "compliance_tags": [],
            },
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.98},
            pre_sales_context={"data_source": "mock_fallback", "product_found": True},
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        self.assertEqual(result[INTERNAL_USAGE_KEY]["total_tokens"], 12228)
        self.assertEqual(result["detected_intent"], "pre_sales")

    def test_attach_customer_service_context_removes_unapproved_catalog_prices(self) -> None:
        inputs = _normalize_inputs(
            {
                "customer": "Alex",
                "person": "Alex",
                "inquiry": "Please share M90 PRO wireless earphones bulk pricing.",
                "product_category": "M90 PRO wireless earphones",
                "channel": "gmail",
                "session_id": "sess-pricing",
            }
        )
        pre_sales_context = {
            "data_source": "mock_fallback",
            "product_found": True,
            "catalog_product_offer": {
                "status": "found",
                "product_found": True,
                "product_name": "M90 PRO wireless earphones",
                "unit_price": "$2.39",
                "carton_quantity": "100 PCS",
                "carton_size": "43.5X24X51cm",
                "carton_weight": "14.1Kg",
                "discount_policy": "Discounts require sales approval.",
                "source": "catalog.pdf",
                "heading": "Page 1",
                "evidence": ["M90 PRO wireless", "1 carton Contains 100 PCS $2.39"],
                "data_source": "local_pdf_catalog",
            },
            "pricing_guardrails": ["Do not invent discount tiers."],
        }

        result = _attach_customer_service_context(
            result={
                "final_response": "Buy 1: $12.99/ea\nBuy 2: $12.34/ea",
                "qa_status": "APPROVED",
                "escalation_needed": False,
            },
            inputs=inputs,
            automation_context={
                "sentiment_analysis": {"requires_human_handoff": False},
                "rma_validation": None,
                "logistics_output": None,
                "escalation_flag": False,
                "compliance_tags": [],
            },
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.98},
            pre_sales_context=pre_sales_context,
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        self.assertIn("$2.39", result["final_response"])
        self.assertNotIn("$12.99", result["final_response"])
        self.assertNotIn("Buy 1", result["final_response"])
        self.assertIn("sales team", result["final_response"])
        self.assertIn("review any discount request", result["final_response"])

    def test_attach_customer_service_context_removes_unverified_catalog_variants_and_features(self) -> None:
        inputs = _normalize_inputs(
            {
                "customer": "Alex",
                "person": "Alex",
                "inquiry": "Please share Wireless Bluetooth headset product specification.",
                "product_category": "Wireless Bluetooth headset",
                "channel": "gmail",
                "session_id": "sess-variant-guard",
            }
        )
        pre_sales_context = {
            "data_source": "mock_fallback",
            "product_found": True,
            "catalog_product_offer": {
                "status": "found",
                "product_found": True,
                "product_name": "B39 wireless Bluetooth headset",
                "unit_price": "$3.35",
                "carton_quantity": "80PCS",
                "carton_size": "76×42×50cm",
                "carton_weight": "24kg",
                "single_product_size": "18.5*15.8*4.6cm",
                "single_product_weight": "200g",
                "discount_policy": "Discounts require sales approval.",
                "source": "catalog.pdf",
                "heading": "Page 1",
                "evidence": ["B39 wireless Bluetooth", "1 carton contains 80pcs $3.35"],
                "data_source": "local_pdf_catalog",
            },
        }

        result = _attach_customer_service_context(
            result={
                "final_response": (
                    "Variants Available:\n"
                    "- Basic Headset (SKU: HEADSET-BASIC) - $6.30 with noise isolation\n"
                    "- Pro Headset (SKU: HEADSET-PRO) - $9.50"
                ),
                "pre_sales_response": {
                    "product_found": True,
                    "data_source": "Product catalog",
                    "variant_options": {
                        "Basic Headset": {"sku": "HEADSET-BASIC", "price": "$6.30"},
                        "Pro Headset": {"sku": "HEADSET-PRO", "price": "$9.50"},
                    },
                    "catalog_product_offer": {
                        "status": "available",
                        "product_name": "B39 Wireless Bluetooth Headset",
                        "unit_price": "$3.35",
                    },
                },
                "qa_status": "APPROVED",
                "escalation_needed": False,
            },
            inputs=inputs,
            automation_context={
                "sentiment_analysis": {"requires_human_handoff": False},
                "rma_validation": None,
                "logistics_output": None,
                "escalation_flag": False,
                "compliance_tags": [],
            },
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.98},
            pre_sales_context=pre_sales_context,
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        self.assertIn("B39 wireless Bluetooth headset", result["final_response"])
        self.assertIn("$3.35", result["final_response"])
        self.assertIn("80PCS", result["final_response"])
        self.assertIn("18.5*15.8*4.6cm", result["final_response"])
        self.assertNotIn("HEADSET-BASIC", result["final_response"])
        self.assertNotIn("$6.30", result["final_response"])
        self.assertNotIn("noise isolation", result["final_response"].lower())
        self.assertEqual(result["pre_sales_response"]["catalog_product_offer"]["status"], "found")
        self.assertEqual(result["pre_sales_response"]["variant_options"], [])
        self.assertIn("UNVERIFIED_PRODUCT_FACT_REWRITTEN", result["compliance_flags"])

    def test_attach_customer_service_context_rewrites_wireless_headset_specs_and_auto_approves(self) -> None:
        inputs = _normalize_inputs(
            {
                "customer": "Alex",
                "person": "Alex",
                "inquiry": "Please share Wireless Bluetooth headset product specification.",
                "product_category": "Wireless Bluetooth headset",
                "channel": "gmail",
                "session_id": "sess-wireless-spec",
            }
        )
        pre_sales_context = {
            "data_source": "mock_fallback",
            "product_found": True,
            "catalog_product_offer": {
                "status": "found",
                "product_found": True,
                "product_name": "Wireless Bluetooth headset",
                "unit_price": "$6.50",
                "carton_quantity": "40PCS",
                "carton_size": "52x38x51cm",
                "carton_weight": "9 kg",
                "single_product_size": None,
                "single_product_weight": None,
                "discount_policy": "Discounts require sales approval.",
                "source": "catalog.pdf",
                "heading": "Page 1",
                "evidence": ["Wireless Bluetooth One carton contains 40PCS", "$6.50"],
                "data_source": "local_pdf_catalog",
            },
        }

        result = _attach_customer_service_context(
            result={
                "final_response": (
                    "The catalog price is $6.50 per headset, and they are sold in cartons of 40. "
                    "Please note that detailed specifications are not available in the catalog."
                ),
                "pre_sales_response": {
                    "product_found": None,
                    "requires_human_review": True,
                    "product_recommendation": "Wireless Bluetooth headset",
                },
                "qa_status": "REVIEW_REQUIRED",
                "escalation_needed": True,
            },
            inputs=inputs,
            automation_context={
                "sentiment_analysis": {"requires_human_handoff": False},
                "rma_validation": None,
                "logistics_output": None,
                "escalation_flag": False,
                "compliance_tags": [],
            },
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.95},
            pre_sales_context=pre_sales_context,
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        self.assertEqual(result["qa_status"], "APPROVED")
        self.assertFalse(result["escalation_needed"])
        self.assertTrue(result["pre_sales_response"]["product_found"])
        self.assertFalse(result["pre_sales_response"]["requires_human_review"])
        self.assertIn("$6.50", result["final_response"])
        self.assertIn("40PCS", result["final_response"])
        self.assertIn("52x38x51cm", result["final_response"])
        self.assertIn("9 kg", result["final_response"])
        self.assertNotIn("detailed specifications are not available", result["final_response"].lower())

    def test_sync_job_result_accepts_customer_service_output(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.92,
                    "final_response": "The Pro model is the best fit.",
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                },
            },
        )

        self.assertEqual(conversation.draft_response, "The Pro model is the best fit.")
        self.assertFalse(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)

    def test_sync_job_result_does_not_handoff_approved_pre_sales_even_if_model_flags_escalation(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.91,
                    "final_response": "The verified catalog price is available, and sales can help complete the purchase.",
                    "qa_status": "APPROVED",
                    "escalation_needed": True,
                    "pre_sales_response": {"product_found": True},
                },
            },
        )

        self.assertEqual(conversation.status, "draft_ready")
        self.assertFalse(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)

    def test_sync_job_result_auto_allows_low_risk_pre_sales_even_if_qa_requests_review(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.95,
                    "final_response": "The headset is available with verified product specifications and purchase guidance.",
                    "qa_status": "REVIEW_REQUIRED",
                    "escalation_needed": True,
                    "compliance_flags": ["GDPR_COMPLIANT_REVIEW", "CCPA_OPT_OUT_AVAILABLE"],
                    "pre_sales_response": {"requires_human_review": True},
                },
            },
        )

        self.assertEqual(conversation.status, "draft_ready")
        self.assertFalse(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)

    def test_sync_job_result_requires_approval_for_low_confidence_pre_sales_without_handoff(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            draft_response=None,
            draft_payload=None,
            requires_approval=False,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.64,
                    "final_response": "A specialist will help.",
                    "qa_status": "REVIEW_REQUIRED",
                    "escalation_needed": True,
                },
            },
        )

        self.assertEqual(conversation.status, "draft_ready")
        self.assertTrue(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)

    def test_sync_job_result_handoffs_pre_sales_with_hard_handoff_flag(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            draft_response=None,
            draft_payload=None,
            requires_approval=False,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.95,
                    "final_response": "A specialist will help.",
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                    "compliance_flags": ["HUMAN_HANDOFF"],
                },
            },
        )

        self.assertEqual(conversation.status, "handoff_required")
        self.assertTrue(conversation.requires_approval)
        self.assertTrue(conversation.escalation_flag)

    def test_service_inquiry_endpoint_queues_support_job(self) -> None:
        class FakeOrchestrator:
            registered_workflows = [WorkflowType.SUPPORT]

            async def submit_job(self, workflow_type, inputs, provider_credentials=None, metadata=None):
                self.workflow_type = workflow_type
                self.inputs = inputs
                self.metadata = metadata
                return "job-123"

            def get_job_status(self, job_id):
                return {"job_id": job_id, "status": "pending", "result": None}

            def get_job_events(self, job_id):
                return []

        orchestrator = FakeOrchestrator()
        app = FastAPI()
        app.include_router(create_router(orchestrator))
        client = TestClient(app)
        response = client.post(
            "/api/v1/service/inquiry",
            json={"customer": "Maria", "inquiry": "Which camera works with HomeKit?", "channel": "whatsapp"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["job_id"], "job-123")
        self.assertEqual(orchestrator.workflow_type, WorkflowType.SUPPORT)
        self.assertEqual(orchestrator.inputs["customer"], "Maria")


if __name__ == "__main__":
    unittest.main()
