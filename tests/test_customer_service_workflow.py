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
from crews.support_crew import CustomerServiceOutput, _attach_customer_service_context, _build_automation_context, _normalize_inputs
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
from support_inbox import SupportInboxStore
from tests.test_whatsapp_omnichannel import FakeSupportSession
from tools.custom.customer_service_tools import IntentRouterTool, OrderTrackingTool, PreSalesProductKnowledgeTool


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

    def test_sync_job_result_escalates_customer_service_output(self) -> None:
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
