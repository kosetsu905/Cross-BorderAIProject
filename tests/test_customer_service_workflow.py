import asyncio
import json
import os
import tempfile
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
    _attach_order_knowledge_context,
    _build_automation_context,
    _build_support_tools,
    _catalog_display_name,
    _normalize_inputs,
    _normalize_order_response,
    _parse_local_tracking_record,
    _rewrite_response_language,
    _should_skip_support_llm_qa,
    run_support_crew,
)
from job_store import InMemoryJobStore
from models import WorkflowType
from runtime_config import apply_runtime_environment, load_runtime_config
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
from utils.workflow_progress import WorkflowProgressRecorder
from tools.custom.customer_service_tools import IntentRouterTool, OrderTrackingTool, PreSalesProductKnowledgeTool
from tools.custom.support_automation_tools import process_rma_request
from tools.custom.support_rag_tools import extract_catalog_product_offer, load_knowledge_chunks, search_knowledge_base
from tools.custom.support_search_tools import build_support_external_search_tools


SUPPORT_FIXTURE_DIR = Path(__file__).parent / "fixtures"
WIRELESS_HEADSET_PRE_SALES_FIXTURE = SUPPORT_FIXTURE_DIR / "support_wireless_headset_pre_sales.json"


def _wireless_headset_pre_sales_fixture() -> dict[str, object]:
    return json.loads(WIRELESS_HEADSET_PRE_SALES_FIXTURE.read_text(encoding="utf-8"))


def _fenced_json_payload(payload: dict[str, object]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


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
            self.assertEqual(result["detected_language"], "en")
            self.assertEqual(result["language_plan"], "English")

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

    def test_rma_denies_worn_hygiene_sensitive_earrings(self) -> None:
        result = process_rma_request(
            order_id="ORDER-NOT-PROVIDED",
            item_sku="SKU-NOT-PROVIDED",
            return_reason=(
                "Hello, I bought an earring, and I have already wear it, but I think "
                "it's not pretty. Therefore I want to refund it."
            ),
            detected_language="en",
            order_history={},
        )

        self.assertFalse(result["rma_validation"]["eligible_for_return"])
        self.assertIsNone(result["logistics_output"])
        self.assertIn("Hygiene-sensitive", result["rma_validation"]["eligibility_reason"])

    def test_rma_denies_tried_on_intimate_change_of_mind_item(self) -> None:
        result = process_rma_request(
            order_id="ORD-US-2026-1001",
            item_sku="UNDERWEAR-BLK",
            return_reason="I tried on the underwear and changed my mind. I want to return it.",
            detected_language="en",
            order_history={"days_since_delivery": 3, "item_condition": "unopened", "region": "US"},
        )

        self.assertFalse(result["rma_validation"]["eligible_for_return"])
        self.assertIsNone(result["logistics_output"])

    def test_rma_denies_worn_bra_change_of_mind_item(self) -> None:
        result = process_rma_request(
            order_id="ORD-US-2026-1004",
            item_sku="BRA-BLK",
            return_reason=(
                "Hello,I bought a bra, and I have already worn it, but I think "
                "it's not pretty. Therefore I want to refund it."
            ),
            detected_language="en",
            order_history={"days_since_delivery": 3, "item_condition": "unopened", "region": "US"},
        )

        self.assertFalse(result["rma_validation"]["eligible_for_return"])
        self.assertIsNone(result["logistics_output"])
        self.assertIn("Hygiene-sensitive", result["rma_validation"]["eligibility_reason"])

    def test_rma_allows_defective_hygiene_sensitive_item_with_real_order_details(self) -> None:
        result = process_rma_request(
            order_id="ORD-US-2026-1002",
            item_sku="EARRING-GOLD",
            return_reason="The earring is broken and defective.",
            detected_language="en",
            order_history={"days_since_delivery": 3, "item_condition": "defective", "region": "US"},
        )

        self.assertTrue(result["rma_validation"]["eligible_for_return"])
        self.assertIsNotNone(result["logistics_output"])
        self.assertEqual(result["rma_validation"]["return_shipping_responsibility"], "BRAND")

    def test_rma_allows_unopened_regular_change_of_mind_return_with_customer_shipping(self) -> None:
        result = process_rma_request(
            order_id="ORD-US-2026-1003",
            item_sku="PHONE-CASE",
            return_reason="I changed my mind and want to return this unopened phone case.",
            detected_language="en",
            order_history={"days_since_delivery": 6, "item_condition": "unopened", "region": "US"},
        )

        self.assertTrue(result["rma_validation"]["eligible_for_return"])
        self.assertEqual(result["rma_validation"]["return_shipping_responsibility"], "CUSTOMER")

    def test_rma_placeholder_order_does_not_generate_customer_visible_label(self) -> None:
        result = process_rma_request(
            order_id="ORDER-NOT-PROVIDED",
            item_sku="SKU-NOT-PROVIDED",
            return_reason="I changed my mind and the item is unopened.",
            detected_language="en",
            order_history={"days_since_delivery": 4, "item_condition": "unopened", "region": "US"},
        )

        self.assertTrue(result["rma_validation"]["eligible_for_return"])
        self.assertIsNone(result["logistics_output"])

    def test_post_sales_guard_rewrites_ineligible_rma_fake_label_response(self) -> None:
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": (
                    "Hello, I bought an earring, and I have already wear it, but I think "
                    "it's not pretty. Therefore I want to refund it."
                ),
                "channel": "gmail",
                "session_id": "sess-rma-denial",
            }
        )
        automation_context = _build_automation_context(inputs)

        result = _attach_customer_service_context(
            result={
                "final_response": (
                    "Since your item falls within our 30-day return window, we can proceed with the return. "
                    "Please print the prepaid return label: "
                    "https://labels.example.local/easypost/return/ORDER-NOT-PROVIDED_SKU-NOT-PROVIDED.pdf"
                ),
                "qa_status": "APPROVED",
                "escalation_needed": False,
            },
            inputs=inputs,
            automation_context=automation_context,
            router_result={"detected_intent": "post_sales_support", "confidence_score": 0.94},
            pre_sales_context={},
            order_context={},
        )

        self.assertFalse(automation_context["rma_validation"]["eligible_for_return"])
        self.assertIn("RMA_POLICY_RESPONSE_REWRITTEN", result["compliance_flags"])
        self.assertNotIn("labels.example.local", result["final_response"])
        self.assertNotIn("proceed with the return", result["final_response"].lower())
        self.assertIn("not able to approve", result["final_response"])

    def test_adaptive_fast_support_qa_skips_second_llm_for_low_risk_post_sales(self) -> None:
        task_counts: list[int] = []

        class FakeCrew:
            def __init__(self, agents, tasks, verbose=False, memory=False):
                task_counts.append(len(tasks))

            def kickoff(self, inputs):
                return SimpleNamespace(
                    json_dict={
                        "final_response": (
                            "We can help with the damaged item. Please print the prepaid return label: "
                            "https://labels.example.local/easypost/return/ORDER-NOT-PROVIDED_SKU-NOT-PROVIDED.pdf"
                        ),
                        "qa_status": "APPROVED",
                        "escalation_needed": False,
                    }
                )

        with patch("crews.support_crew.Crew", FakeCrew):
            result = run_support_crew(
                {
                    "customer": "Alex",
                    "person": "Alex",
                    "inquiry": "I want to return an unopened item.",
                    "channel": "gmail",
                },
                {"support_qa_mode": "adaptive_fast"},
            )

        CustomerServiceOutput.model_validate(result)
        self.assertEqual(task_counts, [1])
        self.assertEqual(result["detected_intent"], "post_sales_support")
        self.assertIn("LLM_QA_SKIPPED_ADAPTIVE_FAST", result["compliance_flags"])
        self.assertNotIn("labels.example.local", result["final_response"])
        self.assertEqual(result["qa_status"], "REVIEW_REQUIRED")

    def test_adaptive_fast_support_qa_maps_stage_fields_for_worn_bra_refund(self) -> None:
        task_counts: list[int] = []
        inquiry = (
            "Hello,I bought a bra, and I have already worn it, but I think "
            "it's not pretty. Therefore I want to refund it."
        )

        class FakeCrew:
            def __init__(self, agents, tasks, verbose=False, memory=False):
                task_counts.append(len(tasks))

            def kickoff(self, inputs):
                return SimpleNamespace(
                    json_dict={
                        "response_type": "post_sales_support",
                        "issue_category": "worn_intimate_apparel_return",
                        "resolution_steps": ["Explain the worn intimate apparel return exclusion."],
                        "policy_reference": "Return policy: hygiene-sensitive items exclusion.",
                        "compensation_offered": "none",
                        "follow_up_required": True,
                        "final_response": (
                            "Since your item falls within our 30-day return window, we can proceed with the return. "
                            "Please print the prepaid return label: "
                            "https://labels.example.local/easypost/return/ORDER-NOT-PROVIDED_SKU-NOT-PROVIDED.pdf"
                        ),
                    }
                )

        with patch("crews.support_crew.Crew", FakeCrew):
            result = run_support_crew(
                {
                    "customer": "Gmail Customer",
                    "person": "Gmail Customer",
                    "inquiry": inquiry,
                    "channel": "gmail",
                },
                {"support_qa_mode": "adaptive_fast"},
            )

        CustomerServiceOutput.model_validate(result)
        self.assertEqual(task_counts, [1])
        self.assertEqual(result["detected_intent"], "post_sales_support")
        self.assertEqual(result["qa_status"], "REVIEW_REQUIRED")
        self.assertIn("RMA_POLICY_RESPONSE_REWRITTEN", result["compliance_flags"])
        self.assertNotIn("labels.example.local", result["final_response"])
        self.assertIn("not able to approve", result["final_response"])
        self.assertIn("worn_intimate_apparel_return", result["support_response"]["internal_notes"])
        self.assertEqual(
            result["support_response"]["resolution_steps"],
            ["Explain the worn intimate apparel return exclusion."],
        )

    def test_adaptive_fast_support_qa_skips_second_llm_for_low_risk_pre_sales(self) -> None:
        task_counts: list[int] = []

        class FakeCrew:
            def __init__(self, agents, tasks, verbose=False, memory=False):
                task_counts.append(len(tasks))

            def kickoff(self, inputs):
                return SimpleNamespace(
                    json_dict={
                        "response_type": "pre_sales",
                        "product_recommendation": "M90 PRO wireless earphones",
                        "feature_explanation": "Catalog facts are available for carton pricing.",
                        "comparison_summary": "Use the catalog unit price for current known pricing.",
                        "next_steps": ["Confirm quantity", "Ask sales to review any discount request"],
                        "confidence_level": 0.92,
                        "requires_human_review": False,
                        "final_response": (
                            "Buy 1: $12.99/ea\n"
                            "Buy 2: $12.34/ea\n"
                            "SKU: HEADSET-BASIC includes noise isolation."
                        ),
                    }
                )

        with patch("crews.support_crew.Crew", FakeCrew):
            result = run_support_crew(
                {
                    "customer": "Alex",
                    "person": "Alex",
                    "inquiry": "Please share M90 PRO wireless earphones bulk pricing.",
                    "product_category": "M90 PRO wireless earphones",
                    "channel": "gmail",
                },
                {"support_qa_mode": "adaptive_fast"},
            )

        CustomerServiceOutput.model_validate(result)
        self.assertEqual(task_counts, [1])
        self.assertEqual(result["detected_intent"], "pre_sales")
        self.assertEqual(result["qa_status"], "APPROVED")
        self.assertFalse(result["escalation_needed"])
        self.assertIn("LLM_QA_SKIPPED_ADAPTIVE_FAST", result["compliance_flags"])
        self.assertIn("UNVERIFIED_PRODUCT_FACT_REWRITTEN", result["compliance_flags"])
        self.assertEqual(result["pre_sales_response"]["product_recommendation"], "M90 PRO wireless earphones")
        self.assertEqual(
            result["pre_sales_response"]["next_steps"],
            ["Confirm quantity", "Ask sales to review any discount request"],
        )
        self.assertEqual(result["pre_sales_response"]["catalog_product_offer"]["status"], "found")
        self.assertNotIn("$12.99", result["final_response"])
        self.assertNotIn("HEADSET-BASIC", result["final_response"])
        self.assertNotIn("noise isolation", result["final_response"].lower())

    def test_run_support_crew_normalizes_object_product_recommendation_from_json_dict(self) -> None:
        crew_payload = _wireless_headset_pre_sales_fixture()

        class FakeCrew:
            def __init__(self, agents, tasks, **kwargs):
                self.task_count = len(tasks)

            def kickoff(self, inputs):
                return SimpleNamespace(json_dict=json.loads(json.dumps(crew_payload)))

        with patch("crews.support_crew.Crew", FakeCrew):
            result = run_support_crew(
                {
                    "customer": "Tonny",
                    "person": "Tonny",
                    "inquiry": "Please share Wireless Bluetooth headset bulk pricing.",
                    "product_category": "Wireless Bluetooth headset",
                    "channel": "gmail",
                    "session_id": "conv-wireless-headset",
                },
                {"support_qa_mode": "adaptive_fast"},
            )

        CustomerServiceOutput.model_validate(result)
        pre_sales = result["pre_sales_response"]
        self.assertEqual(pre_sales["product_recommendation"], "Wireless Bluetooth headset")
        self.assertEqual(pre_sales["catalog_product_offer"]["unit_price"], "$6.50")
        self.assertEqual(pre_sales["catalog_product_offer"]["carton_quantity"], "40PCS")
        self.assertIn("$6.50", result["final_response"])
        self.assertNotIn('"product_recommendation"', result["final_response"])

    def test_run_support_crew_normalizes_object_product_recommendation_from_fenced_raw_json(self) -> None:
        crew_payload = _wireless_headset_pre_sales_fixture()
        raw_payload = _fenced_json_payload(crew_payload)

        class FakeCrew:
            def __init__(self, agents, tasks, **kwargs):
                self.task_count = len(tasks)

            def kickoff(self, inputs):
                return SimpleNamespace(raw=raw_payload)

        with patch("crews.support_crew.Crew", FakeCrew):
            result = run_support_crew(
                {
                    "customer": "Tonny",
                    "person": "Tonny",
                    "inquiry": "Please share Wireless Bluetooth headset bulk pricing.",
                    "product_category": "Wireless Bluetooth headset",
                    "channel": "gmail",
                    "session_id": "conv-wireless-headset",
                },
                {"support_qa_mode": "adaptive_fast"},
            )

        CustomerServiceOutput.model_validate(result)
        pre_sales = result["pre_sales_response"]
        self.assertEqual(pre_sales["product_recommendation"], "Wireless Bluetooth headset")
        self.assertEqual(pre_sales["catalog_product_offer"]["unit_price"], "$6.50")
        self.assertEqual(pre_sales["catalog_product_offer"]["carton_size"], "52x38x51cm")
        self.assertIn("40PCS", result["final_response"])
        self.assertNotIn("```json", result["final_response"])

    def test_adaptive_fast_support_qa_keeps_full_qa_for_unverified_pre_sales_context(self) -> None:
        task_counts: list[int] = []

        class FakeCrew:
            def __init__(self, agents, tasks, verbose=False, memory=False):
                task_counts.append(len(tasks))

            def kickoff(self, inputs):
                return SimpleNamespace(
                    json_dict={
                        "final_response": "A specialist can confirm this product before quoting details.",
                        "qa_status": "REVIEW_REQUIRED",
                        "escalation_needed": True,
                    }
                )

        with tempfile.TemporaryDirectory() as knowledge_dir:
            with patch("crews.support_crew.Crew", FakeCrew):
                result = run_support_crew(
                    {
                        "customer": "Alex",
                        "person": "Alex",
                        "inquiry": "Please compare Foobar Quantum Adapter models for my shop.",
                        "product_category": "Foobar Quantum Adapter",
                        "channel": "gmail",
                    },
                    {
                        "support_qa_mode": "adaptive_fast",
                        "support_knowledge_dir": knowledge_dir,
                    },
                )

        CustomerServiceOutput.model_validate(result)
        self.assertEqual(task_counts, [2])
        self.assertEqual(result["detected_intent"], "pre_sales")
        self.assertNotIn("LLM_QA_SKIPPED_ADAPTIVE_FAST", result["compliance_flags"])

    def test_adaptive_fast_support_qa_keeps_full_qa_for_pre_sales_risk_signals(self) -> None:
        low_risk_context = {
            "data_source": "akeneo",
            "product_found": True,
            "verified_features": ["Verified feature"],
        }
        base_inputs = {"attachments": []}
        base_context = {
            "sentiment_analysis": {
                "requires_human_handoff": False,
                "customer_tier": "STANDARD",
                "sentiment_score": 0.1,
                "sentiment_label": "NEUTRAL",
                "intent_category": "GENERAL",
            },
            "escalation_flag": False,
            "compliance_tags": [],
        }
        scenarios = [
            (
                "vip",
                base_inputs,
                {
                    **base_context,
                    "sentiment_analysis": {
                        **base_context["sentiment_analysis"],
                        "customer_tier": "VIP",
                    },
                },
                {"detected_intent": "pre_sales", "confidence_score": 0.95},
            ),
            (
                "negative",
                base_inputs,
                {
                    **base_context,
                    "sentiment_analysis": {
                        **base_context["sentiment_analysis"],
                        "sentiment_score": -0.4,
                    },
                },
                {"detected_intent": "pre_sales", "confidence_score": 0.95},
            ),
            (
                "low_confidence",
                base_inputs,
                base_context,
                {"detected_intent": "pre_sales", "confidence_score": 0.84},
            ),
            (
                "handoff",
                base_inputs,
                {
                    **base_context,
                    "sentiment_analysis": {
                        **base_context["sentiment_analysis"],
                        "requires_human_handoff": True,
                    },
                },
                {"detected_intent": "pre_sales", "confidence_score": 0.95},
            ),
            (
                "attachments",
                {"attachments": [{"filename": "quote.pdf"}]},
                base_context,
                {"detected_intent": "pre_sales", "confidence_score": 0.95},
            ),
            (
                "hard_flag",
                base_inputs,
                {**base_context, "compliance_tags": ["POLICY_GAP"]},
                {"detected_intent": "pre_sales", "confidence_score": 0.95},
            ),
        ]

        for name, inputs, automation_context, router_result in scenarios:
            with self.subTest(name=name):
                self.assertFalse(
                    _should_skip_support_llm_qa(
                        inputs=inputs,
                        automation_context=automation_context,
                        router_result=router_result,
                        pre_sales_context=low_risk_context,
                        config_context={"support_qa_mode": "adaptive_fast"},
                    )
                )

    def test_adaptive_fast_support_qa_keeps_full_qa_for_vip_post_sales(self) -> None:
        task_counts: list[int] = []

        class FakeCrew:
            def __init__(self, agents, tasks, verbose=False, memory=False):
                task_counts.append(len(tasks))

            def kickoff(self, inputs):
                return SimpleNamespace(
                    json_dict={
                        "final_response": "A human support lead should review this VIP refund request.",
                        "qa_status": "REVIEW_REQUIRED",
                        "escalation_needed": True,
                    }
                )

        with patch("crews.support_crew.Crew", FakeCrew):
            result = run_support_crew(
                {
                    "customer": "Enterprise Buyer",
                    "person": "Alex",
                    "customer_email": "alex@enterprise.com",
                    "inquiry": "I need a refund for a damaged item.",
                    "channel": "gmail",
                    "order_history": {"lifetime_value": 6000},
                },
                {"support_qa_mode": "adaptive_fast"},
            )

        CustomerServiceOutput.model_validate(result)
        self.assertEqual(task_counts, [2])
        self.assertNotIn("LLM_QA_SKIPPED_ADAPTIVE_FAST", result["compliance_flags"])
        self.assertTrue(result["escalation_needed"])

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

    def test_order_context_includes_local_pdf_tracking_knowledge(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Maria",
                "person": "Maria",
                "inquiry": "Can you help me check tracking number C88943021?",
                "order_id": "C88943021",
                "customer_email": "maria@example.com",
                "channel": "gmail",
            }
        )
        context = {"order_found": False, "data_source": "mock_order_db"}
        load_knowledge_chunks.cache_clear()

        _attach_order_knowledge_context(context, inputs, {"support_knowledge_dir": str(knowledge_dir)})
        normalized = _normalize_order_response(context)

        self.assertEqual(context["knowledge_data_source"], "local_order_knowledge_base")
        self.assertTrue(context["tracking_record_found"])
        self.assertTrue(context["order_knowledge_results"])
        self.assertTrue(any("413440868" in item["source"] for item in context["order_knowledge_results"]))
        self.assertEqual(normalized["knowledge_data_source"], "local_order_knowledge_base")
        self.assertTrue(normalized["order_knowledge_results"])
        self.assertEqual(normalized["tracking_lookup_status"], "found")

    def test_local_tracking_pdf_extracts_tracking_number_details(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        load_knowledge_chunks.cache_clear()
        chunks = load_knowledge_chunks(str(knowledge_dir))
        content = next(chunk.content for chunk in chunks if "413440868" in chunk.source)

        record = _parse_local_tracking_record(content, "413440868-Tracking-Details.pdf")

        self.assertEqual(record["tracking_number"], "C88943021")
        self.assertEqual(record["reference_number"], "120399587991")
        self.assertEqual(record["last_status"], "Successfully Delivered")
        self.assertEqual(record["last_status_date"], "13th May 2026")
        self.assertEqual(record["booking_date"], "10th May 2026")
        self.assertEqual(record["origin"], "Chennai")
        self.assertEqual(record["destination"], "Mumbai 400063")
        self.assertEqual(record["pieces"], 1)
        self.assertEqual(record["service_type"], "Lite")
        self.assertEqual(record["package_contents"], "Documents")
        self.assertTrue(any(event["activity"] == "Successfully Delivered" for event in record["tracking_history"]))

    def test_order_context_matches_local_pdf_by_reference_number(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "I want to track my package, the reference number is 120399587991.",
                "channel": "gmail",
            }
        )
        context = {"order_found": False, "data_source": "mock_order_db"}
        load_knowledge_chunks.cache_clear()

        _attach_order_knowledge_context(context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        self.assertTrue(context["tracking_record_found"])
        self.assertEqual(context["local_tracking_record"]["tracking_number"], "C88943021")
        self.assertEqual(context["local_tracking_record"]["reference_number"], "120399587991")

    def test_order_context_wrong_tracking_number_does_not_expose_pdf_facts(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "I want to track my package, the tracking number is C99943021.",
                "channel": "gmail",
            }
        )
        context = {"order_found": False, "data_source": "mock_order_db"}
        load_knowledge_chunks.cache_clear()

        _attach_order_knowledge_context(context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        self.assertFalse(context["tracking_record_found"])
        self.assertEqual(context["tracking_lookup_status"], "not_found")
        self.assertEqual(context["tracking_lookup_query"], ["C99943021"])
        self.assertNotIn("order_knowledge_results", context)
        self.assertNotIn("local_tracking_record", context)

    def test_order_context_wrong_reference_number_does_not_expose_pdf_facts(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "I want to track my package, the reference number is 120399587990.",
                "channel": "gmail",
            }
        )
        context = {"order_found": False, "data_source": "mock_order_db"}
        load_knowledge_chunks.cache_clear()

        _attach_order_knowledge_context(context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        self.assertFalse(context["tracking_record_found"])
        self.assertEqual(context["tracking_lookup_status"], "not_found")
        self.assertEqual(context["tracking_lookup_query"], ["120399587990"])
        self.assertNotIn("order_knowledge_results", context)
        self.assertNotIn("local_tracking_record", context)

    def test_order_context_tracking_record_overrides_missing_order_framing(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "I want to track my package, the tracking number is C88943021. Please provide me more information.",
                "channel": "gmail",
                "session_id": "sess-tracking-local",
            }
        )
        order_context = {"order_found": False, "data_source": "mock_order_db"}
        load_knowledge_chunks.cache_clear()
        _attach_order_knowledge_context(order_context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        result = _attach_customer_service_context(
            result={
                "final_response": "Unfortunately, we could not find an order associated with tracking number C88943021.",
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
            router_result={"detected_intent": "order_fulfillment", "confidence_score": 0.95},
            pre_sales_context={"data_source": "mock_fallback"},
            order_context=order_context,
        )

        self.assertTrue(result["order_response"]["tracking_record_found"])
        self.assertEqual(result["order_response"]["local_tracking_record"]["tracking_number"], "C88943021")
        self.assertEqual(result["order_response"]["local_tracking_record"]["reference_number"], "120399587991")
        self.assertIn("I found the tracking record", result["final_response"])
        self.assertIn("Reference No.: 120399587991", result["final_response"])
        self.assertIn("Successfully Delivered", result["final_response"])
        self.assertNotIn("could not find an order", result["final_response"].lower())
        self.assertIn("LOCAL_TRACKING_RECORD_REWRITTEN", result["compliance_flags"])

    def test_order_tracking_wrong_number_rewrites_to_japanese_not_found_response(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "こんにちは。荷物の追跡を希望します。追跡番号はC99943021です。詳しい情報をお知らせください。",
                "channel": "gmail",
                "session_id": "sess-tracking-ja-miss",
            }
        )
        order_context = {"order_found": False, "data_source": "mock_order_db"}
        load_knowledge_chunks.cache_clear()
        _attach_order_knowledge_context(order_context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        result = _attach_customer_service_context(
            result={
                "final_response": "According to our local tracking document, C88943021 was successfully delivered.",
                "qa_status": "REVIEW_REQUIRED",
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
            router_result={"detected_intent": "order_fulfillment", "confidence_score": 0.95},
            pre_sales_context={"data_source": "mock_fallback"},
            order_context=order_context,
        )

        self.assertEqual(result["detected_language"], "ja")
        self.assertEqual(result["language_plan"], "Japanese")
        self.assertFalse(result["order_response"]["tracking_record_found"])
        self.assertEqual(result["order_response"]["tracking_lookup_status"], "not_found")
        self.assertIn("C99943021", result["final_response"])
        self.assertIn("見つかりませんでした", result["final_response"])
        self.assertNotIn("C88943021", result["final_response"])
        self.assertNotIn("120399587991", result["final_response"])
        self.assertIn("LOCAL_TRACKING_RECORD_REWRITTEN", result["compliance_flags"])

    def test_order_tracking_found_response_uses_japanese_safe_fallback(self) -> None:
        knowledge_dir = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "こんにちは。荷物の追跡を希望します。追跡番号はC88943021です。詳しい情報をお知らせください。",
                "channel": "gmail",
                "session_id": "sess-tracking-ja-hit",
            }
        )
        order_context = {"order_found": False, "data_source": "mock_order_db"}
        load_knowledge_chunks.cache_clear()
        _attach_order_knowledge_context(order_context, inputs, {"support_knowledge_dir": str(knowledge_dir)})

        result = _attach_customer_service_context(
            result={
                "final_response": "Tracking C88943021 was successfully delivered.",
                "qa_status": "REVIEW_REQUIRED",
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
            router_result={"detected_intent": "order_fulfillment", "confidence_score": 0.95},
            pre_sales_context={"data_source": "mock_fallback"},
            order_context=order_context,
        )

        self.assertEqual(result["detected_language"], "ja")
        self.assertEqual(result["language_plan"], "Japanese")
        self.assertTrue(result["order_response"]["tracking_record_found"])
        self.assertIn("C88943021", result["final_response"])
        self.assertIn("120399587991", result["final_response"])
        self.assertIn("Successfully Delivered", result["final_response"])
        self.assertIn("こんにちは", result["final_response"])
        self.assertNotIn("Thank you for reaching out", result["final_response"])

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

    def test_runtime_config_reads_support_serper_stage_switches(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            default_config = load_runtime_config()

        self.assertFalse(default_config.support_serper_pre_sales_enabled)
        self.assertFalse(default_config.support_serper_order_fulfillment_enabled)
        self.assertFalse(default_config.support_serper_post_sales_enabled)

        env = {
            "SUPPORT_SERPER_PRE_SALES_ENABLED": "true",
            "SUPPORT_SERPER_ORDER_FULFILLMENT_ENABLED": "1",
            "SUPPORT_SERPER_POST_SALES_ENABLED": "yes",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()

        self.assertTrue(config.support_serper_pre_sales_enabled)
        self.assertTrue(config.support_serper_order_fulfillment_enabled)
        self.assertTrue(config.support_serper_post_sales_enabled)

    def test_support_external_search_tools_default_disabled_even_with_serper_key(self) -> None:
        config_context = {"serper_api_key": "serper-key"}

        self.assertEqual(build_support_external_search_tools("pre_sales", config_context), [])
        self.assertEqual(build_support_external_search_tools("order_fulfillment", config_context), [])
        self.assertEqual(build_support_external_search_tools("post_sales_support", config_context), [])

    def test_support_external_search_tools_are_stage_specific(self) -> None:
        class FakeSerperTool:
            pass

        config_context = {
            "serper_api_key": "serper-key",
            "support_serper_pre_sales_enabled": True,
            "support_serper_order_fulfillment_enabled": False,
            "support_serper_post_sales_enabled": True,
        }

        with patch("tools.custom.support_search_tools.SerperDevTool", FakeSerperTool):
            pre_sales_tools = build_support_external_search_tools("pre_sales", config_context)
            order_tools = build_support_external_search_tools("order_fulfillment", config_context)
            post_sales_tools = build_support_external_search_tools("post_sales_support", config_context)

        self.assertEqual([type(tool) for tool in pre_sales_tools], [FakeSerperTool])
        self.assertEqual(order_tools, [])
        self.assertEqual([type(tool) for tool in post_sales_tools], [FakeSerperTool])

    def test_support_external_search_tools_need_serper_key(self) -> None:
        config_context = {
            "support_serper_pre_sales_enabled": True,
            "support_serper_order_fulfillment_enabled": True,
            "support_serper_post_sales_enabled": True,
        }

        self.assertEqual(build_support_external_search_tools("pre_sales", config_context), [])
        self.assertEqual(build_support_external_search_tools("order_fulfillment", config_context), [])
        self.assertEqual(build_support_external_search_tools("post_sales_support", config_context), [])

    def test_post_sales_support_tools_keep_local_knowledge_before_optional_search(self) -> None:
        search_tool = object()
        with patch("crews.support_crew.build_support_external_search_tools", return_value=[search_tool]):
            tools = _build_support_tools({})

        self.assertEqual(tools[0].name, "Support Knowledge Base Search")
        self.assertIs(tools[1], search_tool)

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

    def test_runtime_config_prefers_generic_llm_environment(self) -> None:
        env = {
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "openrouter-key",
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_MODEL_NAME": "openai/gpt-4o-mini",
            "LLM_DISABLE_REASONING": "true",
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_MODEL_NAME": "gpt-4o-mini",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_runtime_config()

        self.assertEqual(config.llm_provider, "openrouter")
        self.assertEqual(config.llm_api_key, "openrouter-key")
        self.assertEqual(config.llm_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.llm_model_name, "openai/gpt-4o-mini")
        self.assertTrue(config.llm_disable_reasoning)
        self.assertEqual(config.openai_api_key, "openai-key")
        self.assertEqual(config.openai_model_name, "gpt-4o-mini")

    def test_runtime_config_defaults_openrouter_base_url_by_provider(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openrouter",
                "LLM_API_KEY": "openrouter-key",
                "LLM_MODEL_NAME": "openai/gpt-4o-mini",
            },
            clear=True,
        ):
            config = load_runtime_config()

        self.assertEqual(config.llm_base_url, "https://openrouter.ai/api/v1")

    def test_apply_runtime_environment_sets_openai_compatible_vars(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            apply_runtime_environment(
                {
                    "llm_api_key": "openrouter-key",
                    "llm_model_name": "openai/gpt-4o-mini",
                    "llm_base_url": "https://openrouter.ai/api/v1",
                }
            )

            self.assertEqual(os.environ["OPENAI_API_KEY"], "openrouter-key")
            self.assertEqual(os.environ["OPENAI_MODEL_NAME"], "openai/gpt-4o-mini")
            self.assertEqual(os.environ["OPENAI_API_BASE"], "https://openrouter.ai/api/v1")
            self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://openrouter.ai/api/v1")

    def test_apply_runtime_environment_clears_base_url_when_unset(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
                "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            },
            clear=True,
        ):
            apply_runtime_environment(
                {
                    "llm_api_key": "openai-key",
                    "llm_model_name": "gpt-4o-mini",
                }
            )

            self.assertNotIn("OPENAI_API_BASE", os.environ)
            self.assertNotIn("OPENAI_BASE_URL", os.environ)

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

    @patch("services.intent_router.httpx.post")
    def test_intent_router_llm_fallback_uses_openrouter_config(self, post_mock) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"detected_intent": "pre_sales", "confidence_score": 0.91}'
                            }
                        }
                    ]
                }

        post_mock.return_value = FakeResponse()

        result = classify_intent(
            "Can you help me choose?",
            config_context={
                "llm_api_key": "openrouter-key",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model_name": "openai/gpt-4o-mini",
            },
        )

        self.assertEqual(result["detected_intent"], "pre_sales")
        post_mock.assert_called_once()
        _, kwargs = post_mock.call_args
        self.assertEqual(post_mock.call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer openrouter-key")
        self.assertEqual(kwargs["json"]["model"], "openai/gpt-4o-mini")

    @patch("services.intent_router.httpx.post")
    def test_intent_router_llm_fallback_disables_qwen3_reasoning(self, post_mock) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"detected_intent": "post_sales_support", "confidence_score": 0.9}'
                            }
                        }
                    ]
                }

        post_mock.return_value = FakeResponse()

        result = classify_intent(
            "Can you help me with this?",
            config_context={
                "llm_api_key": "openrouter-key",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model_name": "qwen/qwen3-14b",
                "llm_provider": "openrouter",
            },
        )

        self.assertEqual(result["detected_intent"], "post_sales_support")
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["json"]["reasoning_effort"], "none")

    @patch("crews.support_crew.httpx.post")
    def test_response_language_rewriter_disables_qwen3_reasoning(self, post_mock) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {"message": {"content": '{"response": "Bonjour"}'}}
                    ]
                }

        post_mock.return_value = FakeResponse()

        response = _rewrite_response_language(
            response="Hello",
            inputs={
                "detected_language": "fr",
                "language_plan": "French",
                "channel": "gmail",
                "inquiry_text": "Bonjour",
            },
            structured_facts={"response_type": "test"},
            config_context={
                "llm_api_key": "openrouter-key",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model_name": "qwen/qwen3-14b",
                "llm_provider": "openrouter",
            },
        )

        self.assertEqual(response, "Bonjour")
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["json"]["reasoning_effort"], "none")

    def test_workflow_progress_records_task_duration(self) -> None:
        store = InMemoryJobStore()
        store.create_job("job-1", WorkflowType.SUPPORT, {})
        recorder = WorkflowProgressRecorder(
            job_id="job-1",
            workflow_type="support",
            job_store=store,
            backend="local",
        )

        recorder.task_started(0, 1, "inquiry_resolution", "Senior Support")
        recorder.task_completed(0, 1, "inquiry_resolution", "Senior Support")

        completed_events = [
            event for event in store.get_job_events("job-1")
            if event["event_type"] == "task_completed"
        ]
        self.assertTrue(completed_events)
        self.assertIn("duration_seconds", completed_events[-1]["payload"])
        self.assertIsNotNone(completed_events[-1]["payload"]["duration_seconds"])

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

    def test_attach_customer_service_context_accepts_object_product_recommendation_fixture(self) -> None:
        crew_payload = _wireless_headset_pre_sales_fixture()
        inputs = _normalize_inputs(
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "How much is the Wireless Bluetooth headset?",
                "product_category": "Wireless Bluetooth headset",
                "channel": "gmail",
                "session_id": "conv-wireless-headset",
            }
        )

        result = _attach_customer_service_context(
            result={
                "pre_sales_response": {
                    "product_recommendation": crew_payload["product_recommendation"],
                    "feature_explanation": crew_payload["feature_explanation"],
                    "comparison_summary": crew_payload["comparison_summary"],
                    "next_steps": crew_payload["next_steps"],
                    "confidence_level": crew_payload["confidence_level"],
                    "requires_human_review": crew_payload["requires_human_review"],
                },
                "final_response": crew_payload["final_response"],
                "qa_status": "REVIEW_REQUIRED",
                "escalation_needed": True,
            },
            inputs=inputs,
            automation_context={"escalation_flag": False, "compliance_tags": []},
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.95},
            pre_sales_context={"data_source": "local_pdf_catalog", "product_found": True},
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        CustomerServiceOutput.model_validate(result)
        pre_sales = result["pre_sales_response"]
        self.assertEqual(pre_sales["product_recommendation"], "Wireless Bluetooth headset")
        self.assertEqual(
            pre_sales["next_steps"],
            [
                "Confirm how many units or cartons the customer wants.",
                "Ask sales to review any discount request before quoting a reduced price.",
            ],
        )
        offer = pre_sales["catalog_product_offer"]
        self.assertEqual(offer["status"], "found")
        self.assertEqual(offer["product_name"], "Wireless Bluetooth headset")
        self.assertEqual(offer["unit_price"], "$6.50")
        self.assertEqual(offer["carton_quantity"], "40PCS")
        self.assertEqual(offer["carton_size"], "52x38x51cm")
        self.assertEqual(offer["carton_weight"], "9 kg")

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

    def test_attach_customer_service_context_rewrites_catalog_prices_in_japanese(self) -> None:
        def japanese_rewriter(**kwargs):
            facts = kwargs["structured_facts"]
            return (
                f"こんにちは。\n\n{facts['product_name']}について確認済みのカタログ情報をお伝えします。\n"
                f"現在のカタログ単価: {facts['unit_price']}\n"
                f"カートン入数: {facts['carton_quantity']}\n"
                "割引リクエストは、営業チームによる確認が必要です。"
            )

        inputs = _normalize_inputs(
            {
                "customer": "Yuki",
                "person": "Yuki",
                "inquiry": "M90 PRO wireless earphonesの割引価格を教えてください。",
                "product_category": "M90 PRO wireless earphones",
                "channel": "gmail",
                "session_id": "sess-pricing-ja",
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
            config_context={"response_language_rewriter": japanese_rewriter},
        )

        self.assertEqual(result["detected_language"], "ja")
        self.assertEqual(result["language_plan"], "Japanese")
        self.assertIn("$2.39", result["final_response"])
        self.assertIn("現在のカタログ単価", result["final_response"])
        self.assertIn("割引リクエスト", result["final_response"])
        self.assertNotIn("$12.99", result["final_response"])
        self.assertNotIn("Buy 1", result["final_response"])
        self.assertNotIn("Thanks for your interest", result["final_response"])
        self.assertNotIn("sales team", result["final_response"])

    def test_attach_customer_service_context_rewrites_catalog_prices_in_french(self) -> None:
        def french_rewriter(**kwargs):
            facts = kwargs["structured_facts"]
            return (
                f"Bonjour,\n\nVoici les informations catalogue verifiees pour {facts['product_name']}.\n"
                f"Prix unitaire catalogue actuel : {facts['unit_price']}\n"
                f"Quantite par carton : {facts['carton_quantity']}\n"
                "Toute demande de remise doit etre examinee par l'equipe commerciale."
            )

        inputs = _normalize_inputs(
            {
                "customer": "Claire",
                "person": "Claire",
                "inquiry": "Bonjour. Je souhaite acheter la D18 Smart Watch. Pourriez-vous me donner plus de détails ?",
                "product_category": "D18 Smart Watch",
                "channel": "gmail",
                "session_id": "sess-pricing-fr",
            }
        )
        pre_sales_context = {
            "data_source": "mock_fallback",
            "product_found": True,
            "catalog_product_offer": {
                "status": "found",
                "product_found": True,
                "product_name": "D18 Smart watch /ctn Dz09 smart watch 1 carton contains: 100 pieces $6.20 Please contact me for a discount",
                "unit_price": "$1.68",
                "carton_quantity": "100 PCS",
                "carton_size": "42.5×31.7×26",
                "single_product_size": "12.2×8×2.9",
                "single_product_weight": "40g",
                "source": "catalog.pdf",
                "heading": "Page 1",
                "evidence": ["D18 Smart watch /ctn Dz09 smart watch 1 carton contains: 100 pieces $6.20 Please contact me for a discount"],
                "data_source": "local_pdf_catalog",
            },
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
            config_context={"response_language_rewriter": french_rewriter},
        )

        self.assertEqual(result["detected_language"], "fr")
        self.assertEqual(result["language_plan"], "French")
        self.assertIn("$1.68", result["final_response"])
        self.assertIn("Prix unitaire catalogue actuel", result["final_response"])
        self.assertNotIn("Hello", result["final_response"])
        self.assertNotIn("Thank you for your interest", result["final_response"])
        self.assertNotIn("current catalog unit price", result["final_response"])
        self.assertNotIn("1 carton contains", result["final_response"])
        self.assertNotIn("Please contact me for a discount", result["final_response"])

    def test_catalog_display_name_removes_catalog_noise(self) -> None:
        display_name = _catalog_display_name(
            {
                "product_name": (
                    "D18 Smart watch /ctn Dz09 smart watch 1 carton contains: "
                    "100 pieces $6.20 Please contact me for a discount"
                )
            },
            {"product_category": "D18 Smart Watch"},
        )

        self.assertEqual(display_name, "D18 Smart Watch")

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

    def test_attach_customer_service_context_extracts_fenced_raw_json_final_response(self) -> None:
        result = _attach_customer_service_context(
            result={
                "raw": """```json
{
  "response_type": "pre_sales",
  "final_response": "Hi Tonny, the headset catalog price is $6.50.",
  "qa_status": "APPROVED",
  "escalation_needed": false
}
```"""
            },
            inputs=_normalize_inputs(
                {
                    "customer": "Tonny",
                    "inquiry": "How much is this Bluetooth headset?",
                    "channel": "gmail",
                }
            ),
            automation_context={"escalation_flag": False, "compliance_tags": []},
            router_result={"detected_intent": "pre_sales", "confidence_score": 0.95},
            pre_sales_context={"data_source": "mock_fallback", "product_found": True},
            order_context={"data_source": "mock_order_db", "order_found": False},
        )

        self.assertEqual(result["final_response"], "Hi Tonny, the headset catalog price is $6.50.")

    def test_sync_job_result_does_not_downgrade_sent_conversation_to_draft_ready(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            draft_response="Already sent response.",
            draft_payload=None,
            requires_approval=False,
            escalation_flag=False,
            status="sent",
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
                    "final_response": "The catalog item is available.",
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                },
            },
        )

        self.assertEqual(conversation.status, "sent")
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

    def test_sync_job_result_auto_allows_high_confidence_order_fulfillment_review(self) -> None:
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
                    "detected_intent": "order_fulfillment",
                    "routing_confidence": 0.95,
                    "final_response": "Tracking C88943021 was successfully delivered.",
                    "qa_status": "REVIEW_REQUIRED",
                    "escalation_needed": True,
                    "compliance_flags": ["GDPR_COMPLIANT_REVIEW", "CCPA_OPT_OUT_AVAILABLE"],
                    "order_response": {"tracking_record_found": True},
                },
            },
        )

        self.assertEqual(conversation.status, "draft_ready")
        self.assertFalse(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)

    def test_sync_job_result_requires_approval_for_low_confidence_order_fulfillment(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
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
                    "detected_intent": "order_fulfillment",
                    "routing_confidence": 0.74,
                    "final_response": "A specialist should review this tracking request.",
                    "qa_status": "REVIEW_REQUIRED",
                    "escalation_needed": True,
                },
            },
        )

        self.assertEqual(conversation.status, "handoff_required")
        self.assertTrue(conversation.requires_approval)
        self.assertTrue(conversation.escalation_flag)

    def test_sync_job_result_handoffs_order_fulfillment_with_hard_flag(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
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
                    "detected_intent": "order_fulfillment",
                    "routing_confidence": 0.95,
                    "final_response": "A specialist will review this shipment.",
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                    "compliance_flags": ["HUMAN_HANDOFF"],
                },
            },
        )

        self.assertEqual(conversation.status, "handoff_required")
        self.assertTrue(conversation.requires_approval)
        self.assertTrue(conversation.escalation_flag)

    def test_sync_job_result_requires_approval_for_order_fulfillment_vip_billing_or_negative(self) -> None:
        scenarios = [
            {"customer_tier": "VIP", "sentiment_score": 0.2, "intent_category": "SHIPPING_INQUIRY"},
            {"customer_tier": "STANDARD", "sentiment_score": -0.3, "intent_category": "SHIPPING_INQUIRY"},
            {"customer_tier": "STANDARD", "sentiment_score": 0.2, "intent_category": "BILLING_ISSUE"},
        ]
        for index, sentiment in enumerate(scenarios):
            with self.subTest(sentiment=sentiment):
                conversation = SimpleNamespace(
                    conversation_id=f"conv-{index}",
                    channel="gmail",
                    draft_response=None,
                    draft_payload=None,
                    requires_approval=False,
                    escalation_flag=False,
                    status="processing",
                )
                store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

                store.sync_job_result(
                    conversation.conversation_id,
                    {
                        "job_id": "job-1",
                        "workflow_type": "support",
                        "result": {
                            "session_id": conversation.conversation_id,
                            "detected_intent": "order_fulfillment",
                            "routing_confidence": 0.95,
                            "final_response": "Tracking C88943021 was successfully delivered.",
                            "qa_status": "REVIEW_REQUIRED",
                            "escalation_needed": True,
                            "sentiment_analysis": sentiment,
                        },
                    },
                )

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
