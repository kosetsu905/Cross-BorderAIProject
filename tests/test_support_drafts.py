import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import create_router
from database import get_db_session
from services.support_auto_dispatch import process_completed_support_job
from services.workflow_guardrails import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailSeverity,
    GuardrailStage,
)
from support_inbox import SupportInboxStore
from utils.support_drafts import customer_facing_draft_text


SUPPORT_FIXTURE_DIR = Path(__file__).parent / "fixtures"
WIRELESS_HEADSET_PRE_SALES_FIXTURE = SUPPORT_FIXTURE_DIR / "support_wireless_headset_pre_sales.json"


FENCED_SUPPORT_JSON = """```json
{
  "response_type": "pre_sales",
  "final_response": "Hi Tonny, the Wireless Bluetooth headset is priced at $6.50 each."
}
```"""


def _wireless_headset_pre_sales_fixture() -> dict[str, object]:
    return json.loads(WIRELESS_HEADSET_PRE_SALES_FIXTURE.read_text(encoding="utf-8"))


def _fenced_json_payload(payload: dict[str, object]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _structured_pre_sales_job_result(conversation_id: str = "conv-1") -> dict[str, object]:
    payload = _wireless_headset_pre_sales_fixture()
    return {
        "session_id": conversation_id,
        "detected_intent": "pre_sales",
        "routing_confidence": 0.95,
        "pre_sales_response": {
            "product_recommendation": payload["product_recommendation"],
            "feature_explanation": payload["feature_explanation"],
            "comparison_summary": payload["comparison_summary"],
            "next_steps": payload["next_steps"],
            "confidence_level": payload["confidence_level"],
            "requires_human_review": payload["requires_human_review"],
        },
        "final_response": payload["final_response"],
        "qa_status": "APPROVED",
        "escalation_needed": False,
        "compliance_flags": [],
        "recommended_follow_up": "Follow up if the customer replies.",
    }


def _allow_action_decision() -> GuardrailDecision:
    return GuardrailDecision(
        workflow_type="support",
        stage=GuardrailStage.ACTION,
        action=GuardrailAction.ALLOW,
        severity=GuardrailSeverity.NONE,
        findings=[],
        sanitized_payload={},
    )


def _gmail_conversation(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "conversation_id": "conv-1",
        "channel": "gmail",
        "channel_thread_id": "gmail-thread-1",
        "customer_display_name": "Tonny",
        "customer_handle": "tonny@example.com",
        "customer_handle_masked": "to***@example.com",
        "assigned_to": None,
        "status": "processing",
        "latest_job_id": None,
        "draft_response": None,
        "draft_payload": None,
        "requires_approval": True,
        "escalation_flag": False,
        "last_message_at": None,
        "created_at": None,
        "updated_at": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeScalarResult:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self.records = records

    def __iter__(self) -> object:
        return iter(self.records)

    def first(self) -> SimpleNamespace | None:
        return self.records[0] if self.records else None


class FakeDbSession:
    def __init__(
        self,
        conversation: SimpleNamespace,
        records: list[SimpleNamespace] | None = None,
    ) -> None:
        self.conversation = conversation
        self.records = records or []
        self.added: list[object] = []
        self.commit_count = 0

    def get(self, model: object, key: str) -> SimpleNamespace | None:
        if key == self.conversation.conversation_id:
            return self.conversation
        return None

    def add(self, record: object) -> None:
        self.added.append(record)

    def commit(self) -> None:
        self.commit_count += 1

    def refresh(self, record: object) -> None:
        return None

    def execute(self, statement: object) -> object:
        return SimpleNamespace(scalars=lambda: FakeScalarResult(self.records))


class FallbackConversationDbSession(FakeDbSession):
    def __init__(self, conversation: SimpleNamespace) -> None:
        super().__init__(conversation)
        self.execute_count = 0

    def execute(self, statement: object) -> object:
        self.execute_count += 1
        records = [self.conversation] if self.execute_count == 1 else []
        return SimpleNamespace(scalars=lambda: FakeScalarResult(records))


class FakeSessionContext:
    def __init__(self, session: FakeDbSession) -> None:
        self.session = session

    def __enter__(self) -> FakeDbSession:
        return self.session

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


class FakeSupportStore:
    def __init__(self, db: FakeDbSession) -> None:
        self.db = db
        self.recorded_text: str | None = None

    def sync_job_result(self, conversation_id: str, job_data: dict[str, object] | None) -> None:
        return None

    def record_outbound_message(
        self,
        *,
        conversation_id: str,
        text: str,
        channel_message_id: str | None,
        delivery_status: str,
        raw_payload: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        self.recorded_text = text
        return SimpleNamespace(message_id="outbound-1", channel_message_id=channel_message_id)


class SupportDraftTests(unittest.TestCase):
    def test_extracts_final_response_from_fenced_json(self) -> None:
        self.assertEqual(
            customer_facing_draft_text(FENCED_SUPPORT_JSON),
            "Hi Tonny, the Wireless Bluetooth headset is priced at $6.50 each.",
        )

    def test_preserves_plain_text(self) -> None:
        text = "Hi Tonny,\n\nThe headset is available at the catalog price."

        self.assertEqual(customer_facing_draft_text(text), text)

    def test_extracts_nested_json_like_final_response(self) -> None:
        nested = {
            "final_response": """```json
{"final_response": "A specialist can help confirm bulk pricing."}
```"""
        }

        self.assertEqual(
            customer_facing_draft_text(nested),
            "A specialist can help confirm bulk pricing.",
        )

    def test_structured_dict_without_customer_body_is_not_returned_as_email_text(self) -> None:
        raw_catalog_only = "{'catalog_product_offer': {'unit_price': '$6.50', 'carton_quantity': '40PCS'}}"

        self.assertIsNone(customer_facing_draft_text(raw_catalog_only))

    def test_sync_job_result_stores_plain_draft_and_keeps_payload(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeDbSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.95,
                    "final_response": FENCED_SUPPORT_JSON,
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                },
            },
        )

        self.assertEqual(
            conversation.draft_response,
            "Hi Tonny, the Wireless Bluetooth headset is priced at $6.50 each.",
        )
        self.assertEqual(conversation.draft_payload["final_response"], FENCED_SUPPORT_JSON)

    def test_sync_job_result_requires_approval_for_nested_ineligible_rma(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-rma-nested",
            channel="gmail",
            draft_response=None,
            draft_payload=None,
            requires_approval=False,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeDbSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-rma-nested",
            {
                "job_id": "job-rma",
                "workflow_type": "support",
                "result": {
                    "session_id": "conv-rma-nested",
                    "detected_intent": "post_sales_support",
                    "routing_confidence": 0.95,
                    "final_response": "We are reviewing this return request.",
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                    "support_response": {
                        "rma_validation": json.dumps(
                            {
                                "eligible_for_return": False,
                                "eligibility_reason": "Return window expired.",
                            }
                        )
                    },
                },
            },
        )

        self.assertTrue(conversation.requires_approval)

    def test_sync_job_result_uses_configured_auto_send_confidence_threshold(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-threshold",
            channel="gmail",
            draft_response=None,
            draft_payload=None,
            requires_approval=False,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeDbSession(conversation))  # type: ignore[arg-type]

        with patch("support_inbox._support_auto_send_confidence_threshold", return_value=0.96):
            store.sync_job_result(
                "conv-threshold",
                {
                    "job_id": "job-threshold",
                    "workflow_type": "support",
                    "result": _structured_pre_sales_job_result("conv-threshold"),
                },
            )

        self.assertTrue(conversation.requires_approval)

    def test_get_conversation_syncs_structured_product_recommendation_with_real_store(self) -> None:
        conversation = _gmail_conversation(latest_job_id="job-1")
        fake_db = FakeDbSession(conversation)

        class FakeOrchestrator:
            registered_workflows = []

            def get_job_status(self, job_id: str) -> dict[str, object]:
                return {
                    "job_id": job_id,
                    "workflow_type": "support",
                    "result": _structured_pre_sales_job_result("conv-1"),
                }

        app = FastAPI()
        app.include_router(create_router(FakeOrchestrator()))
        app.dependency_overrides[get_db_session] = lambda: fake_db

        response = TestClient(app).get("/api/v1/support/conversations/conv-1")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        expected_final = _wireless_headset_pre_sales_fixture()["final_response"]
        self.assertEqual(data["draft_response"], expected_final)
        self.assertEqual(
            data["draft_payload"]["pre_sales_response"]["product_recommendation"]["product_name"],
            "Wireless Bluetooth headset",
        )
        self.assertFalse(data["requires_approval"])
        self.assertFalse(data["escalation_flag"])

    @patch("api.routes.send_gmail_reply_message")
    @patch("api.routes._gmail_access_token_from_config", return_value="token")
    @patch("api.routes.load_runtime_config")
    def test_approve_send_normalizes_fenced_json_message(
        self,
        load_config: Mock,
        access_token: Mock,
        send_gmail: Mock,
    ) -> None:
        load_config.return_value = SimpleNamespace(
            gmail_send_enabled=True,
            gmail_sender_email="support@example.com",
        )
        send_gmail.return_value = {"status": "sent", "message_id": "gmail-out-1", "error": None}
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            channel_thread_id="gmail-thread-1",
            customer_handle="tonny@example.com",
            draft_response=FENCED_SUPPORT_JSON,
            draft_payload={"final_response": FENCED_SUPPORT_JSON},
            escalation_flag=False,
            requires_approval=True,
            latest_job_id=None,
            status="draft_ready",
        )
        fake_db = FakeDbSession(
            conversation,
            records=[
                SimpleNamespace(
                    raw_payload={
                        "headers": {
                            "subject": "Bluetooth headset inquiry",
                            "message-id": "<inbound@example.com>",
                        }
                    }
                )
            ],
        )
        fake_store = FakeSupportStore(fake_db)

        class FakeOrchestrator:
            registered_workflows = []

            def get_job_status(self, job_id: str) -> dict[str, object]:
                return {}

        app = FastAPI()
        app.include_router(create_router(FakeOrchestrator()))
        app.dependency_overrides[get_db_session] = lambda: fake_db

        with patch("api.routes.SupportInboxStore", return_value=fake_store), patch(
            "api.routes.WorkflowGuardrailService"
        ) as guardrails:
            guardrails.return_value.evaluate_action.return_value = _allow_action_decision()
            response = TestClient(app).post(
                "/api/v1/support/conversations/conv-1/approve-send",
                json={"message": FENCED_SUPPORT_JSON},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            send_gmail.call_args.kwargs["body"],
            "Hi Tonny, the Wireless Bluetooth headset is priced at $6.50 each.",
        )
        self.assertEqual(
            fake_store.recorded_text,
            "Hi Tonny, the Wireless Bluetooth headset is priced at $6.50 each.",
        )
        access_token.assert_called_once()

    @patch("api.routes.send_gmail_reply_message")
    @patch("api.routes._gmail_access_token_from_config", return_value="token")
    @patch("api.routes.load_runtime_config")
    def test_approve_send_preserves_plain_text_message(
        self,
        load_config: Mock,
        access_token: Mock,
        send_gmail: Mock,
    ) -> None:
        load_config.return_value = SimpleNamespace(
            gmail_send_enabled=True,
            gmail_sender_email="support@example.com",
        )
        send_gmail.return_value = {"status": "sent", "message_id": "gmail-out-1", "error": None}
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            channel_thread_id="gmail-thread-1",
            customer_handle="tonny@example.com",
            draft_response="Plain draft",
            draft_payload={},
            escalation_flag=False,
            requires_approval=True,
            latest_job_id=None,
            status="draft_ready",
        )
        fake_db = FakeDbSession(conversation)
        fake_store = FakeSupportStore(fake_db)

        class FakeOrchestrator:
            registered_workflows = []

            def get_job_status(self, job_id: str) -> dict[str, object]:
                return {}

        app = FastAPI()
        app.include_router(create_router(FakeOrchestrator()))
        app.dependency_overrides[get_db_session] = lambda: fake_db

        with patch("api.routes.SupportInboxStore", return_value=fake_store), patch(
            "api.routes.WorkflowGuardrailService"
        ) as guardrails:
            guardrails.return_value.evaluate_action.return_value = _allow_action_decision()
            response = TestClient(app).post(
                "/api/v1/support/conversations/conv-1/approve-send",
                json={"message": "Plain edited draft"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(send_gmail.call_args.kwargs["body"], "Plain edited draft")
        self.assertEqual(fake_store.recorded_text, "Plain edited draft")
        access_token.assert_called_once()

    @patch("api.routes.send_gmail_reply_message")
    @patch("api.routes._gmail_access_token_from_config", return_value="token")
    @patch("api.routes.load_runtime_config")
    def test_approve_send_uses_real_store_and_sends_final_response_from_structured_json(
        self,
        load_config: Mock,
        access_token: Mock,
        send_gmail: Mock,
    ) -> None:
        load_config.return_value = SimpleNamespace(
            gmail_send_enabled=True,
            gmail_sender_email="support@example.com",
        )
        send_gmail.return_value = {"status": "sent", "message_id": "gmail-out-1", "error": None}
        fixture_payload = _wireless_headset_pre_sales_fixture()
        fenced_payload = _fenced_json_payload(fixture_payload)
        conversation = _gmail_conversation(
            draft_response=fenced_payload,
            draft_payload={"final_response": fenced_payload},
            status="draft_ready",
        )
        fake_db = FakeDbSession(
            conversation,
            records=[
                SimpleNamespace(
                    raw_payload={
                        "headers": {
                            "subject": "Bluetooth headset inquiry",
                            "message-id": "<inbound@example.com>",
                        }
                    }
                )
            ],
        )

        class FakeOrchestrator:
            registered_workflows = []

            def get_job_status(self, job_id: str) -> dict[str, object]:
                return {}

        app = FastAPI()
        app.include_router(create_router(FakeOrchestrator()))
        app.dependency_overrides[get_db_session] = lambda: fake_db

        with patch("api.routes.WorkflowGuardrailService") as guardrails:
            guardrails.return_value.evaluate_action.return_value = _allow_action_decision()
            response = TestClient(app).post(
                "/api/v1/support/conversations/conv-1/approve-send",
                json={"message": fenced_payload},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(send_gmail.call_args.kwargs["body"], fixture_payload["final_response"])
        self.assertEqual(fake_db.added[-1].text, fixture_payload["final_response"])
        self.assertEqual(fake_db.added[-1].raw_payload["draft_payload"]["final_response"], fenced_payload)
        access_token.assert_called_once()

    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    @patch("services.support_auto_dispatch.SessionLocal")
    def test_auto_dispatch_sends_final_response_from_structured_pre_sales_result(
        self,
        session_local: Mock,
        send_gmail: Mock,
    ) -> None:
        send_gmail.return_value = {"status": "sent", "message_id": "gmail-out-1", "error": None}
        conversation = _gmail_conversation()
        fake_db = FakeDbSession(
            conversation,
            records=[
                SimpleNamespace(
                    raw_payload={
                        "headers": {
                            "subject": "Bluetooth headset inquiry",
                            "message-id": "<inbound@example.com>",
                        }
                    }
                )
            ],
        )
        session_local.return_value = FakeSessionContext(fake_db)
        fixture_payload = _wireless_headset_pre_sales_fixture()

        with patch("services.support_auto_dispatch.WorkflowGuardrailService") as guardrails:
            guardrails.return_value.evaluate_action.return_value = _allow_action_decision()
            result = asyncio.run(
                process_completed_support_job(
                    job_id="job-1",
                    inputs={"session_id": "conv-1"},
                    result=_structured_pre_sales_job_result("conv-1"),
                    config_context={
                        "gmail_send_enabled": True,
                        "gmail_access_token": "token",
                        "gmail_sender_email": "support@example.com",
                    },
                )
            )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(send_gmail.call_args.kwargs["body"], fixture_payload["final_response"])
        self.assertEqual(fake_db.added[-1].text, fixture_payload["final_response"])
        self.assertEqual(
            conversation.draft_payload["pre_sales_response"]["product_recommendation"]["product_name"],
            "Wireless Bluetooth headset",
        )

    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    @patch("services.support_auto_dispatch.SessionLocal")
    def test_auto_dispatch_recovers_conversation_by_latest_job_id_and_uses_raw_recipient(
        self,
        session_local: Mock,
        send_gmail: Mock,
    ) -> None:
        send_gmail.return_value = {"status": "sent", "message_id": "gmail-out-1", "error": None}
        conversation = _gmail_conversation(
            conversation_id="bc810e41-b8d5-4627-8563-03b5f5b46659",
            latest_job_id="job-1",
        )
        fake_db = FallbackConversationDbSession(conversation)
        session_local.return_value = FakeSessionContext(fake_db)

        with patch("services.support_auto_dispatch.WorkflowGuardrailService") as guardrails:
            guardrails.return_value.evaluate_action.return_value = _allow_action_decision()
            result = asyncio.run(
                process_completed_support_job(
                    job_id="job-1",
                    inputs={"session_id": "bc810e41-b8d5-4627-[PHONE:8563]-03b5f5b46659"},
                    result=_structured_pre_sales_job_result(conversation.conversation_id),
                    config_context={
                        "gmail_send_enabled": True,
                        "gmail_access_token": "token",
                        "gmail_sender_email": "support@example.com",
                    },
                )
            )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["conversation_id"], conversation.conversation_id)
        self.assertEqual(send_gmail.call_args.kwargs["recipient"], "tonny@example.com")

    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    @patch("services.support_auto_dispatch.SessionLocal")
    def test_auto_dispatch_skips_hard_guardrail_decision(
        self,
        session_local: Mock,
        send_gmail: Mock,
    ) -> None:
        conversation = _gmail_conversation(requires_approval=False)
        fake_db = FakeDbSession(conversation)
        session_local.return_value = FakeSessionContext(fake_db)
        result_payload = _structured_pre_sales_job_result("conv-1")
        result_payload["guardrail_decision"] = {
            "action": "block",
            "severity": "critical",
            "findings": [],
        }

        result = asyncio.run(
            process_completed_support_job(
                job_id="job-1",
                inputs={"session_id": "conv-1"},
                result=result_payload,
                config_context={"gmail_send_enabled": True},
            )
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "guardrail_review_required")
        self.assertTrue(conversation.requires_approval)
        send_gmail.assert_not_called()

    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    @patch("services.support_auto_dispatch.SessionLocal")
    def test_auto_dispatch_skips_when_successful_outbound_already_exists(
        self,
        session_local: Mock,
        send_gmail: Mock,
    ) -> None:
        conversation = _gmail_conversation()
        fake_db = FakeDbSession(
            conversation,
            records=[
                SimpleNamespace(
                    status="sent",
                    raw_payload={
                        "auto_dispatch": True,
                        "source_job_id": "job-earlier",
                        "delivery": {"status": "sent"},
                    },
                )
            ],
        )
        session_local.return_value = FakeSessionContext(fake_db)

        result = asyncio.run(
            process_completed_support_job(
                job_id="job-2",
                inputs={"session_id": "conv-1"},
                result=_structured_pre_sales_job_result("conv-1"),
                config_context={
                    "gmail_send_enabled": True,
                    "gmail_access_token": "token",
                    "gmail_sender_email": "support@example.com",
                },
            )
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "already_dispatched")
        send_gmail.assert_not_called()


if __name__ == "__main__":
    unittest.main()
