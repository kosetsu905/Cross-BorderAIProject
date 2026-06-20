import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import create_router
from database import get_db_session
from support_inbox import SupportInboxStore
from utils.support_drafts import customer_facing_draft_text


FENCED_SUPPORT_JSON = """```json
{
  "response_type": "pre_sales",
  "final_response": "Hi Tonny, the Wireless Bluetooth headset is priced at $6.50 each."
}
```"""


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

        with patch("api.routes.SupportInboxStore", return_value=fake_store):
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

        with patch("api.routes.SupportInboxStore", return_value=fake_store):
            response = TestClient(app).post(
                "/api/v1/support/conversations/conv-1/approve-send",
                json={"message": "Plain edited draft"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(send_gmail.call_args.kwargs["body"], "Plain edited draft")
        self.assertEqual(fake_store.recorded_text, "Plain edited draft")
        access_token.assert_called_once()


if __name__ == "__main__":
    unittest.main()
