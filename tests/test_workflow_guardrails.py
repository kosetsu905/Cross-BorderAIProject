from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import create_router
from database import get_db_session
from job_store import InMemoryJobStore
from models import WorkflowType
from runtime_config import RuntimeConfig
from services.workflow_guardrails import (
    WorkflowGuardrailService,
    apply_output_guardrail_result,
    guardrail_requires_override,
)
from utils.workflow_engine import WorkflowExecutionEngine


def test_input_guardrail_blocks_secret_before_job_storage() -> None:
    store = InMemoryJobStore()
    engine = WorkflowExecutionEngine(store, RuntimeConfig())
    raw_secret = "sk-" + "a" * 28

    prepared = engine.prepare_job(
        WorkflowType.SUPPORT,
        {
            "customer": "Customer",
            "person": "Customer",
            "inquiry": f"please use api_key={raw_secret}",
        },
        provider_credentials=None,
        metadata=None,
        backend="local",
        queued_message="queued",
    )

    job = store.get_job(prepared.job_id)
    serialized_job = json.dumps(job, default=str)
    assert prepared.skip_execution is True
    assert raw_secret not in serialized_job
    assert "[SECRET]" in serialized_job
    assert any(event["event_type"] == "guardrail_input_evaluated" for event in store.get_job_events(prepared.job_id))


def test_workflow_guardrail_masks_pii_and_keeps_raw_out_of_evidence() -> None:
    decision = WorkflowGuardrailService().evaluate_input(
        WorkflowType.SUPPORT,
        {
            "customer": "Tonny",
            "person": "Tonny",
            "inquiry": "Please reply to tonny@example.com or +1 415 555 1212.",
        },
    )

    serialized = json.dumps(decision.model_dump(mode="json"), default=str)
    assert decision.action.value in {"mask", "monitor", "review_required"}
    assert "tonny@example.com" not in serialized
    assert "+1 415 555 1212" not in serialized
    assert "to***@example.com" in serialized


def test_support_output_guardrail_sets_review_required_for_ungrounded_claim() -> None:
    result = {
        "session_id": "conv-1",
        "detected_intent": "post_sales_support",
        "routing_confidence": 0.95,
        "final_response": "Your return is approved and a refund is guaranteed.",
        "qa_status": "APPROVED",
        "escalation_needed": False,
        "compliance_flags": [],
    }

    decision = WorkflowGuardrailService().evaluate_output(WorkflowType.SUPPORT, result, grounding_context=[])
    guarded = apply_output_guardrail_result(WorkflowType.SUPPORT, result, decision)

    assert decision.action.value == "review_required"
    assert guarded["qa_status"] == "REVIEW_REQUIRED"
    assert guarded["requires_approval"] is True
    assert "GUARDRAIL_REVIEW_REQUIRED" in guarded["compliance_flags"]
    assert guardrail_requires_override(guarded) is True


class FakeDbSession:
    def __init__(self, conversation: SimpleNamespace) -> None:
        self.conversation = conversation
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
        return SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: None))


class FakeSupportStore:
    def __init__(self, db: FakeDbSession) -> None:
        self.db = db

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
        record = SimpleNamespace(
            message_id="outbound-1",
            channel_message_id=channel_message_id,
            text=text,
            raw_payload=raw_payload or {},
        )
        self.db.add(record)
        return record


def test_approve_send_requires_reviewer_and_reason_for_high_risk_guardrail() -> None:
    conversation = SimpleNamespace(
        conversation_id="conv-1",
        channel="gmail",
        channel_thread_id="thread-1",
        customer_handle="tonny@example.com",
        draft_response="Safe edited response.",
        draft_payload={
            "guardrail_decision": {
                "action": "review_required",
                "severity": "high",
                "findings": [],
            }
        },
        escalation_flag=False,
        requires_approval=True,
        latest_job_id=None,
        status="draft_ready",
    )
    fake_db = FakeDbSession(conversation)

    class FakeOrchestrator:
        registered_workflows = []

        def get_job_status(self, job_id: str) -> dict[str, object]:
            return {}

    app = FastAPI()
    app.include_router(create_router(FakeOrchestrator()))
    app.dependency_overrides[get_db_session] = lambda: fake_db

    with patch("api.routes.SupportInboxStore", return_value=FakeSupportStore(fake_db)):
        response = TestClient(app).post(
            "/api/v1/support/conversations/conv-1/approve-send",
            json={"message": "Safe edited response."},
        )

    assert response.status_code == 409
    assert "reviewer and override_reason" in response.json()["detail"]


@patch("api.routes.send_gmail_reply_message")
@patch("api.routes._gmail_access_token_from_config", return_value="token")
@patch("api.routes.load_runtime_config")
def test_approve_send_records_guardrail_override_payload(
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
        channel_thread_id="thread-1",
        customer_handle="tonny@example.com",
        draft_response="Safe edited response.",
        draft_payload={
            "guardrail_decision": {
                "action": "review_required",
                "severity": "high",
                "findings": [],
            }
        },
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
            json={
                "message": "Safe edited response.",
                "reviewer": "kosetsu",
                "override_reason": "Reviewed against order policy.",
            },
        )

    assert response.status_code == 200
    raw_payload = fake_db.added[-1].raw_payload
    assert raw_payload["guardrail_approval"]["reviewer"] == "kosetsu"
    assert raw_payload["guardrail_approval"]["override_reason"] == "Reviewed against order policy."
    access_token.assert_called_once()
