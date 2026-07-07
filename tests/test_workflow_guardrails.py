from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import create_router
from database import get_db_session
from job_store import InMemoryJobStore
from models import WorkflowType
from runtime_config import RuntimeConfig
from services.workflow_guardrails import (
    GuardrailAction,
    GuardrailConfigurationError,
    GuardrailDecision,
    GuardrailSeverity,
    GuardrailStage,
    WorkflowGuardrailService,
    apply_output_guardrail_result,
    guardrail_requires_override,
)
from utils.workflow_engine import WorkflowExecutionEngine


class FakeOutcome:
    def __init__(self, passed: bool, reason: str | None = None) -> None:
        self.validation_passed = passed
        self.error = None
        self.validated_output = None
        self.validation_summaries = []
        if reason:
            self.validation_summaries.append(
                {
                    "validator_name": "FakeValidator",
                    "validator_status": "fail",
                    "failure_reason": reason,
                    "error_spans": [],
                }
            )


class FakeGuard:
    calls: list[dict[str, object]] = []
    failures: dict[str, str] = {}

    def __init__(self) -> None:
        self.validator: dict[str, object] | None = None

    def configure(self, **_: object) -> None:
        return None

    def use(self, validator: dict[str, object]) -> "FakeGuard":
        self.validator = validator
        return self

    def validate(self, text: str, metadata: dict[str, object] | None = None) -> FakeOutcome:
        validator = self.validator or {}
        validator_id = str(validator.get("id") or "")
        self.calls.append(
            {
                "validator": validator_id,
                "text": text,
                "metadata": metadata or {},
            }
        )
        reason = self.failures.get(validator_id)
        return FakeOutcome(reason is None, reason)


@contextmanager
def patched_guardrails(failures: dict[str, str]) -> Iterator[type[FakeGuard]]:
    FakeGuard.calls = []
    FakeGuard.failures = failures

    def build_validator(_: WorkflowGuardrailService, validator_config: dict[str, object]) -> dict[str, object]:
        return {
            "hub": validator_config.get("hub"),
            "id": validator_config.get("id"),
        }

    with patch("guardrails.Guard", FakeGuard), patch.object(
        WorkflowGuardrailService,
        "_build_hub_validator",
        build_validator,
    ):
        yield FakeGuard


def test_input_guardrail_blocks_secret_before_job_storage() -> None:
    store = InMemoryJobStore()
    engine = WorkflowExecutionEngine(store, RuntimeConfig())
    raw_secret = "sk-" + "a" * 28

    with patched_guardrails({"secrets_present": f"The following secrets were detected: api_key={raw_secret}"}) as guard:
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
    assert "secrets_present" in {str(call["validator"]) for call in guard.calls}
    assert any(event["event_type"] == "guardrail_input_evaluated" for event in store.get_job_events(prepared.job_id))


def test_workflow_guardrail_masks_pii_and_keeps_raw_out_of_evidence() -> None:
    with patched_guardrails(
        {
            "detect_pii": (
                "The following text contains PII: tonny@example.com and +1 415 555 1212."
            )
        }
    ):
        decision = WorkflowGuardrailService().evaluate_input(
            WorkflowType.SUPPORT,
            {
                "customer": "Tonny",
                "person": "Tonny",
                "inquiry": "Please reply to tonny@example.com or +1 415 555 1212.",
            },
        )

    serialized = json.dumps(decision.model_dump(mode="json"), default=str)
    assert decision.action == GuardrailAction.MASK
    assert "tonny@example.com" not in serialized
    assert "+1 415 555 1212" not in serialized
    assert "to***@example.com" in serialized
    assert "[PHONE:1212]" in serialized


def test_regex_match_forbidden_terms_blocks_input() -> None:
    with patched_guardrails({"forbidden_terms": "Result must match the configured regex."}) as guard:
        decision = WorkflowGuardrailService().evaluate_input(
            WorkflowType.SUPPORT,
            {
                "customer": "Customer",
                "person": "Customer",
                "inquiry": "Please share the internal discount code.",
            },
        )

    assert decision.action == GuardrailAction.BLOCK
    assert "forbidden_terms" in {str(call["validator"]) for call in guard.calls}


def test_support_output_guardrail_skips_provenance_without_grounding_context() -> None:
    result = {
        "session_id": "conv-1",
        "final_response": "Your return is approved and a refund is guaranteed.",
        "qa_status": "APPROVED",
        "compliance_flags": [],
    }

    with patched_guardrails({}) as guard:
        decision = WorkflowGuardrailService().evaluate_output(WorkflowType.SUPPORT, result, grounding_context=[])

    assert decision.action == GuardrailAction.ALLOW
    assert "provenance_llm" not in {str(call["validator"]) for call in guard.calls}
    assert {
        "id": "provenance_llm",
        "hub": "hub://guardrails/provenance_llm",
        "reason": "grounding_context_unavailable",
        "status": "not_applicable",
    } in decision.metadata["skipped_validators"]


def test_toxic_language_output_uses_remote_sidecar_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKFLOW_GUARDRAILS_TOXIC_URL", "http://guardrails-toxic:8011/validate")
    response = httpx.Response(
        200,
        json={
            "validation_passed": False,
            "failure_reasons": ["Toxic language detected."],
            "validator": "hub://guardrails/toxic_language",
            "source": "guardrails_ai_docker",
        },
        request=httpx.Request("POST", "http://guardrails-toxic:8011/validate"),
    )
    result = {
        "session_id": "conv-1",
        "final_response": "You are a stupid idiot.",
        "qa_status": "APPROVED",
        "compliance_flags": [],
    }

    with (
        patched_guardrails({}) as guard,
        patch("services.workflow_guardrails.httpx.Client.post", return_value=response) as post,
    ):
        decision = WorkflowGuardrailService().evaluate_output(WorkflowType.SUPPORT, result, grounding_context=[])

    post.assert_called_once()
    assert post.call_args.args[0] == "http://guardrails-toxic:8011/validate"
    assert post.call_args.kwargs["json"]["threshold"] == 0.5
    assert post.call_args.kwargs["json"]["validation_method"] == "sentence"
    assert decision.action == GuardrailAction.BLOCK
    assert "toxic_language" in {finding.validator for finding in decision.findings}
    assert "toxic_language" not in {str(call["validator"]) for call in guard.calls}


def test_toxic_language_sidecar_unavailable_requires_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKFLOW_GUARDRAILS_TOXIC_URL", "http://guardrails-toxic:8011/validate")
    result = {
        "session_id": "conv-1",
        "final_response": "A normal support response.",
        "qa_status": "APPROVED",
        "compliance_flags": [],
    }

    with (
        patched_guardrails({}),
        patch("services.workflow_guardrails.httpx.Client.post", side_effect=httpx.ConnectError("offline")) as post,
    ):
        decision = WorkflowGuardrailService().evaluate_output(WorkflowType.SUPPORT, result, grounding_context=[])

    assert post.call_count == 2
    assert decision.action == GuardrailAction.REVIEW_REQUIRED
    finding = next(finding for finding in decision.findings if finding.validator == "toxic_language")
    assert finding.metadata["runtime_error"] is True
    assert finding.metadata["source"] == "guardrails_ai_docker"


def test_support_output_guardrail_sets_review_required_for_provenance_failure_with_context() -> None:
    result = {
        "session_id": "conv-1",
        "detected_intent": "post_sales_support",
        "routing_confidence": 0.95,
        "final_response": "Your return is approved and a refund is guaranteed.",
        "qa_status": "APPROVED",
        "escalation_needed": False,
        "compliance_flags": [],
    }

    with patched_guardrails({"provenance_llm": "Generated response is not supported by provided context."}) as guard:
        decision = WorkflowGuardrailService().evaluate_output(
            WorkflowType.SUPPORT,
            result,
            grounding_context=["Only managers can approve refunds after RMA review."],
            embed_function=lambda values: values,
        )
    guarded = apply_output_guardrail_result(WorkflowType.SUPPORT, result, decision)

    provenance_calls = [call for call in guard.calls if call["validator"] == "provenance_llm"]
    assert provenance_calls
    assert provenance_calls[0]["metadata"]["sources"] == ["Only managers can approve refunds after RMA review."]
    assert decision.action == GuardrailAction.REVIEW_REQUIRED
    assert guarded["qa_status"] == "REVIEW_REQUIRED"
    assert guarded["requires_approval"] is True
    assert "GUARDRAIL_REVIEW_REQUIRED" in guarded["compliance_flags"]
    assert guardrail_requires_override(guarded) is True


def test_missing_hub_validator_fails_fast() -> None:
    temp_dir = Path(".tmp")
    temp_dir.mkdir(exist_ok=True)
    config_path = temp_dir / "guardrails-missing-validator.yaml"
    try:
        config_path.write_text(
            """
mode: monitor
guards:
  support:
    input:
      validators:
        - id: missing_validator
          hub: hub://guardrails/not_real
          class_name: NotRealValidator
          policy: block
          severity: high
""",
            encoding="utf-8",
        )

        with pytest.raises(GuardrailConfigurationError, match="not installed|does not expose"):
            WorkflowGuardrailService(config_path).evaluate_input(WorkflowType.SUPPORT, {"inquiry": "hello"})
    finally:
        config_path.unlink(missing_ok=True)


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


class FakeGuardrailService:
    def evaluate_action(
        self,
        workflow_type: WorkflowType | str,
        action_type: str,
        payload: dict[str, object],
    ) -> GuardrailDecision:
        return GuardrailDecision(
            workflow_type=WorkflowType.SUPPORT.value,
            stage=GuardrailStage.ACTION,
            action=GuardrailAction.ALLOW,
            severity=GuardrailSeverity.NONE,
            findings=[],
            sanitized_payload=payload,
            metadata={"action_type": action_type},
        )


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

    with (
        patch("api.routes.SupportInboxStore", return_value=FakeSupportStore(fake_db)),
        patch("api.routes.WorkflowGuardrailService", return_value=FakeGuardrailService()),
    ):
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

    with (
        patch("api.routes.SupportInboxStore", return_value=fake_store),
        patch("api.routes.WorkflowGuardrailService", return_value=FakeGuardrailService()),
    ):
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
