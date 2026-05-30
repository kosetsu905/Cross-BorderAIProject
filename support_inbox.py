from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db_models import SupportConversationRecord, SupportMessageRecord
from models import WorkflowType
from runtime_config import load_runtime_config
from services.language_detector import LanguageDetector
from services.session_manager import SessionManager
from tools.custom.support_handoff_tools import send_handoff_notification
from tools.custom.whatsapp_tools import ChannelMessage, WhatsAppStatusUpdate, local_outbound_message_id


def mask_contact(value: str | None) -> str | None:
    if not value:
        return None
    clean = str(value)
    if "@" in clean:
        local, domain = clean.split("@", 1)
        return f"{local[:2]}***@{domain}"
    digits = "".join(character for character in clean if character.isdigit())
    if len(digits) <= 4:
        return "***"
    return f"{clean[:2]}***{digits[-4:]}"


def _conversation_status_from_result(result: dict[str, Any]) -> str:
    if _requires_handoff(result):
        return "handoff_required"
    if _draft_response_from_result(result):
        return "draft_ready"
    return "processing"


def _next_conversation_status(current_status: str | None, result: dict[str, Any]) -> str:
    next_status = _conversation_status_from_result(result)
    if current_status in {"sent", "send_failed"} and next_status == "draft_ready":
        return current_status
    return next_status


def _requires_approval(result: dict[str, Any]) -> bool:
    sentiment = result.get("sentiment_analysis") or {}
    rma = result.get("rma_validation") or {}
    qa_status = str(result.get("qa_status") or "").upper()
    confidence = float(result.get("routing_confidence") or 1)
    if _is_pre_sales(result):
        compliance_flags = {str(flag).upper() for flag in result.get("compliance_flags") or []}
        return (
            confidence < 0.75
            or qa_status == "REJECTED"
            or _has_pre_sales_hard_blocker(result)
        )
    forced_review = bool(
        _requires_handoff(result)
        or result.get("escalation_flag")
        or qa_status in {"REVIEW_REQUIRED", "REJECTED"}
        or confidence < 0.75
        or sentiment.get("customer_tier") in {"VIP", "PREMIUM"}
        or float(sentiment.get("sentiment_score") or 0) < -0.25
        or sentiment.get("intent_category") == "BILLING_ISSUE"
        or (rma and not rma.get("eligible_for_return", True))
    )
    if forced_review:
        return True
    if "requires_approval" in result and result.get("requires_approval") is not None:
        return bool(result.get("requires_approval"))
    if qa_status == "APPROVED":
        return False
    return True


def _requires_handoff(result: dict[str, Any]) -> bool:
    action = result.get("channel_recommended_action")
    if _is_pre_sales(result):
        return _has_pre_sales_hard_blocker(result)
    return bool(
        result.get("escalation_needed")
        or result.get("escalation_flag")
        or action == "human_handoff"
    )


def _is_pre_sales(result: dict[str, Any]) -> bool:
    return str(result.get("detected_intent") or "").lower() == "pre_sales"


def _has_pre_sales_hard_blocker(result: dict[str, Any]) -> bool:
    if result.get("escalation_flag") or result.get("channel_recommended_action") == "human_handoff":
        return True
    hard_flags = {
        "LOW_ROUTING_CONFIDENCE",
        "HUMAN_HANDOFF",
        "POLICY_GAP",
        "BILLING_DISPUTE",
        "UNSAFE_RESPONSE",
        "VIP_REVIEW",
        "NEGATIVE_SENTIMENT",
    }
    compliance_flags = {str(flag).upper() for flag in result.get("compliance_flags") or []}
    return bool(hard_flags.intersection(compliance_flags))


def _handoff_notification_sent(payload: dict[str, Any] | None) -> bool:
    notification = (payload or {}).get("handoff_notification")
    return isinstance(notification, dict) and notification.get("status") == "sent"


def _handoff_inquiry_preview(result: dict[str, Any]) -> str:
    return str(
        result.get("inquiry_text")
        or result.get("ticket_summary")
        or result.get("final_response")
        or result.get("drafted_response")
        or result.get("internal_notes")
        or ""
    )


def _draft_response_from_result(result: dict[str, Any]) -> str | None:
    value = result.get("final_response") or result.get("drafted_response")
    return str(value) if value else None


class SupportInboxStore:
    def __init__(self, session: Session, session_manager: SessionManager | None = None) -> None:
        self.session = session
        self.session_manager = session_manager or SessionManager.from_config(load_runtime_config())

    def upsert_inbound_message(self, message: ChannelMessage) -> tuple[SupportConversationRecord, SupportMessageRecord, bool]:
        existing = self.session.execute(
            select(SupportMessageRecord).where(
                SupportMessageRecord.channel == message.channel,
                SupportMessageRecord.channel_message_id == message.channel_message_id,
            )
        ).scalar_one_or_none()
        if existing:
            conversation = self.session.get(SupportConversationRecord, existing.conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation {existing.conversation_id} not found for message {existing.message_id}")
            return conversation, existing, False

        conversation = self.session.execute(
            select(SupportConversationRecord).where(
                SupportConversationRecord.channel == message.channel,
                SupportConversationRecord.channel_thread_id == message.channel_thread_id,
            )
        ).scalar_one_or_none()
        if conversation is None:
            profile_name = _profile_display_name(message)
            conversation = SupportConversationRecord(
                conversation_id=str(uuid.uuid4()),
                channel=message.channel,
                channel_thread_id=message.channel_thread_id,
                customer_display_name=profile_name,
                customer_handle=message.sender,
                customer_handle_masked=mask_contact(message.sender),
                status="open",
                requires_approval=True,
                escalation_flag=False,
                last_message_at=message.received_at,
            )
            self.session.add(conversation)
        else:
            conversation.customer_handle = conversation.customer_handle or message.sender
            conversation.customer_handle_masked = conversation.customer_handle_masked or mask_contact(message.sender)
            conversation.last_message_at = message.received_at
            if conversation.status in {"sent", "closed"}:
                conversation.status = "open"

        record = SupportMessageRecord(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation.conversation_id,
            channel=message.channel,
            channel_thread_id=message.channel_thread_id,
            channel_message_id=message.channel_message_id,
            direction="inbound",
            sender=message.sender,
            sender_masked=mask_contact(message.sender),
            recipient=message.recipient,
            recipient_masked=mask_contact(message.recipient),
            text=message.text,
            attachments=message.attachments,
            locale=message.locale,
            raw_payload=message.raw_payload,
            status="received",
            received_at=message.received_at,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(conversation)
        self.session.refresh(record)
        self._record_inbound_session(conversation, record, message)
        return conversation, record, True

    def build_support_inputs(
        self,
        conversation: SupportConversationRecord,
        message: SupportMessageRecord,
    ) -> dict[str, Any]:
        session_state = self.session_manager.load_session(conversation.conversation_id)
        session_history = session_state.get("history") if isinstance(session_state, dict) else None
        history = session_history if isinstance(session_history, list) else self._db_history(conversation.conversation_id)
        display_name = conversation.customer_display_name or conversation.customer_handle_masked or "Channel Customer"
        inquiry_text = message.text or "[Customer sent an attachment or unsupported message.]"
        session_language = session_state.get("language_preference") if isinstance(session_state, dict) else None
        message_locale = getattr(message, "locale", None)
        detected_language = (
            message_locale
            or (LanguageDetector.detect(inquiry_text) if inquiry_text.strip() else None)
            or session_language
            or "en"
        )
        language_plan = LanguageDetector.get_crewai_language_plan(str(detected_language))
        inputs = {
            "customer": display_name,
            "person": display_name,
            "inquiry": inquiry_text,
            "ticket_id": conversation.conversation_id,
            "phone_number": conversation.customer_handle if conversation.channel == "whatsapp" else None,
            "inquiry_text": inquiry_text,
            "channel": conversation.channel,
            "session_id": conversation.conversation_id,
            "channel_thread_id": conversation.channel_thread_id,
            "channel_message_id": message.channel_message_id,
            "sender_profile": {
                "display_name": conversation.customer_display_name,
                "handle_masked": conversation.customer_handle_masked,
            },
            "attachments": message.attachments or [],
            "conversation_history": history[-20:],
            "detected_language": detected_language,
            "language_plan": language_plan,
        }
        if detected_language:
            self.session_manager.update_language_preference(conversation.conversation_id, str(detected_language))
        if conversation.channel == "gmail":
            inputs["customer_email"] = conversation.customer_handle
        return inputs

    def attach_job(self, conversation_id: str, message_id: str, job_id: str) -> None:
        conversation = self.session.get(SupportConversationRecord, conversation_id)
        message = self.session.get(SupportMessageRecord, message_id)
        if conversation:
            conversation.latest_job_id = job_id
            conversation.status = "processing"
        if message:
            message.job_id = job_id
            message.status = "processing"
        self.session.commit()

    def apply_status_update(self, status: WhatsAppStatusUpdate) -> bool:
        message = self.session.execute(
            select(SupportMessageRecord).where(
                SupportMessageRecord.channel == "whatsapp",
                SupportMessageRecord.channel_message_id == status.channel_message_id,
            )
        ).scalar_one_or_none()
        if message is None:
            return False
        message.provider_status = status.status
        message.status = status.status
        message.raw_payload = {
            **(message.raw_payload or {}),
            "latest_status_webhook": status.raw_payload,
        }
        self.session.commit()
        return True

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        records = self.session.execute(
            select(SupportConversationRecord)
            .order_by(SupportConversationRecord.updated_at.desc())
            .limit(max(1, min(limit, 200)))
        ).scalars()
        return [self._conversation_to_dict(record, include_messages=False) for record in records]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        conversation = self.session.get(SupportConversationRecord, conversation_id)
        if conversation is None:
            return None
        return self._conversation_to_dict(conversation, include_messages=True)

    def sync_job_result(self, conversation_id: str, job_data: dict[str, Any] | None) -> None:
        if not job_data or job_data.get("workflow_type") not in {None, WorkflowType.SUPPORT.value, "support"}:
            return
        result = job_data.get("result")
        if not isinstance(result, dict):
            return
        conversation = self.session.get(SupportConversationRecord, conversation_id)
        if conversation is None:
            return
        previous_payload = conversation.draft_payload if isinstance(conversation.draft_payload, dict) else {}
        draft_payload = dict(result)
        conversation.draft_response = _draft_response_from_result(result) or conversation.draft_response
        conversation.requires_approval = _requires_approval(result)
        conversation.escalation_flag = _requires_handoff(result)
        conversation.status = _next_conversation_status(conversation.status, result)
        if conversation.escalation_flag:
            if _handoff_notification_sent(previous_payload):
                draft_payload["handoff_notification"] = previous_payload["handoff_notification"]
            else:
                config = load_runtime_config()
                draft_payload["handoff_notification"] = send_handoff_notification(
                    webhook_url=config.support_handoff_webhook_url,
                    session_id=str(result.get("session_id") or conversation.conversation_id),
                    channel=conversation.channel,
                    inquiry_text=_handoff_inquiry_preview(result),
                    context={
                        "conversation_id": conversation.conversation_id,
                        "job_id": job_data.get("job_id"),
                        "ticket_id": result.get("ticket_id") or result.get("session_id"),
                        "customer_email": result.get("customer_email"),
                        "channel_recommended_action": result.get("channel_recommended_action") or result.get("qa_status"),
                        "escalation_flag": result.get("escalation_flag") or result.get("escalation_needed"),
                    },
                )
        conversation.draft_payload = draft_payload
        self.session.commit()

    def assign_conversation(self, conversation_id: str, assigned_to: str | None) -> dict[str, Any] | None:
        conversation = self.session.get(SupportConversationRecord, conversation_id)
        if conversation is None:
            return None
        conversation.assigned_to = assigned_to
        if assigned_to:
            conversation.status = "assigned"
        self.session.commit()
        return self._conversation_to_dict(conversation, include_messages=True)

    def record_outbound_message(
        self,
        *,
        conversation_id: str,
        text: str,
        channel_message_id: str | None,
        delivery_status: str,
        raw_payload: dict[str, Any] | None = None,
    ) -> SupportMessageRecord:
        conversation = self.session.get(SupportConversationRecord, conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found")
        message = SupportMessageRecord(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation.conversation_id,
            channel=conversation.channel,
            channel_thread_id=conversation.channel_thread_id,
            channel_message_id=channel_message_id or local_outbound_message_id(conversation.channel),
            direction="outbound",
            sender=None,
            recipient=conversation.customer_handle,
            recipient_masked=conversation.customer_handle_masked,
            text=text,
            attachments=[],
            raw_payload=raw_payload or {},
            status=delivery_status,
            provider_status=delivery_status,
            received_at=datetime.now(UTC),
        )
        self.session.add(message)
        conversation.status = "sent" if delivery_status == "sent" else "send_failed"
        self.session.commit()
        self.session.refresh(message)
        self._record_outbound_session(conversation, message)
        return message

    def _db_history(self, conversation_id: str) -> list[dict[str, Any]]:
        return [
            _json_safe(self._message_to_dict(record, include_raw=False))
            for record in self.session.execute(
                select(SupportMessageRecord)
                .where(SupportMessageRecord.conversation_id == conversation_id)
                .order_by(SupportMessageRecord.created_at.asc())
            ).scalars()
        ]

    def _record_inbound_session(
        self,
        conversation: SupportConversationRecord,
        record: SupportMessageRecord,
        message: ChannelMessage,
    ) -> None:
        self.session_manager.record_inbound_message(
            session_id=conversation.conversation_id,
            channel=conversation.channel,
            customer_id=conversation.customer_handle,
            language_preference=message.locale,
            metadata={
                "channel_thread_id": conversation.channel_thread_id,
                "customer_handle_masked": conversation.customer_handle_masked,
            },
            message=_json_safe(self._message_to_dict(record, include_raw=False)),
        )

    def _record_outbound_session(
        self,
        conversation: SupportConversationRecord,
        record: SupportMessageRecord,
    ) -> None:
        self.session_manager.record_outbound_message(
            session_id=conversation.conversation_id,
            channel=conversation.channel,
            customer_id=conversation.customer_handle,
            metadata={
                "channel_thread_id": conversation.channel_thread_id,
                "customer_handle_masked": conversation.customer_handle_masked,
            },
            message=_json_safe(self._message_to_dict(record, include_raw=False)),
        )

    def _conversation_to_dict(
        self,
        record: SupportConversationRecord,
        *,
        include_messages: bool,
    ) -> dict[str, Any]:
        payload = {
            "conversation_id": record.conversation_id,
            "channel": record.channel,
            "channel_thread_id": record.channel_thread_id,
            "customer_display_name": record.customer_display_name,
            "customer_handle_masked": record.customer_handle_masked,
            "assigned_to": record.assigned_to,
            "status": record.status,
            "latest_job_id": record.latest_job_id,
            "draft_response": record.draft_response,
            "draft_payload": record.draft_payload,
            "requires_approval": record.requires_approval,
            "escalation_flag": record.escalation_flag,
            "last_message_at": record.last_message_at,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        if include_messages:
            messages = self.session.execute(
                select(SupportMessageRecord)
                .where(SupportMessageRecord.conversation_id == record.conversation_id)
                .order_by(SupportMessageRecord.created_at.asc())
            ).scalars()
            payload["messages"] = [self._message_to_dict(message) for message in messages]
        return payload

    def _message_to_dict(
        self,
        record: SupportMessageRecord,
        *,
        include_raw: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "message_id": record.message_id,
            "conversation_id": record.conversation_id,
            "channel": record.channel,
            "channel_thread_id": record.channel_thread_id,
            "channel_message_id": record.channel_message_id,
            "direction": record.direction,
            "sender_masked": record.sender_masked,
            "recipient_masked": record.recipient_masked,
            "text": record.text,
            "attachments": record.attachments or [],
            "locale": record.locale,
            "status": record.status,
            "provider_status": record.provider_status,
            "job_id": record.job_id,
            "received_at": record.received_at,
            "created_at": record.created_at,
        }
        if include_raw:
            payload["raw_payload"] = record.raw_payload
        return payload


def _profile_display_name(message: ChannelMessage) -> str | None:
    if not message.sender_profile:
        return None
    if message.channel == "whatsapp":
        return (message.sender_profile.get("profile") or {}).get("name")
    return message.sender_profile.get("display_name") or message.sender_profile.get("email")


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
