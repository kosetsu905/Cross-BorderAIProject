from __future__ import annotations

import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import SessionLocal
from db_models import SupportConversationRecord, SupportMessageRecord
from models import WorkflowType
from runtime_config import RuntimeConfig, load_runtime_config
from services.whatsapp_provider import get_whatsapp_provider
from services.whatsapp_tmpl_mgr import WhatsAppTemplateManager
from services.workflow_guardrails import (
    GuardrailAction,
    WorkflowGuardrailService,
    decision_result_payload,
)
from support_inbox import SupportInboxStore
from tools.custom.gmail_tools import resolve_gmail_access_token, send_gmail_reply_message
from utils.support_drafts import customer_facing_draft_text, is_unsafe_customer_draft, parse_json_like_object

logger = logging.getLogger(__name__)
AUTO_DISPATCH_UNSUPPORTED_RMA_RE = re.compile(
    r"\b(?:eligible\s+for\s+a\s+return|return\s+approved|pleased\s+to\s+inform\s+you|"
    r"prepaid\s+return\s+label|labels\.example\.local|tracking\s+number\s+RTN)\b",
    re.IGNORECASE,
)


async def process_completed_support_job(
    *,
    job_id: str,
    inputs: dict[str, Any],
    result: dict[str, Any],
    config_context: dict[str, Any] | RuntimeConfig | None = None,
) -> dict[str, Any]:
    conversation_id = str(
        inputs.get("session_id")
        or inputs.get("ticket_id")
        or result.get("session_id")
        or result.get("ticket_id")
        or ""
    )
    if not conversation_id:
        return {"status": "skipped", "reason": "missing_conversation_id"}

    with SessionLocal() as db:
        store = SupportInboxStore(db)
        store.sync_job_result(
            conversation_id,
            {
                "job_id": job_id,
                "workflow_type": "support",
                "result": result,
            },
        )
        conversation = db.get(SupportConversationRecord, conversation_id)
        if conversation is None:
            return {"status": "skipped", "reason": "conversation_not_found"}
        if conversation.escalation_flag:
            return {"status": "skipped", "reason": "human_handoff"}
        if conversation.requires_approval:
            return {"status": "skipped", "reason": "requires_approval"}
        if _auto_dispatch_already_recorded(db, conversation_id, job_id):
            return {"status": "skipped", "reason": "already_dispatched"}

        draft_text = customer_facing_draft_text(conversation.draft_response or result.get("drafted_response"))
        if not draft_text:
            return {"status": "skipped", "reason": "missing_draft_response"}
        if is_unsafe_customer_draft(str(draft_text)):
            return {"status": "skipped", "reason": "unsafe_draft_response"}
        if _contains_unsafe_rma_auto_dispatch_claim(str(draft_text), conversation.draft_payload):
            return {"status": "skipped", "reason": "unsafe_rma_claim"}

        action_decision = WorkflowGuardrailService().evaluate_action(
            WorkflowType.SUPPORT,
            f"{conversation.channel}.send",
            {
                "conversation_id": conversation_id,
                "channel": conversation.channel,
                "draft_text": str(draft_text),
                "draft_payload": conversation.draft_payload or {},
            },
        )
        if action_decision.action in {GuardrailAction.REVIEW_REQUIRED, GuardrailAction.BLOCK}:
            conversation.requires_approval = True
            draft_payload = conversation.draft_payload if isinstance(conversation.draft_payload, dict) else {}
            conversation.draft_payload = {
                **draft_payload,
                "guardrail_decision": decision_result_payload(action_decision),
            }
            db.commit()
            return {
                "status": "skipped",
                "reason": "guardrail_review_required",
                "guardrail_decision": decision_result_payload(action_decision),
            }

        config = _config_object(config_context)
        if conversation.channel == "gmail":
            delivery, raw_payload = _send_gmail_auto_reply(
                config=config,
                conversation=conversation,
                conversation_id=conversation_id,
                draft_text=str(draft_text),
                db=db,
                job_id=job_id,
            )
        elif conversation.channel == "whatsapp":
            delivery, raw_payload = await _send_whatsapp_auto_reply(
                config=config,
                store=store,
                conversation=conversation,
                conversation_id=conversation_id,
                draft_text=str(draft_text),
                job_id=job_id,
            )
        else:
            return {"status": "skipped", "reason": f"unsupported_channel:{conversation.channel}"}

        outbound = store.record_outbound_message(
            conversation_id=conversation_id,
            text=str(draft_text),
            channel_message_id=delivery.get("message_id"),
            delivery_status=str(delivery.get("status") or "failed"),
            raw_payload=raw_payload,
        )
    return {
        "status": delivery.get("status"),
        "conversation_id": conversation_id,
        "message_id": outbound.message_id,
        "channel_message_id": outbound.channel_message_id,
        "send_mode": raw_payload.get("send_mode"),
        "provider": raw_payload.get("provider"),
        "error": delivery.get("error"),
    }


def _config_object(config_context: dict[str, Any] | RuntimeConfig | None) -> object:
    if config_context is None:
        return load_runtime_config()
    if isinstance(config_context, RuntimeConfig):
        return config_context
    base = load_runtime_config().as_context()
    base.update(
        {
            key: value
            for key, value in config_context.items()
            if value not in (None, "")
        }
    )
    return SimpleNamespace(**base)


def _auto_dispatch_already_recorded(db: Any, conversation_id: str, job_id: str) -> bool:
    records = db.execute(
        select(SupportMessageRecord)
        .where(SupportMessageRecord.conversation_id == conversation_id)
        .where(SupportMessageRecord.direction == "outbound")
        .order_by(SupportMessageRecord.created_at.desc())
    ).scalars()
    for record in records:
        raw_payload = record.raw_payload if isinstance(record.raw_payload, dict) else {}
        if raw_payload.get("auto_dispatch") and raw_payload.get("source_job_id") == job_id:
            return True
        delivery = raw_payload.get("delivery") if isinstance(raw_payload.get("delivery"), dict) else {}
        if raw_payload.get("auto_dispatch") and str(delivery.get("status") or record.status).lower() == "sent":
            return True
    return False


def _contains_unsafe_rma_auto_dispatch_claim(draft_text: str, draft_payload: Any) -> bool:
    if not AUTO_DISPATCH_UNSUPPORTED_RMA_RE.search(draft_text):
        return False
    payload = draft_payload if isinstance(draft_payload, dict) else {}
    support_response = payload.get("support_response") if isinstance(payload.get("support_response"), dict) else {}
    rma = payload.get("rma_validation") if isinstance(payload.get("rma_validation"), dict) else None
    if rma is None and isinstance(support_response, dict):
        nested = support_response.get("rma_validation")
        if isinstance(nested, dict):
            rma = nested
        elif isinstance(nested, str):
            parsed = parse_json_like_object(nested)
            rma = parsed if isinstance(parsed, dict) else None
    logistics = payload.get("logistics_output") or (support_response.get("logistics_output") if isinstance(support_response, dict) else None)
    if isinstance(rma, dict) and rma.get("eligible_for_return") is False:
        return True
    return logistics in (None, "", "null")


def _send_gmail_auto_reply(
    *,
    config: object,
    conversation: SupportConversationRecord,
    conversation_id: str,
    draft_text: str,
    db: Any,
    job_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _bool_config(config, "gmail_send_enabled"):
        delivery = {
            "status": "disabled",
            "recipient": conversation.customer_handle,
            "message_id": None,
            "error": "GMAIL_SEND_ENABLED is false; no provider call was made.",
        }
        return delivery, _raw_payload(delivery, "text", job_id)

    access_token = resolve_gmail_access_token(
        access_token=getattr(config, "gmail_access_token", None),
        client_id=getattr(config, "gmail_client_id", None),
        client_secret=getattr(config, "gmail_client_secret", None),
        refresh_token=getattr(config, "gmail_refresh_token", None),
    )
    if not access_token or not getattr(config, "gmail_sender_email", None) or not conversation.customer_handle:
        delivery = {
            "status": "missing_credentials",
            "recipient": conversation.customer_handle,
            "message_id": None,
            "error": "Gmail access token or refresh-token credentials, sender email, and recipient are required.",
        }
        return delivery, _raw_payload(delivery, "text", job_id)

    headers = _latest_inbound_headers(conversation_id, db)
    delivery = send_gmail_reply_message(
        access_token=str(access_token),
        sender=str(getattr(config, "gmail_sender_email")),
        recipient=conversation.customer_handle,
        subject=str(headers.get("subject") or f"Support ticket {conversation.conversation_id}"),
        body=draft_text,
        thread_id=conversation.channel_thread_id,
        in_reply_to=headers.get("message-id"),
        references=headers.get("references"),
    )
    return delivery, _raw_payload(delivery, "text", job_id)


async def _send_whatsapp_auto_reply(
    *,
    config: object,
    store: SupportInboxStore,
    conversation: SupportConversationRecord,
    conversation_id: str,
    draft_text: str,
    job_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _bool_config(config, "whatsapp_send_enabled"):
        delivery = {
            "status": "disabled",
            "recipient": conversation.customer_handle,
            "message_id": None,
            "error": "WHATSAPP_SEND_ENABLED is false; no provider call was made.",
        }
        return delivery, _raw_payload(delivery, "text", job_id)
    if not conversation.customer_handle:
        delivery = {
            "status": "missing_credentials",
            "recipient": None,
            "message_id": None,
            "error": "WhatsApp recipient is required.",
        }
        return delivery, _raw_payload(delivery, "text", job_id)

    draft_payload = conversation.draft_payload or {}
    session_state = store.session_manager.load_session(conversation_id) or {}
    provider = get_whatsapp_provider(config)
    if store.session_manager.is_window_expired(conversation_id):
        language = _whatsapp_template_language(draft_payload, session_state)
        template = WhatsAppTemplateManager.from_config(config).template_for_language(language)
        delivery = await provider.send_template_message(
            to=conversation.customer_handle,
            template_name=template["name"],
            language_code=template["lang"],
            parameters=[],
        )
        raw_payload = _raw_payload(delivery, "template", job_id)
        raw_payload["session_language"] = session_state.get("language_preference")
        return delivery, raw_payload

    delivery = await provider.send_text_message(
        to=conversation.customer_handle,
        body=draft_text,
    )
    return delivery, _raw_payload(delivery, "text", job_id)


def _raw_payload(delivery: dict[str, Any], send_mode: str, job_id: str) -> dict[str, Any]:
    return {
        "auto_dispatch": True,
        "source_job_id": job_id,
        "send_mode": send_mode,
        "provider": delivery.get("provider"),
        "delivery": delivery,
    }


def _whatsapp_template_language(draft_payload: dict[str, Any], session_state: dict[str, Any]) -> str:
    sentiment = draft_payload.get("sentiment_analysis")
    if not isinstance(sentiment, dict):
        sentiment = {}
    for value in (
        draft_payload.get("detected_language"),
        draft_payload.get("language_detected"),
        sentiment.get("language_detected"),
        session_state.get("language_preference"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "en"


def _latest_inbound_headers(conversation_id: str, db: Any) -> dict[str, str]:
    record = db.execute(
        select(SupportMessageRecord)
        .where(SupportMessageRecord.conversation_id == conversation_id)
        .where(SupportMessageRecord.direction == "inbound")
        .order_by(SupportMessageRecord.created_at.desc())
    ).scalars().first()
    raw_payload = record.raw_payload if record else None
    headers = (raw_payload or {}).get("headers") if isinstance(raw_payload, dict) else None
    return headers if isinstance(headers, dict) else {}


def _bool_config(config: object, key: str, default: bool = False) -> bool:
    value = getattr(config, key, None)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}
