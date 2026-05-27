from __future__ import annotations

import hashlib
import hmac
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field


class ChannelMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str = Field(..., description="Inbound channel identifier, such as whatsapp")
    channel_thread_id: str
    channel_message_id: str
    sender: str | None = None
    recipient: str | None = None
    text: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    locale: str | None = None
    received_at: datetime
    raw_payload: dict[str, Any]
    sender_profile: dict[str, Any] = Field(default_factory=dict)


class WhatsAppStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_message_id: str
    recipient_id: str | None = None
    status: str
    timestamp: datetime | None = None
    raw_payload: dict[str, Any]


def verify_whatsapp_signature(*, app_secret: str | None, body: bytes, signature: str | None) -> bool:
    if not app_secret:
        return True
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    received = signature.removeprefix("sha256=")
    return hmac.compare_digest(expected, received)


def _timestamp_to_datetime(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(UTC)


def _message_text(message: dict[str, Any]) -> str | None:
    message_type = message.get("type")
    if message_type == "text":
        return (message.get("text") or {}).get("body")
    if message_type == "button":
        return (message.get("button") or {}).get("text")
    if message_type == "interactive":
        interactive = message.get("interactive") or {}
        reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
        return reply.get("title") or reply.get("id")
    return None


def _message_attachments(message: dict[str, Any]) -> list[dict[str, Any]]:
    message_type = message.get("type")
    if message_type in {"image", "audio", "video", "document", "sticker"}:
        media = dict(message.get(message_type) or {})
        media["type"] = message_type
        return [media]
    if message_type == "location":
        return [{"type": "location", **dict(message.get("location") or {})}]
    if message_type == "contacts":
        return [{"type": "contacts", "contacts": message.get("contacts") or []}]
    return []


def parse_whatsapp_webhook(payload: dict[str, Any]) -> tuple[list[ChannelMessage], list[WhatsAppStatusUpdate]]:
    messages: list[ChannelMessage] = []
    statuses: list[WhatsAppStatusUpdate] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            metadata = value.get("metadata") or {}
            business_number = metadata.get("display_phone_number") or metadata.get("phone_number_id")
            contacts_by_wa_id = {
                str(contact.get("wa_id")): contact
                for contact in value.get("contacts") or []
                if contact.get("wa_id")
            }
            for message in value.get("messages") or []:
                sender = str(message.get("from") or "")
                profile = contacts_by_wa_id.get(sender) or {}
                text = _message_text(message)
                attachments = _message_attachments(message)
                if not text and not attachments:
                    text = f"[Unsupported WhatsApp message type: {message.get('type') or 'unknown'}]"
                messages.append(
                    ChannelMessage(
                        channel="whatsapp",
                        channel_thread_id=sender,
                        channel_message_id=str(message.get("id")),
                        sender=sender,
                        recipient=str(business_number) if business_number else None,
                        text=text,
                        attachments=attachments,
                        locale=None,
                        received_at=_timestamp_to_datetime(message.get("timestamp")),
                        raw_payload=message,
                        sender_profile=profile,
                    )
                )
            for status in value.get("statuses") or []:
                statuses.append(
                    WhatsAppStatusUpdate(
                        channel_message_id=str(status.get("id")),
                        recipient_id=status.get("recipient_id"),
                        status=str(status.get("status") or "unknown"),
                        timestamp=_timestamp_to_datetime(status.get("timestamp")) if status.get("timestamp") else None,
                        raw_payload=status,
                    )
                )
    return messages, statuses


def send_whatsapp_text_message(
    *,
    access_token: str,
    phone_number_id: str,
    recipient: str,
    body: str,
    graph_api_version: str = "v23.0",
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    url = f"https://graph.facebook.com/{graph_api_version}/{phone_number_id}/messages"
    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": "text",
                "text": {"preview_url": False, "body": body},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "status": "failed",
            "recipient": recipient,
            "message_id": None,
            "error": f"{exc.response.status_code}: {exc.response.text}",
        }
    except httpx.HTTPError as exc:
        return {
            "status": "failed",
            "recipient": recipient,
            "message_id": None,
            "error": str(exc),
        }

    payload = response.json()
    messages = payload.get("messages") or []
    message_id = messages[0].get("id") if messages else None
    return {
        "status": "sent",
        "recipient": recipient,
        "message_id": message_id,
        "error": None,
        "raw_response": payload,
    }


def local_outbound_message_id(channel: str) -> str:
    return f"{channel}-out-{int(time.time() * 1000)}"
