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


def verify_ycloud_signature(
    *,
    webhook_secret: str | None,
    body: bytes,
    signature: str | None,
    tolerance_seconds: int = 300,
) -> bool:
    if not webhook_secret or not signature:
        return False

    signature_parts = {}
    for part in signature.split(","):
        key, separator, value = part.strip().partition("=")
        if separator:
            signature_parts[key] = value
    timestamp = signature_parts.get("t")
    received = signature_parts.get("s")
    if not timestamp or not received:
        return False

    try:
        timestamp_seconds = int(timestamp)
    except ValueError:
        return False
    if tolerance_seconds > 0 and abs(int(time.time()) - timestamp_seconds) > tolerance_seconds:
        return False

    signed_payload = f"{timestamp}.".encode("utf-8") + body
    expected = hmac.new(webhook_secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received)


def _timestamp_to_datetime(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(UTC)


def _optional_timestamp_to_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value), tz=UTC)
    if isinstance(value, str):
        numeric = value.strip()
        if numeric.isdigit():
            return datetime.fromtimestamp(int(numeric), tz=UTC)
        try:
            return datetime.fromisoformat(numeric.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return None


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _message_text(message: dict[str, Any]) -> str | None:
    message_type = message.get("type")
    if message_type == "text":
        text = message.get("text")
        if isinstance(text, str):
            return text
        return _first_text(_dict_value(text).get("body"), message.get("textBody"), message.get("body"))
    if message_type == "button":
        button = _dict_value(message.get("button"))
        return _first_text(button.get("text"), button.get("payload"))
    if message_type == "interactive":
        interactive = _dict_value(message.get("interactive"))
        reply = _dict_value(interactive.get("button_reply") or interactive.get("list_reply"))
        return _first_text(reply.get("title"), reply.get("id"))
    return None


def _message_attachments(message: dict[str, Any]) -> list[dict[str, Any]]:
    message_type = message.get("type")
    if message_type in {"image", "audio", "video", "document", "sticker"}:
        media = dict(_dict_value(message.get(message_type)))
        media["type"] = message_type
        return [media]
    if message_type == "location":
        return [{"type": "location", **dict(_dict_value(message.get("location")))}]
    if message_type == "contacts":
        return [{"type": "contacts", "contacts": message.get("contacts") or []}]
    return []


def _ycloud_event_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def _ycloud_inbound_message(payload: dict[str, Any]) -> dict[str, Any] | None:
    data = _ycloud_event_data(payload)
    candidate = (
        data.get("whatsappInboundMessage")
        or data.get("inboundMessage")
        or data.get("message")
    )
    return candidate if isinstance(candidate, dict) else None


def _ycloud_status_message(payload: dict[str, Any]) -> dict[str, Any] | None:
    data = _ycloud_event_data(payload)
    candidate = data.get("whatsappMessage") or data.get("message")
    return candidate if isinstance(candidate, dict) else None


def _ycloud_sender_profile(message: dict[str, Any]) -> dict[str, Any]:
    profile = (
        _dict_value(message.get("customerProfile"))
        or _dict_value(message.get("senderProfile"))
        or _dict_value(message.get("profile"))
    )
    name = _first_text(profile.get("name"), profile.get("displayName"), message.get("customerName"))
    if name:
        return {"profile": {"name": name}, **profile}
    return profile


def parse_ycloud_webhook(payload: dict[str, Any]) -> tuple[list[ChannelMessage], list[WhatsAppStatusUpdate]]:
    messages: list[ChannelMessage] = []
    statuses: list[WhatsAppStatusUpdate] = []
    event_type = str(payload.get("type") or payload.get("event") or payload.get("eventType") or "")

    inbound_message = _ycloud_inbound_message(payload)
    if inbound_message and event_type in {"", "whatsapp.inbound_message.received", "whatsapp.inbound.message"}:
        sender = _first_text(inbound_message.get("from"), inbound_message.get("waId"), inbound_message.get("customer"))
        message_id = _first_text(
            inbound_message.get("message_id"),
            inbound_message.get("messageId"),
            inbound_message.get("whatsappMessageId"),
            inbound_message.get("id"),
            inbound_message.get("wamid"),
        )
        if sender and message_id:
            text = _message_text(inbound_message)
            attachments = _message_attachments(inbound_message)
            if not text and not attachments:
                text = f"[Unsupported WhatsApp message type: {inbound_message.get('type') or 'unknown'}]"
            received_at = (
                _optional_timestamp_to_datetime(inbound_message.get("timestamp"))
                or _optional_timestamp_to_datetime(inbound_message.get("createTime"))
                or _optional_timestamp_to_datetime(inbound_message.get("sendTime"))
                or _optional_timestamp_to_datetime(payload.get("createdAt"))
                or _optional_timestamp_to_datetime(payload.get("created"))
                or datetime.now(UTC)
            )
            messages.append(
                ChannelMessage(
                    channel="whatsapp",
                    channel_thread_id=sender,
                    channel_message_id=message_id,
                    sender=sender,
                    recipient=_first_text(inbound_message.get("to"), inbound_message.get("phoneNumber")),
                    text=text,
                    attachments=attachments,
                    locale=_first_text(inbound_message.get("language"), inbound_message.get("locale")),
                    received_at=received_at,
                    raw_payload={"event": payload, "message": inbound_message},
                    sender_profile=_ycloud_sender_profile(inbound_message),
                )
            )

    status_message = _ycloud_status_message(payload)
    if status_message and event_type in {"", "whatsapp.message.updated"}:
        channel_message_id = _first_text(
            status_message.get("id"),
            status_message.get("message_id"),
            status_message.get("messageId"),
            status_message.get("whatsappMessageId"),
            status_message.get("wamid"),
        )
        if channel_message_id:
            statuses.append(
                WhatsAppStatusUpdate(
                    channel_message_id=channel_message_id,
                    recipient_id=_first_text(status_message.get("recipientId"), status_message.get("to")),
                    status=str(status_message.get("status") or status_message.get("messageStatus") or "unknown"),
                    timestamp=(
                        _optional_timestamp_to_datetime(status_message.get("updateTime"))
                        or _optional_timestamp_to_datetime(status_message.get("updatedAt"))
                        or _optional_timestamp_to_datetime(status_message.get("timestamp"))
                    ),
                    raw_payload={"event": payload, "message": status_message},
                )
            )
    return messages, statuses


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
