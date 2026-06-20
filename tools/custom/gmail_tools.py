from __future__ import annotations

import base64
import html
import re
from datetime import UTC, datetime
from email.header import decode_header
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any

import httpx

from tools.custom.whatsapp_tools import ChannelMessage

GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GMAIL_MESSAGES_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
GMAIL_MESSAGE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}"
GMAIL_WATCH_URL = "https://gmail.googleapis.com/gmail/v1/users/me/watch"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"


def refresh_gmail_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    try:
        response = httpx.post(
            GOOGLE_OAUTH_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "status": "failed",
            "access_token": None,
            "expires_in": None,
            "error": f"{exc.response.status_code}: {exc.response.text}",
        }
    except httpx.HTTPError as exc:
        return {
            "status": "failed",
            "access_token": None,
            "expires_in": None,
            "error": str(exc),
        }

    payload = response.json()
    return {
        "status": "refreshed",
        "access_token": payload.get("access_token"),
        "expires_in": payload.get("expires_in"),
        "scope": payload.get("scope"),
        "token_type": payload.get("token_type"),
        "error": None,
    }


def resolve_gmail_access_token(
    *,
    access_token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    refresh_token: str | None = None,
) -> str | None:
    if client_id and client_secret and refresh_token:
        refreshed = refresh_gmail_access_token(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
        )
        if refreshed.get("access_token"):
            return str(refreshed["access_token"])
    return access_token


def build_gmail_raw_message(
    *,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
) -> str:
    message = EmailMessage()
    message["To"] = recipient
    message["From"] = sender
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def build_gmail_reply_raw_message(
    *,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    in_reply_to: str | None = None,
    references: str | None = None,
) -> str:
    message = EmailMessage()
    message["To"] = recipient
    message["From"] = sender
    message["Subject"] = _reply_subject(subject)
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
    if references or in_reply_to:
        message["References"] = " ".join(value for value in [references, in_reply_to] if value)
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def send_gmail_message(
    *,
    access_token: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    raw_message = build_gmail_raw_message(
        sender=sender,
        recipient=recipient,
        subject=subject,
        body=body,
    )
    try:
        response = httpx.post(
            GMAIL_SEND_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"raw": raw_message},
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
    return {
        "status": "sent",
        "recipient": recipient,
        "message_id": payload.get("id"),
        "error": None,
    }


def get_gmail_message(
    *,
    access_token: str,
    message_id: str,
    message_format: str = "full",
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    response = httpx.get(
        GMAIL_MESSAGE_URL.format(message_id=message_id),
        headers={"Authorization": f"Bearer {access_token}"},
        params={"format": message_format},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def list_gmail_messages(
    *,
    access_token: str,
    label_ids: list[str] | None = None,
    max_results: int = 5,
    query: str | None = None,
    timeout_seconds: int = 20,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"maxResults": max(1, min(max_results, 50))}
    if label_ids:
        params["labelIds"] = label_ids
    if query:
        params["q"] = query
    response = httpx.get(
        GMAIL_MESSAGES_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("messages") or [])


def watch_gmail_mailbox(
    *,
    access_token: str,
    topic_name: str,
    label_ids: list[str] | None = None,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"topicName": topic_name}
    if label_ids:
        payload["labelIds"] = label_ids
        payload["labelFilterBehavior"] = "include"
    response = httpx.post(
        GMAIL_WATCH_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def send_gmail_reply_message(
    *,
    access_token: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    thread_id: str | None = None,
    in_reply_to: str | None = None,
    references: str | None = None,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    raw_message = build_gmail_reply_raw_message(
        sender=sender,
        recipient=recipient,
        subject=subject,
        body=body,
        in_reply_to=in_reply_to,
        references=references,
    )
    request_payload: dict[str, Any] = {"raw": raw_message}
    if thread_id:
        request_payload["threadId"] = thread_id
    try:
        response = httpx.post(
            GMAIL_SEND_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=request_payload,
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
    return {
        "status": "sent",
        "recipient": recipient,
        "message_id": payload.get("id"),
        "thread_id": payload.get("threadId") or thread_id,
        "error": None,
        "raw_response": payload,
    }


def parse_gmail_pubsub_payload(payload: dict[str, Any]) -> dict[str, Any]:
    encoded_data = ((payload.get("message") or {}).get("data")) or ""
    if not encoded_data:
        return {}
    padded = encoded_data + "=" * (-len(encoded_data) % 4)
    decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    import json

    parsed = json.loads(decoded)
    return {
        "email_address": parsed.get("emailAddress"),
        "history_id": parsed.get("historyId"),
        "raw": parsed,
    }


def parse_gmail_message(message: dict[str, Any], mailbox_email: str | None = None) -> ChannelMessage:
    payload = message.get("payload") or {}
    headers = _headers_by_name(payload)
    sender = _first_address(headers.get("from"))
    recipient = _first_address(headers.get("to")) or mailbox_email
    text = _extract_text(payload) or message.get("snippet") or ""
    attachments = _extract_attachments(payload)
    received_at = _header_datetime(headers.get("date"))
    message_id = str(message.get("id") or headers.get("message-id") or "")
    thread_id = str(message.get("threadId") or message_id)
    subject = headers.get("subject") or ""
    return ChannelMessage(
        channel="gmail",
        channel_thread_id=thread_id,
        channel_message_id=message_id,
        sender=sender,
        recipient=recipient,
        text=text,
        attachments=attachments,
        locale=None,
        received_at=received_at,
        raw_payload={
            "id": message.get("id"),
            "threadId": thread_id,
            "labelIds": message.get("labelIds") or [],
            "snippet": message.get("snippet"),
            "headers": headers,
        },
        sender_profile={
            "display_name": _first_display_name(headers.get("from")),
            "email": sender,
            "subject": subject,
            "message_id_header": headers.get("message-id"),
            "references": headers.get("references"),
            "mailbox_email": mailbox_email,
            "is_self_sent": is_mailbox_self_sent_message(message, mailbox_email),
        },
    )


def is_mailbox_self_sent_message(message: dict[str, Any], mailbox_email: str | None) -> bool:
    if not mailbox_email:
        return False
    payload = message.get("payload") or {}
    headers = _headers_by_name(payload)
    sender = _first_address(headers.get("from"))
    mailbox = _normalize_email(mailbox_email)
    if sender and _normalize_email(sender) == mailbox:
        return True
    label_ids = {str(label).upper() for label in message.get("labelIds") or []}
    return "SENT" in label_ids and "INBOX" not in label_ids


def gmail_label_ids(raw_label_ids: str | None) -> list[str]:
    return [label.strip() for label in (raw_label_ids or "").split(",") if label.strip()]


def _reply_subject(subject: str | None) -> str:
    clean_subject = (subject or "").strip()
    if clean_subject.lower().startswith("re:"):
        return clean_subject
    return f"Re: {clean_subject or 'Support request'}"


def _headers_by_name(payload: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for header in payload.get("headers") or []:
        name = str(header.get("name") or "").lower()
        value = header.get("value")
        if name and value is not None:
            headers[name] = _decode_mime_header(str(value))
    return headers


def _decode_mime_header(value: str) -> str:
    try:
        decoded_parts = []
        for content, charset in decode_header(value):
            if isinstance(content, bytes):
                decoded_parts.append(content.decode(charset or "utf-8", errors="replace"))
            else:
                decoded_parts.append(content)
        return "".join(decoded_parts)
    except Exception:
        return value


def _first_address(value: str | None) -> str | None:
    if not value:
        return None
    addresses = getaddresses([value])
    return addresses[0][1] if addresses and addresses[0][1] else value


def _first_display_name(value: str | None) -> str | None:
    if not value:
        return None
    addresses = getaddresses([value])
    return addresses[0][0] if addresses and addresses[0][0] else None


def _normalize_email(value: str) -> str:
    return _first_address(value).lower().strip() if _first_address(value) else value.lower().strip()


def _header_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return datetime.now(UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _decode_body_data(data: str | None, charset: str | None = None) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except ValueError:
        return ""
    encodings = [charset, "utf-8", "gb18030", "gbk", "big5", "latin-1"]
    for encoding in encodings:
        if not encoding:
            continue
        try:
            return raw.decode(str(encoding), errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _charset_from_part(part: dict[str, Any]) -> str | None:
    headers = _headers_by_name(part)
    content_type = headers.get("content-type") or str(part.get("mimeType") or "")
    match = re.search(r"charset=[\"']?([^\"';\s]+)", content_type, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _html_to_text(value: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    with_breaks = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", without_scripts)
    without_tags = re.sub(r"<[^>]+>", " ", with_breaks)
    lines = [line.strip() for line in html.unescape(without_tags).splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_text(payload: dict[str, Any]) -> str:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def visit(part: dict[str, Any]) -> None:
        mime_type = str(part.get("mimeType") or "")
        body_data = (part.get("body") or {}).get("data")
        charset = _charset_from_part(part)
        if mime_type == "text/plain":
            plain_parts.append(_decode_body_data(body_data, charset))
        elif mime_type == "text/html":
            html_parts.append(_html_to_text(_decode_body_data(body_data, charset)))
        for child in part.get("parts") or []:
            visit(child)

    visit(payload)
    plain_text = "\n".join(part.strip() for part in plain_parts if part.strip()).strip()
    if plain_text:
        return plain_text
    return "\n".join(part.strip() for part in html_parts if part.strip()).strip()


def _extract_attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    attachments: list[dict[str, Any]] = []

    def visit(part: dict[str, Any]) -> None:
        filename = part.get("filename")
        body = part.get("body") or {}
        attachment_id = body.get("attachmentId")
        if filename or attachment_id:
            attachments.append(
                {
                    "type": "gmail_attachment",
                    "filename": filename,
                    "mime_type": part.get("mimeType"),
                    "attachment_id": attachment_id,
                    "size": body.get("size"),
                }
            )
        for child in part.get("parts") or []:
            visit(child)

    visit(payload)
    return attachments
