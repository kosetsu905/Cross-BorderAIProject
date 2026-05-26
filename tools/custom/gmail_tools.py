from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any

import httpx


GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"


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
