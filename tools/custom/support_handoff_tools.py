from __future__ import annotations

from typing import Any

import httpx


def send_handoff_notification(
    *,
    webhook_url: str | None,
    session_id: str,
    channel: str,
    inquiry_text: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not webhook_url:
        return {"status": "disabled", "error": None}

    preview = inquiry_text.strip()[:200] if inquiry_text else ""
    payload = {
        "text": (
            "Omni-Channel Escalation Required\n"
            f"Session: {session_id}\n"
            f"Channel: {channel}\n"
            f"Inquiry: {preview}..."
        ),
        "session_id": session_id,
        "channel": channel,
        "context": context or {},
    }
    try:
        response = httpx.post(webhook_url, json=payload, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "status": "failed",
            "error": f"{exc.response.status_code}: {exc.response.text}",
        }
    except httpx.HTTPError as exc:
        return {"status": "failed", "error": str(exc)}

    return {"status": "sent", "error": None}
