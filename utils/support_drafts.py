from __future__ import annotations

import ast
import json
import re
from typing import Any

DRAFT_TEXT_FIELDS = ("final_response", "drafted_response", "response", "body")
JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
LIVE_E2E_MARKER_PREFIX = "CB-SUPPORT-E2E"
LIVE_E2E_MARKER_LINE_RE = re.compile(
    rf"(?im)^\s*(?:test\s*marker|测试标记)\s*[:：]\s*{re.escape(LIVE_E2E_MARKER_PREFIX)}-[^\r\n]*\s*$"
)
LIVE_E2E_MARKER_TOKEN_RE = re.compile(rf"{re.escape(LIVE_E2E_MARKER_PREFIX)}-[A-Za-z0-9_-]+-\d{{8}}-\d{{6}}")
MAX_DRAFT_EXTRACTION_DEPTH = 3


def parse_json_like_object(value: str) -> dict[str, Any] | None:
    """Parse a whole-string structured object, accepting JSON fences and Python dict literals."""
    candidate = _strip_json_fence(value)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, TypeError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


def customer_facing_draft_text(value: Any) -> str | None:
    """Return the customer-facing response text from plain or structured draft output."""
    return _customer_facing_draft_text(value, depth=0)


def strip_live_e2e_markers(text: str | None) -> str:
    """Remove live E2E marker lines from text passed into business logic."""
    if not text:
        return ""
    without_marker_lines = LIVE_E2E_MARKER_LINE_RE.sub("", str(text))
    lines = [line.rstrip() for line in without_marker_lines.splitlines()]
    compacted = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", compacted).strip()


def contains_live_e2e_marker(text: str | None) -> bool:
    if not text:
        return False
    return LIVE_E2E_MARKER_PREFIX.lower() in str(text).lower()


def is_structured_draft_text(text: str | None) -> bool:
    if not text:
        return False
    stripped = str(text).strip()
    if stripped.startswith("```"):
        return True
    if stripped.startswith("{") and stripped.endswith("}"):
        return parse_json_like_object(stripped) is not None
    return False


def is_unsafe_customer_draft(text: str | None) -> bool:
    """Return true when text should not be auto-sent as a customer response."""
    if not text or not str(text).strip():
        return True
    return is_structured_draft_text(str(text)) or contains_live_e2e_marker(str(text))


def _customer_facing_draft_text(value: Any, *, depth: int) -> str | None:
    if value is None:
        return None
    if depth > MAX_DRAFT_EXTRACTION_DEPTH:
        return str(value)

    if isinstance(value, dict):
        for field_name in DRAFT_TEXT_FIELDS:
            extracted = _customer_facing_draft_text(value.get(field_name), depth=depth + 1)
            if extracted:
                return extracted
        return None

    text = str(value)
    parsed = parse_json_like_object(text)
    if parsed is None:
        return text

    return _customer_facing_draft_text(parsed, depth=depth + 1)


def _strip_json_fence(value: str) -> str:
    match = JSON_FENCE_RE.match(value)
    if not match:
        return value.strip()
    return match.group("body").strip()
