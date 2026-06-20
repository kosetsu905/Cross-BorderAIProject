from __future__ import annotations

import json
import re
from typing import Any

DRAFT_TEXT_FIELDS = ("final_response", "drafted_response", "response", "body")
JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
MAX_DRAFT_EXTRACTION_DEPTH = 3


def parse_json_like_object(value: str) -> dict[str, Any] | None:
    """Parse a whole-string JSON object, accepting markdown JSON fences."""
    candidate = _strip_json_fence(value)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def customer_facing_draft_text(value: Any) -> str | None:
    """Return the customer-facing response text from plain or structured draft output."""
    return _customer_facing_draft_text(value, depth=0)


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

    extracted = _customer_facing_draft_text(parsed, depth=depth + 1)
    return extracted or text


def _strip_json_fence(value: str) -> str:
    match = JSON_FENCE_RE.match(value)
    if not match:
        return value.strip()
    return match.group("body").strip()
