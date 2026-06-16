import json
import re
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|client[_-]?secret|credential|password|refresh[_-]?token|secret|token)",
    re.IGNORECASE,
)
PII_KEY_RE = re.compile(
    r"(customer[_-]?email|customer[_-]?handle|email|phone|recipient|sender)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
DEFAULT_TEXT_MAX_CHARS = 4000


class CompactContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    content: str = Field(..., description="Redacted compact YAML context.")
    original_chars: int = Field(..., ge=0)
    compacted_chars: int = Field(..., ge=0)
    truncated: bool = False


class ConversationHistoryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    total_messages: int = Field(0, ge=0)
    summarized_messages: int = Field(0, ge=0)


def compact_context(
    name: str,
    payload: Any,
    *,
    max_chars: int,
    text_max_chars: int = DEFAULT_TEXT_MAX_CHARS,
) -> CompactContext:
    redacted_payload = redact_sensitive(payload)
    safe_payload = compact_value(redacted_payload, text_max_chars)
    original_text = _safe_dump(redacted_payload)
    compact_text = _safe_dump(safe_payload)
    truncated = _was_compacted(redacted_payload, safe_payload)

    if len(compact_text) > max_chars:
        tighter_limit = max(250, min(text_max_chars, max_chars // 4))
        safe_payload = compact_value(safe_payload, tighter_limit)
        compact_text = _safe_dump(safe_payload)
        truncated = True

    if len(compact_text) > max_chars:
        compact_text = truncate_text(compact_text, max_chars)
        truncated = True

    return CompactContext(
        name=name,
        content=compact_text,
        original_chars=len(original_text),
        compacted_chars=len(compact_text),
        truncated=truncated,
    )


def compact_value(value: Any, text_max_chars: int = DEFAULT_TEXT_MAX_CHARS) -> Any:
    if isinstance(value, dict):
        return {str(key): compact_value(item, text_max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_value(item, text_max_chars) for item in value]
    if isinstance(value, tuple):
        return [compact_value(item, text_max_chars) for item in value]
    if isinstance(value, str):
        return truncate_text(value, text_max_chars)
    return value


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SECRET_KEY_RE.search(key_text):
                redacted[key_text] = "[REDACTED_SECRET]"
            elif PII_KEY_RE.search(key_text):
                redacted[key_text] = _redact_text(str(item), pii_only=True) if item is not None else None
            else:
                redacted[key_text] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def build_conversation_history_context(
    history: Any,
    *,
    recent_count: int = 3,
    message_max_chars: int = 600,
) -> ConversationHistoryContext:
    messages = history if isinstance(history, list) else []
    safe_messages = [
        compact_value(redact_sensitive(message), message_max_chars)
        for message in messages
        if isinstance(message, dict)
    ]
    recent_messages = safe_messages[-recent_count:] if recent_count > 0 else []
    older_messages = safe_messages[: max(0, len(safe_messages) - len(recent_messages))]
    summary_parts: list[str] = []
    if older_messages:
        directions: dict[str, int] = {}
        for message in older_messages:
            direction = str(message.get("direction") or "unknown")
            directions[direction] = directions.get(direction, 0) + 1
        summary_parts.append(f"{len(older_messages)} older messages summarized")
        summary_parts.append(
            "directions="
            + ", ".join(f"{direction}:{count}" for direction, count in sorted(directions.items()))
        )
        last_text = str(older_messages[-1].get("text") or older_messages[-1].get("body") or "").strip()
        if last_text:
            summary_parts.append(f"latest_older_message={truncate_text(last_text, message_max_chars)}")
    else:
        summary_parts.append("No older conversation messages before the recent window.")

    return ConversationHistoryContext(
        summary="; ".join(summary_parts),
        recent_messages=recent_messages,
        total_messages=len(safe_messages),
        summarized_messages=len(older_messages),
    )


def compact_handoff_payload(
    sections: dict[str, Any],
    *,
    config_context: dict[str, Any],
    name: str = "shared_context",
) -> CompactContext:
    workflow_limit = _positive_int(config_context.get("workflow_context_max_chars"), 12000)
    task_limit = _positive_int(config_context.get("task_context_max_chars"), 4000)
    return compact_context(
        name,
        sections,
        max_chars=workflow_limit,
        text_max_chars=task_limit,
    )


def truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 32:
        return value[:max_chars]
    head_chars = max_chars - 32
    return f"{value[:head_chars]}... [truncated {len(value) - head_chars} chars]"


def _redact_text(value: str, *, pii_only: bool = False) -> str:
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    if pii_only:
        return redacted
    return redacted


def _safe_dump(value: Any) -> str:
    try:
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    except Exception:
        return json.dumps(value, ensure_ascii=False, default=str)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _was_compacted(original: Any, compacted: Any) -> bool:
    if isinstance(original, dict) and isinstance(compacted, dict):
        return any(_was_compacted(original.get(key), compacted.get(key)) for key in original)
    if isinstance(original, (list, tuple)) and isinstance(compacted, list):
        if len(original) != len(compacted):
            return True
        return any(_was_compacted(left, right) for left, right in zip(original, compacted, strict=False))
    if isinstance(original, str) and isinstance(compacted, str):
        return original != compacted
    return False
