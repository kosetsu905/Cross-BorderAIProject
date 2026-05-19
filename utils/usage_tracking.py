import os
import time
from collections.abc import Mapping
from typing import Any


INTERNAL_USAGE_KEY = "_usage_metrics"


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def extract_usage_metrics(raw_result: Any) -> dict[str, Any]:
    """Best-effort extraction across CrewAI/OpenAI result versions."""
    candidates = [
        getattr(raw_result, "usage_metrics", None),
        getattr(raw_result, "token_usage", None),
        getattr(raw_result, "usage", None),
        getattr(raw_result, "metrics", None),
    ]

    if isinstance(raw_result, Mapping):
        candidates.extend(
            [
                raw_result.get("usage_metrics"),
                raw_result.get("token_usage"),
                raw_result.get("usage"),
                raw_result.get("metrics"),
            ]
        )

    for candidate in candidates:
        usage = _to_dict(candidate)
        if usage:
            return usage
    return {}


def attach_usage_metrics(payload: dict[str, Any], raw_result: Any) -> dict[str, Any]:
    usage_metrics = extract_usage_metrics(raw_result)
    if not usage_metrics:
        return payload

    payload_with_usage = dict(payload)
    payload_with_usage[INTERNAL_USAGE_KEY] = usage_metrics
    return payload_with_usage


def pop_usage_metrics(payload: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(payload, dict):
        return payload, {}

    clean_payload = dict(payload)
    usage_metrics = clean_payload.pop(INTERNAL_USAGE_KEY, {}) or {}
    return clean_payload, usage_metrics


def _first_int(metrics: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _float_env(name: str) -> float:
    try:
        return float(os.getenv(name, "0") or 0)
    except ValueError:
        return 0.0


def build_usage_summary(
    usage_metrics: dict[str, Any],
    duration_seconds: float,
    config_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_context = config_context or {}
    prompt_tokens = _first_int(usage_metrics, "prompt_tokens", "input_tokens")
    completion_tokens = _first_int(usage_metrics, "completion_tokens", "output_tokens")
    total_tokens = _first_int(usage_metrics, "total_tokens")

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    input_rate = float(config_context.get("openai_input_cost_per_1m_tokens") or 0)
    output_rate = float(config_context.get("openai_output_cost_per_1m_tokens") or 0)
    if input_rate == 0:
        input_rate = _float_env("OPENAI_INPUT_COST_PER_1M_TOKENS")
    if output_rate == 0:
        output_rate = _float_env("OPENAI_OUTPUT_COST_PER_1M_TOKENS")

    cost_usd = 0.0
    if prompt_tokens is not None:
        cost_usd += prompt_tokens * input_rate / 1_000_000
    if completion_tokens is not None:
        cost_usd += completion_tokens * output_rate / 1_000_000

    return {
        "usage_metrics": usage_metrics or None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 8),
        "duration_seconds": round(duration_seconds, 3),
    }


def monotonic_time() -> float:
    return time.perf_counter()
