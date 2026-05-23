import hashlib
import json
from typing import Any

from models import WorkflowType


DEFAULT_CACHE_TTL_SECONDS = 3600
CACHE_CONTROL_KEYS = {
    "workflow_result_cache_enabled",
    "workflow_result_cache_ttl_seconds",
    "content_language_concurrency",
    "openai_input_cost_per_1m_tokens",
    "openai_output_cost_per_1m_tokens",
}


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def _workflow_value(workflow_type: WorkflowType | str) -> str:
    return workflow_type.value if isinstance(workflow_type, WorkflowType) else str(workflow_type)


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def cache_enabled(config_context: dict[str, Any], metadata: dict[str, Any] | None = None) -> bool:
    if metadata and _bool_value(metadata.get("bypass_cache"), False):
        return False
    return _bool_value(config_context.get("workflow_result_cache_enabled"), True)


def cache_ttl_seconds(config_context: dict[str, Any]) -> int:
    try:
        return max(0, int(config_context.get("workflow_result_cache_ttl_seconds") or DEFAULT_CACHE_TTL_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_CACHE_TTL_SECONDS


def build_workflow_cache_key(
    workflow_type: WorkflowType | str,
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    runtime_context = {
        key: value
        for key, value in config_context.items()
        if key not in CACHE_CONTROL_KEYS
    }
    fingerprint = {
        "workflow_type": _workflow_value(workflow_type),
        "inputs": _json_normalize(inputs),
        "runtime": _json_normalize(runtime_context),
    }
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
