from typing import Any

from crewai import LLM


DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_REASONING_MODEL_MARKERS = (
    "qwen3",
    "deepseek-r1",
    "reasoning",
    "thinking",
)


def llm_api_key(config_context: dict[str, Any]) -> str | None:
    return config_context.get("llm_api_key") or config_context.get("openai_api_key")


def llm_model_name(config_context: dict[str, Any]) -> str:
    return str(
        config_context.get("llm_model_name")
        or config_context.get("openai_model_name")
        or DEFAULT_LLM_MODEL
    )


def llm_base_url(config_context: dict[str, Any]) -> str | None:
    value = config_context.get("llm_base_url")
    return str(value).rstrip("/") if value else None


def llm_chat_completions_url(config_context: dict[str, Any]) -> str:
    base_url = llm_base_url(config_context) or DEFAULT_LLM_BASE_URL
    return f"{base_url}/chat/completions"


def _crewai_model_name(config_context: dict[str, Any]) -> str:
    model_name = llm_model_name(config_context)
    provider = str(config_context.get("llm_provider") or "").lower()
    if provider == "openrouter" and not model_name.startswith("openrouter/"):
        return f"openrouter/{model_name}"
    return model_name


def _bool_context_value(config_context: dict[str, Any], key: str) -> bool:
    value = config_context.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _should_disable_reasoning(config_context: dict[str, Any]) -> bool:
    if _bool_context_value(config_context, "llm_disable_reasoning"):
        return True

    provider = str(config_context.get("llm_provider") or "").lower()
    if provider != "openrouter":
        return False

    normalized_model_name = llm_model_name(config_context).lower()
    return any(marker in normalized_model_name for marker in OPENROUTER_REASONING_MODEL_MARKERS)


def llm_reasoning_compat_params(config_context: dict[str, Any]) -> dict[str, Any]:
    if not _should_disable_reasoning(config_context):
        return {}

    return {
        "reasoning_effort": "none",
    }


def _optional_positive_int(config_context: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config_context.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def llm_token_limit_params(config_context: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    max_tokens = _optional_positive_int(config_context, "llm_max_tokens", "max_tokens")
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    max_completion_tokens = _optional_positive_int(
        config_context,
        "llm_max_completion_tokens",
        "max_completion_tokens",
    )
    if max_completion_tokens is not None:
        params["max_completion_tokens"] = max_completion_tokens
    return params


def build_llm(config_context: dict[str, Any]) -> LLM:
    base_url = llm_base_url(config_context)
    return LLM(
        model=_crewai_model_name(config_context),
        api_key=llm_api_key(config_context),
        base_url=base_url,
        api_base=base_url,
        **llm_token_limit_params(config_context),
        **llm_reasoning_compat_params(config_context),
    )
