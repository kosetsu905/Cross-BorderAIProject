from typing import Any

from crewai import LLM

from runtime_config import apply_llm_profile_context
from utils.llm_config import build_llm


WORKER_TIER = "worker"
REVIEWER_TIER = "reviewer"
DEFAULT_AGENT_TIER = WORKER_TIER
VALID_AGENT_TIERS = {WORKER_TIER, REVIEWER_TIER}
TIER_PROFILE_KEYS = {
    WORKER_TIER: "workflow_worker_llm_profile",
    REVIEWER_TIER: "workflow_reviewer_llm_profile",
}


class ModelTierRouter:
    """Resolve per-agent LLMs using optional worker/reviewer profile routing."""

    def __init__(self, config_context: dict[str, Any]) -> None:
        self.config_context = dict(config_context)
        self._llm_cache: dict[str, LLM] = {}

    def llm_for_agent(self, agent_config: dict[str, Any]) -> LLM:
        tier = agent_llm_tier(agent_config)
        context, cache_key = self._context_for_tier(tier)
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = build_llm(context)
        return self._llm_cache[cache_key]

    def _context_for_tier(self, tier: str) -> tuple[dict[str, Any], str]:
        if not _bool_config(self.config_context, "workflow_model_tiering_enabled", True):
            return self.config_context, "base"

        profile_key = TIER_PROFILE_KEYS[tier]
        profile_name = self.config_context.get(profile_key)
        if profile_name in (None, ""):
            return self.config_context, "base"

        normalized_profile = str(profile_name).strip()
        return (
            apply_llm_profile_context(self.config_context, normalized_profile),
            f"profile:{normalized_profile}",
        )


def llm_for_agent(config_context: dict[str, Any], agent_config: dict[str, Any]) -> LLM:
    return ModelTierRouter(config_context).llm_for_agent(agent_config)


def agent_llm_tier(agent_config: dict[str, Any]) -> str:
    raw_tier = str(agent_config.get("llm_tier") or DEFAULT_AGENT_TIER).strip().lower()
    if raw_tier not in VALID_AGENT_TIERS:
        raise ValueError(
            f"Unsupported llm_tier '{raw_tier}'. Expected one of: "
            f"{', '.join(sorted(VALID_AGENT_TIERS))}."
        )
    return raw_tier


def _bool_config(config_context: dict[str, Any], key: str, default: bool) -> bool:
    value = config_context.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}
