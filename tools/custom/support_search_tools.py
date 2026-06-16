from __future__ import annotations

from typing import Any, Literal

from crewai_tools import SerperDevTool

from utils.tool_cache import build_cached_serper_tool

SupportSearchStage = Literal["pre_sales", "order_fulfillment", "post_sales_support"]

SUPPORT_SERPER_STAGE_FLAGS: dict[SupportSearchStage, str] = {
    "pre_sales": "support_serper_pre_sales_enabled",
    "order_fulfillment": "support_serper_order_fulfillment_enabled",
    "post_sales_support": "support_serper_post_sales_enabled",
}


def build_support_external_search_tools(
    stage: SupportSearchStage,
    config_context: dict[str, Any],
) -> list[Any]:
    """Return optional external search tools for a Customer Service stage."""
    flag_name = SUPPORT_SERPER_STAGE_FLAGS[stage]
    if not _bool_config(config_context, flag_name):
        return []
    if not config_context.get("serper_api_key"):
        return []
    if not str(getattr(SerperDevTool, "__module__", "")).startswith("crewai_tools"):
        return [SerperDevTool()]
    return [build_cached_serper_tool(config_context, purpose=f"support_{stage}")]


def _bool_config(config_context: dict[str, Any], key: str) -> bool:
    value = config_context.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
