from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import ValidationError

from models import (
    WORKFLOW_INPUT_MODELS,
    WorkflowRouteNode,
    WorkflowRoutePlan,
    WorkflowRouteRequest,
    WorkflowType,
)
from runtime_config import apply_llm_profile_context
from utils.llm_config import (
    llm_api_key,
    llm_chat_completions_url,
    llm_model_name,
    llm_reasoning_compat_params,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LANGUAGES = ["en"]
DEFAULT_PLATFORMS = ["blog", "email", "social"]
WORKFLOW_ORDER = [
    WorkflowType.MARKETING,
    WorkflowType.CONTENT,
    WorkflowType.SCHEDULER,
    WorkflowType.ANALYTICS,
    WorkflowType.SALES_IMPROVEMENT,
    WorkflowType.BIZDEV,
    WorkflowType.SUPPORT,
]


class WorkflowRouterAgent:
    """Builds explicit workflow DAGs from a business objective."""

    def __init__(self, config_context: dict[str, Any] | None = None) -> None:
        self.config_context = config_context or {}
        self.threshold = _float_config(
            self.config_context,
            "workflow_router_confidence_threshold",
            0.75,
        )
        self.max_workflows = max(
            1,
            min(_int_config(self.config_context, "workflow_router_max_workflows", 7), 7),
        )

    def plan(self, request: WorkflowRouteRequest) -> WorkflowRoutePlan:
        if not _bool_config(self.config_context, "workflow_router_enabled", True):
            raise ValueError("Workflow router is disabled by WORKFLOW_ROUTER_ENABLED=false.")

        deterministic = self._deterministic_plan(request)
        if (
            deterministic.confidence < self.threshold
            and _bool_config(self.config_context, "workflow_router_llm_fallback_enabled", True)
        ):
            llm_plan = self._llm_plan(request)
            if llm_plan is not None:
                return self._validated_plan(request, llm_plan.nodes, llm_plan.confidence, llm_plan.rationale)
        return deterministic

    def _deterministic_plan(self, request: WorkflowRouteRequest) -> WorkflowRoutePlan:
        selected = self._selected_workflows(request)
        nodes: list[WorkflowRouteNode] = []
        for workflow_type in selected:
            inputs = self._inputs_for_workflow(workflow_type, request)
            node = WorkflowRouteNode(
                name=workflow_type.value,
                workflow_type=workflow_type,
                inputs=inputs,
                depends_on=self._dependencies_for_workflow(workflow_type, selected),
                rationale=self._rationale_for_workflow(workflow_type),
            )
            nodes.append(node)

        confidence = self._confidence(request, selected)
        rationale = self._route_rationale(request, selected)
        return self._validated_plan(request, nodes, confidence, rationale)

    def _selected_workflows(self, request: WorkflowRouteRequest) -> list[WorkflowType]:
        text = request.goal.lower()
        selected: list[WorkflowType] = []
        if _contains_any(text, ("launch", "campaign", "go to market", "go-to-market", "gtm", "promote")):
            selected.extend([WorkflowType.MARKETING, WorkflowType.CONTENT, WorkflowType.SCHEDULER])
        if _contains_any(text, ("performance", "conversion", "optimize", "optimise", "analytics", "roas", "sales funnel")):
            selected.extend([WorkflowType.ANALYTICS, WorkflowType.SALES_IMPROVEMENT])
        if _contains_any(text, ("support", "customer", "refund", "return", "where is my order", "ticket")):
            selected.append(WorkflowType.SUPPORT)
        if _contains_any(text, ("partner", "partnership", "distributor", "reseller", "bizdev", "business development")):
            selected.append(WorkflowType.BIZDEV)

        selected = _dedupe_workflows(selected)
        preferred = request.preferred_workflows or []
        excluded = set(request.excluded_workflows or [])
        if preferred:
            preferred_set = set(preferred)
            selected = [workflow for workflow in selected if workflow in preferred_set and workflow not in excluded]
            if not selected:
                selected = [workflow for workflow in preferred if workflow not in excluded]
        else:
            selected = [workflow for workflow in selected if workflow not in excluded]

        if not selected:
            selected = [workflow for workflow in (preferred or [WorkflowType.MARKETING]) if workflow not in excluded]

        ordered = [workflow for workflow in WORKFLOW_ORDER if workflow in set(selected)]
        return ordered[: self.max_workflows]

    def _inputs_for_workflow(
        self,
        workflow_type: WorkflowType,
        request: WorkflowRouteRequest,
    ) -> dict[str, Any]:
        context = request.context or {}
        product_category = _text_value(context, "product_category", "category", "product", "subject")
        target_markets = _text_value(context, "target_markets", "target_market", "markets", "market", "region")
        date_range = _text_value(context, "date_range", default="Last 30 Days")

        if workflow_type == WorkflowType.MARKETING:
            return {
                "product_category": product_category,
                "product_usp": _text_value(context, "product_usp", "value_proposition", default=request.goal[:240]),
                "target_markets": target_markets,
                "budget": _text_value(context, "budget", default="unspecified"),
                "target_languages": _list_value(context, "target_languages", default=None),
            }
        if workflow_type == WorkflowType.CONTENT:
            return {
                "subject": _text_value(context, "subject", default=request.goal[:240]),
                "product_category": product_category,
                "product_features": _text_value(context, "product_features", "features", default=None),
                "target_markets": target_markets,
                "target_languages": _list_value(context, "target_languages", default=DEFAULT_TARGET_LANGUAGES),
                "platforms": _list_value(context, "platforms", default=DEFAULT_PLATFORMS),
                "brand_voice": _text_value(context, "brand_voice", default=None),
                "brand_name": _text_value(context, "brand_name", default=None),
                "product_url": _text_value(context, "product_url", default=None),
                "primary_keywords": _list_value(context, "primary_keywords", default=None),
                "generate_reddit_geo": bool(context.get("generate_reddit_geo") or False),
                "generate_visual_assets": bool(context.get("generate_visual_assets") or False),
                "image_generation_count": int(context.get("image_generation_count") or 1),
                "image_quality": _text_value(context, "image_quality", default="low"),
                "image_size": _text_value(context, "image_size", default="1024x1024"),
            }
        if workflow_type == WorkflowType.SCHEDULER:
            return {
                "event_type": _text_value(context, "event_type", default="campaign_launch"),
                "target_markets": target_markets,
                "event_list": _text_value(context, "event_list", default=request.goal[:240]),
                "preferred_launch_window": _text_value(
                    context,
                    "preferred_launch_window",
                    "launch_window",
                    default="next 30 days",
                ),
            }
        if workflow_type == WorkflowType.ANALYTICS:
            return {
                "product_category": product_category,
                "target_markets": target_markets,
                "date_range": date_range,
                "currency": _text_value(context, "currency", "base_currency", default="USD"),
                "base_currency": _text_value(context, "base_currency", "currency", default="USD"),
                "historical_metrics": context.get("historical_metrics"),
                "channel_metrics": context.get("channel_metrics"),
            }
        if workflow_type == WorkflowType.SALES_IMPROVEMENT:
            return {
                "product_category": product_category,
                "target_markets": target_markets,
                "current_avg_conversion": _text_value(context, "current_avg_conversion", default="unknown"),
                "target_conversion": _text_value(context, "target_conversion", default="improve by 20%"),
                "date_range": date_range,
            }
        if workflow_type == WorkflowType.BIZDEV:
            return {
                "product_category": product_category,
                "partnership_type": _text_value(context, "partnership_type", default="distribution partnership"),
                "target_markets": target_markets,
                "target_languages": _list_value(context, "target_languages", default=DEFAULT_TARGET_LANGUAGES),
                "key_decision_maker_roles": _text_value(
                    context,
                    "key_decision_maker_roles",
                    default="Partnership Manager, Head of Growth",
                ),
            }
        if workflow_type == WorkflowType.SUPPORT:
            inquiry = _text_value(context, "inquiry", "inquiry_text", "message", default=request.goal)
            return {
                "customer": _text_value(context, "customer", default="Customer"),
                "person": _text_value(context, "person", default=_text_value(context, "customer", default="Customer")),
                "inquiry": inquiry,
                "inquiry_text": inquiry,
                "product_category": _text_value(context, "product_category", default=None),
                "channel": _text_value(context, "channel", default="api"),
                "region": _text_value(context, "region", default="US"),
            }
        raise ValueError(f"Unsupported workflow type: {workflow_type.value}")

    def _dependencies_for_workflow(
        self,
        workflow_type: WorkflowType,
        selected: list[WorkflowType],
    ) -> list[str]:
        if workflow_type == WorkflowType.SCHEDULER and WorkflowType.MARKETING in selected:
            return [WorkflowType.MARKETING.value]
        if workflow_type == WorkflowType.SALES_IMPROVEMENT and WorkflowType.ANALYTICS in selected:
            return [WorkflowType.ANALYTICS.value]
        return []

    def _validated_plan(
        self,
        request: WorkflowRouteRequest,
        nodes: list[WorkflowRouteNode],
        confidence: float,
        rationale: str,
    ) -> WorkflowRoutePlan:
        missing_inputs: list[str] = []
        validated_nodes: list[WorkflowRouteNode] = []
        for node in nodes[: self.max_workflows]:
            try:
                validated_inputs = WORKFLOW_INPUT_MODELS[node.workflow_type].model_validate(node.inputs).model_dump()
                validated_nodes.append(node.model_copy(update={"inputs": validated_inputs}))
            except ValidationError as exc:
                missing_inputs.extend(_missing_inputs_for_node(node, exc))
                validated_nodes.append(node)

        deduped_missing = sorted(set(missing_inputs))
        return WorkflowRoutePlan(
            goal=request.goal,
            confidence=round(max(0.0, min(confidence, 1.0)), 4),
            requires_review=bool(deduped_missing) or confidence < self.threshold,
            missing_inputs=deduped_missing,
            rationale=rationale,
            nodes=validated_nodes,
        )

    def _confidence(self, request: WorkflowRouteRequest, selected: list[WorkflowType]) -> float:
        text = request.goal.lower()
        if not selected:
            return 0.0
        if request.preferred_workflows:
            return 0.82
        if _contains_any(text, ("launch", "performance", "conversion", "support", "refund", "partner")):
            return 0.86
        return 0.58

    def _route_rationale(self, request: WorkflowRouteRequest, selected: list[WorkflowType]) -> str:
        if not selected:
            return "No workflow matched the requested goal."
        workflows = ", ".join(workflow.value for workflow in selected)
        return f"Selected workflows for the goal based on deterministic routing signals: {workflows}."

    def _rationale_for_workflow(self, workflow_type: WorkflowType) -> str:
        labels = {
            WorkflowType.MARKETING: "Build campaign strategy and ad launch package.",
            WorkflowType.CONTENT: "Create localized content assets for the selected markets.",
            WorkflowType.SCHEDULER: "Plan launch timing and reminders across target markets.",
            WorkflowType.ANALYTICS: "Analyze performance and market evidence.",
            WorkflowType.SALES_IMPROVEMENT: "Turn funnel and pricing signals into a sales improvement playbook.",
            WorkflowType.BIZDEV: "Identify partnership opportunities and outreach sequence.",
            WorkflowType.SUPPORT: "Route customer inquiry to the support workflow.",
        }
        return labels[workflow_type]

    def _llm_plan(self, request: WorkflowRouteRequest) -> WorkflowRoutePlan | None:
        context = dict(self.config_context)
        profile = context.get("workflow_router_llm_profile")
        if profile:
            context = apply_llm_profile_context(context, str(profile))
        if not llm_api_key(context):
            return None
        try:
            response = httpx.post(
                llm_chat_completions_url(context),
                headers={
                    "Authorization": f"Bearer {llm_api_key(context)}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_model_name(context),
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return JSON for a workflow route plan. Use only these workflow_type values: "
                                "marketing, content, support, analytics, bizdev, scheduler, sales_improvement. "
                                "Shape: {confidence, rationale, nodes:[{name, workflow_type, inputs, depends_on, rationale}]}."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "goal": request.goal,
                                    "context": request.context,
                                    "preferred_workflows": [item.value for item in request.preferred_workflows or []],
                                    "excluded_workflows": [item.value for item in request.excluded_workflows or []],
                                },
                                default=str,
                            )[:6000],
                        },
                    ],
                    "temperature": 0,
                    **llm_reasoning_compat_params(context),
                },
                timeout=15,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            payload = json.loads(content)
            nodes = [
                WorkflowRouteNode.model_validate(node)
                for node in payload.get("nodes", [])
                if isinstance(node, dict)
            ]
            if not nodes:
                return None
            return WorkflowRoutePlan(
                goal=request.goal,
                confidence=float(payload.get("confidence") or 0.75),
                requires_review=False,
                missing_inputs=[],
                rationale=str(payload.get("rationale") or "LLM workflow routing fallback."),
                nodes=nodes,
            )
        except Exception as exc:
            logger.info("Workflow router LLM fallback skipped: %s", exc)
            return None


def _missing_inputs_for_node(node: WorkflowRouteNode, exc: ValidationError) -> list[str]:
    missing: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        if location:
            missing.append(f"{node.name}.inputs.{location}")
    return missing or [f"{node.name}.inputs"]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _dedupe_workflows(workflows: list[WorkflowType]) -> list[WorkflowType]:
    seen: set[WorkflowType] = set()
    unique: list[WorkflowType] = []
    for workflow in workflows:
        if workflow in seen:
            continue
        seen.add(workflow)
        unique.append(workflow)
    return unique


def _text_value(context: dict[str, Any], *keys: str, default: str | None = "") -> str | None:
    for key in keys:
        value = context.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        elif value not in (None, ""):
            return str(value)
    return default


def _list_value(
    context: dict[str, Any],
    key: str,
    default: list[str] | None,
) -> list[str] | None:
    value = context.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()] or default
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()] or default
    return default


def _bool_config(config_context: dict[str, Any], key: str, default: bool) -> bool:
    value = config_context.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_config(config_context: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config_context.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_config(config_context: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(config_context.get(key, default))
    except (TypeError, ValueError):
        return default
