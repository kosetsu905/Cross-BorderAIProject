from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from models import JobStatus, WorkflowRouteNode, WorkflowRoutePlan, WorkflowRouteRequest


WORKFLOW_ROUTE_MONITOR_TASK = "workflow.route_monitor"
TERMINAL_STATUS_VALUES = {JobStatus.COMPLETED.value, JobStatus.FAILED.value}
SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|client[_-]?secret|credential|password|refresh[_-]?token|secret|token)",
    re.IGNORECASE,
)
PII_KEY_RE = re.compile(
    r"(customer[_-]?email|customer[_-]?handle|support[_-]?handle|handle|email|phone|recipient|sender)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")


def workflow_route_provider_credentials(request: WorkflowRouteRequest) -> dict[str, Any] | None:
    if request.provider_credentials is None:
        return None
    return request.provider_credentials.model_dump(exclude_none=True) or None


def workflow_route_parent_inputs(
    request: WorkflowRouteRequest,
    plan: WorkflowRoutePlan,
) -> dict[str, Any]:
    return {
        "goal": _redact_value(request.goal),
        "context": _redact_value(request.context),
        "preferred_workflows": [item.value for item in request.preferred_workflows or []],
        "excluded_workflows": [item.value for item in request.excluded_workflows or []],
        "metadata": _redact_value(request.metadata or {}),
        "plan": public_workflow_route_plan(plan),
    }


def route_child_metadata(
    parent_job_id: str,
    node: WorkflowRouteNode,
    route_metadata: dict[str, Any] | None,
    wave_index: int,
) -> dict[str, Any]:
    return {
        **_redact_value(route_metadata or {}),
        "workflow_route_job_id": parent_job_id,
        "workflow_route_node_name": node.name,
        "workflow_route_wave_index": wave_index,
    }


def public_workflow_route_plan(plan: WorkflowRoutePlan) -> dict[str, Any]:
    payload = plan.model_dump(mode="json")
    payload["nodes"] = [
        {
            **node,
            "inputs": _redact_value(node.get("inputs") or {}),
        }
        for node in payload.get("nodes", [])
    ]
    payload["missing_inputs"] = list(plan.missing_inputs)
    return payload


def ready_route_nodes(
    plan: WorkflowRoutePlan,
    submitted_names: set[str],
    completed_names: set[str],
) -> list[WorkflowRouteNode]:
    return [
        node
        for wave in plan.waves
        for node_name in wave
        for node in plan.nodes
        if node.name == node_name
        and node.name not in submitted_names
        and set(node.depends_on) <= completed_names
    ]


def route_wave_index(plan: WorkflowRoutePlan, node_name: str) -> int:
    for index, wave in enumerate(plan.waves):
        if node_name in wave:
            return index
    return 0


def build_workflow_route_result(
    plan: WorkflowRoutePlan,
    children: list[dict[str, Any]],
    job_lookup: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any]:
    child_payloads: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "completed": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
        "total": len(plan.nodes),
        "submitted": len(children),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }

    for child in children:
        job = job_lookup(child["job_id"]) or {}
        child_status = status_text(job.get("status", JobStatus.PENDING))
        child_payload: dict[str, Any] = {
            "name": child["name"],
            "workflow_type": child["workflow_type"],
            "job_id": child["job_id"],
            "status": child_status,
            "depends_on": child.get("depends_on", []),
        }
        for key in ("cache_hit", "source_job_id", "error"):
            if job.get(key) not in (None, ""):
                child_payload[key] = job[key]
        child_payloads.append(child_payload)

        if child_status == JobStatus.COMPLETED.value:
            summary["completed"] += 1
        elif child_status == JobStatus.FAILED.value:
            summary["failed"] += 1
        elif child_status == JobStatus.PENDING.value:
            summary["pending"] += 1
        else:
            summary["running"] += 1

        if job.get("result") is not None:
            results[child["name"]] = job["result"]

        summary["prompt_tokens"] += _int_value(job.get("prompt_tokens"))
        summary["completion_tokens"] += _int_value(job.get("completion_tokens"))
        summary["total_tokens"] += _int_value(job.get("total_tokens"))
        summary["cost_usd"] += _float_value(job.get("cost_usd"))

    summary["pending"] += max(0, len(plan.nodes) - len(children))
    summary["cost_usd"] = round(summary["cost_usd"], 8)
    return {
        "plan": public_workflow_route_plan(plan),
        "children": child_payloads,
        "results": results,
        "summary": summary,
    }


def workflow_route_status(result: dict[str, Any]) -> JobStatus:
    summary = result.get("summary") or {}
    if int(summary.get("failed") or 0) > 0:
        return JobStatus.FAILED
    if int(summary.get("completed") or 0) == int(summary.get("total") or 0):
        return JobStatus.COMPLETED
    return JobStatus.RUNNING


def workflow_route_error(result: dict[str, Any]) -> str | None:
    if workflow_route_status(result) != JobStatus.FAILED:
        return None
    failed_children = [
        str(child.get("name"))
        for child in result.get("children", [])
        if child.get("status") == JobStatus.FAILED.value
    ]
    if failed_children:
        return f"One or more route child workflows failed: {', '.join(failed_children)}"
    return "One or more route child workflows failed."


def workflow_route_usage_fields(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary") or {}
    return {
        "usage_metrics": {
            "workflow_route": True,
            "children": result.get("children", []),
        },
        "prompt_tokens": _optional_int(summary.get("prompt_tokens")),
        "completion_tokens": _optional_int(summary.get("completion_tokens")),
        "total_tokens": _optional_int(summary.get("total_tokens")),
        "cost_usd": _float_value(summary.get("cost_usd")),
    }


def status_text(status: Any) -> str:
    if isinstance(status, JobStatus):
        return status.value
    return str(status or JobStatus.PENDING.value)


def route_completed_names(
    children: list[dict[str, Any]],
    job_lookup: Callable[[str], dict[str, Any] | None],
) -> set[str]:
    completed: set[str] = set()
    for child in children:
        job = job_lookup(child["job_id"]) or {}
        if status_text(job.get("status")) == JobStatus.COMPLETED.value:
            completed.add(child["name"])
    return completed


def route_submitted_names(children: list[dict[str, Any]]) -> set[str]:
    return {child["name"] for child in children}


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SECRET_KEY_RE.search(key_text):
                continue
            if PII_KEY_RE.search(key_text):
                redacted[key_text] = "[REDACTED_PII]"
            else:
                redacted[key_text] = _redact_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return PHONE_RE.sub("[REDACTED_PHONE]", EMAIL_RE.sub("[REDACTED_EMAIL]", value))
    return value


def _int_value(value: Any) -> int:
    parsed = _optional_int(value)
    return parsed if parsed is not None else 0


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
