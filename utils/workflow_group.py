from collections.abc import Callable
from typing import Any

from models import JobStatus, WorkflowGroupItem, WorkflowGroupRequest


WORKFLOW_GROUP_TYPE = "workflow_group"
WORKFLOW_GROUP_MONITOR_TASK = "workflow.group_monitor"
TERMINAL_STATUS_VALUES = {JobStatus.COMPLETED.value, JobStatus.FAILED.value}


def workflow_group_item_name(item: WorkflowGroupItem) -> str:
    return item.name or item.workflow_type.value


def workflow_group_provider_credentials(item: WorkflowGroupItem) -> dict[str, Any] | None:
    if item.provider_credentials is None:
        return None
    return item.provider_credentials.model_dump(exclude_none=True) or None


def workflow_group_child_metadata(
    parent_job_id: str,
    child_name: str,
    item: WorkflowGroupItem,
    group_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = {
        **(group_metadata or {}),
        **(item.metadata or {}),
        "workflow_group_job_id": parent_job_id,
        "workflow_group_child_name": child_name,
    }
    return metadata


def workflow_group_parent_inputs(request: WorkflowGroupRequest) -> dict[str, Any]:
    return {
        "metadata": request.metadata or {},
        "workflows": [
            {
                "name": workflow_group_item_name(item),
                "workflow_type": item.workflow_type.value,
                "inputs": item.inputs,
                "metadata": item.metadata or {},
            }
            for item in request.workflows
        ],
    }


def status_text(status: Any) -> str:
    if isinstance(status, JobStatus):
        return status.value
    return str(status or JobStatus.PENDING.value)


def is_terminal_status(status: Any) -> bool:
    return status_text(status) in TERMINAL_STATUS_VALUES


def build_workflow_group_result(
    children: list[dict[str, str]],
    job_lookup: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any]:
    child_payloads: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "completed": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
        "total": len(children),
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

    summary["cost_usd"] = round(summary["cost_usd"], 8)
    return {
        "children": child_payloads,
        "results": results,
        "summary": summary,
    }


def workflow_group_status(result: dict[str, Any]) -> JobStatus:
    summary = result.get("summary") or {}
    if int(summary.get("failed") or 0) > 0:
        return JobStatus.FAILED
    if int(summary.get("completed") or 0) == int(summary.get("total") or 0):
        return JobStatus.COMPLETED
    return JobStatus.RUNNING


def workflow_group_error(result: dict[str, Any]) -> str | None:
    if workflow_group_status(result) != JobStatus.FAILED:
        return None
    failed_children = [
        str(child.get("name"))
        for child in result.get("children", [])
        if child.get("status") == JobStatus.FAILED.value
    ]
    if failed_children:
        return f"One or more child workflows failed: {', '.join(failed_children)}"
    return "One or more child workflows failed."


def workflow_group_usage_fields(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary") or {}
    return {
        "usage_metrics": {
            "workflow_group": True,
            "children": result.get("children", []),
        },
        "prompt_tokens": _optional_int(summary.get("prompt_tokens")),
        "completion_tokens": _optional_int(summary.get("completion_tokens")),
        "total_tokens": _optional_int(summary.get("total_tokens")),
        "cost_usd": _float_value(summary.get("cost_usd")),
    }


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
