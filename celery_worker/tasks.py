import asyncio
import uuid
from typing import Any

from celery.signals import worker_process_init

from celery_worker.celery_app import celery_app
from crews.analytics_crew import run_analytics_crew
from crews.bizdev_crew import run_bizdev_crew
from crews.content_crew import run_content_crew
from crews.marketing_crew import run_marketing_crew
from crews.sales_improvement_crew import run_sales_improvement_crew
from crews.scheduler_crew import run_scheduler_crew
from crews.support_crew import run_support_crew
from database import SessionLocal
from job_store import PostgresJobStore
from models import JobStatus, WorkflowRoutePlan, WorkflowType
from runtime_config import apply_runtime_environment, load_runtime_config, resolve_workflow_runtime_context
from utils.observability import (
    add_span_event,
    flush_observability,
    group_span,
    init_observability,
    record_usage_metrics,
    route_span,
    workflow_span,
)
from utils.result_cache import build_workflow_cache_key
from utils.retry_policy import is_retryable_exception, retry_countdown_seconds
from utils.usage_tracking import build_usage_summary, monotonic_time, pop_usage_metrics
from utils.workflow_engine import create_cached_workflow_job
from utils.workflow_group import (
    WORKFLOW_GROUP_MONITOR_TASK,
    build_workflow_group_result,
    workflow_group_error,
    workflow_group_status,
    workflow_group_usage_fields,
)
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, WorkflowProgressRecorder
from utils.workflow_route import (
    WORKFLOW_ROUTE_MONITOR_TASK,
    build_workflow_route_result,
    ready_route_nodes,
    route_child_metadata,
    route_completed_names,
    route_submitted_names,
    route_wave_index,
    workflow_route_error,
    workflow_route_status,
    workflow_route_usage_fields,
)


job_store = PostgresJobStore(SessionLocal)
TASK_MAP: dict[WorkflowType, str] = {
    WorkflowType.MARKETING: "workflow.marketing",
    WorkflowType.CONTENT: "workflow.content",
    WorkflowType.SUPPORT: "workflow.support",
    WorkflowType.ANALYTICS: "workflow.analytics",
    WorkflowType.SALES_IMPROVEMENT: "workflow.sales_improvement",
    WorkflowType.BIZDEV: "workflow.bizdev",
    WorkflowType.SCHEDULER: "workflow.scheduler",
}


@worker_process_init.connect
def worker_process_init_handler(*args: object, **kwargs: object) -> None:
    init_observability("cross-border-celery-worker")


def _submit_route_child_job(
    parent_job_id: str,
    plan: WorkflowRoutePlan,
    node_name: str,
    provider_credentials: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    node = next(item for item in plan.nodes if item.name == node_name)
    task_name = TASK_MAP[node.workflow_type]
    config_context = resolve_workflow_runtime_context(
        load_runtime_config(),
        node.workflow_type,
        provider_credentials,
    )
    cache_key = build_workflow_cache_key(node.workflow_type, node.inputs, config_context)
    child_metadata = route_child_metadata(
        parent_job_id,
        node,
        metadata,
        route_wave_index(plan, node.name),
    )
    cached_job_id = create_cached_workflow_job(
        job_store,
        node.workflow_type,
        node.inputs,
        config_context,
        child_metadata,
        "celery",
    )
    if cached_job_id:
        return {
            "name": node.name,
            "workflow_type": node.workflow_type.value,
            "job_id": cached_job_id,
            "depends_on": list(node.depends_on),
        }

    job_id = str(uuid.uuid4())
    job_store.create_job(job_id, node.workflow_type, node.inputs, cache_key=cache_key)
    job_store.log_event(
        job_id,
        "queued",
        "Celery task queued.",
        {"workflow_type": node.workflow_type.value, "backend": "celery", "task_name": task_name},
    )
    celery_app.send_task(
        task_name,
        args=[node.inputs],
        kwargs={"config_context": config_context},
        task_id=job_id,
    )
    return {
        "name": node.name,
        "workflow_type": node.workflow_type.value,
        "job_id": job_id,
        "depends_on": list(node.depends_on),
    }


def _run_with_job_state(
    self: object,
    progress: str,
    crew_function: object,
    inputs: dict,
    config_context: dict | None = None,
) -> dict:
    job_id = self.request.id
    config_context = config_context or load_runtime_config().as_context()
    workflow_type = str(getattr(self, "name", "workflow.unknown")).removeprefix("workflow.")
    init_observability("cross-border-celery-worker", config_context=config_context)
    config_context[PROGRESS_CONTEXT_KEY] = WorkflowProgressRecorder(
        job_id=job_id,
        workflow_type=workflow_type,
        job_store=job_store,
        backend="celery",
    )
    apply_runtime_environment(config_context)
    self.update_state(state="PROGRESS", meta={"status": progress})
    job_store.update_job(job_id, status=JobStatus.RUNNING, result={"status": progress}, error=None)
    job_store.log_event(
        job_id,
        "running",
        progress,
        {"backend": "celery", "task_name": getattr(self, "name", None)},
    )
    started_at = monotonic_time()
    try:
        with workflow_span(
            workflow_type,
            job_id=job_id,
            backend="celery",
            config_context=config_context,
            attributes={"task_name": getattr(self, "name", None)},
        ):
            result = crew_function(inputs, config_context)
            clean_result, usage_metrics = pop_usage_metrics(result)
            normalized_result = clean_result if isinstance(clean_result, dict) else {"raw": str(clean_result)}
            usage_summary = build_usage_summary(
                usage_metrics,
                monotonic_time() - started_at,
                config_context,
            )
            record_usage_metrics(usage_summary, config_context)
            job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=normalized_result,
                **usage_summary,
                error=None,
            )
            job_store.log_event(
                job_id,
                "completed",
                "Workflow execution completed.",
                {
                    "backend": "celery",
                    "task_name": getattr(self, "name", None),
                    "total_tokens": usage_summary.get("total_tokens"),
                    "cost_usd": usage_summary.get("cost_usd"),
                    "duration_seconds": usage_summary.get("duration_seconds"),
                },
            )
            if workflow_type == "support":
                try:
                    from services.support_auto_dispatch import process_completed_support_job

                    dispatch_result = asyncio.run(
                        process_completed_support_job(
                            job_id=job_id,
                            inputs=inputs,
                            result=normalized_result,
                            config_context=config_context,
                        )
                    )
                    job_store.log_event(
                        job_id,
                        "support_auto_dispatch",
                        "Support auto-dispatch evaluated.",
                        dispatch_result,
                    )
                except Exception as dispatch_exc:
                    job_store.log_event(
                        job_id,
                        "support_auto_dispatch_failed",
                        "Support auto-dispatch failed after workflow completion.",
                        {"error": str(dispatch_exc)},
                    )
            return {
                "data": normalized_result,
                "meta": usage_summary,
            }
    except Exception as exc:
        request = getattr(self, "request", None)
        retries = int(getattr(request, "retries", 0) or 0)
        max_retries = int(getattr(self, "max_retries", 3) or 3)

        if is_retryable_exception(exc) and retries < max_retries:
            countdown = retry_countdown_seconds(retries)
            retry_meta = {
                "status": "Retrying after transient provider or network error",
                "retry_count": retries + 1,
                "max_retries": max_retries,
                "next_retry_seconds": countdown,
                "error": str(exc),
            }
            job_store.update_job(
                job_id,
                status=JobStatus.RUNNING,
                result=retry_meta,
                error=str(exc),
                duration_seconds=monotonic_time() - started_at,
            )
            job_store.log_event(
                job_id,
                "retrying",
                "Retrying transient provider or network error.",
                retry_meta,
            )
            raise self.retry(exc=exc, countdown=countdown)

        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            result=None,
            error=str(exc),
            duration_seconds=monotonic_time() - started_at,
        )
        job_store.log_event(
            job_id,
            "failed",
            "Workflow execution failed.",
            {
                "backend": "celery",
                "task_name": getattr(self, "name", None),
                "error": str(exc),
            },
        )
        raise
    finally:
        flush_observability()


@celery_app.task(name="health_check")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@celery_app.task(
    bind=True,
    name=WORKFLOW_GROUP_MONITOR_TASK,
    max_retries=360,
    soft_time_limit=1800,
    time_limit=2100,
)
def run_workflow_group_monitor_task(
    self: object,
    parent_job_id: str,
    children: list[dict[str, str]],
    poll_interval_seconds: int = 5,
) -> dict:
    init_observability("cross-border-celery-worker")
    with group_span(
        parent_job_id=parent_job_id,
        backend="celery",
        attributes={"children_count": len(children)},
    ):
        result = build_workflow_group_result(children, job_store.get_job)
        group_status = workflow_group_status(result)
        group_error = workflow_group_error(result)

        if group_status == JobStatus.RUNNING:
            add_span_event(
                "workflow_group.poll",
                {"status": group_status.value, "children_count": len(children)},
            )
            job_store.update_job(
                parent_job_id,
                status=JobStatus.RUNNING,
                result=result,
                error=None,
            )
            self.update_state(
                state="PROGRESS",
                meta={
                    "status": "Waiting for child workflows.",
                    "summary": result.get("summary"),
                    "children": result.get("children"),
                },
            )
            request = getattr(self, "request", None)
            retries = int(getattr(request, "retries", 0) or 0)
            max_retries = int(getattr(self, "max_retries", 360) or 360)
            if retries >= max_retries:
                error = "Workflow group monitor exceeded the maximum polling retries."
                job_store.update_job(
                    parent_job_id,
                    status=JobStatus.FAILED,
                    result=result,
                    error=error,
                )
                job_store.log_event(
                    parent_job_id,
                    "failed",
                    error,
                    {"backend": "celery", "children": result.get("children")},
                )
                raise RuntimeError(error)
            raise self.retry(countdown=poll_interval_seconds)

        usage_fields = workflow_group_usage_fields(result)
        record_usage_metrics(usage_fields)
        job_store.update_job(
            parent_job_id,
            status=group_status,
            result=result,
            **usage_fields,
            error=group_error,
        )
        job_store.log_event(
            parent_job_id,
            "completed" if group_status == JobStatus.COMPLETED else "failed",
            "Workflow group execution completed."
            if group_status == JobStatus.COMPLETED
            else "Workflow group execution failed.",
            {
                "backend": "celery",
                "summary": result.get("summary"),
                "children": result.get("children"),
            },
        )
        if group_status == JobStatus.FAILED:
            raise RuntimeError(group_error or "Workflow group execution failed.")

        return {
            "data": result,
            "meta": usage_fields,
        }


@celery_app.task(
    bind=True,
    name=WORKFLOW_ROUTE_MONITOR_TASK,
    max_retries=360,
    soft_time_limit=1800,
    time_limit=2100,
)
def run_workflow_route_monitor_task(
    self: object,
    parent_job_id: str,
    plan_payload: dict[str, Any],
    children: list[dict[str, Any]],
    provider_credentials: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    poll_interval_seconds: int = 5,
) -> dict:
    init_observability("cross-border-celery-worker")
    with route_span(
        parent_job_id=parent_job_id,
        backend="celery",
        attributes={"children_count": len(children)},
    ):
        plan = WorkflowRoutePlan.model_validate(plan_payload)
        result = build_workflow_route_result(plan, children, job_store.get_job)
        route_status = workflow_route_status(result)
        route_error = workflow_route_error(result)

        if route_status == JobStatus.FAILED:
            usage_fields = workflow_route_usage_fields(result)
            record_usage_metrics(usage_fields)
            job_store.update_job(
                parent_job_id,
                status=JobStatus.FAILED,
                result=result,
                **usage_fields,
                error=route_error,
            )
            job_store.log_event(
                parent_job_id,
                "failed",
                "Workflow route execution failed.",
                {"backend": "celery", "summary": result.get("summary"), "children": result.get("children")},
            )
            raise RuntimeError(route_error or "Workflow route execution failed.")

        submitted_names = route_submitted_names(children)
        completed_names = route_completed_names(children, job_store.get_job)
        ready_nodes = ready_route_nodes(plan, submitted_names, completed_names)
        if ready_nodes:
            for node in ready_nodes:
                child = _submit_route_child_job(
                    parent_job_id,
                    plan,
                    node.name,
                    provider_credentials,
                    metadata,
                )
                children.append(child)
            result = build_workflow_route_result(plan, children, job_store.get_job)
            add_span_event(
                "route.wave_submitted",
                {
                    "ready_nodes": [node.name for node in ready_nodes],
                    "children_count": len(children),
                },
            )
            job_store.update_job(parent_job_id, status=JobStatus.RUNNING, result=result, error=None)
            job_store.log_event(
                parent_job_id,
                "route_wave_submitted",
                "Workflow route submitted another ready worker wave.",
                {"backend": "celery", "children": children},
            )
            raise self.retry(countdown=poll_interval_seconds)

        if workflow_route_status(result) == JobStatus.COMPLETED:
            usage_fields = workflow_route_usage_fields(result)
            record_usage_metrics(usage_fields)
            job_store.update_job(
                parent_job_id,
                status=JobStatus.COMPLETED,
                result=result,
                **usage_fields,
                error=None,
            )
            job_store.log_event(
                parent_job_id,
                "completed",
                "Workflow route execution completed.",
                {"backend": "celery", "summary": result.get("summary"), "children": result.get("children")},
            )
            return {"data": result, "meta": usage_fields}

        job_store.update_job(parent_job_id, status=JobStatus.RUNNING, result=result, error=None)
        add_span_event(
            "workflow_route.poll",
            {"status": JobStatus.RUNNING.value, "children_count": len(children)},
        )
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "Waiting for route child workflows.",
                "summary": result.get("summary"),
                "children": result.get("children"),
            },
        )
        request = getattr(self, "request", None)
        retries = int(getattr(request, "retries", 0) or 0)
        max_retries = int(getattr(self, "max_retries", 360) or 360)
        if retries >= max_retries:
            error = "Workflow route monitor exceeded the maximum polling retries."
            job_store.update_job(parent_job_id, status=JobStatus.FAILED, result=result, error=error)
            job_store.log_event(
                parent_job_id,
                "failed",
                error,
                {"backend": "celery", "children": result.get("children")},
            )
            raise RuntimeError(error)
        raise self.retry(countdown=poll_interval_seconds)


@celery_app.task(bind=True, name="workflow.marketing", soft_time_limit=1200, time_limit=1500)
def run_marketing_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(self, "Initializing Marketing Crew...", run_marketing_crew, inputs, config_context)


@celery_app.task(bind=True, name="workflow.content", soft_time_limit=1200, time_limit=1500)
def run_content_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(self, "Generating localized content...", run_content_crew, inputs, config_context)


@celery_app.task(bind=True, name="workflow.support", soft_time_limit=900, time_limit=1200)
def run_support_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(self, "Drafting and QA support response...", run_support_crew, inputs, config_context)


@celery_app.task(bind=True, name="workflow.analytics", soft_time_limit=1500, time_limit=1800)
def run_analytics_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Aggregating platform metrics and benchmarking...",
        run_analytics_crew,
        inputs,
        config_context,
    )


@celery_app.task(bind=True, name="workflow.sales_improvement", soft_time_limit=1200, time_limit=1500)
def run_sales_improvement_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Analyzing funnel and generating CRO playbook...",
        run_sales_improvement_crew,
        inputs,
        config_context,
    )


@celery_app.task(bind=True, name="workflow.bizdev", soft_time_limit=1200, time_limit=1500)
def run_bizdev_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Prospecting leads and drafting outreach...",
        run_bizdev_crew,
        inputs,
        config_context,
    )


@celery_app.task(bind=True, name="workflow.scheduler", soft_time_limit=900, time_limit=1200)
def run_scheduler_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Mapping timezones and resolving conflicts...",
        run_scheduler_crew,
        inputs,
        config_context,
    )
