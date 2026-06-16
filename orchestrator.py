import asyncio
import logging
import uuid
from typing import Any

from celery.result import AsyncResult

from celery_worker.celery_app import celery_app
from job_store import InMemoryJobStore, JobStore
from models import JobStatus, WorkflowGroupRequest, WorkflowRoutePlan, WorkflowRouteRequest, WorkflowType, WORKFLOW_ROUTE_TYPE
from runtime_config import RuntimeConfig, apply_runtime_environment, resolve_workflow_runtime_context
from services.workflow_router import WorkflowRouterAgent
from utils.observability import (
    group_span,
    init_observability,
    record_usage_metrics,
    route_span,
    workflow_span,
)
from utils.usage_tracking import build_usage_summary, monotonic_time, pop_usage_metrics
from utils.workflow_engine import CrewFunction, WorkflowExecutionEngine
from utils.workflow_group import (
    WORKFLOW_GROUP_MONITOR_TASK,
    WORKFLOW_GROUP_TYPE,
    build_workflow_group_result,
    workflow_group_child_metadata,
    workflow_group_error,
    workflow_group_item_name,
    workflow_group_parent_inputs,
    workflow_group_provider_credentials,
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
    workflow_route_parent_inputs,
    workflow_route_provider_credentials,
    workflow_route_status,
    workflow_route_usage_fields,
)

logger = logging.getLogger(__name__)


USAGE_FIELD_NAMES = {
    "usage_metrics",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "duration_seconds",
}


def _split_task_result(task_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        isinstance(task_result, dict)
        and isinstance(task_result.get("data"), dict)
        and isinstance(task_result.get("meta"), dict)
    ):
        usage_fields = {
            key: value
            for key, value in task_result["meta"].items()
            if key in USAGE_FIELD_NAMES
        }
        return task_result["data"], usage_fields

    result_payload = task_result if isinstance(task_result, dict) else {"raw": str(task_result)}
    return result_payload, {}


def _ensure_submittable_route_plan(plan: WorkflowRoutePlan) -> None:
    if not plan.nodes:
        raise ValueError("Workflow route plan did not select any child workflows.")
    if plan.missing_inputs:
        raise ValueError(
            "Workflow route plan is missing required child inputs: "
            + ", ".join(plan.missing_inputs)
        )


class MasterOrchestrator:
    """Central router for local CrewAI execution with persistent job state."""

    def __init__(
        self,
        job_store: JobStore | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self._job_store = job_store or InMemoryJobStore()
        self._runtime_config = runtime_config or RuntimeConfig()
        init_observability("cross-border-local-orchestrator", config_context=self._runtime_config.as_context())
        self._engine = WorkflowExecutionEngine(self._job_store, self._runtime_config)
        logger.info("Master orchestrator initialized with %s job store.", self.job_store_name)

    @property
    def job_store_name(self) -> str:
        return self._job_store.name

    @property
    def registered_workflows(self) -> list[WorkflowType]:
        return self._engine.registered_workflows()

    def register_crew(self, workflow_type: WorkflowType, crew_function: CrewFunction) -> None:
        self._engine.register_crew(workflow_type, crew_function)
        logger.info("Registered crew: %s", workflow_type.value)

    async def submit_job(
        self,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
        provider_credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self._engine.has_workflow(workflow_type):
            raise ValueError(f"Workflow '{workflow_type.value}' is not registered.")

        prepared = self._engine.prepare_job(
            workflow_type,
            inputs,
            provider_credentials,
            metadata,
            "local",
            "Local background execution scheduled.",
        )
        if prepared.cache_hit:
            return prepared.job_id

        asyncio.create_task(self._run_job(prepared.job_id, workflow_type, inputs, prepared.config_context))
        logger.info("Submitted job %s for workflow %s", prepared.job_id, workflow_type.value)
        return prepared.job_id

    async def submit_workflow_group(self, request: WorkflowGroupRequest) -> str:
        parent_job_id = str(uuid.uuid4())
        children: list[dict[str, str]] = []
        self._job_store.create_job(
            parent_job_id,
            WORKFLOW_GROUP_TYPE,
            workflow_group_parent_inputs(request),
        )
        self._job_store.log_event(
            parent_job_id,
            "queued",
            "Workflow group submitted.",
            {"workflow_type": WORKFLOW_GROUP_TYPE, "backend": "local"},
        )

        try:
            with group_span(
                parent_job_id=parent_job_id,
                backend="local",
                attributes={"children_requested": len(request.workflows)},
                config_context=self._runtime_config.as_context(),
            ):
                for item in request.workflows:
                    child_name = workflow_group_item_name(item)
                    child_job_id = await self.submit_job(
                        item.workflow_type,
                        item.inputs,
                        provider_credentials=workflow_group_provider_credentials(item),
                        metadata=workflow_group_child_metadata(
                            parent_job_id,
                            child_name,
                            item,
                            request.metadata,
                        ),
                    )
                    children.append(
                        {
                            "name": child_name,
                            "workflow_type": item.workflow_type.value,
                            "job_id": child_job_id,
                        }
                    )

                result = build_workflow_group_result(children, self._job_store.get_job)
                self._job_store.update_job(
                    parent_job_id,
                    status=JobStatus.RUNNING,
                    result=result,
                    error=None,
                )
                self._job_store.log_event(
                    parent_job_id,
                    "running",
                    "Workflow group child jobs submitted.",
                    {"backend": "local", "children": children},
                )
                asyncio.create_task(self._monitor_workflow_group(parent_job_id, children))
        except Exception as exc:
            logger.exception("Workflow group %s failed during submission", parent_job_id)
            result = build_workflow_group_result(children, self._job_store.get_job)
            self._job_store.update_job(
                parent_job_id,
                status=JobStatus.FAILED,
                result=result,
                error=str(exc),
            )
            self._job_store.log_event(
                parent_job_id,
                "failed",
                "Workflow group submission failed.",
                {"backend": "local", "error": str(exc), "children": children},
            )
        return parent_job_id

    def plan_workflow_route(self, request: WorkflowRouteRequest) -> WorkflowRoutePlan:
        config_context = resolve_workflow_runtime_context(
            self._runtime_config,
            WORKFLOW_ROUTE_TYPE,
            workflow_route_provider_credentials(request),
        )
        return WorkflowRouterAgent(config_context).plan(request)

    async def submit_workflow_route(self, request: WorkflowRouteRequest) -> str:
        plan = self.plan_workflow_route(request)
        _ensure_submittable_route_plan(plan)
        parent_job_id = str(uuid.uuid4())
        children: list[dict[str, Any]] = []
        self._job_store.create_job(
            parent_job_id,
            WORKFLOW_ROUTE_TYPE,
            workflow_route_parent_inputs(request, plan),
        )
        initial_result = build_workflow_route_result(plan, children, self._job_store.get_job)
        self._job_store.update_job(parent_job_id, status=JobStatus.RUNNING, result=initial_result, error=None)
        self._job_store.log_event(
            parent_job_id,
            "queued",
            "Workflow route submitted.",
            {
                "workflow_type": WORKFLOW_ROUTE_TYPE,
                "backend": "local",
                "plan": initial_result.get("plan"),
            },
        )
        try:
            with route_span(
                parent_job_id=parent_job_id,
                backend="local",
                attributes={"planned_nodes": len(plan.nodes)},
                config_context=self._runtime_config.as_context(),
            ):
                await self._submit_ready_route_children(parent_job_id, request, plan, children)
                result = build_workflow_route_result(plan, children, self._job_store.get_job)
                self._job_store.update_job(
                    parent_job_id,
                    status=workflow_route_status(result),
                    result=result,
                    error=None,
                )
                self._job_store.log_event(
                    parent_job_id,
                    "running",
                    "Workflow route initial child jobs submitted.",
                    {"backend": "local", "children": children},
                )
                asyncio.create_task(self._monitor_workflow_route(parent_job_id, request, plan, children))
        except Exception as exc:
            logger.exception("Workflow route %s failed during submission", parent_job_id)
            result = build_workflow_route_result(plan, children, self._job_store.get_job)
            self._job_store.update_job(
                parent_job_id,
                status=JobStatus.FAILED,
                result=result,
                error=str(exc),
            )
            self._job_store.log_event(
                parent_job_id,
                "failed",
                "Workflow route submission failed.",
                {"backend": "local", "error": str(exc), "children": children},
            )
        return parent_job_id

    async def _monitor_workflow_group(
        self,
        parent_job_id: str,
        children: list[dict[str, str]],
    ) -> None:
        try:
            while True:
                result = build_workflow_group_result(children, self._job_store.get_job)
                group_status = workflow_group_status(result)
                update_fields: dict[str, Any] = {
                    "status": group_status,
                    "result": result,
                    "error": workflow_group_error(result),
                }
                if group_status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                    update_fields.update(workflow_group_usage_fields(result))
                self._job_store.update_job(parent_job_id, **update_fields)

                if group_status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                    self._job_store.log_event(
                        parent_job_id,
                        "completed" if group_status == JobStatus.COMPLETED else "failed",
                        "Workflow group execution completed."
                        if group_status == JobStatus.COMPLETED
                        else "Workflow group execution failed.",
                        {
                            "backend": "local",
                            "summary": result.get("summary"),
                            "children": result.get("children"),
                        },
                    )
                    return

                await asyncio.sleep(1.0)
        except Exception as exc:
            logger.exception("Workflow group monitor failed for %s", parent_job_id)
            self._job_store.update_job(
                parent_job_id,
                status=JobStatus.FAILED,
                error=str(exc),
            )
            self._job_store.log_event(
                parent_job_id,
                "failed",
                "Workflow group monitor failed.",
                {"backend": "local", "error": str(exc)},
            )

    async def _run_job(
        self,
        job_id: str,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
        config_context: dict[str, Any],
    ) -> None:
        try:
            with workflow_span(
                workflow_type.value,
                job_id=job_id,
                backend="local",
                config_context=config_context,
            ):
                self._job_store.update_job(job_id, status=JobStatus.RUNNING)
                self._job_store.log_event(
                    job_id,
                    "running",
                    "Workflow execution started.",
                    {"workflow_type": workflow_type.value, "backend": "local"},
                )
                crew_function = self._engine.crew_function(workflow_type)
                config_context[PROGRESS_CONTEXT_KEY] = WorkflowProgressRecorder(
                    job_id=job_id,
                    workflow_type=workflow_type.value,
                    job_store=self._job_store,
                    backend="local",
                )
                apply_runtime_environment(config_context)
                started_at = monotonic_time()
                result = await asyncio.to_thread(crew_function, inputs, config_context)
                clean_result, usage_metrics = pop_usage_metrics(result)
                usage_summary = build_usage_summary(
                    usage_metrics,
                    monotonic_time() - started_at,
                    config_context,
                )
                record_usage_metrics(usage_summary, config_context)
                self._job_store.update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    result=clean_result if isinstance(clean_result, dict) else {"raw": str(clean_result)},
                    **usage_summary,
                    error=None,
                )
                self._job_store.log_event(
                    job_id,
                    "completed",
                    "Workflow execution completed.",
                    {
                        "workflow_type": workflow_type.value,
                        "backend": "local",
                        "total_tokens": usage_summary.get("total_tokens"),
                        "cost_usd": usage_summary.get("cost_usd"),
                        "duration_seconds": usage_summary.get("duration_seconds"),
                    },
                )
                if workflow_type == WorkflowType.SUPPORT and isinstance(clean_result, dict):
                    try:
                        from services.support_auto_dispatch import process_completed_support_job

                        dispatch_result = await process_completed_support_job(
                            job_id=job_id,
                            inputs=inputs,
                            result=clean_result,
                            config_context=config_context,
                        )
                        self._job_store.log_event(
                            job_id,
                            "support_auto_dispatch",
                            "Support auto-dispatch evaluated.",
                            dispatch_result,
                        )
                    except Exception as dispatch_exc:
                        logger.exception("Support auto-dispatch failed for job %s", job_id)
                        self._job_store.log_event(
                            job_id,
                            "support_auto_dispatch_failed",
                            "Support auto-dispatch failed after workflow completion.",
                            {"error": str(dispatch_exc)},
                        )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                result=None,
                duration_seconds=None,
                error=str(exc),
            )
            self._job_store.log_event(
                job_id,
                "failed",
                "Workflow execution failed.",
                {"workflow_type": workflow_type.value, "backend": "local", "error": str(exc)},
            )

    async def _submit_ready_route_children(
        self,
        parent_job_id: str,
        request: WorkflowRouteRequest,
        plan: WorkflowRoutePlan,
        children: list[dict[str, Any]],
    ) -> None:
        submitted_names = route_submitted_names(children)
        completed_names = route_completed_names(children, self._job_store.get_job)
        for node in ready_route_nodes(plan, submitted_names, completed_names):
            child_job_id = await self.submit_job(
                node.workflow_type,
                node.inputs,
                provider_credentials=workflow_route_provider_credentials(request),
                metadata=route_child_metadata(
                    parent_job_id,
                    node,
                    request.metadata,
                    route_wave_index(plan, node.name),
                ),
            )
            children.append(
                {
                    "name": node.name,
                    "workflow_type": node.workflow_type.value,
                    "job_id": child_job_id,
                    "depends_on": list(node.depends_on),
                }
            )
            submitted_names.add(node.name)

    async def _monitor_workflow_route(
        self,
        parent_job_id: str,
        request: WorkflowRouteRequest,
        plan: WorkflowRoutePlan,
        children: list[dict[str, Any]],
    ) -> None:
        try:
            while True:
                result = build_workflow_route_result(plan, children, self._job_store.get_job)
                route_status = workflow_route_status(result)
                if route_status == JobStatus.FAILED:
                    self._job_store.update_job(
                        parent_job_id,
                        status=JobStatus.FAILED,
                        result=result,
                        **workflow_route_usage_fields(result),
                        error=workflow_route_error(result),
                    )
                    self._job_store.log_event(
                        parent_job_id,
                        "failed",
                        "Workflow route execution failed.",
                        {"backend": "local", "summary": result.get("summary"), "children": result.get("children")},
                    )
                    return

                before_count = len(children)
                await self._submit_ready_route_children(parent_job_id, request, plan, children)
                if len(children) != before_count:
                    result = build_workflow_route_result(plan, children, self._job_store.get_job)
                    self._job_store.update_job(
                        parent_job_id,
                        status=JobStatus.RUNNING,
                        result=result,
                        error=None,
                    )
                    self._job_store.log_event(
                        parent_job_id,
                        "route_wave_submitted",
                        "Workflow route submitted another ready worker wave.",
                        {"backend": "local", "children": children},
                    )

                route_status = workflow_route_status(result)
                if route_status == JobStatus.COMPLETED:
                    self._job_store.update_job(
                        parent_job_id,
                        status=JobStatus.COMPLETED,
                        result=result,
                        **workflow_route_usage_fields(result),
                        error=None,
                    )
                    self._job_store.log_event(
                        parent_job_id,
                        "completed",
                        "Workflow route execution completed.",
                        {"backend": "local", "summary": result.get("summary"), "children": result.get("children")},
                    )
                    return

                self._job_store.update_job(parent_job_id, status=JobStatus.RUNNING, result=result, error=None)
                await asyncio.sleep(1.0)
        except Exception as exc:
            logger.exception("Workflow route monitor failed for %s", parent_job_id)
            self._job_store.update_job(parent_job_id, status=JobStatus.FAILED, error=str(exc))
            self._job_store.log_event(
                parent_job_id,
                "failed",
                "Workflow route monitor failed.",
                {"backend": "local", "error": str(exc)},
            )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return self._job_store.get_job(job_id) or {
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "result": None,
            "error": "Job not found",
        }

    def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
        return self._job_store.get_job_events(job_id)


class CeleryOrchestrator:
    """Celery-backed workflow router for Redis/message-broker deployments."""

    TASK_MAP: dict[WorkflowType, str] = {
        WorkflowType.MARKETING: "workflow.marketing",
        WorkflowType.CONTENT: "workflow.content",
        WorkflowType.SUPPORT: "workflow.support",
        WorkflowType.ANALYTICS: "workflow.analytics",
        WorkflowType.SALES_IMPROVEMENT: "workflow.sales_improvement",
        WorkflowType.BIZDEV: "workflow.bizdev",
        WorkflowType.SCHEDULER: "workflow.scheduler",
    }

    def __init__(self, job_store: JobStore, runtime_config: RuntimeConfig | None = None) -> None:
        self._job_store = job_store
        self._runtime_config = runtime_config or RuntimeConfig()
        init_observability("cross-border-celery-orchestrator", config_context=self._runtime_config.as_context())
        self._engine = WorkflowExecutionEngine(self._job_store, self._runtime_config)
        logger.info("Celery orchestrator initialized with %s job store.", self.job_store_name)

    @property
    def job_store_name(self) -> str:
        return self._job_store.name

    @property
    def registered_workflows(self) -> list[WorkflowType]:
        return list(self.TASK_MAP.keys())

    async def submit_job(
        self,
        workflow_type: WorkflowType,
        inputs: dict[str, Any],
        provider_credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        task_name = self.TASK_MAP.get(workflow_type)
        if task_name is None:
            raise ValueError(f"Workflow '{workflow_type.value}' is not registered.")

        prepared = self._engine.prepare_job(
            workflow_type,
            inputs,
            provider_credentials,
            metadata,
            "celery",
            "Celery task queued.",
            {"task_name": task_name},
        )
        if prepared.cache_hit:
            return prepared.job_id

        task = celery_app.send_task(
            task_name,
            args=[inputs],
            kwargs={"config_context": prepared.config_context},
            task_id=prepared.job_id,
        )
        logger.info("Queued Celery job %s for workflow %s", task.id, workflow_type.value)
        return prepared.job_id

    async def submit_workflow_group(self, request: WorkflowGroupRequest) -> str:
        parent_job_id = str(uuid.uuid4())
        children: list[dict[str, str]] = []
        self._job_store.create_job(
            parent_job_id,
            WORKFLOW_GROUP_TYPE,
            workflow_group_parent_inputs(request),
        )
        self._job_store.log_event(
            parent_job_id,
            "queued",
            "Workflow group submitted.",
            {"workflow_type": WORKFLOW_GROUP_TYPE, "backend": "celery"},
        )

        try:
            with group_span(
                parent_job_id=parent_job_id,
                backend="celery",
                attributes={"children_requested": len(request.workflows)},
                config_context=self._runtime_config.as_context(),
            ):
                for item in request.workflows:
                    child_name = workflow_group_item_name(item)
                    child_job_id = await self.submit_job(
                        item.workflow_type,
                        item.inputs,
                        provider_credentials=workflow_group_provider_credentials(item),
                        metadata=workflow_group_child_metadata(
                            parent_job_id,
                            child_name,
                            item,
                            request.metadata,
                        ),
                    )
                    children.append(
                        {
                            "name": child_name,
                            "workflow_type": item.workflow_type.value,
                            "job_id": child_job_id,
                        }
                    )

                result = build_workflow_group_result(children, self._job_store.get_job)
                self._job_store.update_job(
                    parent_job_id,
                    status=JobStatus.RUNNING,
                    result=result,
                    error=None,
                )
                monitor_task = celery_app.send_task(
                    WORKFLOW_GROUP_MONITOR_TASK,
                    args=[parent_job_id, children],
                    task_id=parent_job_id,
                )
                self._job_store.log_event(
                    parent_job_id,
                    "running",
                    "Workflow group child jobs submitted and monitor queued.",
                    {
                        "backend": "celery",
                        "monitor_task_id": monitor_task.id,
                        "children": children,
                    },
                )
        except Exception as exc:
            logger.exception("Workflow group %s failed during Celery submission", parent_job_id)
            result = build_workflow_group_result(children, self._job_store.get_job)
            self._job_store.update_job(
                parent_job_id,
                status=JobStatus.FAILED,
                result=result,
                error=str(exc),
            )
            self._job_store.log_event(
                parent_job_id,
                "failed",
                "Workflow group submission failed.",
                {"backend": "celery", "error": str(exc), "children": children},
            )
        return parent_job_id

    def plan_workflow_route(self, request: WorkflowRouteRequest) -> WorkflowRoutePlan:
        config_context = resolve_workflow_runtime_context(
            self._runtime_config,
            WORKFLOW_ROUTE_TYPE,
            workflow_route_provider_credentials(request),
        )
        return WorkflowRouterAgent(config_context).plan(request)

    async def submit_workflow_route(self, request: WorkflowRouteRequest) -> str:
        plan = self.plan_workflow_route(request)
        _ensure_submittable_route_plan(plan)
        parent_job_id = str(uuid.uuid4())
        children: list[dict[str, Any]] = []
        self._job_store.create_job(
            parent_job_id,
            WORKFLOW_ROUTE_TYPE,
            workflow_route_parent_inputs(request, plan),
        )
        initial_result = build_workflow_route_result(plan, children, self._job_store.get_job)
        self._job_store.update_job(parent_job_id, status=JobStatus.RUNNING, result=initial_result, error=None)
        self._job_store.log_event(
            parent_job_id,
            "queued",
            "Workflow route submitted.",
            {"workflow_type": WORKFLOW_ROUTE_TYPE, "backend": "celery", "plan": initial_result.get("plan")},
        )

        try:
            with route_span(
                parent_job_id=parent_job_id,
                backend="celery",
                attributes={"planned_nodes": len(plan.nodes)},
                config_context=self._runtime_config.as_context(),
            ):
                children.extend(
                    await self._submit_ready_route_children(
                        parent_job_id,
                        request,
                        plan,
                        children,
                    )
                )
                result = build_workflow_route_result(plan, children, self._job_store.get_job)
                self._job_store.update_job(parent_job_id, status=JobStatus.RUNNING, result=result, error=None)
                monitor_task = celery_app.send_task(
                    WORKFLOW_ROUTE_MONITOR_TASK,
                    args=[
                        parent_job_id,
                        plan.model_dump(mode="json"),
                        children,
                    ],
                    kwargs={
                        "provider_credentials": workflow_route_provider_credentials(request),
                        "metadata": request.metadata or {},
                    },
                    task_id=parent_job_id,
                )
                self._job_store.log_event(
                    parent_job_id,
                    "running",
                    "Workflow route initial child jobs submitted and monitor queued.",
                    {
                        "backend": "celery",
                        "monitor_task_id": monitor_task.id,
                        "children": children,
                    },
                )
        except Exception as exc:
            logger.exception("Workflow route %s failed during Celery submission", parent_job_id)
            result = build_workflow_route_result(plan, children, self._job_store.get_job)
            self._job_store.update_job(parent_job_id, status=JobStatus.FAILED, result=result, error=str(exc))
            self._job_store.log_event(
                parent_job_id,
                "failed",
                "Workflow route submission failed.",
                {"backend": "celery", "error": str(exc), "children": children},
            )
        return parent_job_id

    async def _submit_ready_route_children(
        self,
        parent_job_id: str,
        request: WorkflowRouteRequest,
        plan: WorkflowRoutePlan,
        children: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        submitted: list[dict[str, Any]] = []
        submitted_names = route_submitted_names(children)
        completed_names = route_completed_names(children, self._job_store.get_job)
        for node in ready_route_nodes(plan, submitted_names, completed_names):
            child_job_id = await self.submit_job(
                node.workflow_type,
                node.inputs,
                provider_credentials=workflow_route_provider_credentials(request),
                metadata=route_child_metadata(
                    parent_job_id,
                    node,
                    request.metadata,
                    route_wave_index(plan, node.name),
                ),
            )
            child = {
                "name": node.name,
                "workflow_type": node.workflow_type.value,
                "job_id": child_job_id,
                "depends_on": list(node.depends_on),
            }
            submitted.append(child)
            submitted_names.add(node.name)
        return submitted

    def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
        return self._job_store.get_job_events(job_id)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        result = AsyncResult(job_id, app=celery_app)

        if result.state == "PENDING":
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "result": None,
                "error": None,
            }

        if result.state in {"STARTED", "PROGRESS", "RETRY"}:
            progress = result.info if isinstance(result.info, dict) else None
            update_fields: dict[str, Any] = {
                "status": JobStatus.RUNNING,
                "error": None,
            }
            if progress is not None:
                update_fields["result"] = progress
            self._job_store.update_job(job_id, **update_fields)
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.RUNNING,
                "result": progress,
                "error": None,
            }

        if result.state == "SUCCESS":
            task_result = result.result
            result_payload, usage_fields = _split_task_result(task_result)
            self._job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=result_payload,
                **usage_fields,
                error=None,
            )
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.COMPLETED,
                "result": result_payload,
                **usage_fields,
                "error": None,
            }

        if result.state == "FAILURE":
            existing_job = self._job_store.get_job(job_id)
            self._job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=str(result.result),
            )
            return self._job_store.get_job(job_id) or {
                "job_id": job_id,
                "status": JobStatus.FAILED,
                "result": existing_job.get("result") if existing_job else None,
                "error": str(result.result),
            }

        self._job_store.update_job(
            job_id,
            status=JobStatus.RUNNING,
            result={"celery_state": result.state},
            error=None,
        )
        return {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "result": {"celery_state": result.state},
            "error": None,
        }
