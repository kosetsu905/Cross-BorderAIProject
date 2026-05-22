import logging
from typing import Any, Callable

from models import JobStatus

logger = logging.getLogger(__name__)

PROGRESS_CONTEXT_KEY = "_workflow_progress_recorder"
PROGRESS_START = 0.2
PROGRESS_SPAN = 0.7


class WorkflowProgressRecorder:
    def __init__(self, job_id: str, workflow_type: str, job_store: Any, backend: str) -> None:
        self.job_id = job_id
        self.workflow_type = workflow_type
        self.job_store = job_store
        self.backend = backend

    def emit_plan(self, task_names: list[str]) -> None:
        self._update_progress(
            event_type="task_plan",
            message=f"Workflow has {len(task_names)} planned tasks.",
            progress=PROGRESS_START,
            payload={
                "task_names": task_names,
                "total_tasks": len(task_names),
            },
        )

    def task_started(self, index: int, total: int, task_name: str, agent_role: str | None) -> None:
        progress = PROGRESS_START + (PROGRESS_SPAN * index / max(total, 1))
        self._update_progress(
            event_type="task_started",
            message=f"Task {index + 1}/{total} started: {task_name}",
            progress=progress,
            payload={
                "task_index": index + 1,
                "total_tasks": total,
                "task_name": task_name,
                "agent_role": agent_role,
            },
        )

    def task_completed(self, index: int, total: int, task_name: str, agent_role: str | None) -> None:
        progress = PROGRESS_START + (PROGRESS_SPAN * (index + 1) / max(total, 1))
        self._update_progress(
            event_type="task_completed",
            message=f"Task {index + 1}/{total} completed: {task_name}",
            progress=progress,
            payload={
                "task_index": index + 1,
                "total_tasks": total,
                "task_name": task_name,
                "agent_role": agent_role,
            },
        )

    def _update_progress(
        self,
        event_type: str,
        message: str,
        progress: float,
        payload: dict[str, Any],
    ) -> None:
        event_payload = {
            "workflow_type": self.workflow_type,
            "backend": self.backend,
            "progress": round(progress, 3),
            **payload,
        }
        self.job_store.update_job(
            self.job_id,
            status=JobStatus.RUNNING,
            result={
                "status": message,
                "progress": round(progress, 3),
                **payload,
            },
            error=None,
        )
        self.job_store.log_event(self.job_id, event_type, message, event_payload)


def attach_task_progress(
    config_context: dict[str, Any],
    workflow_type: str,
    tasks: list[Any],
    task_names: list[str],
) -> None:
    recorder = config_context.get(PROGRESS_CONTEXT_KEY)
    if not isinstance(recorder, WorkflowProgressRecorder) or not tasks:
        return

    total = len(tasks)
    safe_task_names = task_names[:total]
    if len(safe_task_names) < total:
        safe_task_names.extend(f"task_{index + 1}" for index in range(len(safe_task_names), total))

    recorder.emit_plan(safe_task_names)
    recorder.task_started(0, total, safe_task_names[0], _agent_role(tasks[0]))

    for index, task in enumerate(tasks):
        original_callback = getattr(task, "callback", None)
        task.callback = _build_task_callback(
            recorder=recorder,
            task=task,
            task_name=safe_task_names[index],
            task_index=index,
            total_tasks=total,
            next_task=tasks[index + 1] if index + 1 < total else None,
            next_task_name=safe_task_names[index + 1] if index + 1 < total else None,
            original_callback=original_callback,
        )


def _build_task_callback(
    recorder: WorkflowProgressRecorder,
    task: Any,
    task_name: str,
    task_index: int,
    total_tasks: int,
    next_task: Any | None,
    next_task_name: str | None,
    original_callback: Callable[[Any], Any] | None,
) -> Callable[[Any], Any]:
    def callback(output: Any) -> Any:
        callback_result = None
        if original_callback:
            callback_result = original_callback(output)

        try:
            recorder.task_completed(task_index, total_tasks, task_name, _agent_role(task))
            if next_task is not None and next_task_name is not None:
                recorder.task_started(
                    task_index + 1,
                    total_tasks,
                    next_task_name,
                    _agent_role(next_task),
                )
        except Exception:
            logger.exception("Failed to record workflow progress for task %s", task_name)

        return callback_result

    return callback


def _agent_role(task: Any) -> str | None:
    agent = getattr(task, "agent", None)
    role = getattr(agent, "role", None)
    return str(role) if role else None
