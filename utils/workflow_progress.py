import logging
import time
from threading import Lock
from typing import Any, Callable

from models import JobStatus
from utils.observability import add_span_event, end_span, start_agent_span

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
        self._lock = Lock()
        self._task_started_at: dict[tuple[int, str], float] = {}
        self._task_spans: dict[tuple[int, str], Any] = {}

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
        started_at = time.monotonic()
        self._task_started_at[(index, task_name)] = started_at
        self._task_spans[(index, task_name)] = start_agent_span(
            job_id=self.job_id,
            workflow_type=self.workflow_type,
            task_name=task_name,
            agent_role=agent_role,
            backend=self.backend,
        )
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
        started_at = self._task_started_at.pop((index, task_name), None)
        task_span = self._task_spans.pop((index, task_name), None)
        duration_seconds = None if started_at is None else round(time.monotonic() - started_at, 3)
        end_span(
            task_span,
            attributes={
                "task_index": index + 1,
                "total_tasks": total,
                "duration_ms": round(duration_seconds * 1000, 3)
                if duration_seconds is not None
                else None,
            },
        )
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
                "duration_seconds": duration_seconds,
            },
        )

    def _update_progress(
        self,
        event_type: str,
        message: str,
        progress: float,
        payload: dict[str, Any],
    ) -> None:
        self.emit_progress(event_type, message, progress, payload)

    def emit_progress(
        self,
        event_type: str,
        message: str,
        progress: float,
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
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
            add_span_event(event_type, event_payload)


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

    state_lock = Lock()
    started_indices: set[int] = set()
    completed_indices: set[int] = set()

    def start_task(task_index: int) -> None:
        if task_index in started_indices or task_index >= total:
            return
        started_indices.add(task_index)
        recorder.task_started(
            task_index,
            total,
            safe_task_names[task_index],
            _agent_role(tasks[task_index]),
        )

    def start_async_block(first_index: int) -> None:
        task_index = first_index
        while task_index < total and _is_async_task(tasks[task_index]):
            start_task(task_index)
            task_index += 1

    recorder.emit_plan(safe_task_names)
    if _is_async_task(tasks[0]):
        start_async_block(0)
    else:
        start_task(0)

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
            tasks=tasks,
            task_names=safe_task_names,
            state_lock=state_lock,
            started_indices=started_indices,
            completed_indices=completed_indices,
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
    tasks: list[Any],
    task_names: list[str],
    state_lock: Lock,
    started_indices: set[int],
    completed_indices: set[int],
) -> Callable[[Any], Any]:
    def callback(output: Any) -> Any:
        callback_result = None
        if original_callback:
            callback_result = original_callback(output)

        try:
            with state_lock:
                completed_indices.add(task_index)
                recorder.task_completed(task_index, total_tasks, task_name, _agent_role(task))

                if _is_async_task(task):
                    block_start, block_end = _async_block_bounds(tasks, task_index)
                    if all(index in completed_indices for index in range(block_start, block_end)):
                        _start_next_progress_task(
                            recorder,
                            tasks,
                            task_names,
                            block_end,
                            total_tasks,
                            started_indices,
                        )
                elif next_task is not None and next_task_name is not None:
                    _start_next_progress_task(
                        recorder,
                        tasks,
                        task_names,
                        task_index + 1,
                        total_tasks,
                        started_indices,
                    )
        except Exception:
            logger.exception("Failed to record workflow progress for task %s", task_name)

        return callback_result

    return callback


def _start_next_progress_task(
    recorder: WorkflowProgressRecorder,
    tasks: list[Any],
    task_names: list[str],
    task_index: int,
    total_tasks: int,
    started_indices: set[int],
) -> None:
    if task_index >= total_tasks:
        return

    if _is_async_task(tasks[task_index]):
        while task_index < total_tasks and _is_async_task(tasks[task_index]):
            _start_progress_task(
                recorder,
                tasks,
                task_names,
                task_index,
                total_tasks,
                started_indices,
            )
            task_index += 1
        return

    _start_progress_task(
        recorder,
        tasks,
        task_names,
        task_index,
        total_tasks,
        started_indices,
    )


def _start_progress_task(
    recorder: WorkflowProgressRecorder,
    tasks: list[Any],
    task_names: list[str],
    task_index: int,
    total_tasks: int,
    started_indices: set[int],
) -> None:
    if task_index in started_indices:
        return

    started_indices.add(task_index)
    recorder.task_started(
        task_index,
        total_tasks,
        task_names[task_index],
        _agent_role(tasks[task_index]),
    )


def _async_block_bounds(tasks: list[Any], task_index: int) -> tuple[int, int]:
    block_start = task_index
    while block_start > 0 and _is_async_task(tasks[block_start - 1]):
        block_start -= 1

    block_end = task_index + 1
    while block_end < len(tasks) and _is_async_task(tasks[block_end]):
        block_end += 1

    return block_start, block_end


def _is_async_task(task: Any) -> bool:
    return bool(getattr(task, "async_execution", False))


def _agent_role(task: Any) -> str | None:
    agent = getattr(task, "agent", None)
    role = getattr(agent, "role", None)
    return str(role) if role else None
