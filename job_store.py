import json
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from db_models import JobEventRecord, JobRecord
from models import JobStatus, WorkflowType


class JobStore(Protocol):
    name: str

    def create_job(
        self,
        job_id: str,
        workflow_type: WorkflowType | str,
        inputs: dict[str, Any],
        cache_key: str | None = None,
    ) -> None:
        ...

    def update_job(self, job_id: str, **fields: Any) -> None:
        ...

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        ...

    def find_cached_job(self, cache_key: str, ttl_seconds: int) -> dict[str, Any] | None:
        ...

    def log_event(
        self,
        job_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ...

    def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
        ...


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _status_value(status: JobStatus | str) -> str:
    return status.value if isinstance(status, JobStatus) else status


def _workflow_type_value(workflow_type: WorkflowType | str) -> str:
    return workflow_type.value if isinstance(workflow_type, WorkflowType) else str(workflow_type)


class InMemoryJobStore:
    name = "memory"

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(
        self,
        job_id: str,
        workflow_type: WorkflowType | str,
        inputs: dict[str, Any],
        cache_key: str | None = None,
    ) -> None:
        workflow_value = _workflow_type_value(workflow_type)
        self._jobs[job_id] = {
            "job_id": job_id,
            "workflow_type": workflow_value,
            "status": JobStatus.PENDING,
            "inputs": inputs,
            "result": None,
            "cache_key": cache_key,
            "cache_hit": False,
            "source_job_id": None,
            "usage_metrics": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
            "duration_seconds": None,
            "error": None,
            "events": [
                {
                    "event_id": 1,
                    "job_id": job_id,
                    "event_type": "submitted",
                    "message": f"Workflow job submitted: {workflow_value}",
                    "payload": {"workflow_type": workflow_value},
                    "created_at": None,
                }
            ],
        }

    def update_job(self, job_id: str, **fields: Any) -> None:
        job = self._jobs.setdefault(
            job_id,
            {
                "job_id": job_id,
                "workflow_type": "",
                "status": JobStatus.PENDING,
                "inputs": None,
                "result": None,
                "usage_metrics": None,
                "cache_key": None,
                "cache_hit": None,
                "source_job_id": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cost_usd": None,
                "duration_seconds": None,
                "error": None,
            },
        )
        job.update(fields)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def find_cached_job(self, cache_key: str, ttl_seconds: int) -> dict[str, Any] | None:
        for job in reversed(list(self._jobs.values())):
            if (
                job.get("cache_key") == cache_key
                and job.get("status") == JobStatus.COMPLETED
                and job.get("result") is not None
                and not job.get("cache_hit")
            ):
                return job
        return None

    def log_event(
        self,
        job_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        job = self._jobs.setdefault(job_id, {"job_id": job_id, "events": []})
        events = job.setdefault("events", [])
        events.append(
            {
                "event_id": len(events) + 1,
                "job_id": job_id,
                "event_type": event_type,
                "message": message,
                "payload": _json_safe(payload),
                "created_at": None,
            }
        )

    def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
        job = self._jobs.get(job_id) or {}
        return list(job.get("events", []))


class PostgresJobStore:
    name = "postgres"

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_job(
        self,
        job_id: str,
        workflow_type: WorkflowType | str,
        inputs: dict[str, Any],
        cache_key: str | None = None,
    ) -> None:
        workflow_value = _workflow_type_value(workflow_type)
        with self._session_factory() as session:
            session.add(
                JobRecord(
                    job_id=job_id,
                    workflow_type=workflow_value,
                    status=JobStatus.PENDING.value,
                    inputs=_json_safe(inputs),
                    result=None,
                    cache_key=cache_key,
                    cache_hit=False,
                    source_job_id=None,
                    usage_metrics=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    cost_usd=None,
                    duration_seconds=None,
                    error=None,
                )
            )
            session.commit()

        self.log_event(
            job_id,
            "submitted",
            f"Workflow job submitted: {workflow_value}",
            {"workflow_type": workflow_value},
        )

    def update_job(self, job_id: str, **fields: Any) -> None:
        with self._session_factory() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                record = JobRecord(
                    job_id=job_id,
                    workflow_type=str(fields.pop("workflow_type", "")),
                    status=JobStatus.PENDING.value,
                )
                session.add(record)

            if "status" in fields:
                record.status = _status_value(fields["status"])
            if "result" in fields:
                record.result = _json_safe(fields["result"])
            if "cache_key" in fields:
                record.cache_key = fields["cache_key"]
            if "cache_hit" in fields:
                record.cache_hit = fields["cache_hit"]
            if "source_job_id" in fields:
                record.source_job_id = fields["source_job_id"]
            if "usage_metrics" in fields:
                record.usage_metrics = _json_safe(fields["usage_metrics"])
            if "prompt_tokens" in fields:
                record.prompt_tokens = fields["prompt_tokens"]
            if "completion_tokens" in fields:
                record.completion_tokens = fields["completion_tokens"]
            if "total_tokens" in fields:
                record.total_tokens = fields["total_tokens"]
            if "cost_usd" in fields:
                record.cost_usd = fields["cost_usd"]
            if "duration_seconds" in fields:
                record.duration_seconds = fields["duration_seconds"]
            if "error" in fields:
                record.error = fields["error"]
            if "inputs" in fields:
                record.inputs = _json_safe(fields["inputs"])
            if "workflow_type" in fields:
                workflow_type = fields["workflow_type"]
                record.workflow_type = (
                    workflow_type.value if isinstance(workflow_type, WorkflowType) else str(workflow_type)
                )

            session.commit()

    def log_event(
        self,
        job_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._session_factory() as session:
            session.add(
                JobEventRecord(
                    job_id=job_id,
                    event_type=event_type,
                    message=message,
                    payload=_json_safe(payload),
                )
            )
            session.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                return None

            return {
                "job_id": record.job_id,
                "status": JobStatus(record.status),
                "result": record.result,
                "cache_hit": record.cache_hit,
                "source_job_id": record.source_job_id,
                "usage_metrics": record.usage_metrics,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
                "cost_usd": record.cost_usd,
                "duration_seconds": record.duration_seconds,
                "error": record.error,
            }

    def find_cached_job(self, cache_key: str, ttl_seconds: int) -> dict[str, Any] | None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        with self._session_factory() as session:
            query = (
                session.query(JobRecord)
                .filter(JobRecord.cache_key == cache_key)
                .filter(JobRecord.status == JobStatus.COMPLETED.value)
                .filter(JobRecord.result.isnot(None))
                .filter(JobRecord.cache_hit.isnot(True))
            )
            if ttl_seconds > 0:
                query = query.filter(JobRecord.updated_at >= cutoff)
            record = query.order_by(JobRecord.updated_at.desc()).first()
            if record is None:
                return None

            return {
                "job_id": record.job_id,
                "status": JobStatus(record.status),
                "result": record.result,
                "cache_hit": record.cache_hit,
                "source_job_id": record.source_job_id,
                "usage_metrics": record.usage_metrics,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
                "cost_usd": record.cost_usd,
                "duration_seconds": record.duration_seconds,
                "error": record.error,
            }

    def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            records = (
                session.query(JobEventRecord)
                .filter(JobEventRecord.job_id == job_id)
                .order_by(JobEventRecord.event_id.asc())
                .all()
            )
            return [
                {
                    "event_id": record.event_id,
                    "job_id": record.job_id,
                    "event_type": record.event_type,
                    "message": record.message,
                    "payload": record.payload,
                    "created_at": record.created_at,
                }
                for record in records
            ]
