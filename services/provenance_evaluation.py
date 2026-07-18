from __future__ import annotations

from typing import Any

from job_store import JobStore
from models import WorkflowType
from runtime_config import load_runtime_config, resolve_workflow_runtime_context
from services.workflow_guardrails import (
    WorkflowGuardrailService,
    decision_event_payload,
    support_provenance_claim,
    support_provenance_grounding_context,
)


def evaluate_support_job_provenance(
    job_id: str,
    *,
    job_store: JobStore,
    service: WorkflowGuardrailService | None = None,
    config_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_events = job_store.get_job_events(job_id)
    if any(event.get("event_type") == "guardrail_provenance_evaluated" for event in existing_events):
        return {"status": "already_evaluated", "job_id": job_id}

    job = job_store.get_job(job_id)
    if not isinstance(job, dict) or str(job.get("workflow_type") or "") != WorkflowType.SUPPORT.value:
        return {"status": "not_applicable", "job_id": job_id}
    result = job.get("result")
    if not isinstance(result, dict):
        return {"status": "missing_result", "job_id": job_id}

    claim = support_provenance_claim(result)
    sources = support_provenance_grounding_context(result)
    if not claim:
        job_store.log_event(
            job_id,
            "guardrail_provenance_evaluated",
            "Asynchronous provenance evaluation was not applicable.",
            {"status": "not_applicable", "reason": "customer_facing_reply_unavailable", "advisory": True},
        )
        return {"status": "not_applicable", "job_id": job_id}

    inputs = job.get("inputs") if isinstance(job.get("inputs"), dict) else {}
    conversation_id = str(
        inputs.get("conversation_id")
        or inputs.get("session_id")
        or result.get("conversation_id")
        or result.get("session_id")
        or ""
    )
    context = dict(
        config_context
        or resolve_workflow_runtime_context(
            load_runtime_config(),
            WorkflowType.SUPPORT,
        )
    )
    context.update(
        {
            "job_id": job_id,
            "conversation_id": conversation_id,
            "workflow_type": WorkflowType.SUPPORT.value,
        }
    )
    decision = (service or WorkflowGuardrailService()).evaluate_provenance(
        WorkflowType.SUPPORT,
        claim,
        grounding_context=sources,
        config_context=context,
    )
    skipped = decision.metadata.get("skipped_validators") or []
    status = "not_applicable" if skipped and not decision.findings else "evaluated"
    job_store.log_event(
        job_id,
        "guardrail_provenance_evaluated",
        "Asynchronous provenance evaluation completed.",
        {
            **decision_event_payload(decision),
            "status": status,
            "advisory": True,
            "claim_present": True,
            "source_count": len(sources),
        },
    )
    return {
        "status": status,
        "job_id": job_id,
        "finding_count": len(decision.findings),
    }
