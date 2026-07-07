from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import ExitStack, contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from fastapi import FastAPI


logger = logging.getLogger(__name__)

SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|client[_-]?secret|credential|password|refresh[_-]?token|secret|token)",
    re.IGNORECASE,
)
PII_KEY_RE = re.compile(
    r"(customer[_-]?email|customer[_-]?handle|email|phone|recipient|sender|raw[_-]?handle)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
MAX_ATTRIBUTE_LENGTH = 2048

_INIT_LOCK = Lock()
_INITIALIZED_SERVICES: set[str] = set()
_INSTRUMENTED_FASTAPI_APP_IDS: set[int] = set()
_INSTRUMENTED_GLOBALS = False
_OTEL_CONFIGURED = False
_LANGFUSE_CLIENT: Any | None = None


class NoOpSpan:
    """Small span-like object used when observability is disabled or unavailable."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None

    def record_exception(self, exception: BaseException) -> None:
        return None

    def end(self) -> None:
        return None


class ManagedObservationSpan:
    """Span-like wrapper that updates both OpenTelemetry and Langfuse observations."""

    def __init__(
        self,
        *,
        otel_span: Any | None = None,
        langfuse_context: Any | None = None,
        langfuse_observation: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._otel_span = otel_span
        self._langfuse_context = langfuse_context
        self._langfuse_observation = langfuse_observation
        self._attributes = dict(attributes or {})
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        if value is None:
            return
        self._attributes[str(key)] = value
        if self._otel_span is not None:
            self._otel_span.set_attribute(key, value)
        self._update_langfuse_metadata()

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        if self._otel_span is not None:
            self._otel_span.add_event(name, attributes=_safe_span_attributes(attributes or {}))
        observation = self._langfuse_observation
        if observation is not None and hasattr(observation, "create_event"):
            try:
                observation.create_event(
                    name=name,
                    metadata=_json_safe(redact_observability_payload(attributes or {}, capture_raw=False)),
                )
            except Exception:
                logger.debug("Failed to create Langfuse observation event %s", name, exc_info=True)

    def record_exception(self, exception: BaseException) -> None:
        if self._otel_span is not None:
            self._otel_span.record_exception(exception)
        observation = self._langfuse_observation
        if observation is not None and hasattr(observation, "update"):
            try:
                observation.update(level="ERROR", status_message=str(exception))
            except Exception:
                logger.debug("Failed to record Langfuse observation exception", exc_info=True)

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._update_langfuse_metadata()
        if self._otel_span is not None:
            self._otel_span.end()
        if self._langfuse_observation is not None and hasattr(self._langfuse_observation, "end"):
            try:
                self._langfuse_observation.end()
            except Exception:
                logger.debug("Failed to end Langfuse observation", exc_info=True)
        if self._langfuse_context is not None and hasattr(self._langfuse_context, "__exit__"):
            try:
                self._langfuse_context.__exit__(None, None, None)
            except Exception:
                logger.debug("Failed to exit Langfuse observation context", exc_info=True)

    def _update_langfuse_metadata(self) -> None:
        observation = self._langfuse_observation
        if observation is None or not hasattr(observation, "update"):
            return
        try:
            observation.update(metadata=_json_safe(redact_observability_payload(self._attributes, capture_raw=False)))
        except Exception:
            logger.debug("Failed to update Langfuse observation metadata", exc_info=True)


def init_observability(
    service_name: str,
    app: FastAPI | None = None,
    config_context: dict[str, Any] | None = None,
) -> None:
    """Initialize optional tracing/instrumentation for FastAPI or Celery processes."""
    if not _observability_enabled(config_context):
        return
    _apply_observability_environment(config_context)
    otel_enabled = _otel_enabled(config_context)

    with _INIT_LOCK:
        if service_name not in _INITIALIZED_SERVICES:
            if otel_enabled:
                _init_otel(service_name, config_context)
                _init_openinference()
            _init_langfuse(config_context)
            _INITIALIZED_SERVICES.add(service_name)

        if otel_enabled and app is not None and id(app) not in _INSTRUMENTED_FASTAPI_APP_IDS:
            _instrument_fastapi(app)
            _INSTRUMENTED_FASTAPI_APP_IDS.add(id(app))


def flush_observability() -> None:
    """Flush buffered telemetry clients after short-lived Celery task execution."""
    client = _LANGFUSE_CLIENT
    if client is not None and hasattr(client, "flush"):
        try:
            client.flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)


@contextmanager
def workflow_span(
    workflow_type: str,
    *,
    job_id: str | None = None,
    backend: str | None = None,
    config_context: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    span_name = f"workflow.{workflow_type}"
    span_attributes = {
        "job_id": job_id,
        "workflow_type": workflow_type,
        "backend": backend,
        **(attributes or {}),
    }
    with _observation_span(
        span_name,
        config_context=config_context,
        attributes=span_attributes,
        observation_type="chain",
    ) as span:
        yield span


@contextmanager
def group_span(
    *,
    parent_job_id: str,
    backend: str,
    attributes: dict[str, Any] | None = None,
    config_context: dict[str, Any] | None = None,
) -> Iterator[Any]:
    span_attributes = {
        "job_id": parent_job_id,
        "workflow_type": "workflow_group",
        "backend": backend,
        **(attributes or {}),
    }
    with _observation_span(
        "workflow_group",
        config_context=config_context,
        attributes=span_attributes,
        observation_type="chain",
    ) as span:
        yield span


@contextmanager
def route_span(
    *,
    parent_job_id: str,
    backend: str,
    attributes: dict[str, Any] | None = None,
    config_context: dict[str, Any] | None = None,
) -> Iterator[Any]:
    span_attributes = {
        "job_id": parent_job_id,
        "workflow_type": "workflow_route",
        "backend": backend,
        **(attributes or {}),
    }
    with _observation_span(
        "workflow_route",
        config_context=config_context,
        attributes=span_attributes,
        observation_type="chain",
    ) as span:
        yield span


@contextmanager
def agent_span(
    *,
    job_id: str,
    workflow_type: str,
    task_name: str,
    agent_role: str | None = None,
    backend: str | None = None,
    config_context: dict[str, Any] | None = None,
) -> Iterator[Any]:
    span_attributes = {
        "job_id": job_id,
        "workflow_type": workflow_type,
        "task_name": task_name,
        "agent_role": agent_role,
        "backend": backend,
    }
    with _observation_span(
        f"agent.{task_name}",
        config_context=config_context,
        attributes=span_attributes,
        observation_type="agent",
    ) as span:
        yield span


@contextmanager
def tool_span(
    tool_name: str,
    *,
    config_context: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    span_attributes = {"tool_name": tool_name, **(attributes or {})}
    with _observation_span(
        f"tool.{tool_name}",
        config_context=config_context,
        attributes=span_attributes,
        observation_type="tool",
    ) as span:
        yield span


@contextmanager
def stage_span(
    stage: str,
    *,
    job_id: str | None = None,
    workflow_type: str | None = None,
    task_name: str | None = None,
    agent_role: str | None = None,
    language: str | None = None,
    target_market: str | None = None,
    status: str | None = None,
    backend: str | None = None,
    config_context: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    stage_name = str(stage).strip() or "unknown"
    observation_name = f"stage.{stage_name}:{language}" if language else f"stage.{stage_name}"
    span_attributes = _safe_stage_attributes(
        {
            "job_id": job_id,
            "workflow_type": workflow_type,
            "task_name": task_name,
            "agent_role": agent_role,
            "language": language,
            "target_market": target_market,
            "stage": stage_name,
            "status": status,
            "backend": backend,
            **(attributes or {}),
        }
    )
    with _observation_span(
        observation_name,
        config_context=config_context,
        attributes=span_attributes,
        observation_type="span",
    ) as span:
        yield span


@contextmanager
def evaluation_span(
    eval_name: str,
    *,
    job_id: str | None = None,
    workflow_type: str | None = None,
    config_context: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    span_attributes = {
        "job_id": job_id,
        "workflow_type": workflow_type,
        "evaluation_name": eval_name,
        **(attributes or {}),
    }
    with _observation_span(
        f"eval.{eval_name}",
        config_context=config_context,
        attributes=span_attributes,
        observation_type="evaluator",
    ) as span:
        yield span


def start_agent_span(
    *,
    job_id: str,
    workflow_type: str,
    task_name: str,
    agent_role: str | None = None,
    backend: str | None = None,
    config_context: dict[str, Any] | None = None,
) -> Any:
    if not _observability_enabled(config_context):
        return NoOpSpan()
    span_name = f"agent.{task_name}"
    span_attributes = {
        "job_id": job_id,
        "workflow_type": workflow_type,
        "task_name": task_name,
        "agent_role": agent_role,
        "backend": backend,
    }
    otel_span = None
    if _otel_enabled(config_context):
        tracer = _get_tracer()
        if tracer is not None:
            otel_span = tracer.start_span(span_name, attributes=_safe_span_attributes(span_attributes))
    langfuse_context, langfuse_observation = _start_langfuse_observation(
        span_name,
        config_context=config_context,
        attributes=span_attributes,
        observation_type="agent",
    )
    if otel_span is None and langfuse_observation is None:
        return NoOpSpan()
    return ManagedObservationSpan(
        otel_span=otel_span,
        langfuse_context=langfuse_context,
        langfuse_observation=langfuse_observation,
        attributes=span_attributes,
    )


def end_span(
    span: Any,
    *,
    attributes: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    if span is None:
        return
    try:
        if attributes:
            for key, value in _safe_span_attributes(attributes).items():
                span.set_attribute(key, value)
        if error is not None:
            span.record_exception(error)
            _set_error_status(span)
        span.end()
    except Exception:
        logger.debug("Failed to end observability span", exc_info=True)


def add_span_event(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    config_context: dict[str, Any] | None = None,
) -> None:
    if not _observability_enabled(config_context):
        return
    span = _current_span()
    if span is None:
        return
    try:
        span.add_event(name, attributes=_safe_span_attributes(attributes or {}))
    except Exception:
        logger.debug("Failed to add observability event %s", name, exc_info=True)


def set_span_attributes(
    attributes: dict[str, Any],
    *,
    config_context: dict[str, Any] | None = None,
) -> None:
    if not _observability_enabled(config_context):
        return
    _update_current_langfuse_span(attributes, config_context)
    span = _current_span()
    if span is None:
        return
    for key, value in _safe_span_attributes(attributes).items():
        try:
            span.set_attribute(key, value)
        except Exception:
            logger.debug("Failed to set observability attribute %s", key, exc_info=True)


def record_usage_metrics(
    usage_summary: dict[str, Any],
    config_context: dict[str, Any] | None = None,
) -> None:
    duration_seconds = usage_summary.get("duration_seconds")
    attributes = {
        "prompt_tokens": usage_summary.get("prompt_tokens"),
        "completion_tokens": usage_summary.get("completion_tokens"),
        "total_tokens": usage_summary.get("total_tokens"),
        "cost_usd": usage_summary.get("cost_usd"),
        "duration_ms": round(float(duration_seconds) * 1000, 3)
        if duration_seconds is not None
        else None,
        "model_name": (config_context or {}).get("llm_model_name"),
        "llm_profile": (config_context or {}).get("llm_profile"),
        "llm_provider": (config_context or {}).get("llm_provider"),
    }
    set_span_attributes(attributes, config_context=config_context)


def record_workflow_result_observability(
    result: dict[str, Any],
    config_context: dict[str, Any] | None = None,
) -> None:
    """Attach high-signal workflow result metadata and Langfuse scores."""
    if not _observability_enabled(config_context) or not isinstance(result, dict):
        return
    summary = _workflow_result_summary(result)
    if summary:
        set_span_attributes(summary, config_context=config_context)
    _record_langfuse_scores(_workflow_result_scores(result), config_context)


def redact_observability_payload(value: Any, *, capture_raw: bool | None = None) -> Any:
    """Return a JSON-safe payload with secrets and PII-like values redacted."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            if SENSITIVE_KEY_RE.search(key):
                redacted[key] = "[REDACTED_SECRET]"
            elif PII_KEY_RE.search(key):
                redacted[key] = "[REDACTED_PII]"
            else:
                redacted[key] = redact_observability_payload(raw_item, capture_raw=capture_raw)
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [redact_observability_payload(item, capture_raw=capture_raw) for item in value]
    if isinstance(value, str):
        text = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
        text = PHONE_RE.sub("[REDACTED_PHONE]", text)
        if capture_raw is False and len(text) > MAX_ATTRIBUTE_LENGTH:
            return text[:MAX_ATTRIBUTE_LENGTH] + "...[TRUNCATED]"
        return text
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _workflow_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary_fields = {
        "detected_intent": result.get("detected_intent"),
        "routing_confidence": result.get("routing_confidence"),
        "qa_status": result.get("qa_status"),
        "escalation_needed": result.get("escalation_needed"),
        "requires_approval": result.get("requires_approval"),
        "channel_recommended_action": result.get("channel_recommended_action"),
        "data_sources": result.get("data_sources"),
        "compliance_flags": result.get("compliance_flags"),
    }
    customer_context = result.get("customer_context")
    if isinstance(customer_context, dict):
        summary_fields["customer_tier"] = customer_context.get("tier")
        summary_fields["customer_language"] = customer_context.get("language")
        summary_fields["customer_channel"] = customer_context.get("channel")
    guardrail_summary = _guardrail_decision_summary(result.get("guardrail_decision"))
    summary_fields.update(guardrail_summary)
    return {key: value for key, value in summary_fields.items() if value not in (None, "", [])}


def _workflow_result_scores(result: dict[str, Any]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    routing_confidence = _optional_float(result.get("routing_confidence"))
    if routing_confidence is not None:
        scores.append(
            {
                "name": "routing_confidence",
                "value": routing_confidence,
                "data_type": "NUMERIC",
                "comment": "Intent router confidence for this workflow result.",
            }
        )
    qa_status = result.get("qa_status")
    if qa_status:
        scores.append(
            {
                "name": "qa_status",
                "value": str(qa_status),
                "data_type": "CATEGORICAL",
                "comment": "Support QA decision for customer-facing response.",
            }
        )
    if "escalation_needed" in result:
        scores.append(
            {
                "name": "escalation_needed",
                "value": bool(result.get("escalation_needed")),
                "data_type": "BOOLEAN",
                "comment": "Whether the workflow result requires human handoff.",
            }
        )
    pre_sales = result.get("pre_sales_response")
    if isinstance(pre_sales, dict):
        confidence_level = _optional_float(pre_sales.get("confidence_level"))
        if confidence_level is not None:
            scores.append(
                {
                    "name": "pre_sales_confidence_level",
                    "value": confidence_level,
                    "data_type": "NUMERIC",
                    "comment": "Pre-sales specialist confidence level from structured output.",
                }
            )
    guardrail = _guardrail_decision_summary(result.get("guardrail_decision"))
    if guardrail:
        scores.extend(
            [
                {
                    "name": "guardrail_review_required",
                    "value": guardrail.get("guardrail_action") == "review_required",
                    "data_type": "BOOLEAN",
                    "comment": "Whether workflow guardrails require human review.",
                },
                {
                    "name": "guardrail_blocked",
                    "value": guardrail.get("guardrail_action") == "block",
                    "data_type": "BOOLEAN",
                    "comment": "Whether workflow guardrails blocked this workflow result.",
                },
                {
                    "name": "guardrail_finding_count",
                    "value": int(guardrail.get("guardrail_finding_count") or 0),
                    "data_type": "NUMERIC",
                    "comment": "Number of masked workflow guardrail findings.",
                },
            ]
        )
    return scores


def _guardrail_decision_summary(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    findings = decision.get("findings")
    return {
        "guardrail_action": decision.get("action"),
        "guardrail_severity": decision.get("severity"),
        "guardrail_stage": decision.get("stage"),
        "guardrail_finding_count": len(findings) if isinstance(findings, list) else decision.get("finding_count"),
    }


def _record_langfuse_scores(scores: list[dict[str, Any]], config_context: dict[str, Any] | None) -> None:
    if not scores or not _langfuse_enabled(config_context):
        return
    client = _langfuse_client()
    if client is None or not hasattr(client, "score_current_trace"):
        return
    for score in scores:
        try:
            client.score_current_trace(
                name=str(score["name"]),
                value=score["value"],
                data_type=score.get("data_type"),
                comment=score.get("comment"),
                metadata=_json_safe(
                    redact_observability_payload(
                        {
                            "source": "cross_border_ai.workflow_result",
                            "workflow_type": (config_context or {}).get("workflow_type"),
                        },
                        capture_raw=False,
                    )
                ),
            )
        except Exception:
            logger.debug("Failed to record Langfuse score %s", score.get("name"), exc_info=True)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _update_current_langfuse_span(
    attributes: dict[str, Any],
    config_context: dict[str, Any] | None,
) -> None:
    if not attributes or not _langfuse_enabled(config_context):
        return
    client = _langfuse_client()
    if client is None or not hasattr(client, "update_current_span"):
        return
    try:
        client.update_current_span(
            metadata=_json_safe(redact_observability_payload(attributes, capture_raw=False))
        )
    except Exception:
        logger.debug("Failed to update current Langfuse span", exc_info=True)


@contextmanager
def _observation_span(
    name: str,
    *,
    config_context: dict[str, Any] | None,
    attributes: dict[str, Any] | None = None,
    observation_type: str = "span",
) -> Iterator[Any]:
    if not _observability_enabled(config_context):
        yield NoOpSpan()
        return

    tracer = _get_tracer() if _otel_enabled(config_context) else None
    span_cm = (
        tracer.start_as_current_span(name, attributes=_safe_span_attributes(attributes or {}))
        if tracer is not None
        else _noop_span_context()
    )
    start_time = time.perf_counter()
    with ExitStack() as stack:
        span = stack.enter_context(span_cm)
        stack.enter_context(_langfuse_observation(name, config_context, attributes or {}, observation_type))
        try:
            yield span
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            span.set_attribute("duration_ms", duration_ms)
            set_span_attributes({"duration_ms": duration_ms}, config_context=config_context)
        except Exception as exc:
            if exc.__class__.__name__ != "Retry":
                try:
                    span.record_exception(exc)
                    _set_error_status(span)
                finally:
                    raise
            raise


@contextmanager
def _noop_span_context() -> Iterator[NoOpSpan]:
    yield NoOpSpan()


@contextmanager
def _langfuse_observation(
    name: str,
    config_context: dict[str, Any] | None,
    attributes: dict[str, Any],
    observation_type: str = "span",
) -> Iterator[None]:
    if not _langfuse_enabled(config_context):
        yield
        return
    client = _langfuse_client()
    if client is None or not hasattr(client, "start_as_current_observation"):
        yield
        return
    try:
        observation = client.start_as_current_observation(
            name=name,
            as_type=observation_type,
            metadata=_json_safe(redact_observability_payload(attributes, capture_raw=False)),
        )
    except Exception:
        logger.debug("Langfuse observation failed for %s", name, exc_info=True)
        yield
        return
    with observation:
        yield


def _start_langfuse_observation(
    name: str,
    *,
    config_context: dict[str, Any] | None,
    attributes: dict[str, Any],
    observation_type: str = "span",
) -> tuple[Any | None, Any | None]:
    if not _langfuse_enabled(config_context):
        return None, None
    client = _langfuse_client()
    if client is None or not hasattr(client, "start_as_current_observation"):
        return None, None
    try:
        context = client.start_as_current_observation(
            name=name,
            as_type=observation_type,
            metadata=_json_safe(redact_observability_payload(attributes, capture_raw=False)),
            end_on_exit=False,
        )
        observation = context.__enter__()
        return context, observation
    except Exception:
        logger.debug("Langfuse observation failed for %s", name, exc_info=True)
        return None, None


def _init_otel(service_name: str, config_context: dict[str, Any] | None) -> None:
    global _INSTRUMENTED_GLOBALS, _OTEL_CONFIGURED

    if not _otel_enabled(config_context):
        return
    if _OTEL_CONFIGURED:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.info("OpenTelemetry packages are not installed; observability spans are disabled.")
        return

    endpoint = _string_config(config_context, "otel_exporter_otlp_traces_endpoint")
    protocol = _string_config(config_context, "otel_exporter_otlp_protocol", "http/protobuf")
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "cross-border-ai",
            "deployment.environment": _string_config(config_context, "observability_environment", "local"),
        }
    )
    provider = TracerProvider(resource=resource)

    if endpoint:
        exporter = _build_otlp_exporter(endpoint, protocol)
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))

    try:
        trace.set_tracer_provider(provider)
        _OTEL_CONFIGURED = True
    except Exception:
        logger.debug("OpenTelemetry tracer provider was already configured", exc_info=True)
        _OTEL_CONFIGURED = True

    if not _INSTRUMENTED_GLOBALS:
        _instrument_global_libraries()
        _INSTRUMENTED_GLOBALS = True


def _build_otlp_exporter(endpoint: str, protocol: str) -> Any | None:
    try:
        if protocol.lower() in {"grpc", "http/grpc"}:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcOTLPSpanExporter,
            )

            return GrpcOTLPSpanExporter(endpoint=endpoint, insecure=True)

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpOTLPSpanExporter,
        )

        return HttpOTLPSpanExporter(endpoint=endpoint)
    except ImportError:
        logger.info("OpenTelemetry OTLP exporter package is not installed.")
    except Exception as exc:
        logger.warning("Failed to configure OTLP exporter: %s", exc)
    return None


def _instrument_fastapi(app: FastAPI) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="/health,/ready")
    except ImportError:
        logger.info("FastAPI OpenTelemetry instrumentation is not installed.")
    except Exception:
        logger.debug("FastAPI instrumentation failed", exc_info=True)


def _instrument_global_libraries() -> None:
    instrumentors = [
        ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("opentelemetry.instrumentation.redis", "RedisInstrumentor"),
        ("opentelemetry.instrumentation.celery", "CeleryInstrumentor"),
    ]
    for module_name, class_name in instrumentors:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)().instrument()
        except ImportError:
            logger.debug("%s is not installed", module_name)
        except Exception:
            logger.debug("%s instrumentation failed", class_name, exc_info=True)

    try:
        from database import engine
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(engine=engine)
    except ImportError:
        logger.debug("SQLAlchemy OpenTelemetry instrumentation is not installed")
    except Exception:
        logger.debug("SQLAlchemy instrumentation failed", exc_info=True)


def _init_openinference() -> None:
    instrumentors = [
        ("openinference.instrumentation.crewai", "CrewAIInstrumentor", {"skip_dep_check": True}),
        ("openinference.instrumentation.litellm", "LiteLLMInstrumentor", {}),
    ]
    for module_name, class_name, kwargs in instrumentors:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)().instrument(**kwargs)
        except ImportError:
            logger.debug("%s is not installed", module_name)
        except Exception:
            logger.debug("%s instrumentation failed", class_name, exc_info=True)


def _init_langfuse(config_context: dict[str, Any] | None) -> None:
    if not _langfuse_enabled(config_context):
        return
    _langfuse_client()


def _apply_observability_environment(config_context: dict[str, Any] | None) -> None:
    if not config_context:
        return
    env_map = {
        "LANGFUSE_BASE_URL": config_context.get("langfuse_base_url"),
        "MLFLOW_TRACKING_URI": config_context.get("mlflow_tracking_uri"),
        "MLFLOW_EXPERIMENT_NAME": config_context.get("mlflow_experiment_name"),
        "PHOENIX_PROJECT_NAME": config_context.get("phoenix_project_name"),
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": config_context.get("otel_exporter_otlp_traces_endpoint"),
        "OTEL_EXPORTER_OTLP_PROTOCOL": config_context.get("otel_exporter_otlp_protocol"),
    }
    for env_name, value in env_map.items():
        if value not in (None, ""):
            os.environ[env_name] = str(value)


def _langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT

    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT
    if os.getenv("LANGFUSE_HOST") and not os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_HOST", "")
    try:
        from langfuse import get_client

        _LANGFUSE_CLIENT = get_client()
        return _LANGFUSE_CLIENT
    except ImportError:
        logger.info("Langfuse package is not installed; Langfuse observations are disabled.")
    except Exception as exc:
        logger.warning("Langfuse client initialization failed: %s", exc)
    return None


def _get_tracer() -> Any | None:
    try:
        from opentelemetry import trace

        return trace.get_tracer("cross_border_ai.observability")
    except ImportError:
        return None


def _current_span() -> Any | None:
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        return None if isinstance(span, NoOpSpan) else span
    except ImportError:
        return None


def _set_error_status(span: Any) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR))
    except Exception:
        return


def _safe_span_attributes(attributes: dict[str, Any]) -> dict[str, str | int | float | bool]:
    safe_payload = redact_observability_payload(attributes, capture_raw=False)
    flattened: dict[str, str | int | float | bool] = {}
    if not isinstance(safe_payload, dict):
        return flattened

    for key, value in safe_payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flattened[str(key)] = _truncate_attribute(value)
        else:
            flattened[str(key)] = _truncate_attribute(json.dumps(value, sort_keys=True, default=str))
    return flattened


def _safe_stage_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "agent_role",
        "asset_count",
        "backend",
        "duration_ms",
        "job_id",
        "language",
        "score_count",
        "stage",
        "status",
        "target_market",
        "task_name",
        "workflow_type",
    }
    return {
        key: value
        for key, value in attributes.items()
        if key in allowed_keys and value not in (None, "", [])
    }


def _truncate_attribute(value: str | int | float | bool) -> str | int | float | bool:
    if isinstance(value, str) and len(value) > MAX_ATTRIBUTE_LENGTH:
        return value[:MAX_ATTRIBUTE_LENGTH] + "...[TRUNCATED]"
    return value


def _observability_enabled(config_context: dict[str, Any] | None = None) -> bool:
    return _bool_config(config_context, "observability_enabled", False)


def _otel_enabled(config_context: dict[str, Any] | None = None) -> bool:
    return _bool_config(config_context, "otel_enabled", _observability_enabled(config_context))


def _langfuse_enabled(config_context: dict[str, Any] | None = None) -> bool:
    if not _observability_enabled(config_context):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _bool_config(
    config_context: dict[str, Any] | None,
    key: str,
    default: bool = False,
) -> bool:
    if config_context and key in config_context:
        value = config_context.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
    env_name = key.upper()
    value = os.getenv(env_name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _string_config(
    config_context: dict[str, Any] | None,
    key: str,
    default: str | None = None,
) -> str | None:
    if config_context and config_context.get(key) not in (None, ""):
        return str(config_context[key])
    value = os.getenv(key.upper())
    return value if value not in (None, "") else default


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
