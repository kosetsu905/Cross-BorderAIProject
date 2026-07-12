import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import utils.observability as observability
from utils.observability import (
    NoOpSpan,
    end_span,
    guardrail_span,
    record_workflow_result_observability,
    redact_observability_payload,
    start_agent_span,
    stage_span,
    workflow_span,
)


class FakeLangfuseObservation:
    def __init__(self, name: str, as_type: str, metadata: dict[str, object] | None) -> None:
        self.name = name
        self.as_type = as_type
        self.metadata = metadata or {}
        self.metadata_updates: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self.ended = False

    def update(self, **kwargs: object) -> "FakeLangfuseObservation":
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            self.metadata_updates.append(metadata)
        return self

    def end(self) -> "FakeLangfuseObservation":
        self.ended = True
        return self

    def create_event(self, name: str, metadata: dict[str, object] | None = None) -> None:
        self.events.append({"name": name, "metadata": metadata or {}})


class FakeLangfuseContext:
    def __init__(self, observation: FakeLangfuseObservation) -> None:
        self.observation = observation
        self.exited = False

    def __enter__(self) -> FakeLangfuseObservation:
        return self.observation

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exited = True


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[FakeLangfuseObservation] = []
        self.current_span_updates: list[dict[str, object]] = []
        self.scores: list[dict[str, object]] = []

    def start_as_current_observation(self, **kwargs: object) -> FakeLangfuseContext:
        observation = FakeLangfuseObservation(
            name=str(kwargs["name"]),
            as_type=str(kwargs.get("as_type") or "span"),
            metadata=kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None,
        )
        self.observations.append(observation)
        return FakeLangfuseContext(observation)

    def update_current_span(self, **kwargs: object) -> None:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            self.current_span_updates.append(metadata)

    def score_current_trace(self, **kwargs: object) -> None:
        self.scores.append(dict(kwargs))


class FakeMlflowSpan:
    def __init__(self, name: str, span_type: str | None, attributes: dict[str, object] | None) -> None:
        self.name = name
        self.span_type = span_type
        self.attributes = dict(attributes or {})
        self.events: list[dict[str, object]] = []
        self.exceptions: list[BaseException] = []
        self.ended = False

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, object]) -> None:
        self.attributes.update(attributes)

    def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None:
        self.events.append({"name": name, "attributes": attributes or {}})

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)

    def end(self) -> None:
        self.ended = True


class FakeMlflowContext:
    def __init__(self, module: "FakeMlflowModule", span: FakeMlflowSpan) -> None:
        self.module = module
        self.span = span
        self.exited = False
        self.exit_exception_type: object | None = None

    def __enter__(self) -> FakeMlflowSpan:
        self.module.active_spans.append(self.span)
        return self.span

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exited = True
        self.exit_exception_type = exc_type
        if self.module.active_spans and self.module.active_spans[-1] is self.span:
            self.module.active_spans.pop()
        self.span.end()


class FakeMlflowModule:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.contexts: list[FakeMlflowContext] = []
        self.active_spans: list[FakeMlflowSpan] = []
        self.flushed = False

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(
        self,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
        trace_location: object | None = None,
    ) -> object:
        self.experiment_name = experiment_name
        return SimpleNamespace(name=experiment_name, experiment_id=experiment_id, trace_location=trace_location)

    def start_span(
        self,
        name: str = "span",
        span_type: str | None = "UNKNOWN",
        attributes: dict[str, object] | None = None,
        **_: object,
    ) -> FakeMlflowContext:
        context = FakeMlflowContext(self, FakeMlflowSpan(name, span_type, attributes))
        self.contexts.append(context)
        return context

    def get_current_active_span(self) -> FakeMlflowSpan | None:
        return self.active_spans[-1] if self.active_spans else None

    def flush_trace_async_logging(self, terminate: bool = False) -> None:
        self.flushed = not terminate


class ObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        observability._INITIALIZED_SERVICES.clear()
        observability._INSTRUMENTED_FASTAPI_APP_IDS.clear()
        observability._OTEL_CONFIGURED = False
        observability._OTEL_PROVIDER = None
        observability._INSTRUMENTED_GLOBALS = False
        observability._OPENINFERENCE_CONFIGURED = False
        observability._LANGFUSE_CLIENT = None
        observability._MLFLOW_CONFIGURED = False
        observability._MLFLOW_INIT_FAILED = False
        for env_name in (
            "MLFLOW_EXPERIMENT_NAME",
            "MLFLOW_HTTP_REQUEST_TIMEOUT",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACING_ENABLED",
        ):
            os.environ.pop(env_name, None)

    def test_workflow_span_is_noop_when_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with workflow_span("analytics", job_id="job-1", backend="local") as span:
                self.assertIsInstance(span, NoOpSpan)

    def test_redacts_secrets_and_pii_like_fields(self) -> None:
        payload = {
            "api_key": "sk-secret",
            "customer_email": "buyer@example.com",
            "phone_number": "+1 555 123 4567",
            "nested": {
                "authorization": "Bearer token",
                "message": "Contact buyer@example.com at +1 555 123 4567",
            },
        }

        redacted = redact_observability_payload(payload, capture_raw=False)

        self.assertEqual(redacted["api_key"], "[REDACTED_SECRET]")
        self.assertEqual(redacted["customer_email"], "[REDACTED_PII]")
        self.assertEqual(redacted["phone_number"], "[REDACTED_PII]")
        self.assertEqual(redacted["nested"]["authorization"], "[REDACTED_SECRET]")
        self.assertIn("[REDACTED_EMAIL]", redacted["nested"]["message"])
        self.assertIn("[REDACTED_PHONE]", redacted["nested"]["message"])

    @patch("utils.observability._instrument_fastapi")
    @patch("utils.observability._init_langfuse")
    @patch("utils.observability._init_openinference")
    @patch("utils.observability._init_otel")
    def test_otel_disabled_skips_otel_instrumentation_but_initializes_langfuse(
        self,
        init_otel,
        init_openinference,
        init_langfuse,
        instrument_fastapi,
    ) -> None:
        observability.init_observability(
            "cross-border-fastapi",
            app=object(),
            config_context={"observability_enabled": True, "otel_enabled": False},
        )

        init_otel.assert_not_called()
        init_openinference.assert_not_called()
        instrument_fastapi.assert_not_called()
        init_langfuse.assert_called_once()

    def test_langfuse_enabled_does_not_require_otel_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_SECRET_KEY": "secret",
            },
            clear=True,
        ):
            self.assertTrue(
                observability._langfuse_enabled(
                    {"observability_enabled": True, "otel_enabled": False}
                )
            )

    @patch("utils.observability._instrument_fastapi")
    @patch("utils.observability._init_langfuse")
    @patch("utils.observability._init_openinference")
    @patch("utils.observability._init_otel")
    def test_otel_enabled_does_not_auto_instrument_fastapi_routes_by_default(
        self,
        init_otel,
        init_openinference,
        init_langfuse,
        instrument_fastapi,
    ) -> None:
        observability.init_observability(
            "cross-border-fastapi",
            app=object(),
            config_context={"observability_enabled": True, "otel_enabled": True},
        )

        init_otel.assert_called_once()
        init_openinference.assert_called_once()
        init_langfuse.assert_called_once()
        instrument_fastapi.assert_not_called()

    @patch("utils.observability._instrument_fastapi")
    @patch("utils.observability._init_langfuse")
    @patch("utils.observability._init_openinference")
    @patch("utils.observability._init_otel")
    def test_fastapi_route_auto_instrumentation_requires_explicit_opt_in(
        self,
        init_otel,
        init_openinference,
        init_langfuse,
        instrument_fastapi,
    ) -> None:
        app = object()
        observability.init_observability(
            "cross-border-fastapi",
            app=app,
            config_context={
                "observability_enabled": True,
                "otel_enabled": True,
                "fastapi_otel_auto_instrumentation_enabled": True,
            },
        )

        init_otel.assert_called_once()
        init_openinference.assert_called_once()
        init_langfuse.assert_called_once()
        instrument_fastapi.assert_called_once_with(app)

    @patch("builtins.__import__")
    def test_global_auto_instrumentation_is_disabled_by_default(self, import_module) -> None:
        observability._instrument_global_libraries(
            {"observability_enabled": True, "otel_enabled": True}
        )

        import_module.assert_not_called()

    @patch("builtins.__import__")
    def test_low_level_instrumentation_requires_per_library_opt_in(self, import_module) -> None:
        instrumented: list[str] = []

        def fake_import(module_name: str, *args: object, **kwargs: object) -> object:
            class FakeInstrumentor:
                def instrument(self) -> None:
                    instrumented.append(module_name)

            return SimpleNamespace(HTTPXClientInstrumentor=FakeInstrumentor)

        import_module.side_effect = fake_import

        observability._instrument_global_libraries(
            {
                "observability_enabled": True,
                "otel_enabled": True,
                "otel_httpx_instrumentation_enabled": True,
            }
        )

        self.assertEqual(instrumented, ["opentelemetry.instrumentation.httpx"])

    @patch("utils.observability.find_spec", return_value=object())
    @patch("builtins.__import__")
    def test_openinference_defaults_to_litellm_only(self, import_module, find_spec) -> None:
        instrumented: list[str] = []

        def fake_import(module_name: str, *args: object, **kwargs: object) -> object:
            class FakeInstrumentor:
                def instrument(self, **_: object) -> None:
                    instrumented.append(module_name)

            return SimpleNamespace(
                CrewAIInstrumentor=FakeInstrumentor,
                LiteLLMInstrumentor=FakeInstrumentor,
            )

        import_module.side_effect = fake_import

        observability._init_openinference({"observability_enabled": True, "otel_enabled": True})

        self.assertEqual(instrumented, ["openinference.instrumentation.litellm"])
        self.assertEqual(find_spec.call_args_list[-1].args[0], "litellm")

    def test_langfuse_client_reuses_project_otel_provider_when_enabled(self) -> None:
        provider = object()
        observability._OTEL_PROVIDER = provider
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "public",
                "LANGFUSE_SECRET_KEY": "secret",
            },
            clear=True,
        ), patch("langfuse.Langfuse") as langfuse:
            client = observability._langfuse_client(
                {"observability_enabled": True, "otel_enabled": True}
            )

        self.assertEqual(client, langfuse.return_value)
        langfuse.assert_called_once_with(tracer_provider=provider)

    def test_flush_observability_flushes_otel_provider(self) -> None:
        class FakeProvider:
            def __init__(self) -> None:
                self.flushed = False

            def force_flush(self) -> None:
                self.flushed = True

        provider = FakeProvider()
        observability._OTEL_PROVIDER = provider

        observability.flush_observability()

        self.assertTrue(provider.flushed)

    def test_init_observability_initializes_mlflow_when_enabled(self) -> None:
        fake_mlflow = FakeMlflowModule()
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            observability.init_observability(
                "cross-border-fastapi",
                config_context={
                    "observability_enabled": True,
                    "otel_enabled": False,
                    "mlflow_tracing_enabled": True,
                    "mlflow_tracking_uri": "http://mlflow:5000",
                    "mlflow_experiment_name": "cross-border-ai",
                },
            )

        self.assertTrue(observability._MLFLOW_CONFIGURED)
        self.assertEqual(fake_mlflow.tracking_uri, "http://mlflow:5000")
        self.assertEqual(fake_mlflow.experiment_name, "cross-border-ai")

    def test_workflow_span_creates_mlflow_span_with_safe_attributes(self) -> None:
        fake_mlflow = FakeMlflowModule()
        config_context = {
            "observability_enabled": True,
            "otel_enabled": False,
            "mlflow_tracing_enabled": True,
        }

        with patch("utils.observability._mlflow_module", return_value=fake_mlflow):
            with workflow_span(
                "support",
                job_id="job-1",
                backend="celery",
                config_context=config_context,
                attributes={
                    "customer_email": "buyer@example.com",
                    "api_key": "sk-secret",
                },
            ):
                observability.set_span_attributes({"cost_usd": 0.12}, config_context=config_context)

        span = fake_mlflow.contexts[0].span
        self.assertEqual(span.name, "workflow.support")
        self.assertEqual(span.span_type, "CHAIN")
        self.assertEqual(span.attributes["job_id"], "job-1")
        self.assertEqual(span.attributes["customer_email"], "[REDACTED_PII]")
        self.assertEqual(span.attributes["api_key"], "[REDACTED_SECRET]")
        self.assertEqual(span.attributes["cost_usd"], 0.12)
        self.assertIn("duration_ms", span.attributes)
        self.assertTrue(span.ended)

    @patch("utils.observability._get_tracer")
    def test_start_agent_span_creates_mlflow_agent_when_enabled(self, get_tracer) -> None:
        fake_mlflow = FakeMlflowModule()
        with patch("utils.observability._mlflow_module", return_value=fake_mlflow):
            span = start_agent_span(
                job_id="job-1",
                workflow_type="support",
                task_name="pre_sales_consultation",
                agent_role="E-commerce Pre-Sales Product Consultant",
                backend="celery",
                config_context={
                    "observability_enabled": True,
                    "otel_enabled": False,
                    "mlflow_tracing_enabled": True,
                },
            )
            span.add_event("agent_completed", {"customer_email": "buyer@example.com"})
            end_span(span, attributes={"duration_ms": 123})

        get_tracer.assert_not_called()
        mlflow_span = fake_mlflow.contexts[0].span
        self.assertEqual(mlflow_span.name, "agent.pre_sales_consultation")
        self.assertEqual(mlflow_span.span_type, "AGENT")
        self.assertEqual(mlflow_span.attributes["duration_ms"], 123)
        self.assertEqual(mlflow_span.events[0]["attributes"]["customer_email"], "[REDACTED_PII]")
        self.assertTrue(mlflow_span.ended)

    def test_guardrail_span_writes_redacted_mlflow_metadata(self) -> None:
        fake_mlflow = FakeMlflowModule()
        config_context = {
            "observability_enabled": True,
            "otel_enabled": False,
            "mlflow_tracing_enabled": True,
        }

        with patch("utils.observability._mlflow_module", return_value=fake_mlflow):
            with guardrail_span(
                "guardrail_input_evaluated",
                workflow_type="support",
                stage="input",
                job_id="job-1",
                conversation_id="buyer@example.com",
                config_context=config_context,
                attributes={
                    "guardrail_action": "review_required",
                    "guardrail_severity": "high",
                },
            ):
                pass

        serialized_attributes = str(fake_mlflow.contexts[0].span.attributes)
        self.assertIn("guardrail_action", serialized_attributes)
        self.assertIn("[REDACTED_EMAIL]", serialized_attributes)
        self.assertNotIn("buyer@example.com", serialized_attributes)

    def test_flush_observability_flushes_mlflow_when_configured(self) -> None:
        fake_mlflow = FakeMlflowModule()
        observability._MLFLOW_CONFIGURED = True

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            observability.flush_observability()

        self.assertTrue(fake_mlflow.flushed)

    def test_mlflow_initialization_failure_does_not_raise(self) -> None:
        class FailingMlflow(FakeMlflowModule):
            def set_experiment(
                self,
                experiment_name: str | None = None,
                experiment_id: str | None = None,
                trace_location: object | None = None,
            ) -> object:
                raise RuntimeError("mlflow unavailable")

        with patch.dict("sys.modules", {"mlflow": FailingMlflow()}):
            observability.init_observability(
                "cross-border-fastapi",
                config_context={
                    "observability_enabled": True,
                    "otel_enabled": False,
                    "mlflow_tracing_enabled": True,
                    "mlflow_tracking_uri": "http://mlflow:5000",
                    "mlflow_experiment_name": "cross-border-ai",
                },
            )

        self.assertFalse(observability._MLFLOW_CONFIGURED)
        self.assertTrue(observability._MLFLOW_INIT_FAILED)

    @patch("utils.observability._get_tracer")
    def test_start_agent_span_creates_langfuse_agent_when_otel_disabled(self, get_tracer) -> None:
        client = FakeLangfuseClient()
        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "public", "LANGFUSE_SECRET_KEY": "secret"},
            clear=True,
        ), patch("utils.observability._langfuse_client", return_value=client):
            span = start_agent_span(
                job_id="job-1",
                workflow_type="support",
                task_name="pre_sales_consultation",
                agent_role="E-commerce Pre-Sales Product Consultant",
                backend="celery",
                config_context={"observability_enabled": True, "otel_enabled": False},
            )
            span.set_attribute("duration_ms", 123)
            end_span(span)

        get_tracer.assert_not_called()
        self.assertEqual(client.observations[0].name, "agent.pre_sales_consultation")
        self.assertEqual(client.observations[0].as_type, "agent")
        self.assertEqual(client.observations[0].metadata["task_name"], "pre_sales_consultation")
        self.assertEqual(
            client.observations[0].metadata["agent_role"],
            "E-commerce Pre-Sales Product Consultant",
        )
        self.assertTrue(client.observations[0].ended)

    @patch("utils.observability._get_tracer")
    def test_stage_span_creates_langfuse_span_with_safe_metadata_when_otel_disabled(
        self,
        get_tracer,
    ) -> None:
        client = FakeLangfuseClient()
        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "public", "LANGFUSE_SECRET_KEY": "secret"},
            clear=True,
        ), patch("utils.observability._langfuse_client", return_value=client):
            with stage_span(
                "content_generation",
                job_id="job-1",
                workflow_type="content",
                task_name="content_creation_and_qa:zh",
                agent_role="Multilingual Content Creator & Quality Editor",
                language="zh",
                target_market="China",
                status="running",
                backend="celery",
                config_context={"observability_enabled": True, "otel_enabled": False},
                attributes={
                    "asset_count": 2,
                    "raw_content": "full article body should not be stored",
                    "prompt": "do not store this prompt",
                },
            ):
                pass

        get_tracer.assert_not_called()
        self.assertEqual(client.observations[0].name, "stage.content_generation:zh")
        self.assertEqual(client.observations[0].as_type, "span")
        self.assertEqual(client.observations[0].metadata["stage"], "content_generation")
        self.assertEqual(client.observations[0].metadata["language"], "zh")
        self.assertEqual(client.observations[0].metadata["target_market"], "China")
        self.assertEqual(client.observations[0].metadata["asset_count"], 2)
        serialized_metadata = str(client.observations[0].metadata)
        self.assertNotIn("raw_content", serialized_metadata)
        self.assertNotIn("full article body", serialized_metadata)
        self.assertNotIn("prompt", serialized_metadata)
        self.assertTrue(client.current_span_updates)
        self.assertIn("duration_ms", client.current_span_updates[-1])

    def test_record_workflow_result_observability_updates_langfuse_scores(self) -> None:
        client = FakeLangfuseClient()
        with patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "public", "LANGFUSE_SECRET_KEY": "secret"},
            clear=True,
        ), patch("utils.observability._langfuse_client", return_value=client):
            record_workflow_result_observability(
                {
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.95,
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                    "data_sources": ["local_pdf_catalog"],
                },
                {"observability_enabled": True, "otel_enabled": False, "workflow_type": "support"},
            )

        self.assertEqual(client.current_span_updates[0]["conversation_id"], "conv-1")
        self.assertEqual(client.current_span_updates[0]["detected_intent"], "pre_sales")
        self.assertEqual(client.current_span_updates[0]["qa_status"], "APPROVED")
        score_names = {score["name"] for score in client.scores}
        self.assertIn("routing_confidence", score_names)
        self.assertIn("qa_status", score_names)
        self.assertIn("escalation_needed", score_names)


if __name__ == "__main__":
    unittest.main()
