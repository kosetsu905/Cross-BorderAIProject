import os
import unittest
from unittest.mock import patch

import utils.observability as observability
from utils.observability import (
    NoOpSpan,
    end_span,
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


class ObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        observability._INITIALIZED_SERVICES.clear()
        observability._INSTRUMENTED_FASTAPI_APP_IDS.clear()
        observability._OTEL_CONFIGURED = False
        observability._OTEL_PROVIDER = None
        observability._INSTRUMENTED_GLOBALS = False
        observability._OPENINFERENCE_CONFIGURED = False
        observability._LANGFUSE_CLIENT = None

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
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.95,
                    "qa_status": "APPROVED",
                    "escalation_needed": False,
                    "data_sources": ["local_pdf_catalog"],
                },
                {"observability_enabled": True, "otel_enabled": False, "workflow_type": "support"},
            )

        self.assertEqual(client.current_span_updates[0]["detected_intent"], "pre_sales")
        self.assertEqual(client.current_span_updates[0]["qa_status"], "APPROVED")
        score_names = {score["name"] for score in client.scores}
        self.assertIn("routing_confidence", score_names)
        self.assertIn("qa_status", score_names)
        self.assertIn("escalation_needed", score_names)


if __name__ == "__main__":
    unittest.main()
