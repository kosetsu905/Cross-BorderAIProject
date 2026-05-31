import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import create_router
from job_store import InMemoryJobStore
from models import JobStatus, WorkflowType
from orchestrator import MasterOrchestrator
from runtime_config import LLMProfileConfig, RuntimeConfig


class FakeOrchestrator:
    registered_workflows = [WorkflowType.SUPPORT]

    async def submit_job(
        self,
        workflow_type: WorkflowType,
        inputs: dict,
        provider_credentials: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        self.workflow_type = workflow_type
        self.inputs = inputs
        self.provider_credentials = provider_credentials
        self.metadata = metadata
        return "job-123"

    def get_job_status(self, job_id: str) -> dict:
        return {"job_id": job_id, "status": "pending", "result": None}

    def get_job_events(self, job_id: str) -> list:
        return []


class LLMProfileSwitchingTests(unittest.TestCase):
    def test_workflow_endpoint_passes_llm_profile_provider_credentials(self) -> None:
        orchestrator = FakeOrchestrator()
        app = FastAPI()
        app.include_router(create_router(orchestrator))
        client = TestClient(app)

        response = client.post(
            "/api/v1/workflow",
            json={
                "workflow_type": "support",
                "inputs": {
                    "customer": "Maria",
                    "person": "Maria",
                    "inquiry": "I need help with a return.",
                },
                "provider_credentials": {"llm_profile": "openrouter_gpt4o_mini"},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(orchestrator.workflow_type, WorkflowType.SUPPORT)
        self.assertEqual(
            orchestrator.provider_credentials,
            {"llm_profile": "openrouter_gpt4o_mini"},
        )

    def test_workflow_endpoint_passes_support_serper_stage_flags(self) -> None:
        orchestrator = FakeOrchestrator()
        app = FastAPI()
        app.include_router(create_router(orchestrator))
        client = TestClient(app)

        response = client.post(
            "/api/v1/workflow",
            json={
                "workflow_type": "support",
                "inputs": {
                    "customer": "Maria",
                    "person": "Maria",
                    "inquiry": "Which camera works with HomeKit?",
                },
                "provider_credentials": {
                    "serper_api_key": "serper-key",
                    "support_serper_pre_sales_enabled": True,
                    "support_serper_order_fulfillment_enabled": False,
                    "support_serper_post_sales_enabled": True,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            orchestrator.provider_credentials,
            {
                "serper_api_key": "serper-key",
                "support_serper_pre_sales_enabled": True,
                "support_serper_order_fulfillment_enabled": False,
                "support_serper_post_sales_enabled": True,
            },
        )

    def test_workflow_endpoint_rejects_unknown_provider_credentials(self) -> None:
        orchestrator = FakeOrchestrator()
        app = FastAPI()
        app.include_router(create_router(orchestrator))
        client = TestClient(app)

        response = client.post(
            "/api/v1/workflow",
            json={
                "workflow_type": "support",
                "inputs": {
                    "customer": "Maria",
                    "person": "Maria",
                    "inquiry": "I need help with a return.",
                },
                "provider_credentials": {"support_serper_unknown_enabled": True},
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_service_inquiry_extracts_llm_profile_without_leaking_to_inputs(self) -> None:
        orchestrator = FakeOrchestrator()
        app = FastAPI()
        app.include_router(create_router(orchestrator))
        client = TestClient(app)

        response = client.post(
            "/api/v1/service/inquiry",
            json={
                "customer": "Maria",
                "inquiry": "The item arrived damaged. Can I return it?",
                "channel": "whatsapp",
                "llm_profile": "openrouter_gpt4o_mini",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(orchestrator.workflow_type, WorkflowType.SUPPORT)
        self.assertEqual(
            orchestrator.provider_credentials,
            {"llm_profile": "openrouter_gpt4o_mini"},
        )
        self.assertNotIn("llm_profile", orchestrator.inputs)

    def test_service_inquiry_extracts_nested_provider_credentials_without_leaking_to_inputs(self) -> None:
        orchestrator = FakeOrchestrator()
        app = FastAPI()
        app.include_router(create_router(orchestrator))
        client = TestClient(app)

        response = client.post(
            "/api/v1/service/inquiry",
            json={
                "customer": "Maria",
                "inquiry": "Which camera works with HomeKit?",
                "channel": "whatsapp",
                "llm_profile": "openrouter_gpt4o_mini",
                "provider_credentials": {
                    "serper_api_key": "serper-key",
                    "support_serper_pre_sales_enabled": True,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            orchestrator.provider_credentials,
            {
                "serper_api_key": "serper-key",
                "support_serper_pre_sales_enabled": True,
                "llm_profile": "openrouter_gpt4o_mini",
            },
        )
        self.assertNotIn("llm_profile", orchestrator.inputs)
        self.assertNotIn("provider_credentials", orchestrator.inputs)

    def test_master_orchestrator_applies_support_profile_and_request_override(self) -> None:
        async def run_case(provider_credentials: dict | None = None) -> dict:
            captured_context: dict = {}

            def fake_support_crew(inputs: dict, config_context: dict) -> dict:
                captured_context.update(config_context)
                return {"ok": True}

            config = RuntimeConfig(
                llm_profiles={
                    "support_openai": LLMProfileConfig(
                        llm_provider="openai",
                        llm_model_name="gpt-4o-mini",
                        llm_api_key_env="OPENAI_API_KEY",
                    ),
                    "request_openrouter": LLMProfileConfig(
                        llm_provider="openrouter",
                        llm_model_name="openai/gpt-4o-mini",
                        llm_api_key_env="OPENROUTER_API_KEY",
                    ),
                },
                support_llm_profile="support_openai",
                workflow_result_cache_enabled=False,
            )
            store = InMemoryJobStore()
            orchestrator = MasterOrchestrator(job_store=store, runtime_config=config)
            orchestrator.register_crew(WorkflowType.SUPPORT, fake_support_crew)
            job_id = await orchestrator.submit_job(
                WorkflowType.SUPPORT,
                {},
                provider_credentials=provider_credentials,
            )
            for _ in range(50):
                job = store.get_job(job_id)
                if job and job.get("status") == JobStatus.COMPLETED:
                    break
                await asyncio.sleep(0.01)
            return captured_context

        env = {
            "OPENAI_API_KEY": "openai-key",
            "OPENROUTER_API_KEY": "openrouter-key",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "services.support_auto_dispatch.process_completed_support_job",
            new_callable=AsyncMock,
        ) as dispatch_mock:
            dispatch_mock.return_value = {"status": "skipped"}
            default_context = asyncio.run(run_case())
            override_context = asyncio.run(
                run_case({"llm_profile": "request_openrouter"})
            )

        self.assertEqual(default_context["llm_profile"], "support_openai")
        self.assertEqual(default_context["llm_provider"], "openai")
        self.assertEqual(default_context["llm_api_key"], "openai-key")
        self.assertEqual(override_context["llm_profile"], "request_openrouter")
        self.assertEqual(override_context["llm_provider"], "openrouter")
        self.assertEqual(override_context["llm_base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(override_context["llm_api_key"], "openrouter-key")


if __name__ == "__main__":
    unittest.main()
