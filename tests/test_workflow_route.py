import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.routes import create_router
from job_store import InMemoryJobStore
from models import JobStatus, WorkflowRouteRequest, WorkflowType
from orchestrator import CeleryOrchestrator, MasterOrchestrator
from runtime_config import RuntimeConfig
from services.workflow_router import WorkflowRouterAgent
from utils.usage_tracking import INTERNAL_USAGE_KEY
from utils.workflow_route import status_text


ROUTE_CONTEXT = {
    "product_category": "Smart Cameras",
    "product_usp": "Fast setup",
    "target_markets": "US, JP",
    "budget": "5000",
    "target_languages": ["en", "ja"],
    "preferred_launch_window": "2026-07-01 to 2026-07-31",
    "current_avg_conversion": "2.5%",
    "target_conversion": "4%",
}


def _fake_result(name: str, total_tokens: int = 5) -> dict[str, Any]:
    return {
        "workflow": name,
        INTERNAL_USAGE_KEY: {
            "prompt_tokens": 2,
            "completion_tokens": total_tokens - 2,
            "total_tokens": total_tokens,
        },
    }


async def _wait_for_terminal(orchestrator: MasterOrchestrator, job_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        job = orchestrator.get_job_status(job_id)
        if status_text(job.get("status")) in {JobStatus.COMPLETED.value, JobStatus.FAILED.value}:
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not reach a terminal status")


class WorkflowRouteModelTests(unittest.TestCase):
    def test_route_request_rejects_overlapping_filters(self) -> None:
        with self.assertRaises(ValidationError):
            WorkflowRouteRequest.model_validate(
                {
                    "goal": "Launch the product",
                    "context": ROUTE_CONTEXT,
                    "preferred_workflows": ["marketing"],
                    "excluded_workflows": ["marketing"],
                }
            )

    def test_provider_credentials_are_not_stored_in_parent_inputs(self) -> None:
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Launch the product",
                "context": {**ROUTE_CONTEXT, "customer_email": "buyer@example.com", "support_handle": "@buyer123"},
                "provider_credentials": {"serper_api_key": "secret"},
            }
        )
        plan = WorkflowRouterAgent({"workflow_router_enabled": True}).plan(request)
        from utils.workflow_route import workflow_route_parent_inputs

        parent_inputs = workflow_route_parent_inputs(request, plan)
        serialized = str(parent_inputs)

        self.assertNotIn("provider_credentials", serialized)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("buyer@example.com", serialized)
        self.assertNotIn("@buyer123", serialized)


class WorkflowRouterAgentTests(unittest.TestCase):
    def test_launch_goal_routes_to_marketing_content_and_scheduler(self) -> None:
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Launch Smart Cameras with a go-to-market campaign",
                "context": ROUTE_CONTEXT,
            }
        )

        plan = WorkflowRouterAgent({"workflow_router_enabled": True}).plan(request)

        self.assertEqual([node.workflow_type for node in plan.nodes], [
            WorkflowType.MARKETING,
            WorkflowType.CONTENT,
            WorkflowType.SCHEDULER,
        ])
        scheduler = next(node for node in plan.nodes if node.workflow_type == WorkflowType.SCHEDULER)
        self.assertEqual(scheduler.depends_on, ["marketing"])
        self.assertEqual(plan.missing_inputs, [])

    def test_performance_goal_routes_to_analytics_then_sales(self) -> None:
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Optimize conversion performance and sales funnel",
                "context": ROUTE_CONTEXT,
            }
        )

        plan = WorkflowRouterAgent({"workflow_router_enabled": True}).plan(request)

        self.assertEqual([node.workflow_type for node in plan.nodes], [
            WorkflowType.ANALYTICS,
            WorkflowType.SALES_IMPROVEMENT,
        ])
        sales = next(node for node in plan.nodes if node.workflow_type == WorkflowType.SALES_IMPROVEMENT)
        self.assertEqual(sales.depends_on, ["analytics"])

    def test_support_goal_routes_to_support(self) -> None:
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Customer asks where their order is",
                "context": {"customer": "Alex", "person": "Alex", "inquiry": "Where is my order?"},
            }
        )

        plan = WorkflowRouterAgent({"workflow_router_enabled": True}).plan(request)

        self.assertEqual([node.workflow_type for node in plan.nodes], [WorkflowType.SUPPORT])
        self.assertEqual(plan.nodes[0].inputs["inquiry"], "Where is my order?")

    def test_missing_required_seed_inputs_are_reported(self) -> None:
        request = WorkflowRouteRequest.model_validate(
            {"goal": "Launch a go-to-market campaign", "context": {}}
        )

        plan = WorkflowRouterAgent({"workflow_router_enabled": True}).plan(request)

        self.assertTrue(plan.requires_review)
        self.assertTrue(any("product_category" in item for item in plan.missing_inputs))
        self.assertTrue(any("target_markets" in item for item in plan.missing_inputs))

    def test_preferred_and_excluded_workflows_are_respected(self) -> None:
        with self.assertRaises(ValidationError):
            # Overlap is rejected at the model boundary.
            WorkflowRouteRequest.model_validate(
                {
                    "goal": "Launch and promote Smart Cameras",
                    "context": ROUTE_CONTEXT,
                    "preferred_workflows": ["marketing", "content"],
                    "excluded_workflows": ["content"],
                }
            )

        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Launch and promote Smart Cameras",
                "context": ROUTE_CONTEXT,
                "preferred_workflows": ["marketing", "content"],
                "excluded_workflows": ["scheduler"],
            }
        )
        plan = WorkflowRouterAgent({"workflow_router_enabled": True}).plan(request)
        self.assertEqual([node.workflow_type for node in plan.nodes], [WorkflowType.MARKETING, WorkflowType.CONTENT])

    @patch("services.workflow_router.httpx.post")
    def test_invalid_llm_json_falls_back_to_deterministic_plan(self, post_mock) -> None:
        post_mock.return_value.raise_for_status.return_value = None
        post_mock.return_value.json.return_value = {
            "choices": [{"message": {"content": "not-json"}}],
        }
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Ambiguous growth objective",
                "context": ROUTE_CONTEXT,
            }
        )

        plan = WorkflowRouterAgent(
            {
                "workflow_router_enabled": True,
                "workflow_router_llm_fallback_enabled": True,
                "workflow_router_confidence_threshold": 0.75,
                "llm_api_key": "test-key",
                "llm_model_name": "gpt-4o-mini",
            }
        ).plan(request)

        self.assertEqual([node.workflow_type for node in plan.nodes], [WorkflowType.MARKETING])
        self.assertTrue(plan.requires_review)


class WorkflowRouteOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_route_completes_parent_and_dependency_waves(self) -> None:
        store = InMemoryJobStore()
        orchestrator = MasterOrchestrator(
            job_store=store,
            runtime_config=RuntimeConfig(workflow_result_cache_enabled=False),
        )
        orchestrator.register_crew(WorkflowType.MARKETING, lambda inputs, context: _fake_result("marketing"))
        orchestrator.register_crew(WorkflowType.CONTENT, lambda inputs, context: _fake_result("content", 6))
        orchestrator.register_crew(WorkflowType.SCHEDULER, lambda inputs, context: _fake_result("scheduler", 4))
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Launch Smart Cameras with a go-to-market campaign",
                "context": ROUTE_CONTEXT,
                "provider_credentials": {"workflow_router_enabled": True},
            }
        )

        parent_job_id = await orchestrator.submit_workflow_route(request)
        parent_job = await _wait_for_terminal(orchestrator, parent_job_id)

        self.assertEqual(parent_job["status"], JobStatus.COMPLETED)
        self.assertEqual(parent_job["result"]["summary"]["completed"], 3)
        self.assertEqual(parent_job["result"]["summary"]["total_tokens"], 15)
        self.assertEqual(set(parent_job["result"]["results"]), {"marketing", "content", "scheduler"})
        scheduler_child = next(child for child in parent_job["result"]["children"] if child["name"] == "scheduler")
        self.assertEqual(scheduler_child["depends_on"], ["marketing"])

    async def test_local_route_child_failure_marks_parent_failed_with_partial_results(self) -> None:
        store = InMemoryJobStore()
        orchestrator = MasterOrchestrator(
            job_store=store,
            runtime_config=RuntimeConfig(workflow_result_cache_enabled=False),
        )
        orchestrator.register_crew(WorkflowType.ANALYTICS, lambda inputs, context: _fake_result("analytics"))

        def failing_sales(inputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("sales failed")

        orchestrator.register_crew(WorkflowType.SALES_IMPROVEMENT, failing_sales)
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Optimize conversion performance and sales funnel",
                "context": ROUTE_CONTEXT,
            }
        )

        parent_job_id = await orchestrator.submit_workflow_route(request)
        parent_job = await _wait_for_terminal(orchestrator, parent_job_id)

        self.assertEqual(parent_job["status"], JobStatus.FAILED)
        self.assertEqual(parent_job["result"]["summary"]["completed"], 1)
        self.assertEqual(parent_job["result"]["summary"]["failed"], 1)
        self.assertIn("analytics", parent_job["result"]["results"])
        self.assertIn("sales_improvement", parent_job["error"])

    async def test_local_route_rejects_missing_child_inputs_before_submission(self) -> None:
        orchestrator = MasterOrchestrator(job_store=InMemoryJobStore())
        orchestrator.register_crew(WorkflowType.MARKETING, lambda inputs, context: _fake_result("marketing"))
        request = WorkflowRouteRequest.model_validate(
            {"goal": "Launch a go-to-market campaign", "context": {}}
        )

        with self.assertRaises(ValueError):
            await orchestrator.submit_workflow_route(request)

    async def test_local_route_includes_cached_child_jobs(self) -> None:
        store = InMemoryJobStore()
        orchestrator = MasterOrchestrator(job_store=store)
        orchestrator.register_crew(WorkflowType.MARKETING, lambda inputs, context: _fake_result("marketing"))
        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Launch the product",
                "context": ROUTE_CONTEXT,
                "preferred_workflows": ["marketing"],
            }
        )
        plan = orchestrator.plan_workflow_route(request)
        source_job_id = await orchestrator.submit_job(WorkflowType.MARKETING, plan.nodes[0].inputs)
        await _wait_for_terminal(orchestrator, source_job_id)

        parent_job_id = await orchestrator.submit_workflow_route(request)
        parent_job = await _wait_for_terminal(orchestrator, parent_job_id)
        child = parent_job["result"]["children"][0]

        self.assertEqual(parent_job["status"], JobStatus.COMPLETED)
        self.assertTrue(child["cache_hit"])
        self.assertEqual(child["source_job_id"], source_job_id)


class WorkflowRouteApiTests(unittest.TestCase):
    def test_plan_and_submit_endpoints_delegate_to_orchestrator(self) -> None:
        class FakeOrchestrator:
            registered_workflows = [WorkflowType.MARKETING]

            def __init__(self) -> None:
                self.plan = WorkflowRouterAgent({"workflow_router_enabled": True}).plan(
                    WorkflowRouteRequest.model_validate(
                        {"goal": "Launch the product", "context": ROUTE_CONTEXT}
                    )
                )

            def plan_workflow_route(self, request: WorkflowRouteRequest):
                return self.plan

            async def submit_workflow_route(self, request: WorkflowRouteRequest) -> str:
                return "route-job-1"

            def get_job_status(self, job_id: str) -> dict[str, Any]:
                return {"job_id": job_id, "status": JobStatus.PENDING, "result": None}

            def get_job_events(self, job_id: str) -> list[dict[str, Any]]:
                return []

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(create_router(FakeOrchestrator()))
        client = TestClient(app)

        plan_response = client.post(
            "/api/v1/workflow-route/plan",
            json={"goal": "Launch the product", "context": ROUTE_CONTEXT},
        )
        submit_response = client.post(
            "/api/v1/workflow-route",
            json={"goal": "Launch the product", "context": ROUTE_CONTEXT},
        )

        self.assertEqual(plan_response.status_code, 200)
        self.assertEqual(submit_response.status_code, 200)
        self.assertEqual(submit_response.json()["job_id"], "route-job-1")


class WorkflowRouteCeleryTests(unittest.IsolatedAsyncioTestCase):
    async def test_celery_route_submission_enqueues_initial_children_and_monitor(self) -> None:
        store = InMemoryJobStore()
        orchestrator = CeleryOrchestrator(
            job_store=store,
            runtime_config=RuntimeConfig(workflow_result_cache_enabled=False),
        )
        sent_tasks: list[dict[str, Any]] = []

        class SentTask:
            def __init__(self, task_id: str) -> None:
                self.id = task_id

        def fake_send_task(name: str, args=None, kwargs=None, task_id=None):
            sent_tasks.append({"name": name, "args": args or [], "kwargs": kwargs or {}, "task_id": task_id})
            return SentTask(str(task_id))

        request = WorkflowRouteRequest.model_validate(
            {
                "goal": "Launch Smart Cameras with a go-to-market campaign",
                "context": ROUTE_CONTEXT,
            }
        )
        with patch("orchestrator.celery_app.send_task", side_effect=fake_send_task):
            parent_job_id = await orchestrator.submit_workflow_route(request)

        parent_job = store.get_job(parent_job_id)
        task_names = [task["name"] for task in sent_tasks]

        self.assertEqual(parent_job["status"], JobStatus.RUNNING)
        self.assertIn("workflow.marketing", task_names)
        self.assertIn("workflow.content", task_names)
        self.assertIn("workflow.route_monitor", task_names)
        self.assertNotIn("workflow.scheduler", task_names)
