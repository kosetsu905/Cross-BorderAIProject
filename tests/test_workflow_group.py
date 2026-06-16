import asyncio
import json
import unittest
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError

from crews import analytics_crew, sales_improvement_crew
from job_store import InMemoryJobStore
from models import JobStatus, WorkflowGroupRequest, WorkflowType
from orchestrator import MasterOrchestrator
from runtime_config import RuntimeConfig
from utils.usage_tracking import INTERNAL_USAGE_KEY
from utils.workflow_group import status_text
from utils.workflow_progress import PROGRESS_CONTEXT_KEY, WorkflowProgressRecorder, attach_task_progress


MARKETING_INPUTS = {
    "product_category": "Smart Cameras",
    "product_usp": "Fast setup",
    "target_markets": "US",
    "budget": "1000",
}
CONTENT_INPUTS = {
    "subject": "Smart Camera Launch",
    "product_category": "Smart Cameras",
    "target_markets": "US",
    "target_languages": ["en"],
    "platforms": ["Instagram"],
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


class WorkflowGroupRequestTests(unittest.TestCase):
    def test_valid_group_request_validates_child_inputs(self) -> None:
        request = WorkflowGroupRequest.model_validate(
            {
                "workflows": [
                    {
                        "workflow_type": "marketing",
                        "inputs": MARKETING_INPUTS,
                    },
                    {
                        "workflow_type": "content",
                        "inputs": CONTENT_INPUTS,
                    },
                ]
            }
        )

        self.assertEqual(len(request.workflows), 2)
        self.assertEqual(request.workflows[0].inputs["product_category"], "Smart Cameras")

    def test_group_request_rejects_invalid_child_inputs(self) -> None:
        with self.assertRaises(ValidationError):
            WorkflowGroupRequest.model_validate(
                {
                    "workflows": [
                        {
                            "workflow_type": "marketing",
                            "inputs": MARKETING_INPUTS,
                        },
                        {
                            "workflow_type": "content",
                            "inputs": {"product_category": "Smart Cameras"},
                        },
                    ]
                }
            )

    def test_group_request_rejects_duplicate_resolved_names(self) -> None:
        with self.assertRaises(ValidationError):
            WorkflowGroupRequest.model_validate(
                {
                    "workflows": [
                        {
                            "workflow_type": "marketing",
                            "inputs": MARKETING_INPUTS,
                        },
                        {
                            "workflow_type": "marketing",
                            "inputs": MARKETING_INPUTS,
                        },
                    ]
                }
            )


class WorkflowGroupOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_workflow_group_completes_parent_and_children(self) -> None:
        store = InMemoryJobStore()
        orchestrator = MasterOrchestrator(
            job_store=store,
            runtime_config=RuntimeConfig(workflow_result_cache_enabled=False),
        )
        orchestrator.register_crew(WorkflowType.MARKETING, lambda inputs, context: _fake_result("marketing"))
        orchestrator.register_crew(WorkflowType.CONTENT, lambda inputs, context: _fake_result("content", 7))
        request = WorkflowGroupRequest.model_validate(
            {
                "metadata": {"source": "test"},
                "workflows": [
                    {
                        "workflow_type": "marketing",
                        "inputs": MARKETING_INPUTS,
                        "provider_credentials": {"workflow_async_execution_enabled": False},
                    },
                    {
                        "workflow_type": "content",
                        "inputs": CONTENT_INPUTS,
                    },
                ],
            }
        )

        parent_job_id = await orchestrator.submit_workflow_group(request)
        parent_job = await _wait_for_terminal(orchestrator, parent_job_id)

        self.assertEqual(parent_job["status"], JobStatus.COMPLETED)
        self.assertEqual(parent_job["result"]["summary"]["completed"], 2)
        self.assertEqual(parent_job["result"]["summary"]["total_tokens"], 12)
        self.assertEqual(set(parent_job["result"]["results"]), {"marketing", "content"})
        serialized_parent_inputs = json.dumps(store.get_job(parent_job_id)["inputs"])
        self.assertNotIn("provider_credentials", serialized_parent_inputs)
        self.assertNotIn("workflow_async_execution_enabled", serialized_parent_inputs)

    async def test_local_workflow_group_failure_preserves_partial_results(self) -> None:
        store = InMemoryJobStore()
        orchestrator = MasterOrchestrator(
            job_store=store,
            runtime_config=RuntimeConfig(workflow_result_cache_enabled=False),
        )
        orchestrator.register_crew(WorkflowType.MARKETING, lambda inputs, context: _fake_result("marketing"))

        def failing_crew(inputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("content failed")

        orchestrator.register_crew(WorkflowType.CONTENT, failing_crew)
        request = WorkflowGroupRequest.model_validate(
            {
                "workflows": [
                    {
                        "workflow_type": "marketing",
                        "inputs": MARKETING_INPUTS,
                    },
                    {
                        "workflow_type": "content",
                        "inputs": CONTENT_INPUTS,
                    },
                ],
            }
        )

        parent_job_id = await orchestrator.submit_workflow_group(request)
        parent_job = await _wait_for_terminal(orchestrator, parent_job_id)

        self.assertEqual(parent_job["status"], JobStatus.FAILED)
        self.assertEqual(parent_job["result"]["summary"]["completed"], 1)
        self.assertEqual(parent_job["result"]["summary"]["failed"], 1)
        self.assertIn("marketing", parent_job["result"]["results"])
        self.assertIn("content", parent_job["error"])

    async def test_local_workflow_group_includes_cached_child_jobs(self) -> None:
        store = InMemoryJobStore()
        orchestrator = MasterOrchestrator(job_store=store)
        orchestrator.register_crew(WorkflowType.MARKETING, lambda inputs, context: _fake_result("marketing"))
        orchestrator.register_crew(WorkflowType.CONTENT, lambda inputs, context: _fake_result("content"))

        request = WorkflowGroupRequest.model_validate(
            {
                "workflows": [
                    {
                        "name": "cached_marketing",
                        "workflow_type": "marketing",
                        "inputs": MARKETING_INPUTS,
                    },
                    {
                        "workflow_type": "content",
                        "inputs": CONTENT_INPUTS,
                    },
                ],
            }
        )
        source_job_id = await orchestrator.submit_job(
            WorkflowType.MARKETING,
            request.workflows[0].inputs,
        )
        await _wait_for_terminal(orchestrator, source_job_id)

        parent_job_id = await orchestrator.submit_workflow_group(request)
        parent_job = await _wait_for_terminal(orchestrator, parent_job_id)
        cached_child = next(
            child for child in parent_job["result"]["children"] if child["name"] == "cached_marketing"
        )

        self.assertEqual(parent_job["status"], JobStatus.COMPLETED)
        self.assertTrue(cached_child["cache_hit"])
        self.assertEqual(cached_child["source_job_id"], source_job_id)


class AsyncProgressTests(unittest.TestCase):
    def test_async_sibling_progress_starts_together_and_joins_before_next_sync(self) -> None:
        class Agent:
            def __init__(self, role: str) -> None:
                self.role = role

        class Task:
            def __init__(self, role: str, async_execution: bool = False) -> None:
                self.agent = Agent(role)
                self.async_execution = async_execution
                self.callback = None

        tasks = [
            Task("Collector"),
            Task("Analyst", async_execution=True),
            Task("Researcher", async_execution=True),
            Task("Reporter"),
        ]
        store = InMemoryJobStore()
        store.create_job("job-1", WorkflowType.ANALYTICS, {})
        recorder = WorkflowProgressRecorder(
            job_id="job-1",
            workflow_type="analytics",
            job_store=store,
            backend="local",
        )

        attach_task_progress(
            {PROGRESS_CONTEXT_KEY: recorder},
            "analytics",
            tasks,
            ["collect", "analyze", "research", "report"],
        )
        tasks[0].callback(None)
        started_after_collect = [
            event["payload"]["task_name"]
            for event in store.get_job_events("job-1")
            if event["event_type"] == "task_started"
        ]
        tasks[1].callback(None)
        started_after_first_async = [
            event["payload"]["task_name"]
            for event in store.get_job_events("job-1")
            if event["event_type"] == "task_started"
        ]
        tasks[2].callback(None)
        started_after_second_async = [
            event["payload"]["task_name"]
            for event in store.get_job_events("job-1")
            if event["event_type"] == "task_started"
        ]

        self.assertEqual(started_after_collect, ["collect", "analyze", "research"])
        self.assertNotIn("report", started_after_first_async)
        self.assertEqual(started_after_second_async, ["collect", "analyze", "research", "report"])


class CrewAsyncConstructionTests(unittest.TestCase):
    def test_analytics_uses_async_sibling_tasks_when_enabled(self) -> None:
        captured_tasks = _run_analytics_with_fakes({"workflow_async_execution_enabled": True})

        self.assertFalse(captured_tasks[0].async_execution)
        self.assertTrue(captured_tasks[1].async_execution)
        self.assertTrue(captured_tasks[2].async_execution)
        self.assertFalse(captured_tasks[3].async_execution)

    def test_analytics_can_disable_async_sibling_tasks(self) -> None:
        captured_tasks = _run_analytics_with_fakes({"workflow_async_execution_enabled": False})

        self.assertFalse(captured_tasks[1].async_execution)
        self.assertFalse(captured_tasks[2].async_execution)

    def test_sales_improvement_uses_async_sibling_tasks_when_enabled(self) -> None:
        captured_tasks = _run_sales_with_fakes({"workflow_async_execution_enabled": True})

        self.assertFalse(captured_tasks[0].async_execution)
        self.assertTrue(captured_tasks[1].async_execution)
        self.assertTrue(captured_tasks[2].async_execution)
        self.assertFalse(captured_tasks[3].async_execution)

    def test_analytics_passes_memory_from_builder_to_crew(self) -> None:
        with patch.object(analytics_crew, "build_crew_memory", return_value="MEMORY"):
            _run_analytics_with_fakes({})

        self.assertEqual(FakeCrew.captured_kwargs["memory"], "MEMORY")

    def test_sales_passes_memory_from_builder_to_crew(self) -> None:
        with patch.object(sales_improvement_crew, "build_crew_memory", return_value="MEMORY"):
            _run_sales_with_fakes({})

        self.assertEqual(FakeCrew.captured_kwargs["memory"], "MEMORY")

    def test_analytics_routes_worker_and_reviewer_llms_by_agent_tier(self) -> None:
        captured_tasks = _run_analytics_with_fakes({})

        self.assertEqual(captured_tasks[0].agent.llm, "worker:Cross-Border E-commerce Data & Forecast Collector")
        self.assertEqual(captured_tasks[1].agent.llm, "worker:E-commerce Performance, Attribution & ChatBI Analyst")
        self.assertEqual(captured_tasks[2].agent.llm, "worker:Global Market, Competitor & Macro Risk Researcher")
        self.assertEqual(captured_tasks[3].agent.llm, "worker:Closed-Loop Automation Dry-Run Planner")
        self.assertEqual(captured_tasks[4].agent.llm, "reviewer:Executive Insights & Reporting Specialist")

    def test_sales_routes_worker_and_reviewer_llms_by_agent_tier(self) -> None:
        captured_tasks = _run_sales_with_fakes({})

        self.assertEqual(captured_tasks[0].agent.llm, "worker:Cross-Border Sales Funnel Analyst")
        self.assertEqual(captured_tasks[1].agent.llm, "worker:Conversion Rate Optimization Expert")
        self.assertEqual(captured_tasks[2].agent.llm, "worker:Dynamic Pricing & Margin Optimization Strategist")
        self.assertEqual(captured_tasks[3].agent.llm, "reviewer:Sales Playbook & Implementation Coach")


class FakeAgent:
    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        self.role = str(config.get("role") or "Fake Agent")
        self.llm = kwargs.get("llm")


class FakeTask:
    def __init__(
        self,
        config: dict[str, Any],
        agent: FakeAgent,
        context: list[Any] | None = None,
        output_pydantic: Any | None = None,
        async_execution: bool = False,
    ) -> None:
        self.config = config
        self.agent = agent
        self.context = context
        self.output_pydantic = output_pydantic
        self.async_execution = async_execution
        self.callback = None


class FakeCrew:
    captured_tasks: list[FakeTask] = []
    captured_kwargs: dict[str, Any] = {}

    def __init__(self, agents: list[FakeAgent], tasks: list[FakeTask], **kwargs: Any) -> None:
        self.agents = agents
        self.tasks = tasks
        self.kwargs = kwargs
        FakeCrew.captured_tasks = tasks
        FakeCrew.captured_kwargs = kwargs

    def kickoff(self, inputs: dict[str, Any]) -> object:
        return object()


class FakeModelTierRouter:
    def __init__(self, config_context: dict[str, Any]) -> None:
        self.config_context = config_context

    def llm_for_agent(self, agent_config: dict[str, Any]) -> str:
        return f"{agent_config.get('llm_tier')}:{agent_config.get('role')}"


def _run_analytics_with_fakes(config_context: dict[str, Any]) -> list[FakeTask]:
    with patch.object(analytics_crew, "Agent", FakeAgent):
        with patch.object(analytics_crew, "Task", FakeTask):
            with patch.object(analytics_crew, "Crew", FakeCrew):
                with patch.object(analytics_crew, "ModelTierRouter", FakeModelTierRouter):
                    with patch.object(analytics_crew, "attach_task_progress", return_value=None):
                        with patch.object(analytics_crew, "_serialize_crew_result", return_value={}):
                            with patch.object(analytics_crew, "_apply_provider_status", return_value={}):
                                analytics_crew.run_analytics_crew(
                                    {
                                        "product_category": "Smart Cameras",
                                        "target_markets": "US",
                                        "date_range": "Last 30 Days",
                                    },
                                    config_context,
                                )
    return list(FakeCrew.captured_tasks)


def _run_sales_with_fakes(config_context: dict[str, Any]) -> list[FakeTask]:
    with patch.object(sales_improvement_crew, "Agent", FakeAgent):
        with patch.object(sales_improvement_crew, "Task", FakeTask):
            with patch.object(sales_improvement_crew, "Crew", FakeCrew):
                with patch.object(sales_improvement_crew, "ModelTierRouter", FakeModelTierRouter):
                    with patch.object(sales_improvement_crew, "attach_task_progress", return_value=None):
                        with patch.object(sales_improvement_crew, "_serialize_crew_result", return_value={}):
                            with patch.object(sales_improvement_crew, "_apply_provider_status", return_value={}):
                                sales_improvement_crew.run_sales_improvement_crew(
                                    {
                                        "product_category": "Smart Cameras",
                                        "target_markets": "US",
                                        "current_avg_conversion": "1.5%",
                                        "target_conversion": "3%",
                                        "date_range": "Last 30 Days",
                                    },
                                    config_context,
                                )
    return list(FakeCrew.captured_tasks)


if __name__ == "__main__":
    unittest.main()
