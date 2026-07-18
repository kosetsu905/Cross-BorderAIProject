from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from services.mlflow_governance import (
    SUPPORT_AGENT_PROMPT_FIELDS,
    _parse_prompt_template,
    _serialize_prompt_template,
    build_official_support_scorers,
    load_support_prompts,
    log_support_review,
    register_official_support_scorers,
)


@pytest.fixture
def cache_dir() -> Path:
    path = Path("tmp") / f"mlflow-governance-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class FakePrompt:
    def __init__(self, template: str, version: int = 2) -> None:
        self.template = template
        self.version = version

    def to_single_brace_format(self) -> str:
        return self.template.replace("{{customer}}", "{customer}")


class FakeMlflow:
    def __init__(self, prompt: FakePrompt | None = None) -> None:
        self.genai = SimpleNamespace(load_prompt=Mock(return_value=prompt))
        self.trace_updates: list[dict[str, object]] = []
        self.feedback: list[dict[str, object]] = []
        self.expectations: list[dict[str, object]] = []
        self.active_span = SimpleNamespace(set_attribute=Mock())

    def update_current_trace(self, **kwargs: object) -> None:
        self.trace_updates.append(dict(kwargs))

    def get_current_active_span(self) -> object:
        return self.active_span

    def search_traces(self, **_: object) -> list[object]:
        return [SimpleNamespace(info=SimpleNamespace(trace_id="trace-1"))]

    def log_feedback(self, **kwargs: object) -> None:
        self.feedback.append(dict(kwargs))

    def log_expectation(self, **kwargs: object) -> None:
        self.expectations.append(dict(kwargs))


def test_prompt_template_round_trip_uses_mlflow_double_braces() -> None:
    serialized = _serialize_prompt_template(
        {
            "role": "Support agent for {customer}",
            "goal": "Answer accurately",
            "backstory": "Use verified context",
        }
    )

    assert "{{customer}}" in serialized
    parsed = _parse_prompt_template(
        serialized.replace("{{customer}}", "{customer}"),
        SUPPORT_AGENT_PROMPT_FIELDS,
    )
    assert parsed["role"] == "Support agent for {customer}"


def test_load_support_prompts_uses_official_registry_and_records_lineage(cache_dir: Path) -> None:
    prompt = FakePrompt(
        _serialize_prompt_template(
            {
                "role": "Governed role for {customer}",
                "goal": "Governed goal",
                "backstory": "Governed backstory",
            }
        )
    )
    fake_mlflow = FakeMlflow(prompt)
    agents = {
        "agent_one": {
            "llm_tier": "worker",
            "role": "Local role",
            "goal": "Local goal",
            "backstory": "Local backstory",
        }
    }

    with patch(
        "services.mlflow_governance.configure_mlflow",
        return_value=(fake_mlflow, "1"),
    ):
        loaded = load_support_prompts(
            agents,
            {},
            {
                "mlflow_prompt_registry_enabled": True,
                "mlflow_support_prompt_alias": "production",
                "mlflow_prompt_cache_dir": str(cache_dir),
            },
        )

    assert loaded.agents["agent_one"]["role"] == "Governed role for {customer}"
    assert loaded.agents["agent_one"]["llm_tier"] == "worker"
    lineage = loaded.lineage["support-agent-agent-one"]
    assert lineage.version == 2
    assert lineage.source == "mlflow"
    assert fake_mlflow.trace_updates


def test_load_support_prompts_uses_last_successful_cache_when_registry_fails(cache_dir: Path) -> None:
    prompt = FakePrompt(
        _serialize_prompt_template(
            {
                "role": "Cached governed role",
                "goal": "Cached goal",
                "backstory": "Cached backstory",
            }
        )
    )
    agents = {
        "agent_one": {
            "llm_tier": "worker",
            "role": "Local role",
            "goal": "Local goal",
            "backstory": "Local backstory",
        }
    }
    context = {
        "mlflow_prompt_registry_enabled": True,
        "mlflow_support_prompt_alias": "production",
        "mlflow_prompt_cache_dir": str(cache_dir),
    }

    with patch(
        "services.mlflow_governance.configure_mlflow",
        return_value=(FakeMlflow(prompt), "1"),
    ):
        load_support_prompts(agents, {}, context)
    with patch(
        "services.mlflow_governance.configure_mlflow",
        side_effect=ConnectionError("offline"),
    ):
        loaded = load_support_prompts(agents, {}, context)

    assert loaded.agents["agent_one"]["role"] == "Cached governed role"
    assert loaded.lineage["support-agent-agent-one"].source == "cache"


def test_load_support_prompts_fails_when_registry_and_cache_are_unavailable(
    cache_dir: Path,
) -> None:
    agents = {
        "agent_one": {
            "role": "Ungoverned local role",
            "goal": "Local goal",
            "backstory": "Local backstory",
        }
    }
    with patch(
        "services.mlflow_governance.configure_mlflow",
        side_effect=ConnectionError("offline"),
    ), pytest.raises(RuntimeError, match="no governed cache exists"):
        load_support_prompts(
            agents,
            {},
            {
                "mlflow_prompt_registry_enabled": True,
                "mlflow_support_prompt_alias": "production",
                "mlflow_prompt_cache_dir": str(cache_dir),
            },
        )


def test_official_support_scorers_are_mlflow_builtin_types() -> None:
    scorers = build_official_support_scorers("openai:/gpt-4o-mini")

    assert [scorer.name for scorer in scorers] == [
        "relevance_to_query",
        "completeness",
        "safety",
        "pii_detection",
        "conversational_role_adherence",
        "user_frustration",
        "support_guidelines",
    ]
    assert all(scorer.__class__.__module__.startswith("mlflow.genai.scorers") for scorer in scorers)


def test_automatic_evaluation_requires_official_gateway_model() -> None:
    with pytest.raises(ValueError, match="AI Gateway"):
        register_official_support_scorers(
            {
                "mlflow_genai_judge_default_model": "openai:/gpt-4o-mini",
                "mlflow_automatic_evaluation_enabled": True,
            },
            "1",
        )


def test_log_support_review_uses_official_feedback_and_expectation() -> None:
    fake_mlflow = FakeMlflow()
    with patch(
        "services.mlflow_governance.configure_mlflow",
        return_value=(fake_mlflow, "1"),
    ):
        trace_id = log_support_review(
            job_id="job-1",
            decision="override",
            reviewer="reviewer-1",
            rationale="Policy checked",
            approved_response="Send the revised reply to buyer@example.com",
            original_response="Original reply",
            config_context={"mlflow_tracing_enabled": True},
        )

    assert trace_id == "trace-1"
    assert fake_mlflow.feedback[0]["name"] == "support_review_decision"
    assert fake_mlflow.feedback[0]["value"] == "override"
    assert fake_mlflow.expectations[0]["name"] == "approved_support_response"
    assert "buyer@example.com" not in str(fake_mlflow.expectations[0]["value"])
