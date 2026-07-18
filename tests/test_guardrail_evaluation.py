from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from services.guardrail_evaluation import (
    GuardrailCaseResult,
    GuardrailEvaluationConfig,
    GuardrailJudgeMetrics,
    _CachedValidationResult,
    _FaultInjectingGuardrailService,
    _judge_payload,
    calculate_guardrail_metrics,
    evaluate_guardrail_case,
    evaluate_guardrail_gates,
    guardrail_coverage_report,
    guardrail_service_for_case,
    load_guardrail_cases,
    mlflow_dataset_records,
    score_guardrail_case,
)
from services.mlflow_guardrail_evaluation import (
    _select_judge_cases,
    bootstrap_guardrail_evaluation,
    build_guardrail_code_scorers,
    build_guardrail_judges,
    run_mlflow_guardrail_evaluation,
)
from services.workflow_guardrails import _PayloadEvaluation


def _ideal_result(case) -> GuardrailCaseResult:
    expectation = case.policy_expectation
    sanitized_payload = " ".join(case.assertions.sanitized_contains)
    return GuardrailCaseResult(
        case_id=case.case_id,
        workflow_type=case.workflow_type,
        stage=case.stage,
        action_type=case.action_type,
        triggered=expectation.should_trigger,
        policy_ids=sorted(expectation.policy_ids),
        action=expectation.action,
        severity=expectation.severity,
        sanitized_payload=sanitized_payload,
        findings=[],
        skipped_validator_ids=list(case.assertions.skipped_validator_ids),
        runtime_error=bool(case.assertions.runtime_error),
        cache_hit=bool(case.assertions.cache_hit),
        latency_ms=10.0,
        judge_payload={"redacted": True},
    )


def test_fixed_dataset_shape_language_split_and_dual_labels() -> None:
    cases = load_guardrail_cases()

    assert len(cases) == 200
    assert [case.case_id for case in cases] == [
        f"G{index:03d}" for index in range(1, 201)
    ]
    assert sum(case.tags["language"] == "en" for case in cases) == 140
    assert sum(case.tags["language"] == "zh" for case in cases) == 60
    assert (
        sum(case.policy_expectation != case.current_expectation for case in cases) == 46
    )
    assert {case.workflow_type for case in cases} == {"support", "content"}
    assert {case.stage for case in cases} == {"input", "output", "action", "provenance"}


def test_dataset_covers_every_configured_guardrail_and_action() -> None:
    coverage = guardrail_coverage_report(load_guardrail_cases())

    assert coverage.ratio == 1.0
    assert coverage.missing_targets == []
    assert all(target.positive_case_ids for target in coverage.targets)
    assert all(target.negative_case_ids for target in coverage.targets)


def test_mlflow_dataset_records_exclude_raw_payloads() -> None:
    records = mlflow_dataset_records(load_guardrail_cases())
    serialized = json.dumps(records, ensure_ascii=False)

    assert len(records) == 200
    assert '"payload"' not in serialized
    assert "TestPassword9988" not in serialized
    assert "buyer@example.test" not in serialized
    assert all(set(record["inputs"]) == {"case_id"} for record in records)


def test_judge_boundary_force_masks_pii_even_for_unrecognized_field_names() -> None:
    case = load_guardrail_cases()[32]
    payload = _judge_payload(case, case.payload)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "buyer@example.test" not in serialized
    assert "bu***@example.test" in serialized


def test_judge_configuration_requires_provider_model_uri() -> None:
    config = GuardrailEvaluationConfig()
    assert config.judge_model == "openrouter:/qwen/qwen3.7-plus"
    context_config = GuardrailEvaluationConfig.from_context(
        {
            "mlflow_guardrail_judge_model": "openrouter:/qwen/qwen3.7-plus",
            "mlflow_genai_judge_default_model": "openai:/gpt-4o-mini",
        }
    )
    assert context_config.judge_model == "openrouter:/qwen/qwen3.7-plus"

    with pytest.raises(ValidationError):
        GuardrailEvaluationConfig(judge_model="qwen3.7-plus")
    with pytest.raises(ValidationError):
        GuardrailEvaluationConfig(judge_model="unknown:/qwen3.7-plus")


def test_policy_and_current_contract_are_scored_independently() -> None:
    case = load_guardrail_cases()[13]
    result = _ideal_result(case)

    assert score_guardrail_case(case, result, target="policy").passed
    assert not score_guardrail_case(case, result, target="current").passed


def test_metrics_and_balanced_gates_pass_for_ideal_policy_results() -> None:
    cases = load_guardrail_cases()
    results = [_ideal_result(case) for case in cases]
    metrics = calculate_guardrail_metrics(cases, results, duration_seconds=100.0)
    judge_metrics = GuardrailJudgeMetrics(
        attempted_calls=96,
        completed_calls=96,
        passed_calls=90,
        pass_rate=90 / 96,
        error_rate=0.0,
    )

    assert metrics.desired_policy_pass_rate == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.config_coverage == 1.0
    assert metrics.current_contract_pass_rate < 1.0
    assert evaluate_guardrail_gates(metrics, judge_metrics=judge_metrics).passed


def test_fault_injection_uses_real_policy_path_and_marks_runtime_error() -> None:
    case = load_guardrail_cases()[190]
    service = _FaultInjectingGuardrailService("prompt_injection")
    with patch.object(
        service.__class__.__mro__[1],
        "_guardrails_ai_findings",
        return_value=_PayloadEvaluation(findings=[], skipped_validators=[]),
    ):
        result = evaluate_guardrail_case(case, service=service)

    assert result.runtime_error
    assert result.policy_ids == ["prompt_injection"]
    assert result.action == "review_required"


def test_cache_fixture_returns_deterministic_cached_result() -> None:
    from services.guardrail_evaluation import _CacheInjectingGuardrailService

    service = _CacheInjectingGuardrailService(validation_passed=False)
    cached = service._read_prompt_injection_cache("key", {})

    assert isinstance(cached, _CachedValidationResult)
    assert not cached.validation_passed
    assert service.evaluation_cache_hit


def test_fixture_services_share_loaded_validator_state() -> None:
    from services.workflow_guardrails import WorkflowGuardrailService

    shared = WorkflowGuardrailService()
    shared._validator_cache["sentinel"] = object()
    shared._embed_model = object()
    shared._redis_clients["sentinel"] = object()

    for case in load_guardrail_cases()[190:196]:
        isolated = guardrail_service_for_case(case, shared)
        assert isolated is not shared
        assert isolated._validator_cache is shared._validator_cache
        assert isolated._embed_model is shared._embed_model
        assert isolated._redis_clients is shared._redis_clients


def test_mlflow_scorers_and_bounded_qwen_judge_selection() -> None:
    cases = load_guardrail_cases()
    code_names = {scorer.name for scorer in build_guardrail_code_scorers()}
    judges = build_guardrail_judges("openrouter:/qwen/qwen3.7-plus")
    selected = _select_judge_cases(cases, max_calls=96)

    assert "guardrail_policy_passed" in code_names
    assert "guardrail_output_pii_detection" in code_names
    assert all(judge.model == "openrouter:/qwen/qwen3.7-plus" for judge in judges)
    assert sum(len(group) for group in selected.values()) == 96
    assert all(
        {case.tags["language"] for case in group} == {"en", "zh"}
        for group in selected.values()
    )


def test_bootstrap_merges_only_safe_records_and_is_idempotent() -> None:
    cases = load_guardrail_cases()
    dataset = MagicMock(dataset_id="dataset-1")
    fake_mlflow = SimpleNamespace(
        genai=SimpleNamespace(
            datasets=SimpleNamespace(get_dataset=MagicMock(return_value=dataset))
        )
    )
    config = GuardrailEvaluationConfig()
    with (
        patch(
            "services.mlflow_guardrail_evaluation.configure_guardrail_mlflow",
            return_value=(fake_mlflow, "experiment-1"),
        ),
        patch(
            "services.mlflow_guardrail_evaluation._register_scorers",
            return_value=[],
        ),
    ):
        report = bootstrap_guardrail_evaluation(
            cases=cases,
            config_context={"mlflow_automatic_evaluation_enabled": False},
            evaluation_config=config,
        )

    merged = dataset.merge_records.call_args.args[0]
    assert report["dataset"]["status"] == "existing"
    assert report["dataset"]["raw_payloads_uploaded"] is False
    assert '"payload"' not in json.dumps(merged)


def test_mlflow_evaluation_logs_precomputed_redacted_scores() -> None:
    case = load_guardrail_cases()[0]
    result = _ideal_result(case)
    active_run = MagicMock()
    active_run.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1")
    )
    fake_mlflow = SimpleNamespace(
        start_run=MagicMock(return_value=active_run),
        log_dict=MagicMock(),
        set_tags=MagicMock(),
    )
    with patch(
        "services.mlflow_guardrail_evaluation.configure_guardrail_mlflow",
        return_value=(fake_mlflow, "experiment-1"),
    ):
        run_id, judge_metrics = run_mlflow_guardrail_evaluation(
            cases=[case],
            results=[result],
            config_context={},
            evaluation_config=GuardrailEvaluationConfig(),
            run_judges=False,
        )

    assert run_id == "run-1"
    assert judge_metrics is None
    logged = fake_mlflow.log_dict.call_args.args[0]
    assert logged["records"][0]["outputs"]["policy_ids"] == result.policy_ids
    assert '"payload"' not in json.dumps(logged)
