from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402

from runtime_config import load_runtime_config  # noqa: E402
from services.guardrail_evaluation import (  # noqa: E402
    GuardrailEvaluationConfig,
    calculate_guardrail_metrics,
    evaluate_guardrail_case,
    evaluate_guardrail_gates,
    guardrail_coverage_report,
    guardrail_service_for_case,
    load_guardrail_cases,
    score_guardrail_case,
)
from services.mlflow_guardrail_evaluation import (  # noqa: E402
    bootstrap_guardrail_evaluation,
    log_guardrail_suite_summary,
    run_mlflow_guardrail_evaluation,
)
from services.workflow_guardrails import WorkflowGuardrailService  # noqa: E402


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed guardrail regression suite and bounded MLflow Qwen judges."
    )
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-judge-calls", type=int, default=None)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only this case ID; repeat the option to select multiple cases.",
    )
    parser.add_argument("--start-case", default=None, help="First case ID to run.")
    parser.add_argument("--end-case", default=None, help="Last case ID to run.")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-judges", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit zero for quality-gate failures; infrastructure failures still exit two.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "artifacts" / "guardrail_evaluation" / "latest.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        exit_code = _run(args)
    except Exception as exc:
        logger.exception("Guardrail evaluation infrastructure failed")
        print(
            json.dumps(
                {"status": "infrastructure_error", "error_type": type(exc).__name__},
                ensure_ascii=False,
            )
        )
        raise SystemExit(2) from exc
    raise SystemExit(exit_code)


def _run(args: argparse.Namespace) -> int:
    evaluator_openrouter_key = os.getenv("OPENROUTER_API_KEY")
    load_dotenv(BASE_DIR / ".env")
    if evaluator_openrouter_key is None:
        # A shared application .env is not an authorized source for judge credentials.
        os.environ.pop("OPENROUTER_API_KEY", None)
    load_dotenv(BASE_DIR / ".env.guardrail-eval")
    config_context = load_runtime_config().as_context()
    # Source payload execution is local to this harness. Only the later,
    # explicitly redacted MLflow evaluation phase is allowed to create traces.
    config_context.update(
        {
            "observability_enabled": False,
            "otel_enabled": False,
            "mlflow_tracing_enabled": False,
            "langfuse_base_url": None,
        }
    )
    if args.tracking_uri:
        config_context["mlflow_tracking_uri"] = args.tracking_uri
    if args.judge_model:
        config_context["mlflow_guardrail_judge_model"] = args.judge_model
    if args.max_cases is not None:
        config_context["mlflow_guardrail_max_cases"] = args.max_cases
    if args.max_judge_calls is not None:
        config_context["mlflow_guardrail_max_judge_calls"] = args.max_judge_calls
    evaluation_config = GuardrailEvaluationConfig.from_context(config_context)

    dataset_path = (
        args.dataset_path
        or BASE_DIR / "config" / "mlflow" / "guardrail_evaluation_dataset.json"
    )
    all_cases = load_guardrail_cases(dataset_path)
    if args.case_id and (args.start_case or args.end_case):
        raise ValueError("--case-id cannot be combined with --start-case or --end-case")
    if args.case_id:
        requested_ids = set(args.case_id)
        known_ids = {case.case_id for case in all_cases}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"Unknown guardrail case IDs: {', '.join(unknown_ids)}")
        cases = [case for case in all_cases if case.case_id in requested_ids]
    elif args.start_case or args.end_case:
        index_by_id = {case.case_id: index for index, case in enumerate(all_cases)}
        start_case = args.start_case or all_cases[0].case_id
        end_case = args.end_case or all_cases[-1].case_id
        unknown_ids = [
            case_id for case_id in (start_case, end_case) if case_id not in index_by_id
        ]
        if unknown_ids:
            raise ValueError(f"Unknown guardrail case IDs: {', '.join(unknown_ids)}")
        start_index = index_by_id[start_case]
        end_index = index_by_id[end_case]
        if start_index > end_index:
            raise ValueError("--start-case must not be after --end-case")
        cases = all_cases[start_index : end_index + 1]
    else:
        cases = all_cases[: evaluation_config.max_cases]
    if not cases:
        raise ValueError("No guardrail evaluation cases selected")
    coverage = guardrail_coverage_report(all_cases)
    if coverage.missing_targets:
        raise ValueError(
            "Guardrail dataset does not cover configured targets: "
            + ", ".join(coverage.missing_targets)
        )
    if not args.skip_bootstrap:
        bootstrap_guardrail_evaluation(
            cases=all_cases,
            config_context=config_context,
            evaluation_config=evaluation_config,
        )

    started = time.perf_counter()
    results = []
    deadline = started + evaluation_config.suite_timeout_seconds
    shared_service = WorkflowGuardrailService()
    for case in cases:
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"Guardrail suite exceeded {evaluation_config.suite_timeout_seconds} seconds"
            )
        logger.info("Evaluating guardrail case %s", case.case_id)
        try:
            results.append(
                evaluate_guardrail_case(
                    case,
                    service=guardrail_service_for_case(case, shared_service),
                    config_context=config_context,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Guardrail case {case.case_id} failed with {type(exc).__name__}"
            ) from exc
    deterministic_duration = time.perf_counter() - started
    metrics = calculate_guardrail_metrics(
        cases,
        results,
        duration_seconds=deterministic_duration,
    )
    mlflow_run_id, judge_metrics = run_mlflow_guardrail_evaluation(
        cases=cases,
        results=results,
        config_context=config_context,
        evaluation_config=evaluation_config,
        run_judges=not args.skip_judges,
    )
    total_duration = time.perf_counter() - started
    metrics = metrics.model_copy(update={"duration_seconds": total_duration})
    gates = evaluate_guardrail_gates(
        metrics,
        judge_metrics=judge_metrics,
        judges_required=not args.skip_judges,
    )
    failed_cases = []
    for case, result in zip(cases, results, strict=True):
        policy_score = score_guardrail_case(case, result, target="policy")
        current_score = score_guardrail_case(case, result, target="current")
        if not policy_score.passed or not current_score.passed:
            failed_cases.append(
                {
                    "case_id": case.case_id,
                    "policy_passed": policy_score.passed,
                    "current_passed": current_score.passed,
                    "actual": {
                        "triggered": result.triggered,
                        "policy_ids": result.policy_ids,
                        "action": result.action,
                        "severity": result.severity,
                    },
                }
            )
    report: dict[str, Any] = {
        "status": "passed" if gates.passed else "quality_gate_failed",
        "mlflow_run_id": mlflow_run_id,
        "experiment": evaluation_config.experiment_name,
        "dataset": evaluation_config.dataset_name,
        "judge_model": evaluation_config.judge_model,
        "raw_payloads_uploaded": False,
        "metrics": metrics.model_dump(mode="json"),
        "judge_metrics": judge_metrics.model_dump(mode="json")
        if judge_metrics
        else None,
        "gates": gates.model_dump(mode="json"),
        "coverage": coverage.model_dump(mode="json"),
        "failed_cases": failed_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log_guardrail_suite_summary(
        config_context=config_context,
        evaluation_config=evaluation_config,
        run_id=mlflow_run_id,
        metrics=_flat_metrics(metrics.model_dump(mode="json"), judge_metrics),
        report=report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if gates.passed or args.report_only:
        return 0
    return 1


def _flat_metrics(payload: dict[str, Any], judge_metrics: Any) -> dict[str, float]:
    excluded = {"case_count", "dataset_digest", "policy_metrics"}
    metrics = {
        f"guardrail.{key}": float(value)
        for key, value in payload.items()
        if key not in excluded and isinstance(value, int | float)
    }
    for policy_id, values in payload["policy_metrics"].items():
        for key in ("precision", "recall", "f1"):
            metrics[f"guardrail.policy.{policy_id}.{key}"] = float(values[key])
    if judge_metrics is not None:
        metrics["guardrail.judge.pass_rate"] = judge_metrics.pass_rate
        metrics["guardrail.judge.error_rate"] = judge_metrics.error_rate
        metrics["guardrail.judge.calls"] = float(judge_metrics.attempted_calls)
    return metrics


if __name__ == "__main__":
    main()
