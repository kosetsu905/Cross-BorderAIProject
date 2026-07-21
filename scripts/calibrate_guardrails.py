from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.guardrail_evaluation import (  # noqa: E402
    calculate_guardrail_metrics,
    evaluate_guardrail_case,
    evaluate_guardrail_gates,
    guardrail_service_for_case,
    load_guardrail_cases,
)
from runtime_config import load_runtime_config  # noqa: E402
from services.workflow_guardrails import WorkflowGuardrailService  # noqa: E402


DEFAULT_DATASET_PATH = (
    BASE_DIR / "config" / "mlflow" / "guardrail_calibration_dataset.json"
)
DEFAULT_OUTPUT_PATH = (
    BASE_DIR / "artifacts" / "guardrail_evaluation" / "calibration.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local guardrail detectors against the 600-case calibration corpus."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--expected-case-count", type=int, default=600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--with-hub", action="store_true")
    parser.add_argument("--with-qwen", action="store_true")
    parser.add_argument("--suite-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    cases = load_guardrail_cases(
        args.dataset_path,
        expected_count=args.expected_case_count,
    )
    service = WorkflowGuardrailService()
    context = {
        **load_runtime_config().as_context(),
        "langfuse_base_url": None,
        "mlflow_tracing_enabled": False,
        "observability_enabled": False,
        "otel_enabled": False,
        "workflow_guardrails_hub_enabled": args.with_hub,
        "workflow_guardrails_semantic_enabled": args.with_qwen,
    }
    started = time.perf_counter()
    deadline = started + args.suite_timeout_seconds
    results = []
    for case in cases:
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"Guardrail calibration exceeded {args.suite_timeout_seconds} seconds"
            )
        results.append(
            evaluate_guardrail_case(
                case,
                service=guardrail_service_for_case(case, service),
                config_context=context,
            )
        )
    metrics = calculate_guardrail_metrics(
        cases,
        results,
        duration_seconds=time.perf_counter() - started,
    )
    first_stage_passed = (
        metrics.false_positive_rate < 0.10
        and all(rate < 0.10 for rate in metrics.policy_false_positive_rates.values())
        and metrics.secrets_recall >= 0.90
        and metrics.pii_detection_recall >= 0.90
        and metrics.pii_masking_recall >= 0.90
        and metrics.toxicity_recall >= 0.90
        and metrics.privacy_leakage_rate < 0.10
    )
    formal_gate = evaluate_guardrail_gates(
        metrics,
        judge_metrics=None,
        judges_required=False,
    )
    report = {
        "dataset": str(args.dataset_path),
        "first_stage_passed": first_stage_passed,
        "formal_gate": formal_gate.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json"),
        "raw_payloads_uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": metrics.case_count,
                "false_positive_rate": metrics.false_positive_rate,
                "first_stage_passed": first_stage_passed,
                "formal_gate_passed": formal_gate.passed,
                "pii_detection_recall": metrics.pii_detection_recall,
                "pii_masking_recall": metrics.pii_masking_recall,
                "privacy_leakage_rate": metrics.privacy_leakage_rate,
                "secrets_recall": metrics.secrets_recall,
                "toxicity_recall": metrics.toxicity_recall,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if first_stage_passed else 1)


if __name__ == "__main__":
    main()
