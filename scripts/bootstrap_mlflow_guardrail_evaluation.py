from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402

from runtime_config import load_runtime_config  # noqa: E402
from services.guardrail_evaluation import (  # noqa: E402
    GuardrailEvaluationConfig,
    load_guardrail_cases,
)
from services.mlflow_guardrail_evaluation import (  # noqa: E402
    bootstrap_guardrail_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the redacted guardrail dataset and MLflow scorers/judges."
    )
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    config_context = load_runtime_config().as_context()
    if args.tracking_uri:
        config_context["mlflow_tracking_uri"] = args.tracking_uri
    evaluation_config = GuardrailEvaluationConfig.from_context(config_context)
    cases = load_guardrail_cases(
        args.dataset_path
        or BASE_DIR / "config" / "mlflow" / "guardrail_evaluation_dataset.json"
    )
    report = bootstrap_guardrail_evaluation(
        cases=cases,
        config_context=config_context,
        evaluation_config=evaluation_config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
