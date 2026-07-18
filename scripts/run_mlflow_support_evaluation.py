from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

from crews.support_crew import run_support_crew
from runtime_config import load_runtime_config
from services.mlflow_governance import (
    SUPPORT_GUIDELINES,
    build_official_support_scorers,
    configure_mlflow,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the support workflow with MLflow's official GenAI evaluation API."
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional tracking URI override, such as http://localhost:5000.",
    )
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    config_context = load_runtime_config().as_context()
    if args.tracking_uri:
        config_context["mlflow_tracking_uri"] = args.tracking_uri
    mlflow, _ = configure_mlflow(config_context)
    dataset_name = str(
        config_context.get("mlflow_support_evaluation_dataset_name") or "support-governance"
    )
    dataset = mlflow.genai.datasets.get_dataset(name=dataset_name)
    model = str(
        config_context.get("mlflow_genai_judge_default_model")
        or "openrouter:/qwen/qwen3.7-plus"
    )
    scorers = [
        scorer
        for scorer in build_official_support_scorers(model)
        if scorer.name
        in {
            "relevance_to_query",
            "completeness",
            "safety",
            "pii_detection",
            "support_guidelines",
        }
    ]

    def predict_fn(**inputs: Any) -> str:
        result = run_support_crew(inputs, dict(config_context))
        if isinstance(result, dict):
            return str(result.get("final_response") or result)
        return str(result)

    evaluation = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=scorers,
    )
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "guidelines": SUPPORT_GUIDELINES,
                "result": str(evaluation),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
