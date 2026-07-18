from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import yaml
from dotenv import load_dotenv

from runtime_config import load_runtime_config
from services.mlflow_governance import bootstrap_support_governance

def _load_yaml(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a list of records in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed support prompts, evaluation data, and official MLflow scorers."
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

    report = bootstrap_support_governance(
        agents=_load_yaml(BASE_DIR / "config" / "support" / "agents.yaml"),
        tasks=_load_yaml(BASE_DIR / "config" / "support" / "tasks.yaml"),
        dataset_records=_load_dataset(
            BASE_DIR / "config" / "mlflow" / "support_evaluation_dataset.json"
        ),
        config_context=config_context,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
