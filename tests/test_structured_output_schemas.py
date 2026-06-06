import importlib
import sys
import unittest
from pathlib import Path

import yaml


OUTPUT_PYDANTIC_MODELS = [
    ("crews.analytics_crew", "AnalyticsReportOutput"),
    ("crews.bizdev_crew", "BizDevOutput"),
    ("crews.content_crew", "ContentOutput"),
    ("crews.content_crew", "PerLanguageContentOutput"),
    ("crews.marketing_crew", "PerMarketCampaignOutput"),
    ("crews.sales_improvement_crew", "SalesImprovementOutput"),
    ("crews.scheduler_crew", "SchedulerOutput"),
    ("crews.support_crew", "CustomerServiceOutput"),
]

FINAL_OUTPUT_TASKS = [
    (
        "crews.analytics_crew",
        "AnalyticsReportOutput",
        "config/analytics/tasks.yaml",
        "insight_report",
    ),
    (
        "crews.bizdev_crew",
        "BizDevOutput",
        "config/business_development/tasks.yaml",
        "pipeline_sync",
    ),
    (
        "crews.content_crew",
        "PerLanguageContentOutput",
        "config/content/tasks.yaml",
        "content_creation_and_qa",
    ),
    (
        "crews.marketing_crew",
        "PerMarketCampaignOutput",
        "config/marketing/tasks.yaml",
        "creative_compliance_package",
    ),
    (
        "crews.sales_improvement_crew",
        "SalesImprovementOutput",
        "config/sales_improvement/tasks.yaml",
        "playbook_generation",
    ),
    (
        "crews.scheduler_crew",
        "SchedulerOutput",
        "config/scheduler/tasks.yaml",
        "notification_reminder_setup",
    ),
    (
        "crews.support_crew",
        "CustomerServiceOutput",
        "config/support/tasks.yaml",
        "quality_assurance_review",
    ),
]

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _open_object_paths(schema: dict) -> list[str]:
    missing: list[str] = []

    def walk(node, path: str = "root") -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and node.get("additionalProperties") is not False:
                missing.append(path)
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(schema)
    return missing


class StructuredOutputSchemaTests(unittest.TestCase):
    def test_crewai_output_pydantic_models_are_openai_strict_schema_safe(self) -> None:
        failures: dict[str, list[str]] = {}
        for module_name, class_name in OUTPUT_PYDANTIC_MODELS:
            model = getattr(importlib.import_module(module_name), class_name)
            missing_paths = _open_object_paths(model.model_json_schema())
            if missing_paths:
                failures[f"{module_name}.{class_name}"] = missing_paths

        self.assertEqual(failures, {})

    def test_final_task_prompts_name_required_output_fields(self) -> None:
        failures: dict[str, list[str]] = {}
        for module_name, class_name, task_config_path, task_name in FINAL_OUTPUT_TASKS:
            model = getattr(importlib.import_module(module_name), class_name)
            tasks_config = yaml.safe_load(
                (BASE_DIR / task_config_path).read_text(encoding="utf-8")
            )
            task_config = tasks_config[task_name]
            prompt_text = "\n".join(
                [
                    str(task_config.get("description", "")),
                    str(task_config.get("expected_output", "")),
                ]
            )
            missing_fields = [
                field_name
                for field_name, field_info in model.model_fields.items()
                if field_info.is_required() and field_name not in prompt_text
            ]
            if missing_fields:
                failures[f"{module_name}.{class_name}:{task_name}"] = missing_fields

        self.assertEqual(failures, {})


if __name__ == "__main__":
    unittest.main()
