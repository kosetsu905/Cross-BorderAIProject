import importlib
import unittest


OUTPUT_PYDANTIC_MODELS = [
    ("crews.analytics_crew", "AnalyticsReportOutput"),
    ("crews.bizdev_crew", "BizDevOutput"),
    ("crews.content_crew", "PerLanguageContentOutput"),
    ("crews.marketing_crew", "PerMarketCampaignOutput"),
    ("crews.sales_improvement_crew", "SalesImprovementOutput"),
    ("crews.scheduler_crew", "SchedulerOutput"),
    ("crews.support_crew", "CustomerServiceOutput"),
]


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


if __name__ == "__main__":
    unittest.main()
