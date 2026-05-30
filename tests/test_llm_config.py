import ast
import unittest
from pathlib import Path

from utils.llm_config import build_llm, llm_chat_completions_url


CREW_FILES = [
    "analytics_crew.py",
    "bizdev_crew.py",
    "content_crew.py",
    "marketing_crew.py",
    "sales_improvement_crew.py",
    "scheduler_crew.py",
    "support_crew.py",
]


class LLMConfigTests(unittest.TestCase):
    def test_build_llm_uses_generic_openai_compatible_config(self) -> None:
        llm = build_llm(
            {
                "llm_api_key": "openrouter-key",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model_name": "openai/gpt-4o-mini",
                "llm_provider": "openrouter",
            }
        )

        self.assertEqual(llm.model, "openai/gpt-4o-mini")
        self.assertEqual(llm.provider, "openrouter")
        self.assertEqual(llm.api_key, "openrouter-key")
        self.assertEqual(llm.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(llm.api_base, "https://openrouter.ai/api/v1")

    def test_build_llm_disables_reasoning_for_openrouter_qwen3(self) -> None:
        llm = build_llm(
            {
                "llm_api_key": "openrouter-key",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model_name": "qwen/qwen3-14b",
                "llm_provider": "openrouter",
            }
        )

        self.assertEqual(llm.model, "qwen/qwen3-14b")
        self.assertEqual(llm.reasoning_effort, "none")
        self.assertEqual(llm.additional_params, {})

    def test_build_llm_does_not_add_reasoning_params_for_openrouter_gpt4o_mini(self) -> None:
        llm = build_llm(
            {
                "llm_api_key": "openrouter-key",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_model_name": "openai/gpt-4o-mini",
                "llm_provider": "openrouter",
            }
        )

        self.assertIsNone(llm.reasoning_effort)
        self.assertEqual(llm.additional_params, {})

    def test_build_llm_manual_disable_reasoning_override(self) -> None:
        llm = build_llm(
            {
                "llm_model_name": "openai/gpt-4o-mini",
                "llm_disable_reasoning": True,
            }
        )

        self.assertEqual(llm.reasoning_effort, "none")
        self.assertEqual(llm.additional_params, {})

    def test_chat_completions_url_defaults_to_openai_compatible_endpoint(self) -> None:
        self.assertEqual(
            llm_chat_completions_url({"llm_base_url": "https://openrouter.ai/api/v1/"}),
            "https://openrouter.ai/api/v1/chat/completions",
        )

    def test_all_crew_agents_receive_configured_llm(self) -> None:
        root = Path(__file__).resolve().parents[1]
        missing: list[str] = []
        for file_name in CREW_FILES:
            path = root / "crews" / file_name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name) or node.func.id != "Agent":
                    continue
                if not any(keyword.arg == "llm" for keyword in node.keywords):
                    missing.append(f"{file_name}:{node.lineno}")

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
