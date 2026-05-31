import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from models import WorkflowType
from runtime_config import load_runtime_config, resolve_workflow_runtime_context
from utils.llm_config import build_llm, llm_chat_completions_url
from utils.result_cache import build_workflow_cache_key


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

    def test_runtime_config_loads_support_llm_profiles(self) -> None:
        profiles_json = (
            '{"openai_gpt4o_mini":{"llm_provider":"openai","llm_model_name":"gpt-4o-mini",'
            '"llm_api_key_env":"OPENAI_API_KEY"},"openrouter_gpt4o_mini":{'
            '"llm_provider":"openrouter","llm_model_name":"openai/gpt-4o-mini",'
            '"llm_base_url":"https://openrouter.ai/api/v1","llm_api_key_env":"OPENROUTER_API_KEY",'
            '"llm_disable_reasoning":false}}'
        )
        env = {
            "OPENAI_API_KEY": "openai-key",
            "OPENROUTER_API_KEY": "openrouter-key",
            "LLM_PROFILES_JSON": profiles_json,
            "SUPPORT_LLM_PROFILE": "openai_gpt4o_mini",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()
            context = resolve_workflow_runtime_context(config, WorkflowType.SUPPORT)

        self.assertIn("openrouter_gpt4o_mini", config.llm_profiles)
        self.assertEqual(config.support_llm_profile, "openai_gpt4o_mini")
        self.assertEqual(context["llm_profile"], "openai_gpt4o_mini")
        self.assertEqual(context["llm_provider"], "openai")
        self.assertEqual(context["llm_model_name"], "gpt-4o-mini")
        self.assertEqual(context["llm_api_key"], "openai-key")
        self.assertIsNone(context["llm_base_url"])

    def test_support_llm_profile_unknown_name_fails_fast(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROFILES_JSON": "{}",
                "SUPPORT_LLM_PROFILE": "missing_profile",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Unknown LLM profile"):
                load_runtime_config()

    def test_support_llm_profile_missing_key_env_fails_fast(self) -> None:
        profiles_json = (
            '{"openai_gpt4o_mini":{"llm_provider":"openai","llm_model_name":"gpt-4o-mini",'
            '"llm_api_key_env":"OPENAI_API_KEY"}}'
        )
        with patch.dict(
            os.environ,
            {
                "LLM_PROFILES_JSON": profiles_json,
                "SUPPORT_LLM_PROFILE": "openai_gpt4o_mini",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                load_runtime_config()

    def test_request_llm_profile_overrides_support_default_profile(self) -> None:
        profiles_json = (
            '{"openai_gpt4o_mini":{"llm_provider":"openai","llm_model_name":"gpt-4o-mini",'
            '"llm_api_key_env":"OPENAI_API_KEY"},"openrouter_qwen":{'
            '"llm_provider":"openrouter","llm_model_name":"qwen/qwen3-14b",'
            '"llm_api_key_env":"OPENROUTER_API_KEY"}}'
        )
        env = {
            "OPENAI_API_KEY": "openai-key",
            "OPENROUTER_API_KEY": "openrouter-key",
            "LLM_PROFILES_JSON": profiles_json,
            "SUPPORT_LLM_PROFILE": "openai_gpt4o_mini",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()
            context = resolve_workflow_runtime_context(
                config,
                WorkflowType.SUPPORT,
                {"llm_profile": "openrouter_qwen"},
            )

        self.assertEqual(context["llm_profile"], "openrouter_qwen")
        self.assertEqual(context["llm_provider"], "openrouter")
        self.assertEqual(context["llm_model_name"], "qwen/qwen3-14b")
        self.assertEqual(context["llm_base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(context["llm_api_key"], "openrouter-key")

    def test_runtime_config_reads_support_qa_mode(self) -> None:
        with patch.dict(os.environ, {"SUPPORT_QA_MODE": "adaptive_fast"}, clear=True):
            config = load_runtime_config()

        self.assertEqual(config.support_qa_mode, "adaptive_fast")

    def test_runtime_config_rejects_unknown_support_qa_mode(self) -> None:
        with patch.dict(os.environ, {"SUPPORT_QA_MODE": "skip_everything"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SUPPORT_QA_MODE"):
                load_runtime_config()

    def test_workflow_cache_key_ignores_profile_secret_material(self) -> None:
        base_context = {
            "llm_profile": "openrouter_gpt4o_mini",
            "llm_provider": "openrouter",
            "llm_model_name": "openai/gpt-4o-mini",
            "llm_base_url": "https://openrouter.ai/api/v1",
            "llm_api_key": "key-a",
            "openai_api_key": "openai-key-a",
            "support_llm_profile": "openai_gpt4o_mini",
            "llm_profiles": {
                "openrouter_gpt4o_mini": {
                    "llm_api_key_env": "OPENROUTER_API_KEY",
                    "llm_model_name": "openai/gpt-4o-mini",
                    "llm_provider": "openrouter",
                }
            },
        }
        changed_secret_context = {
            **base_context,
            "llm_api_key": "key-b",
            "openai_api_key": "openai-key-b",
            "llm_profiles": {
                "openrouter_gpt4o_mini": {
                    "llm_api_key_env": "ANOTHER_ENV",
                    "llm_model_name": "openai/gpt-4o-mini",
                    "llm_provider": "openrouter",
                }
            },
        }
        changed_model_context = {
            **base_context,
            "llm_model_name": "qwen/qwen3-14b",
        }

        self.assertEqual(
            build_workflow_cache_key(WorkflowType.SUPPORT, {"inquiry": "hello"}, base_context),
            build_workflow_cache_key(WorkflowType.SUPPORT, {"inquiry": "hello"}, changed_secret_context),
        )
        self.assertNotEqual(
            build_workflow_cache_key(WorkflowType.SUPPORT, {"inquiry": "hello"}, base_context),
            build_workflow_cache_key(WorkflowType.SUPPORT, {"inquiry": "hello"}, changed_model_context),
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
