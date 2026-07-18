import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from models import ProviderCredentials, WorkflowType
from runtime_config import (
    RuntimeConfig,
    load_runtime_config,
    resolve_workflow_runtime_context,
)
from utils.llm_config import build_llm, llm_chat_completions_url
from utils.model_tiering import ModelTierRouter, agent_llm_tier
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
EXPECTED_REVIEWER_AGENTS = {
    "config/analytics/agents.yaml": {"report_generator"},
    "config/business_development/agents.yaml": {"pipeline_manager"},
    "config/content/agents.yaml": {"multilingual_editor"},
    "config/marketing/agents.yaml": {"creative_compliance_specialist"},
    "config/sales_improvement/agents.yaml": {"playbook_coach"},
    "config/scheduler/agents.yaml": {"notification_coordinator"},
    "config/support/agents.yaml": {"support_qa_specialist"},
}


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

    def test_build_llm_does_not_add_reasoning_params_for_openrouter_gpt4o_mini(
        self,
    ) -> None:
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
            "WORKFLOW_GUARDRAILS_MODEL": "openrouter_gpt4o_mini",
            "WORKFLOW_GUARDRAILS_PROMPT_INJECTION_MODEL": "openai_gpt4o_mini",
            "WORKFLOW_GUARDRAILS_PROMPT_INJECTION_TIMEOUT_SECONDS": "5",
            "WORKFLOW_GUARDRAILS_PROMPT_INJECTION_CACHE_TTL_SECONDS": "43200",
            "WORKFLOW_GUARDRAILS_NATIVE_TRACING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()
            context = resolve_workflow_runtime_context(config, WorkflowType.SUPPORT)

        self.assertIn("openrouter_gpt4o_mini", config.llm_profiles)
        self.assertEqual(config.support_llm_profile, "openai_gpt4o_mini")
        self.assertEqual(config.workflow_guardrails_model, "openrouter_gpt4o_mini")
        self.assertEqual(
            config.workflow_guardrails_prompt_injection_model, "openai_gpt4o_mini"
        )
        self.assertEqual(
            config.workflow_guardrails_prompt_injection_timeout_seconds, 5.0
        )
        self.assertEqual(
            config.workflow_guardrails_prompt_injection_cache_ttl_seconds, 43200
        )
        self.assertFalse(config.workflow_guardrails_native_tracing_enabled)
        self.assertEqual(context["llm_profile"], "openai_gpt4o_mini")
        self.assertEqual(context["llm_provider"], "openai")
        self.assertEqual(context["llm_model_name"], "gpt-4o-mini")
        self.assertEqual(context["llm_api_key"], "openai-key")

    def test_runtime_config_loads_tool_cache_and_execution_flags(self) -> None:
        env = {
            "CELERY_BROKER_URL": "redis://localhost:6379/7",
            "TOOL_CACHE_ENABLED": "false",
            "TOOL_CACHE_BACKEND": "redis_postgres",
            "TOOL_CACHE_TTL_SECONDS": "123",
            "TOOL_CACHE_DB_ENABLED": "true",
            "TOOL_CACHE_MAX_VALUE_BYTES": "4096",
            "TOOL_EXECUTION_ASYNC_ENABLED": "false",
            "TOOL_EXECUTION_MAX_WORKERS": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()

        self.assertFalse(config.tool_cache_enabled)
        self.assertEqual(config.tool_cache_backend, "redis_postgres")
        self.assertEqual(config.tool_cache_redis_url, "redis://localhost:6379/7")
        self.assertEqual(config.tool_cache_ttl_seconds, 123)
        self.assertTrue(config.tool_cache_db_enabled)
        self.assertEqual(config.tool_cache_max_value_bytes, 4096)
        self.assertFalse(config.tool_execution_async_enabled)
        self.assertEqual(config.tool_execution_max_workers, 3)

    def test_runtime_config_loads_observability_flags(self) -> None:
        env = {
            "OBSERVABILITY_ENABLED": "true",
            "OBSERVABILITY_CAPTURE_INPUT_OUTPUT": "false",
            "OBSERVABILITY_ENVIRONMENT": "local",
            "OTEL_ENABLED": "true",
            "OTEL_GLOBAL_AUTO_INSTRUMENTATION_ENABLED": "true",
            "OTEL_HTTPX_INSTRUMENTATION_ENABLED": "true",
            "OTEL_REDIS_INSTRUMENTATION_ENABLED": "true",
            "OTEL_SQLALCHEMY_INSTRUMENTATION_ENABLED": "true",
            "OTEL_CELERY_INSTRUMENTATION_ENABLED": "true",
            "FASTAPI_OTEL_AUTO_INSTRUMENTATION_ENABLED": "true",
            "OPENINFERENCE_CREWAI_ENABLED": "true",
            "OPENINFERENCE_LITELLM_ENABLED": "false",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://phoenix:6006/v1/traces",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "PHOENIX_PROJECT_NAME": "cross-border-ai-dev",
            "LANGFUSE_BASE_URL": "http://langfuse-web:3000",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_EXPERIMENT_NAME": "cross-border-ai",
            "MLFLOW_TRACING_ENABLED": "true",
            "MLFLOW_PROMPT_REGISTRY_ENABLED": "true",
            "MLFLOW_SUPPORT_PROMPT_ALIAS": "production",
            "MLFLOW_PROMPT_CACHE_DIR": "artifacts/mlflow_prompt_cache",
            "MLFLOW_SUPPORT_EVALUATION_DATASET_NAME": "support-governance",
            "MLFLOW_GUARDRAIL_EXPERIMENT_NAME": "guardrail-eval-test",
            "MLFLOW_GUARDRAIL_EVALUATION_DATASET_NAME": "guardrail-dataset-test",
            "MLFLOW_GUARDRAIL_MAX_CASES": "200",
            "MLFLOW_GUARDRAIL_MAX_JUDGE_CALLS": "96",
            "MLFLOW_GUARDRAIL_SUITE_TIMEOUT_SECONDS": "1800",
            "MLFLOW_GUARDRAIL_JUDGE_MODEL": "openrouter:/qwen/qwen3.7-plus",
            "MLFLOW_AUTOMATIC_EVALUATION_ENABLED": "false",
            "MLFLOW_GENAI_JUDGE_DEFAULT_MODEL": "openai:/gpt-4o-mini",
            "MLFLOW_GIT_VERSION_TRACKING_ENABLED": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()

        self.assertTrue(config.observability_enabled)
        self.assertFalse(config.observability_capture_input_output)
        self.assertEqual(config.observability_environment, "local")
        self.assertTrue(config.otel_enabled)
        self.assertTrue(config.otel_global_auto_instrumentation_enabled)
        self.assertTrue(config.otel_httpx_instrumentation_enabled)
        self.assertTrue(config.otel_redis_instrumentation_enabled)
        self.assertTrue(config.otel_sqlalchemy_instrumentation_enabled)
        self.assertTrue(config.otel_celery_instrumentation_enabled)
        self.assertTrue(config.fastapi_otel_auto_instrumentation_enabled)
        self.assertTrue(config.openinference_crewai_enabled)
        self.assertFalse(config.openinference_litellm_enabled)
        self.assertEqual(
            config.otel_exporter_otlp_traces_endpoint, "http://phoenix:6006/v1/traces"
        )
        self.assertEqual(config.otel_exporter_otlp_protocol, "http/protobuf")
        self.assertEqual(config.phoenix_project_name, "cross-border-ai-dev")
        self.assertEqual(config.langfuse_base_url, "http://langfuse-web:3000")
        self.assertEqual(config.mlflow_tracking_uri, "http://mlflow:5000")
        self.assertEqual(config.mlflow_experiment_name, "cross-border-ai")
        self.assertTrue(config.mlflow_tracing_enabled)
        self.assertTrue(config.mlflow_prompt_registry_enabled)
        self.assertEqual(config.mlflow_support_prompt_alias, "production")
        self.assertEqual(
            config.mlflow_prompt_cache_dir, "artifacts/mlflow_prompt_cache"
        )
        self.assertEqual(
            config.mlflow_support_evaluation_dataset_name, "support-governance"
        )
        self.assertEqual(config.mlflow_guardrail_experiment_name, "guardrail-eval-test")
        self.assertEqual(
            config.mlflow_guardrail_evaluation_dataset_name,
            "guardrail-dataset-test",
        )
        self.assertEqual(config.mlflow_guardrail_max_cases, 200)
        self.assertEqual(config.mlflow_guardrail_max_judge_calls, 96)
        self.assertEqual(config.mlflow_guardrail_suite_timeout_seconds, 1800)
        self.assertEqual(
            config.mlflow_guardrail_judge_model,
            "openrouter:/qwen/qwen3.7-plus",
        )
        self.assertFalse(config.mlflow_automatic_evaluation_enabled)
        self.assertEqual(config.mlflow_genai_judge_default_model, "openai:/gpt-4o-mini")
        self.assertTrue(config.mlflow_git_version_tracking_enabled)

    def test_provider_credentials_override_tool_cache_without_infra_settings(
        self,
    ) -> None:
        credentials = ProviderCredentials.model_validate(
            {
                "tool_cache_enabled": False,
                "tool_cache_ttl_seconds": 30,
                "tool_execution_async_enabled": False,
            }
        )
        context = resolve_workflow_runtime_context(
            RuntimeConfig(tool_cache_enabled=True, tool_cache_ttl_seconds=60),
            WorkflowType.ANALYTICS,
            credentials.model_dump(exclude_none=True),
        )

        self.assertFalse(context["tool_cache_enabled"])
        self.assertEqual(context["tool_cache_ttl_seconds"], 30)
        self.assertFalse(context["tool_execution_async_enabled"])

        with self.assertRaises(ValidationError):
            ProviderCredentials.model_validate(
                {"tool_cache_redis_url": "redis://malicious.local/0"}
            )

    def test_runtime_config_loads_workflow_router_flags(self) -> None:
        env = {
            "WORKFLOW_ROUTER_ENABLED": "false",
            "WORKFLOW_ROUTER_LLM_FALLBACK_ENABLED": "false",
            "WORKFLOW_ROUTER_CONFIDENCE_THRESHOLD": "0.67",
            "WORKFLOW_ROUTER_MAX_WORKFLOWS": "3",
            "WORKFLOW_ROUTER_LLM_PROFILE": "router_profile",
            "LLM_PROFILES_JSON": (
                '{"router_profile":{"llm_provider":"openai",'
                '"llm_model_name":"gpt-4o-mini","llm_api_key_env":"OPENAI_API_KEY"}}'
            ),
            "OPENAI_API_KEY": "openai-key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()

        self.assertFalse(config.workflow_router_enabled)
        self.assertFalse(config.workflow_router_llm_fallback_enabled)
        self.assertEqual(config.workflow_router_confidence_threshold, 0.67)
        self.assertEqual(config.workflow_router_max_workflows, 3)
        self.assertEqual(config.workflow_router_llm_profile, "router_profile")

    def test_runtime_config_loads_support_auto_send_confidence_threshold(self) -> None:
        with patch.dict(
            os.environ,
            {"SUPPORT_AUTO_SEND_CONFIDENCE_THRESHOLD": "0.82"},
            clear=True,
        ):
            config = load_runtime_config()

        self.assertEqual(config.support_auto_send_confidence_threshold, 0.82)
        self.assertEqual(
            config.as_context()["support_auto_send_confidence_threshold"], 0.82
        )

    def test_provider_credentials_override_workflow_router_safely(self) -> None:
        credentials = ProviderCredentials.model_validate(
            {
                "workflow_router_enabled": False,
                "workflow_router_llm_fallback_enabled": False,
                "workflow_router_confidence_threshold": 0.9,
                "workflow_router_llm_profile": "router_profile",
            }
        )
        context = resolve_workflow_runtime_context(
            RuntimeConfig(),
            "workflow_route",
            credentials.model_dump(exclude_none=True),
        )

        self.assertFalse(context["workflow_router_enabled"])
        self.assertFalse(context["workflow_router_llm_fallback_enabled"])
        self.assertEqual(context["workflow_router_confidence_threshold"], 0.9)
        self.assertEqual(context["workflow_router_llm_profile"], "router_profile")
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

    def test_runtime_config_reads_workflow_async_execution_flag(self) -> None:
        with patch.dict(
            os.environ, {"WORKFLOW_ASYNC_EXECUTION_ENABLED": "false"}, clear=True
        ):
            config = load_runtime_config()

        self.assertFalse(config.workflow_async_execution_enabled)

    def test_runtime_config_reads_model_tiering_flags(self) -> None:
        profiles_json = (
            '{"worker_gpt4o_mini":{"llm_provider":"openai","llm_model_name":"gpt-4o-mini",'
            '"llm_api_key_env":"OPENAI_API_KEY"},"reviewer_gpt4o":{'
            '"llm_provider":"openai","llm_model_name":"gpt-4o",'
            '"llm_api_key_env":"OPENAI_API_KEY"}}'
        )
        env = {
            "OPENAI_API_KEY": "openai-key",
            "LLM_PROFILES_JSON": profiles_json,
            "WORKFLOW_MODEL_TIERING_ENABLED": "true",
            "WORKFLOW_WORKER_LLM_PROFILE": "worker_gpt4o_mini",
            "WORKFLOW_REVIEWER_LLM_PROFILE": "reviewer_gpt4o",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()

        self.assertTrue(config.workflow_model_tiering_enabled)
        self.assertEqual(config.workflow_worker_llm_profile, "worker_gpt4o_mini")
        self.assertEqual(config.workflow_reviewer_llm_profile, "reviewer_gpt4o")

    def test_request_credentials_override_model_tiering_settings(self) -> None:
        config = RuntimeConfig(
            workflow_model_tiering_enabled=True,
            workflow_worker_llm_profile="worker_default",
            workflow_reviewer_llm_profile="reviewer_default",
        )

        context = resolve_workflow_runtime_context(
            config,
            WorkflowType.ANALYTICS,
            {
                "workflow_model_tiering_enabled": False,
                "workflow_worker_llm_profile": "worker_request",
                "workflow_reviewer_llm_profile": "reviewer_request",
            },
        )

        self.assertFalse(context["workflow_model_tiering_enabled"])
        self.assertEqual(context["workflow_worker_llm_profile"], "worker_request")
        self.assertEqual(context["workflow_reviewer_llm_profile"], "reviewer_request")

    def test_model_tier_router_uses_worker_and_reviewer_profiles(self) -> None:
        context = {
            "llm_provider": "openai",
            "llm_model_name": "gpt-base",
            "openai_api_key": "base-key",
            "workflow_model_tiering_enabled": True,
            "workflow_worker_llm_profile": "worker_gpt4o_mini",
            "workflow_reviewer_llm_profile": "reviewer_gpt4o",
            "llm_profiles": {
                "worker_gpt4o_mini": {
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o-mini",
                    "llm_api_key_env": "OPENAI_API_KEY",
                },
                "reviewer_gpt4o": {
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "llm_api_key_env": "OPENAI_API_KEY",
                },
            },
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            router = ModelTierRouter(context)
            worker_llm = router.llm_for_agent({"llm_tier": "worker"})
            reviewer_llm = router.llm_for_agent({"llm_tier": "reviewer"})

        self.assertEqual(worker_llm.model, "gpt-4o-mini")
        self.assertEqual(reviewer_llm.model, "gpt-4o")
        self.assertEqual(worker_llm.api_key, "openai-key")
        self.assertEqual(reviewer_llm.api_key, "openai-key")

    def test_model_tier_router_falls_back_to_base_when_disabled_or_unset(self) -> None:
        disabled_router = ModelTierRouter(
            {
                "llm_model_name": "gpt-base",
                "workflow_model_tiering_enabled": False,
                "workflow_worker_llm_profile": "missing_worker",
                "workflow_reviewer_llm_profile": "missing_reviewer",
            }
        )
        self.assertIs(
            disabled_router.llm_for_agent({"llm_tier": "worker"}),
            disabled_router.llm_for_agent({"llm_tier": "reviewer"}),
        )
        self.assertEqual(
            disabled_router.llm_for_agent({"llm_tier": "worker"}).model, "gpt-base"
        )

        reviewer_only_router = ModelTierRouter(
            {
                "llm_provider": "openai",
                "llm_model_name": "gpt-base",
                "openai_api_key": "base-key",
                "workflow_model_tiering_enabled": True,
                "workflow_reviewer_llm_profile": "reviewer_gpt4o",
                "llm_profiles": {
                    "reviewer_gpt4o": {
                        "llm_provider": "openai",
                        "llm_model_name": "gpt-4o",
                        "llm_api_key_env": "OPENAI_API_KEY",
                    }
                },
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            worker_llm = reviewer_only_router.llm_for_agent({"llm_tier": "worker"})
            reviewer_llm = reviewer_only_router.llm_for_agent({"llm_tier": "reviewer"})

        self.assertEqual(worker_llm.model, "gpt-base")
        self.assertEqual(reviewer_llm.model, "gpt-4o")

    def test_model_tier_router_fails_for_unknown_profile_or_tier(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown LLM profile"):
            ModelTierRouter(
                {
                    "workflow_model_tiering_enabled": True,
                    "workflow_worker_llm_profile": "missing_profile",
                    "llm_profiles": {},
                }
            ).llm_for_agent({"llm_tier": "worker"})

        with self.assertRaisesRegex(ValueError, "Unsupported llm_tier"):
            agent_llm_tier({"llm_tier": "manager"})

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
        changed_tier_profile_context = {
            **base_context,
            "workflow_worker_llm_profile": "worker_gpt4o_mini",
        }

        self.assertEqual(
            build_workflow_cache_key(
                WorkflowType.SUPPORT, {"inquiry": "hello"}, base_context
            ),
            build_workflow_cache_key(
                WorkflowType.SUPPORT, {"inquiry": "hello"}, changed_secret_context
            ),
        )
        self.assertNotEqual(
            build_workflow_cache_key(
                WorkflowType.SUPPORT, {"inquiry": "hello"}, base_context
            ),
            build_workflow_cache_key(
                WorkflowType.SUPPORT, {"inquiry": "hello"}, changed_model_context
            ),
        )
        self.assertNotEqual(
            build_workflow_cache_key(
                WorkflowType.SUPPORT, {"inquiry": "hello"}, base_context
            ),
            build_workflow_cache_key(
                WorkflowType.SUPPORT, {"inquiry": "hello"}, changed_tier_profile_context
            ),
        )

    def test_agents_yaml_declares_expected_model_tiers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        failures: dict[str, object] = {}
        for relative_path, expected_reviewers in EXPECTED_REVIEWER_AGENTS.items():
            agents_config = yaml.safe_load(
                (root / relative_path).read_text(encoding="utf-8")
            )
            tiers = {
                agent_name: str(agent_config.get("llm_tier") or "")
                for agent_name, agent_config in agents_config.items()
            }
            invalid = {
                agent_name: tier
                for agent_name, tier in tiers.items()
                if tier not in {"worker", "reviewer"}
            }
            actual_reviewers = {
                agent_name for agent_name, tier in tiers.items() if tier == "reviewer"
            }
            if invalid or actual_reviewers != expected_reviewers:
                failures[relative_path] = {
                    "invalid": invalid,
                    "actual_reviewers": sorted(actual_reviewers),
                    "expected_reviewers": sorted(expected_reviewers),
                }

        self.assertEqual(failures, {})

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
