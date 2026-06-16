import os
import tempfile
import unittest
from unittest.mock import patch

from crewai.memory import Memory

from crews.support_crew import _normalize_inputs
from runtime_config import load_runtime_config
from utils.crew_memory import build_crew_memory
from utils.shared_context import build_conversation_history_context, compact_context


class CrewMemoryBuilderTests(unittest.TestCase):
    def test_disabled_memory_returns_false(self) -> None:
        self.assertFalse(build_crew_memory({"crewai_memory_enabled": False}, "analytics"))

    def test_support_is_not_allowlisted_by_default(self) -> None:
        self.assertFalse(build_crew_memory({"crewai_memory_enabled": True}, "support"))

    def test_enabled_allowlisted_workflow_returns_memory(self) -> None:
        with tempfile.TemporaryDirectory() as memory_dir:
            with patch("utils.crew_memory.build_llm", return_value=object()):
                memory = build_crew_memory(
                    {
                        "crewai_memory_enabled": True,
                        "openai_api_key": "test-openai-key",
                        "crewai_memory_storage_path": memory_dir,
                    },
                    "analytics",
                )

        self.assertIsInstance(memory, Memory)

    def test_enabled_allowlisted_workflow_requires_embedding_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "CREWAI_MEMORY_ENABLED requires OPENAI_API_KEY"):
            build_crew_memory({"crewai_memory_enabled": True}, "analytics")

    def test_runtime_config_reads_memory_and_context_flags(self) -> None:
        env = {
            "CREWAI_MEMORY_ENABLED": "true",
            "CREWAI_MEMORY_WORKFLOWS": "analytics,sales_improvement",
            "CREWAI_MEMORY_STORAGE_PATH": "artifacts/custom_memory",
            "CREWAI_MEMORY_EMBEDDER_MODEL": "text-embedding-3-large",
            "WORKFLOW_CONTEXT_MAX_CHARS": "9000",
            "TASK_CONTEXT_MAX_CHARS": "2500",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_runtime_config()

        self.assertTrue(config.crewai_memory_enabled)
        self.assertEqual(config.crewai_memory_workflows, "analytics,sales_improvement")
        self.assertEqual(config.crewai_memory_storage_path, "artifacts/custom_memory")
        self.assertEqual(config.crewai_memory_embedder_model, "text-embedding-3-large")
        self.assertEqual(config.workflow_context_max_chars, 9000)
        self.assertEqual(config.task_context_max_chars, 2500)


class SharedContextTests(unittest.TestCase):
    def test_compact_context_truncates_oversized_outputs(self) -> None:
        compacted = compact_context(
            "oversized",
            {"report": "x" * 500},
            max_chars=120,
            text_max_chars=80,
        )

        self.assertTrue(compacted.truncated)
        self.assertLessEqual(compacted.compacted_chars, 120)
        self.assertIn("[truncated", compacted.content)

    def test_compact_context_redacts_sensitive_values_and_preserves_evidence(self) -> None:
        compacted = compact_context(
            "evidence",
            {
                "api_key": "sk-secret",
                "claim": "Contact buyer@example.com for source details.",
                "market": "AU",
                "confidence": "high",
                "source_ids": ["S1"],
                "source_urls": ["https://example.com/report"],
                "nested": {"customer_email": "buyer@example.com"},
            },
            max_chars=2000,
        )

        self.assertIn("[REDACTED_SECRET]", compacted.content)
        self.assertIn("[REDACTED_EMAIL]", compacted.content)
        self.assertNotIn("sk-secret", compacted.content)
        self.assertNotIn("buyer@example.com", compacted.content)
        self.assertIn("market: AU", compacted.content)
        self.assertIn("confidence: high", compacted.content)
        self.assertIn("S1", compacted.content)
        self.assertIn("https://example.com/report", compacted.content)

    def test_conversation_history_summary_redacts_old_messages_and_keeps_recent_window(self) -> None:
        history = [
            {"direction": "customer", "text": "old email buyer@example.com"},
            {"direction": "agent", "text": "old phone +1 415 555 1212"},
            {"direction": "customer", "text": "recent question"},
            {"direction": "agent", "text": "recent answer"},
        ]

        context = build_conversation_history_context(history, recent_count=2, message_max_chars=80)

        self.assertEqual(context.total_messages, 4)
        self.assertEqual(context.summarized_messages, 2)
        self.assertEqual(len(context.recent_messages), 2)
        self.assertIn("2 older messages summarized", context.summary)
        serialized = f"{context.summary} {context.recent_messages}"
        self.assertNotIn("buyer@example.com", serialized)
        self.assertNotIn("+1 415 555 1212", serialized)

    def test_support_normalizer_uses_summary_plus_latest_messages(self) -> None:
        history = [
            {"direction": "customer", "text": f"old message {index} user{index}@example.com"}
            for index in range(5)
        ]

        normalized = _normalize_inputs(
            {
                "customer": "Example Store",
                "person": "Alex",
                "inquiry": "Where is my order?",
                "conversation_history": history,
            }
        )

        self.assertEqual(len(normalized["conversation_history"]), 3)
        self.assertEqual(normalized["conversation_history"], normalized["recent_conversation_history"])
        self.assertIn("2 older messages summarized", normalized["conversation_history_summary"])
        self.assertNotIn("user0@example.com", normalized["conversation_history_summary"])
        self.assertNotIn("user4@example.com", str(normalized["recent_conversation_history"]))


if __name__ == "__main__":
    unittest.main()
