from __future__ import annotations

import importlib.util
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "support-inbox-live-e2e"
    / "scripts"
    / "support_inbox_live_e2e_runner.py"
)


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("support_inbox_live_e2e_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict[str, object]:
    return {
        "recipient": "",
        "marker_prefix": "CB-SUPPORT-E2E",
        "suites": {"smoke": ["pre_sales_catalog_headset"]},
        "scenarios": {
            "pre_sales_catalog_headset": {
                "subject": "{marker} Headset question",
                "body": "Hello,\n\nTest marker: {marker}\n\nWhat is the headset price?",
            }
        },
    }


class SupportInboxLiveE2ESkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _load_runner()

    def test_print_messages_uses_gmail_sender_email_from_environment(self) -> None:
        manifest = _manifest()
        with patch.dict(os.environ, {"GMAIL_SENDER_EMAIL": "test-support@example.com"}):
            resolved = self.runner.resolve_recipient(manifest)

        output = io.StringIO()
        with redirect_stdout(output):
            self.runner.print_messages(
                manifest,
                "20260620-000000",
                "smoke",
                "",
                ["pre_sales_catalog_headset"],
                resolved,
            )

        text = output.getvalue()
        self.assertIn("Recipient: test-support@example.com", text)
        self.assertIn("To: test-support@example.com", text)

    def test_cli_recipient_overrides_gmail_sender_email(self) -> None:
        with patch.dict(os.environ, {"GMAIL_SENDER_EMAIL": "test-support@example.com"}):
            resolved = self.runner.resolve_recipient(
                _manifest(),
                explicit_recipient="override@example.com",
            )

        self.assertEqual(resolved, "override@example.com")

    def test_missing_recipient_fails_with_clear_message(self) -> None:
        with patch.dict(os.environ, {"GMAIL_SENDER_EMAIL": ""}):
            with self.assertRaisesRegex(ValueError, "GMAIL_SENDER_EMAIL"):
                self.runner.resolve_recipient(_manifest())


if __name__ == "__main__":
    unittest.main()
