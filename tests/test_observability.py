import os
import unittest
from unittest.mock import patch

from utils.observability import NoOpSpan, redact_observability_payload, workflow_span


class ObservabilityTests(unittest.TestCase):
    def test_workflow_span_is_noop_when_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with workflow_span("analytics", job_id="job-1", backend="local") as span:
                self.assertIsInstance(span, NoOpSpan)

    def test_redacts_secrets_and_pii_like_fields(self) -> None:
        payload = {
            "api_key": "sk-secret",
            "customer_email": "buyer@example.com",
            "phone_number": "+1 555 123 4567",
            "nested": {
                "authorization": "Bearer token",
                "message": "Contact buyer@example.com at +1 555 123 4567",
            },
        }

        redacted = redact_observability_payload(payload, capture_raw=False)

        self.assertEqual(redacted["api_key"], "[REDACTED_SECRET]")
        self.assertEqual(redacted["customer_email"], "[REDACTED_PII]")
        self.assertEqual(redacted["phone_number"], "[REDACTED_PII]")
        self.assertEqual(redacted["nested"]["authorization"], "[REDACTED_SECRET]")
        self.assertIn("[REDACTED_EMAIL]", redacted["nested"]["message"])
        self.assertIn("[REDACTED_PHONE]", redacted["nested"]["message"])


if __name__ == "__main__":
    unittest.main()
