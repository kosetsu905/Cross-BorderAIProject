import hashlib
import hmac
import unittest
from datetime import UTC, datetime

from models import SupportInputs
from support_inbox import _json_safe, mask_contact
from tools.custom.whatsapp_tools import parse_whatsapp_webhook, verify_whatsapp_signature


class WhatsAppOmniChannelTests(unittest.TestCase):
    def test_signature_verification_accepts_valid_signature(self) -> None:
        body = b'{"object":"whatsapp_business_account"}'
        secret = "app-secret"
        digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

        self.assertTrue(
            verify_whatsapp_signature(
                app_secret=secret,
                body=body,
                signature=f"sha256={digest}",
            )
        )

    def test_signature_verification_rejects_invalid_signature(self) -> None:
        self.assertFalse(
            verify_whatsapp_signature(
                app_secret="app-secret",
                body=b"{}",
                signature="sha256=bad",
            )
        )

    def test_parse_inbound_text_message(self) -> None:
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "metadata": {"display_phone_number": "15551234567"},
                                "contacts": [
                                    {"wa_id": "61412345678", "profile": {"name": "Maria"}}
                                ],
                                "messages": [
                                    {
                                        "from": "61412345678",
                                        "id": "wamid.1",
                                        "timestamp": "1779878400",
                                        "type": "text",
                                        "text": {"body": "Where is my order?"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        messages, statuses = parse_whatsapp_webhook(payload)

        self.assertEqual(statuses, [])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].channel, "whatsapp")
        self.assertEqual(messages[0].channel_thread_id, "61412345678")
        self.assertEqual(messages[0].text, "Where is my order?")
        self.assertEqual(messages[0].sender_profile["profile"]["name"], "Maria")

    def test_parse_media_message_uses_attachment_placeholder(self) -> None:
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "metadata": {"phone_number_id": "123"},
                                "messages": [
                                    {
                                        "from": "61412345678",
                                        "id": "wamid.media",
                                        "timestamp": "1779878400",
                                        "type": "image",
                                        "image": {"id": "media-id", "mime_type": "image/jpeg"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        messages, _ = parse_whatsapp_webhook(payload)

        self.assertEqual(messages[0].text, None)
        self.assertEqual(messages[0].attachments[0]["type"], "image")
        self.assertEqual(messages[0].attachments[0]["id"], "media-id")

    def test_support_inputs_accept_channel_fields(self) -> None:
        parsed = SupportInputs.model_validate(
            {
                "customer": "Maria",
                "person": "Maria",
                "inquiry": "Where is my order?",
                "channel": "whatsapp",
                "channel_thread_id": "61412345678",
                "channel_message_id": "wamid.1",
                "sender_profile": {"display_name": "Maria"},
                "attachments": [],
                "conversation_history": [{"direction": "inbound", "text": "Where is my order?"}],
            }
        )

        self.assertEqual(parsed.channel, "whatsapp")
        self.assertEqual(parsed.channel_message_id, "wamid.1")

    def test_mask_contact_hides_phone_and_email(self) -> None:
        self.assertEqual(mask_contact("+61 412 345 678"), "+6***5678")
        self.assertEqual(mask_contact("maria@example.com"), "ma***@example.com")

    def test_json_safe_converts_datetime_for_crew_inputs(self) -> None:
        payload = {"created_at": datetime(2026, 5, 27, 1, 2, 3, tzinfo=UTC)}

        self.assertEqual(_json_safe(payload), {"created_at": "2026-05-27T01:02:03+00:00"})


if __name__ == "__main__":
    unittest.main()
