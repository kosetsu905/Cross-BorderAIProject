import base64
import json
import unittest
from unittest.mock import Mock, patch

from tools.custom.gmail_tools import (
    build_gmail_reply_raw_message,
    gmail_label_ids,
    list_gmail_messages,
    parse_gmail_message,
    parse_gmail_pubsub_payload,
    refresh_gmail_access_token,
    resolve_gmail_access_token,
)


def _b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


class GmailOmniChannelTests(unittest.TestCase):
    def test_parse_plain_text_message(self) -> None:
        message = {
            "id": "msg-1",
            "threadId": "thread-1",
            "snippet": "snippet",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [
                    {"name": "From", "value": "Maria Chen <maria@example.com>"},
                    {"name": "To", "value": "Support <support@example.com>"},
                    {"name": "Subject", "value": "Order status"},
                    {"name": "Message-ID", "value": "<msg-1@example.com>"},
                    {"name": "Date", "value": "Wed, 27 May 2026 10:00:00 +1000"},
                ],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _b64("Where is my order?")},
                    }
                ],
            },
        }

        parsed = parse_gmail_message(message, mailbox_email="support@example.com")

        self.assertEqual(parsed.channel, "gmail")
        self.assertEqual(parsed.channel_thread_id, "thread-1")
        self.assertEqual(parsed.channel_message_id, "msg-1")
        self.assertEqual(parsed.sender, "maria@example.com")
        self.assertEqual(parsed.recipient, "support@example.com")
        self.assertEqual(parsed.text, "Where is my order?")
        self.assertEqual(parsed.sender_profile["display_name"], "Maria Chen")
        self.assertEqual(parsed.sender_profile["subject"], "Order status")

    def test_parse_html_fallback_and_attachment_metadata(self) -> None:
        message = {
            "id": "msg-html",
            "threadId": "thread-html",
            "payload": {
                "headers": [
                    {"name": "From", "value": "maria@example.com"},
                    {"name": "Subject", "value": "Damaged item"},
                ],
                "parts": [
                    {
                        "mimeType": "text/html",
                        "body": {"data": _b64("<p>The item is <b>damaged</b>.</p>")},
                    },
                    {
                        "filename": "photo.jpg",
                        "mimeType": "image/jpeg",
                        "body": {"attachmentId": "att-1", "size": 1234},
                    },
                ],
            },
        }

        parsed = parse_gmail_message(message)

        self.assertEqual(parsed.text, "The item is  damaged .")
        self.assertEqual(parsed.attachments[0]["filename"], "photo.jpg")
        self.assertEqual(parsed.attachments[0]["attachment_id"], "att-1")

    def test_parse_pubsub_payload(self) -> None:
        data = _b64(json.dumps({"emailAddress": "support@example.com", "historyId": "12345"}))

        parsed = parse_gmail_pubsub_payload({"message": {"data": data}})

        self.assertEqual(parsed["email_address"], "support@example.com")
        self.assertEqual(parsed["history_id"], "12345")

    def test_label_ids(self) -> None:
        self.assertEqual(gmail_label_ids("INBOX, CATEGORY_PERSONAL"), ["INBOX", "CATEGORY_PERSONAL"])

    def test_reply_raw_message_contains_headers(self) -> None:
        raw = build_gmail_reply_raw_message(
            sender="support@example.com",
            recipient="maria@example.com",
            subject="Order status",
            body="Thanks, we are checking.",
            in_reply_to="<msg-1@example.com>",
            references="<root@example.com>",
        )
        decoded = base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8")

        self.assertIn("Subject: Re: Order status", decoded)
        self.assertIn("In-Reply-To: <msg-1@example.com>", decoded)
        self.assertIn("References: <root@example.com> <msg-1@example.com>", decoded)

    @patch("tools.custom.gmail_tools.httpx.post")
    def test_refresh_access_token(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"access_token": "new-token", "expires_in": 3600}
        response.raise_for_status.return_value = None
        post.return_value = response

        result = refresh_gmail_access_token(
            client_id="client",
            client_secret="secret",
            refresh_token="refresh",
        )

        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(result["access_token"], "new-token")
        post.assert_called_once()

    @patch("tools.custom.gmail_tools.refresh_gmail_access_token")
    def test_resolve_access_token_prefers_refresh_credentials(self, refresh: Mock) -> None:
        refresh.return_value = {"access_token": "fresh"}

        token = resolve_gmail_access_token(
            access_token="old",
            client_id="client",
            client_secret="secret",
            refresh_token="refresh",
        )

        self.assertEqual(token, "fresh")

    def test_resolve_access_token_falls_back_to_static_token(self) -> None:
        self.assertEqual(resolve_gmail_access_token(access_token="static"), "static")

    @patch("tools.custom.gmail_tools.httpx.get")
    def test_list_gmail_messages(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {"messages": [{"id": "msg-1", "threadId": "thread-1"}]}
        response.raise_for_status.return_value = None
        get.return_value = response

        messages = list_gmail_messages(
            access_token="token",
            label_ids=["INBOX"],
            max_results=5,
        )

        self.assertEqual(messages, [{"id": "msg-1", "threadId": "thread-1"}])
        self.assertEqual(get.call_args.kwargs["params"]["labelIds"], ["INBOX"])


if __name__ == "__main__":
    unittest.main()
