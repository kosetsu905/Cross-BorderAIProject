import asyncio
import hashlib
import hmac
import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import yaml

from api.routes import _send_whatsapp_approval_delivery
from crews.support_crew import _normalize_inputs
from models import SupportInputs
from runtime_config import load_runtime_config
from services.language_detector import LanguageDetector
from services.session_manager import AsyncSessionManager, SessionManager
from services.support_auto_dispatch import _send_whatsapp_auto_reply, process_completed_support_job
from services.whatsapp_provider import (
    MetaCloudWhatsAppProvider,
    YCloudWhatsAppProvider,
    get_whatsapp_provider,
    send_rma_label_message,
)
from services.whatsapp_tmpl_mgr import WhatsAppTemplateManager
from support_inbox import SupportInboxStore, _json_safe, mask_contact
from tools.custom.support_automation_tools import detect_language
from tools.custom.support_handoff_tools import send_handoff_notification
from tools.custom.whatsapp_tools import parse_whatsapp_webhook, verify_whatsapp_signature


class FakeSupportSession:
    def __init__(self, conversation: SimpleNamespace, records: list[SimpleNamespace] | None = None) -> None:
        self.conversation = conversation
        self.records = records or []
        self.added: list[object] = []
        self.commit_count = 0

    def get(self, model: object, key: str) -> SimpleNamespace | None:
        if key == self.conversation.conversation_id:
            return self.conversation
        return None

    def add(self, record: object) -> None:
        self.added.append(record)

    def commit(self) -> None:
        self.commit_count += 1

    def refresh(self, record: object) -> None:
        return None

    def execute(self, statement: object) -> object:
        return SimpleNamespace(scalars=lambda: FakeScalarResult(self.records))


class FakeScalarResult:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self.records = records

    def __iter__(self) -> object:
        return iter(self.records)

    def first(self) -> SimpleNamespace | None:
        return self.records[0] if self.records else None


class FakeSessionContext:
    def __init__(self, session: FakeSupportSession) -> None:
        self.session = session

    def __enter__(self) -> FakeSupportSession:
        return self.session

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.ttls[key] = ttl
        self.values[key] = value


class FakeAsyncRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.ttls[key] = ttl
        self.values[key] = value


class FakeAsyncResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, text=self.text)
            request = httpx.Request("GET", "https://api.example.test")
            response.request = request
            raise httpx.HTTPStatusError(str(self.status_code), request=request, response=response)
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeAsyncClient:
    def __init__(
        self,
        *,
        post_response: dict[str, object] | None = None,
        get_response: dict[str, object] | None = None,
        get_status_code: int = 200,
    ) -> None:
        self.post_response = post_response or {}
        self.get_response = get_response or {}
        self.get_status_code = get_status_code
        self.posts: list[dict[str, object]] = []
        self.gets: list[dict[str, object]] = []

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]) -> FakeAsyncResponse:
        self.posts.append({"url": url, "json": json, "headers": headers})
        return FakeAsyncResponse(self.post_response)

    async def get(self, url: str, *, headers: dict[str, str], params: dict[str, str] | None = None) -> FakeAsyncResponse:
        self.gets.append({"url": url, "headers": headers, "params": params or {}})
        return FakeAsyncResponse(self.get_response, self.get_status_code)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object] | None = None,
    ) -> FakeAsyncResponse:
        if method.upper() == "GET":
            self.gets.append({"url": url, "headers": headers, "params": {}})
            return FakeAsyncResponse(self.get_response, self.get_status_code)
        self.posts.append({"url": url, "json": json or {}, "headers": headers})
        return FakeAsyncResponse(self.post_response)


class FakeWindowSessionManager:
    def __init__(self, *, expired: bool, session: dict[str, object] | None = None) -> None:
        self.expired = expired
        self.session = session or {}

    def load_session(self, session_id: str) -> dict[str, object]:
        return self.session

    def is_window_expired(self, session_id: str) -> bool:
        return self.expired


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
                "session_id": "session-123",
                "channel_thread_id": "61412345678",
                "channel_message_id": "wamid.1",
                "sender_profile": {"display_name": "Maria"},
                "attachments": [],
                "conversation_history": [{"direction": "inbound", "text": "Where is my order?"}],
            }
        )

        self.assertEqual(parsed.channel, "whatsapp")
        self.assertEqual(parsed.session_id, "session-123")
        self.assertEqual(parsed.channel_message_id, "wamid.1")

    def test_support_input_normalization_defaults_session_id(self) -> None:
        normalized = _normalize_inputs(
            {
                "customer": "Maria",
                "person": "Maria",
                "inquiry": "商品什么时候到？",
            }
        )

        self.assertEqual(normalized["channel"], "email")
        self.assertEqual(normalized["session_id"], "unknown")
        self.assertEqual(normalized["detected_language"], "zh")
        self.assertEqual(normalized["language_plan"], "Chinese")

    def test_language_detector_handles_short_multilingual_text(self) -> None:
        self.assertEqual(LanguageDetector.detect("商品什么时候到？"), "zh")
        self.assertEqual(LanguageDetector.detect("注文はどこですか？"), "ja")
        self.assertEqual(LanguageDetector.detect("주문 어디?"), "ko")
        self.assertEqual(LanguageDetector.detect("أين طلبي؟"), "ar")
        self.assertEqual(LanguageDetector.detect("", fallback="es"), "es")
        self.assertEqual(LanguageDetector.detect("ok", fallback="fr"), "fr")

    def test_language_detector_maps_crewai_language_plan(self) -> None:
        self.assertEqual(LanguageDetector.get_crewai_language_plan("ja"), "Japanese")
        self.assertEqual(LanguageDetector.get_crewai_language_plan("pt-BR"), "Portuguese")
        self.assertEqual(LanguageDetector.get_crewai_language_plan("xx"), "English")

    def test_support_automation_language_detection_uses_service(self) -> None:
        self.assertEqual(detect_language("商品什么时候到？"), "zh")
        self.assertEqual(detect_language("注文はどこですか？"), "ja")

    def test_support_task_prompt_includes_channel_adaptation_rules(self) -> None:
        tasks_path = Path(__file__).resolve().parents[1] / "config" / "support" / "tasks.yaml"
        tasks = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))
        resolution_task = tasks["inquiry_resolution"]
        description = resolution_task["description"]

        self.assertIn("{channel}", description)
        self.assertIn("WhatsApp", description)
        self.assertIn("Email/Gmail", description)
        self.assertIn("WebChat/Social", description)
        self.assertIn("<2000 chars", description)
        self.assertIn("channel-optimized draft response", resolution_task["expected_output"])

    def test_mask_contact_hides_phone_and_email(self) -> None:
        self.assertEqual(mask_contact("+61 412 345 678"), "+6***5678")
        self.assertEqual(mask_contact("maria@example.com"), "ma***@example.com")

    def test_json_safe_converts_datetime_for_crew_inputs(self) -> None:
        payload = {"created_at": datetime(2026, 5, 27, 1, 2, 3, tzinfo=UTC)}

        self.assertEqual(_json_safe(payload), {"created_at": "2026-05-27T01:02:03+00:00"})

    def test_runtime_config_supports_deployment_env_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WHATSAPP_TOKEN": "legacy-whatsapp-token",
                "WHATSAPP_PHONE_ID": "legacy-phone-id",
                "SLACK_WEBHOOK_URL": "https://hooks.example.test/legacy",
            },
            clear=True,
        ):
            config = load_runtime_config()

        self.assertEqual(config.whatsapp_access_token, "legacy-whatsapp-token")
        self.assertEqual(config.whatsapp_phone_number_id, "legacy-phone-id")
        self.assertEqual(config.support_handoff_webhook_url, "https://hooks.example.test/legacy")

    def test_runtime_config_prefers_current_env_names_over_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WHATSAPP_ACCESS_TOKEN": "current-whatsapp-token",
                "WHATSAPP_TOKEN": "legacy-whatsapp-token",
                "WHATSAPP_PHONE_NUMBER_ID": "current-phone-id",
                "WHATSAPP_PHONE_ID": "legacy-phone-id",
                "SUPPORT_HANDOFF_WEBHOOK_URL": "https://hooks.example.test/current",
                "SLACK_WEBHOOK_URL": "https://hooks.example.test/legacy",
            },
            clear=True,
        ):
            config = load_runtime_config()

        self.assertEqual(config.whatsapp_access_token, "current-whatsapp-token")
        self.assertEqual(config.whatsapp_phone_number_id, "current-phone-id")
        self.assertEqual(config.support_handoff_webhook_url, "https://hooks.example.test/current")

    def test_runtime_config_loads_session_manager_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SUPPORT_SESSION_REDIS_URL": "redis://session-cache:6379/2",
                "SUPPORT_SESSION_TTL_SECONDS": "3600",
                "SUPPORT_SESSION_HISTORY_LIMIT": "7",
            },
            clear=True,
        ):
            config = load_runtime_config()

        self.assertEqual(config.support_session_redis_url, "redis://session-cache:6379/2")
        self.assertEqual(config.support_session_ttl_seconds, 3600)
        self.assertEqual(config.support_session_history_limit, 7)

    def test_runtime_config_session_settings_use_defaults_for_invalid_numbers(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CELERY_BROKER_URL": "redis://broker:6379/0",
                "SUPPORT_SESSION_TTL_SECONDS": "not-a-number",
                "SUPPORT_SESSION_HISTORY_LIMIT": "also-bad",
            },
            clear=True,
        ):
            config = load_runtime_config()

        self.assertEqual(config.support_session_redis_url, "redis://broker:6379/0")
        self.assertEqual(config.support_session_ttl_seconds, 86400)
        self.assertEqual(config.support_session_history_limit, 20)

    def test_runtime_config_loads_ycloud_provider_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WHATSAPP_PROVIDER": "ycloud",
                "YCLOUD_API_KEY": "yc-key",
                "YCLOUD_WHATSAPP_FROM": "+15551234567",
                "YCLOUD_WABA_ID": "waba-1",
                "YCLOUD_BASE_URL": "https://api.example.test/v2",
            },
            clear=True,
        ):
            config = load_runtime_config()

        self.assertEqual(config.whatsapp_provider, "ycloud")
        self.assertEqual(config.ycloud_api_key, "yc-key")
        self.assertEqual(config.ycloud_whatsapp_from, "+15551234567")
        self.assertEqual(config.ycloud_waba_id, "waba-1")
        self.assertEqual(config.ycloud_base_url, "https://api.example.test/v2")

    def test_session_manager_records_inbound_session_state(self) -> None:
        redis_client = FakeRedis()
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=5, redis_client=redis_client)

        result = manager.record_inbound_message(
            session_id="conv-1",
            channel="whatsapp",
            customer_id="+61412345678",
            language_preference="en",
            metadata={"channel_thread_id": "61412345678"},
            message={"message_id": "msg-1", "text": "Where is my order?"},
        )

        self.assertEqual(result["status"], "ok")
        stored = manager.load_session("conv-1")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["session_id"], "conv-1")
        self.assertEqual(stored["language_preference"], "en")
        self.assertEqual(stored["metadata"]["channel_thread_id"], "61412345678")
        self.assertEqual(stored["history"][0]["direction"], "inbound")
        self.assertIn("window_expiry", stored)

    def test_session_manager_rotates_history_to_limit(self) -> None:
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=2, redis_client=FakeRedis())

        for index in range(3):
            manager.record_inbound_message(
                session_id="conv-1",
                channel="whatsapp",
                customer_id="+61412345678",
                message={"message_id": f"msg-{index}", "text": f"message {index}"},
            )

        stored = manager.load_session("conv-1")
        self.assertEqual([item["message_id"] for item in stored["history"]], ["msg-1", "msg-2"])

    def test_session_manager_compat_methods_record_turns_and_window_status(self) -> None:
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=5, redis_client=FakeRedis())

        self.assertTrue(manager.is_window_expired("missing"))
        manager.create_or_update(
            "conv-1",
            {"channel": "whatsapp", "customer_id": "+61412345678", "language": "en"},
            "Where is my order?",
        )
        manager.log_ai_response("conv-1", "A support lead will check this.")

        stored = manager.load_session("conv-1")
        self.assertFalse(manager.is_window_expired("conv-1"))
        self.assertEqual([item["role"] for item in stored["history"]], ["user", "ai"])
        self.assertEqual(stored["history"][0]["direction"], "inbound")
        self.assertEqual(stored["history"][1]["direction"], "outbound")

    def test_async_session_manager_records_and_rotates_history(self) -> None:
        async def scenario() -> dict[str, object]:
            manager = AsyncSessionManager(redis_url=None, ttl_seconds=3600, history_limit=2, redis_client=FakeAsyncRedis())
            self.assertTrue(await manager.is_window_expired("missing"))
            await manager.create_or_update(
                "conv-1",
                {"channel": "whatsapp", "customer_id": "+61412345678", "language": "en"},
                "First user turn",
            )
            await manager.create_or_update(
                "conv-1",
                {"channel": "whatsapp", "customer_id": "+61412345678", "language": "en"},
                "Second user turn",
            )
            await manager.log_ai_response("conv-1", "AI reply")
            return {
                "expired": await manager.is_window_expired("conv-1"),
                "session": await manager.load_session("conv-1"),
            }

        result = asyncio.run(scenario())
        stored = result["session"]

        self.assertFalse(result["expired"])
        self.assertEqual([item["content"] for item in stored["history"]], ["Second user turn", "AI reply"])
        self.assertEqual(stored["language_preference"], "en")
        self.assertIn("window_expiry", stored)

    def test_build_support_inputs_prefers_redis_session_history(self) -> None:
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=5, redis_client=FakeRedis())
        manager.record_inbound_message(
            session_id="conv-1",
            channel="whatsapp",
            customer_id="+61412345678",
            language_preference="ja",
            message={"message_id": "redis-msg", "text": "Redis history"},
        )
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            channel_thread_id="61412345678",
            customer_display_name="Maria",
            customer_handle="+61412345678",
            customer_handle_masked="+6***5678",
        )
        message = SimpleNamespace(
            channel_message_id="wamid.1",
            text="Where is my order?",
            attachments=[],
        )
        db_record = SimpleNamespace(
            message_id="db-msg",
            conversation_id="conv-1",
            channel="whatsapp",
            channel_thread_id="61412345678",
            channel_message_id="db-msg",
            direction="inbound",
            sender_masked="+6***5678",
            recipient_masked=None,
            text="DB history",
            attachments=[],
            locale=None,
            status="received",
            provider_status=None,
            job_id=None,
            received_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
            raw_payload={},
        )
        store = SupportInboxStore(FakeSupportSession(conversation, [db_record]), session_manager=manager)  # type: ignore[arg-type]

        inputs = store.build_support_inputs(conversation, message)  # type: ignore[arg-type]
        stored = manager.load_session("conv-1")

        self.assertEqual(inputs["conversation_history"][0]["message_id"], "redis-msg")
        self.assertEqual(inputs["detected_language"], "en")
        self.assertEqual(inputs["language_plan"], "English")
        self.assertEqual(stored["language_preference"], "en")

    def test_build_support_inputs_falls_back_to_db_history_without_redis(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            channel_thread_id="61412345678",
            customer_display_name="Maria",
            customer_handle="+61412345678",
            customer_handle_masked="+6***5678",
        )
        message = SimpleNamespace(channel_message_id="wamid.1", text="Where is my order?", attachments=[])
        db_record = SimpleNamespace(
            message_id="db-msg",
            conversation_id="conv-1",
            channel="whatsapp",
            channel_thread_id="61412345678",
            channel_message_id="db-msg",
            direction="inbound",
            sender_masked="+6***5678",
            recipient_masked=None,
            text="DB history",
            attachments=[],
            locale=None,
            status="received",
            provider_status=None,
            job_id=None,
            received_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
            raw_payload={},
        )
        manager = SessionManager(redis_url=None)
        store = SupportInboxStore(FakeSupportSession(conversation, [db_record]), session_manager=manager)  # type: ignore[arg-type]

        inputs = store.build_support_inputs(conversation, message)  # type: ignore[arg-type]

        self.assertEqual(inputs["conversation_history"][0]["message_id"], "db-msg")

    def test_build_support_inputs_auto_detects_language_and_updates_session(self) -> None:
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=5, redis_client=FakeRedis())
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            channel_thread_id="61412345678",
            customer_display_name="Maria",
            customer_handle="+61412345678",
            customer_handle_masked="+6***5678",
        )
        message = SimpleNamespace(
            channel_message_id="wamid.1",
            text="商品什么时候到？",
            attachments=[],
            locale=None,
        )
        store = SupportInboxStore(FakeSupportSession(conversation), session_manager=manager)  # type: ignore[arg-type]

        inputs = store.build_support_inputs(conversation, message)  # type: ignore[arg-type]
        stored = manager.load_session("conv-1")

        self.assertEqual(inputs["detected_language"], "zh")
        self.assertEqual(inputs["language_plan"], "Chinese")
        self.assertEqual(stored["language_preference"], "zh")

    def test_build_support_inputs_prefers_current_message_language(self) -> None:
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=5, redis_client=FakeRedis())
        manager.update_language_preference("conv-1", "ja")
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            channel_thread_id="61412345678",
            customer_display_name="Maria",
            customer_handle="+61412345678",
            customer_handle_masked="+6***5678",
        )
        message = SimpleNamespace(
            channel_message_id="wamid.1",
            text="商品什么时候到？",
            attachments=[],
            locale=None,
        )
        store = SupportInboxStore(FakeSupportSession(conversation), session_manager=manager)  # type: ignore[arg-type]

        inputs = store.build_support_inputs(conversation, message)  # type: ignore[arg-type]
        stored = manager.load_session("conv-1")

        self.assertEqual(inputs["detected_language"], "zh")
        self.assertEqual(inputs["language_plan"], "Chinese")
        self.assertEqual(stored["language_preference"], "zh")

    def test_build_support_inputs_current_japanese_overrides_english_session(self) -> None:
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=5, redis_client=FakeRedis())
        manager.update_language_preference("conv-1", "en")
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            channel_thread_id="thread-1",
            customer_display_name="Yuki",
            customer_handle="yuki@example.com",
            customer_handle_masked="yu***@example.com",
        )
        message = SimpleNamespace(
            channel_message_id="msg-1",
            text="この商品の割引はありますか？",
            attachments=[],
            locale=None,
        )
        store = SupportInboxStore(FakeSupportSession(conversation), session_manager=manager)  # type: ignore[arg-type]

        inputs = store.build_support_inputs(conversation, message)  # type: ignore[arg-type]
        stored = manager.load_session("conv-1")

        self.assertEqual(inputs["detected_language"], "ja")
        self.assertEqual(inputs["language_plan"], "Japanese")
        self.assertEqual(stored["language_preference"], "ja")

    def test_record_outbound_message_appends_to_redis_session_history(self) -> None:
        manager = SessionManager(redis_url=None, ttl_seconds=3600, history_limit=5, redis_client=FakeRedis())
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            channel_thread_id="61412345678",
            customer_handle="+61412345678",
            customer_handle_masked="+6***5678",
            status="draft_ready",
        )
        store = SupportInboxStore(FakeSupportSession(conversation), session_manager=manager)  # type: ignore[arg-type]

        store.record_outbound_message(
            conversation_id="conv-1",
            text="Thanks, a support lead will follow up.",
            channel_message_id="wamid.out",
            delivery_status="sent",
            raw_payload={"provider": "test"},
        )

        stored = manager.load_session("conv-1")
        self.assertEqual(stored["history"][0]["direction"], "outbound")
        self.assertEqual(stored["history"][0]["text"], "Thanks, a support lead will follow up.")

    def test_handoff_notification_is_disabled_without_webhook(self) -> None:
        result = send_handoff_notification(
            webhook_url=None,
            session_id="conv-1",
            channel="whatsapp",
            inquiry_text="Please get me a manager.",
            context={"ticket_id": "TKT-1"},
        )

        self.assertEqual(result["status"], "disabled")

    @patch("tools.custom.support_handoff_tools.httpx.post")
    def test_handoff_notification_posts_slack_compatible_payload(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        post.return_value = response

        result = send_handoff_notification(
            webhook_url="https://hooks.example.test/support",
            session_id="conv-1",
            channel="whatsapp",
            inquiry_text="Please get me a manager.",
            context={"ticket_id": "TKT-1"},
        )

        self.assertEqual(result["status"], "sent")
        post.assert_called_once()
        self.assertIn("Omni-Channel Escalation Required", post.call_args.kwargs["json"]["text"])
        self.assertEqual(post.call_args.kwargs["json"]["session_id"], "conv-1")

    def test_whatsapp_template_manager_selects_template_by_language(self) -> None:
        manager = WhatsAppTemplateManager(access_token="token", phone_number_id="phone")

        self.assertEqual(manager.template_for_language("ja")["name"], "support_reengagement_ja")
        self.assertEqual(manager.template_for_language("es-MX")["name"], "support_reengagement_es")
        self.assertEqual(manager.template_for_language("zh")["name"], "support_reengagement")

    def test_whatsapp_template_send_returns_missing_credentials(self) -> None:
        manager = WhatsAppTemplateManager(access_token=None, phone_number_id="phone")

        result = asyncio.run(manager.send_out_of_window_message("+61412345678", "en"))

        self.assertEqual(result["status"], "missing_credentials")

    def test_whatsapp_template_send_posts_expected_payload(self) -> None:
        client = FakeAsyncClient(post_response={"messages": [{"id": "wamid.template"}]})
        manager = WhatsAppTemplateManager(
            access_token="token",
            phone_number_id="phone-id",
            graph_api_version="v23.0",
            http_client_factory=lambda **_: client,
        )

        result = asyncio.run(manager.send_out_of_window_message("+61412345678", "ja"))

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["message_id"], "wamid.template")
        self.assertEqual(client.posts[0]["url"], "https://graph.facebook.com/v23.0/phone-id/messages")
        payload = client.posts[0]["json"]
        self.assertEqual(payload["type"], "template")
        self.assertEqual(payload["template"]["name"], "support_reengagement_ja")
        self.assertEqual(payload["template"]["language"]["code"], "ja")

    def test_whatsapp_template_status_handles_empty_and_present_api_data(self) -> None:
        empty_client = FakeAsyncClient(get_response={"data": []})
        empty_manager = WhatsAppTemplateManager(
            access_token="token",
            phone_number_id="phone-id",
            http_client_factory=lambda **_: empty_client,
        )
        approved_client = FakeAsyncClient(get_response={"data": [{"status": "APPROVED"}]})
        approved_manager = WhatsAppTemplateManager(
            access_token="token",
            phone_number_id="phone-id",
            http_client_factory=lambda **_: approved_client,
        )

        self.assertEqual(asyncio.run(empty_manager.check_template_status("missing")), "NOT_FOUND")
        self.assertEqual(asyncio.run(approved_manager.check_template_status("support_reengagement_en")), "APPROVED")

    def test_whatsapp_template_submit_returns_template_id(self) -> None:
        client = FakeAsyncClient(post_response={"id": "tmpl-123"})
        manager = WhatsAppTemplateManager(
            access_token="token",
            phone_number_id="phone-id",
            http_client_factory=lambda **_: client,
        )

        template_id = asyncio.run(
            manager.submit_template_for_approval(
                name="support_reengagement_en",
                category="UTILITY",
                body_text="A support agent has an update for you.",
                lang="en",
            )
        )

        self.assertEqual(template_id, "tmpl-123")
        payload = client.posts[0]["json"]
        self.assertEqual(payload["category"], "UTILITY")
        self.assertEqual(payload["components"][0]["type"], "BODY")

    def test_whatsapp_provider_factory_defaults_to_ycloud(self) -> None:
        self.assertIsInstance(
            get_whatsapp_provider(
                SimpleNamespace(
                    ycloud_api_key="yc-key",
                    ycloud_whatsapp_from="+15551234567",
                    ycloud_waba_id="waba-1",
                    ycloud_base_url="https://api.example.test/v2",
                )
            ),
            YCloudWhatsAppProvider,
        )

    def test_whatsapp_provider_factory_can_select_meta(self) -> None:
        self.assertIsInstance(get_whatsapp_provider(SimpleNamespace(whatsapp_provider="meta")), MetaCloudWhatsAppProvider)

    def test_whatsapp_provider_factory_returns_ycloud(self) -> None:
        provider = get_whatsapp_provider(
            SimpleNamespace(
                whatsapp_provider="ycloud",
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            )
        )

        self.assertIsInstance(provider, YCloudWhatsAppProvider)

    @patch("services.whatsapp_provider.send_whatsapp_text_message")
    def test_meta_provider_wraps_text_sender(self, send_text: Mock) -> None:
        send_text.return_value = {"status": "sent", "message_id": "wamid.text", "error": None}
        provider = MetaCloudWhatsAppProvider(
            SimpleNamespace(
                whatsapp_access_token="token",
                whatsapp_phone_number_id="phone-id",
                whatsapp_graph_api_version="v23.0",
            )
        )

        result = asyncio.run(provider.send_text_message("+61412345678", "Hello"))

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["provider"], "meta")
        send_text.assert_called_once()

    @patch("services.whatsapp_provider.WhatsAppTemplateManager.from_config")
    def test_meta_provider_wraps_template_manager(self, from_config: Mock) -> None:
        manager = Mock()
        manager.send_template_message = AsyncMock(
            return_value={"status": "sent", "message_id": "wamid.template", "error": None}
        )
        from_config.return_value = manager
        provider = MetaCloudWhatsAppProvider(SimpleNamespace())

        result = asyncio.run(
            provider.send_template_message(
                to="+61412345678",
                template_name="support_reengagement_en",
                language_code="en_US",
            )
        )

        self.assertEqual(result["provider"], "meta")
        manager.send_template_message.assert_awaited_once()

    def test_ycloud_provider_missing_credentials_returns_error(self) -> None:
        provider = YCloudWhatsAppProvider(SimpleNamespace(ycloud_api_key=None, ycloud_whatsapp_from="+15551234567"))

        result = asyncio.run(provider.send_text_message("+61412345678", "Hello"))

        self.assertEqual(result["status"], "missing_credentials")
        self.assertEqual(result["provider"], "ycloud")

    def test_ycloud_provider_posts_text_payload(self) -> None:
        client = FakeAsyncClient(post_response={"id": "ycloud-msg-1"})
        provider = YCloudWhatsAppProvider(
            SimpleNamespace(
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            ),
            http_client_factory=lambda **_: client,
        )

        result = asyncio.run(provider.send_text_message("+61412345678", "Hello"))

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["provider"], "ycloud")
        self.assertEqual(client.posts[0]["url"], "https://api.example.test/v2/whatsapp/messages/sendDirectly")
        self.assertEqual(client.posts[0]["headers"]["X-API-Key"], "yc-key")
        self.assertEqual(client.posts[0]["headers"]["User-Agent"], "CrossBorderAI/1.0 (YCLOUD-Adapter)")
        payload = client.posts[0]["json"]
        self.assertEqual(payload["from"], "+15551234567")
        self.assertEqual(payload["to"], "+61412345678")
        self.assertEqual(payload["type"], "text")

    def test_ycloud_provider_posts_template_payload(self) -> None:
        client = FakeAsyncClient(post_response={"id": "ycloud-template-msg"})
        provider = YCloudWhatsAppProvider(
            SimpleNamespace(
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            ),
            http_client_factory=lambda **_: client,
        )

        result = asyncio.run(
            provider.send_template_message(
                "+61412345678",
                "support_reengagement_en",
                "en_US",
            )
        )

        self.assertEqual(result["message_id"], "ycloud-template-msg")
        payload = client.posts[0]["json"]
        self.assertEqual(payload["type"], "template")
        self.assertEqual(payload["template"]["name"], "support_reengagement_en")
        self.assertEqual(payload["template"]["language"]["code"], "en_US")

    def test_ycloud_provider_normalizes_standard_response_message_id(self) -> None:
        client = FakeAsyncClient(post_response={"success": True, "data": {"message_id": "wamid.ycloud"}})
        provider = YCloudWhatsAppProvider(
            SimpleNamespace(
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            ),
            http_client_factory=lambda **_: client,
        )

        result = asyncio.run(provider.send_text_message("+61412345678", "Hello"))

        self.assertEqual(result["message_id"], "wamid.ycloud")
        self.assertEqual(result["status"], "sent")

    def test_ycloud_provider_posts_document_media_payload(self) -> None:
        client = FakeAsyncClient(post_response={"success": True, "data": {"message_id": "doc-msg"}})
        provider = YCloudWhatsAppProvider(
            SimpleNamespace(
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            ),
            http_client_factory=lambda **_: client,
        )

        result = asyncio.run(
            provider.send_media_message(
                to="+61412345678",
                media_type="document",
                url="https://labels.example.test/rma.pdf",
                filename="RMA_123_Label.pdf",
                caption="Print this label.",
            )
        )

        self.assertEqual(result["message_id"], "doc-msg")
        payload = client.posts[0]["json"]
        self.assertEqual(payload["type"], "document")
        self.assertEqual(payload["document"]["link"], "https://labels.example.test/rma.pdf")
        self.assertEqual(payload["document"]["filename"], "RMA_123_Label.pdf")
        self.assertEqual(payload["document"]["caption"], "Print this label.")

    def test_ycloud_provider_posts_interactive_payload_and_truncates_buttons(self) -> None:
        client = FakeAsyncClient(post_response={"success": True, "data": {"message_id": "interactive-msg"}})
        provider = YCloudWhatsAppProvider(
            SimpleNamespace(
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            ),
            http_client_factory=lambda **_: client,
        )

        result = asyncio.run(
            provider.send_interactive_message(
                to="+61412345678",
                body="Choose an option",
                buttons=[
                    {"id": "track", "title": "Track"},
                    {"id": "return", "title": "Return"},
                    {"id": "agent", "title": "Agent"},
                    {"id": "extra", "title": "Extra"},
                ],
            )
        )

        self.assertEqual(result["message_id"], "interactive-msg")
        buttons = client.posts[0]["json"]["interactive"]["action"]["buttons"]
        self.assertEqual(len(buttons), 3)
        self.assertEqual(buttons[0]["reply"]["id"], "track")
        self.assertEqual(buttons[2]["reply"]["title"], "Agent")

    def test_send_rma_label_uses_media_inside_window_and_template_outside(self) -> None:
        provider = Mock()
        provider.send_media_message = AsyncMock(return_value={"status": "sent", "message_id": "doc-msg"})
        provider.send_template_message = AsyncMock(return_value={"status": "sent", "message_id": "tmpl-msg"})

        inside = asyncio.run(
            send_rma_label_message(
                provider=provider,
                to="+61412345678",
                label_url="https://labels.example.test/rma.pdf",
                order_id="ORDER-123",
                is_window_expired=False,
            )
        )
        outside = asyncio.run(
            send_rma_label_message(
                provider=provider,
                to="+61412345678",
                label_url="https://labels.example.test/rma.pdf",
                order_id="ORDER-123",
                is_window_expired=True,
            )
        )

        self.assertEqual(inside["message_id"], "doc-msg")
        self.assertEqual(outside["message_id"], "tmpl-msg")
        provider.send_media_message.assert_awaited_once()
        provider.send_template_message.assert_awaited_once()

    def test_ycloud_provider_template_status_and_create(self) -> None:
        not_found_client = FakeAsyncClient(get_response={}, get_status_code=404)
        not_found_provider = YCloudWhatsAppProvider(
            SimpleNamespace(
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            ),
            http_client_factory=lambda **_: not_found_client,
        )
        approved_client = FakeAsyncClient(get_response={"status": "APPROVED"}, post_response={"id": "tmpl-123"})
        approved_provider = YCloudWhatsAppProvider(
            SimpleNamespace(
                ycloud_api_key="yc-key",
                ycloud_whatsapp_from="+15551234567",
                ycloud_waba_id="waba-1",
                ycloud_base_url="https://api.example.test/v2",
            ),
            http_client_factory=lambda **_: approved_client,
        )

        self.assertEqual(asyncio.run(not_found_provider.check_template_status("missing", "en_US")), "NOT_FOUND")
        self.assertEqual(asyncio.run(approved_provider.check_template_status("support_reengagement_en", "en_US")), "APPROVED")
        template_id = asyncio.run(
            approved_provider.submit_template_for_approval(
                "support_reengagement_en",
                "UTILITY",
                "A support agent has an update for you.",
                "en",
            )
        )
        self.assertEqual(template_id, "tmpl-123")
        self.assertEqual(approved_client.posts[0]["url"], "https://api.example.test/v2/whatsapp/templates")

    @patch("api.routes.get_whatsapp_provider")
    def test_whatsapp_approval_uses_template_when_window_expired(self, provider_factory: Mock) -> None:
        provider = Mock(provider_name="meta")
        provider.send_template_message = AsyncMock(
            return_value={"status": "sent", "message_id": "wamid.template", "error": None}
        )
        provider_factory.return_value = provider
        store = SimpleNamespace(session_manager=FakeWindowSessionManager(expired=True, session={"language_preference": "ja"}))
        conversation = SimpleNamespace(
            customer_handle="+61412345678",
            draft_payload={"detected_language": "es"},
        )

        delivery, raw_payload = asyncio.run(
            _send_whatsapp_approval_delivery(
                config=SimpleNamespace(),
                store=store,  # type: ignore[arg-type]
                conversation=conversation,  # type: ignore[arg-type]
                conversation_id="conv-1",
                draft_text="Thanks for your message.",
            )
        )

        self.assertEqual(delivery["status"], "sent")
        self.assertEqual(raw_payload["send_mode"], "template")
        self.assertEqual(raw_payload["provider"], "meta")
        provider.send_template_message.assert_awaited_once()

    @patch("api.routes.get_whatsapp_provider")
    def test_whatsapp_approval_uses_text_when_window_is_active(self, provider_factory: Mock) -> None:
        provider = Mock(provider_name="ycloud")
        provider.send_text_message = AsyncMock(
            return_value={"status": "sent", "message_id": "wamid.text", "error": None, "provider": "ycloud"}
        )
        provider_factory.return_value = provider
        store = SimpleNamespace(session_manager=FakeWindowSessionManager(expired=False, session={"language_preference": "ja"}))
        conversation = SimpleNamespace(customer_handle="+61412345678", draft_payload={})

        delivery, raw_payload = asyncio.run(
            _send_whatsapp_approval_delivery(
                config=SimpleNamespace(
                    whatsapp_access_token="token",
                    whatsapp_phone_number_id="phone-id",
                    whatsapp_graph_api_version="v23.0",
                ),
                store=store,  # type: ignore[arg-type]
                conversation=conversation,  # type: ignore[arg-type]
                conversation_id="conv-1",
                draft_text="Thanks for your message.",
            )
        )

        self.assertEqual(delivery["message_id"], "wamid.text")
        self.assertEqual(raw_payload["send_mode"], "text")
        self.assertEqual(raw_payload["provider"], "ycloud")
        provider.send_text_message.assert_awaited_once()

    @patch("support_inbox.load_runtime_config")
    def test_sync_job_result_records_disabled_handoff_notification(self, load_config: Mock) -> None:
        load_config.return_value = SimpleNamespace(support_handoff_webhook_url=None)
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "ticket_id": "TKT-1",
                    "ticket_summary": "VIP customer is asking for a manager.",
                    "drafted_response": "A human support lead will follow up.",
                    "escalation_flag": True,
                    "channel_recommended_action": "human_handoff",
                },
            },
        )

        self.assertEqual(conversation.status, "handoff_required")
        self.assertTrue(conversation.escalation_flag)
        self.assertEqual(conversation.draft_payload["handoff_notification"]["status"], "disabled")

    @patch("support_inbox.load_runtime_config")
    @patch("support_inbox.send_handoff_notification")
    def test_sync_job_result_sends_handoff_notification_once(self, notify: Mock, load_config: Mock) -> None:
        load_config.return_value = SimpleNamespace(support_handoff_webhook_url="https://hooks.example.test/support")
        notify.return_value = {"status": "sent", "error": None}
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]
        job_data = {
            "job_id": "job-1",
            "workflow_type": "support",
            "result": {
                "ticket_id": "TKT-1",
                "ticket_summary": "VIP customer is asking for a manager.",
                "drafted_response": "A human support lead will follow up.",
                "escalation_flag": True,
                "channel_recommended_action": "human_handoff",
            },
        }

        store.sync_job_result("conv-1", job_data)
        store.sync_job_result("conv-1", job_data)

        notify.assert_called_once()
        self.assertEqual(conversation.draft_payload["handoff_notification"]["status"], "sent")

    def test_sync_job_result_respects_low_risk_auto_send_decision(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "ticket_id": "conv-1",
                    "ticket_summary": "Simple shipping update request.",
                    "drafted_response": "Your order is on the way.",
                    "requires_approval": False,
                    "escalation_flag": False,
                    "channel_recommended_action": "auto_send",
                    "sentiment_analysis": {
                        "customer_tier": "STANDARD",
                        "sentiment_score": 0.2,
                        "intent_category": "SHIPPING_INQUIRY",
                    },
                },
            },
        )

        self.assertFalse(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)

    def test_sync_job_result_forces_vip_review_even_when_crew_allows_auto_send(self) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="whatsapp",
            draft_response=None,
            draft_payload=None,
            requires_approval=False,
            escalation_flag=False,
            status="processing",
        )
        store = SupportInboxStore(FakeSupportSession(conversation))  # type: ignore[arg-type]

        store.sync_job_result(
            "conv-1",
            {
                "job_id": "job-1",
                "workflow_type": "support",
                "result": {
                    "ticket_id": "conv-1",
                    "drafted_response": "I can help with that.",
                    "requires_approval": False,
                    "escalation_flag": False,
                    "channel_recommended_action": "auto_send",
                    "sentiment_analysis": {
                        "customer_tier": "VIP",
                        "sentiment_score": 0.5,
                        "intent_category": "SHIPPING_INQUIRY",
                    },
                },
            },
        )

        self.assertTrue(conversation.requires_approval)

    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    @patch("services.support_auto_dispatch.SessionLocal")
    def test_auto_dispatch_sends_low_risk_gmail_reply(self, session_local: Mock, send_gmail: Mock) -> None:
        send_gmail.return_value = {"status": "sent", "message_id": "gmail-out-1", "error": None}
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            channel_thread_id="gmail-thread-1",
            customer_handle="maria@example.com",
            customer_handle_masked="ma***@example.com",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        fake_session = FakeSupportSession(conversation)
        session_local.return_value = FakeSessionContext(fake_session)

        result = asyncio.run(
            process_completed_support_job(
                job_id="job-1",
                inputs={"session_id": "conv-1"},
                result={
                    "ticket_id": "conv-1",
                    "drafted_response": "Your order is on the way.",
                    "requires_approval": False,
                    "escalation_flag": False,
                    "channel_recommended_action": "auto_send",
                    "sentiment_analysis": {
                        "customer_tier": "STANDARD",
                        "sentiment_score": 0.1,
                        "intent_category": "SHIPPING_INQUIRY",
                    },
                },
                config_context={
                    "gmail_send_enabled": True,
                    "gmail_access_token": "token",
                    "gmail_sender_email": "support@example.com",
                },
            )
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(fake_session.added[-1].raw_payload["auto_dispatch"], True)
        self.assertEqual(fake_session.added[-1].raw_payload["send_mode"], "text")
        send_gmail.assert_called_once()

    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    @patch("services.support_auto_dispatch.SessionLocal")
    def test_auto_dispatch_records_disabled_gmail_for_high_confidence_pre_sales(self, session_local: Mock, send_gmail: Mock) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            channel_thread_id="gmail-thread-1",
            customer_handle="maria@example.com",
            customer_handle_masked="ma***@example.com",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        fake_session = FakeSupportSession(conversation)
        session_local.return_value = FakeSessionContext(fake_session)

        result = asyncio.run(
            process_completed_support_job(
                job_id="job-1",
                inputs={"session_id": "conv-1"},
                result={
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.95,
                    "final_response": "The catalog item is available.",
                    "qa_status": "REVIEW_REQUIRED",
                    "escalation_needed": True,
                    "pre_sales_response": {"requires_human_review": True},
                },
                config_context={"gmail_send_enabled": False},
            )
        )

        self.assertEqual(result["status"], "disabled")
        self.assertEqual(fake_session.added[-1].status, "disabled")
        self.assertEqual(fake_session.added[-1].raw_payload["auto_dispatch"], True)
        self.assertEqual(fake_session.added[-1].raw_payload["delivery"]["status"], "disabled")
        self.assertFalse(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)
        send_gmail.assert_not_called()

    @patch("services.support_auto_dispatch.SessionLocal")
    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    def test_auto_dispatch_skips_low_confidence_pre_sales(self, send_gmail: Mock, session_local: Mock) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            channel_thread_id="gmail-thread-1",
            customer_handle="maria@example.com",
            customer_handle_masked="ma***@example.com",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        session_local.return_value = FakeSessionContext(FakeSupportSession(conversation))

        result = asyncio.run(
            process_completed_support_job(
                job_id="job-1",
                inputs={"session_id": "conv-1"},
                result={
                    "session_id": "conv-1",
                    "detected_intent": "pre_sales",
                    "routing_confidence": 0.74,
                    "final_response": "A specialist should review this.",
                    "qa_status": "REVIEW_REQUIRED",
                    "escalation_needed": True,
                },
                config_context={"gmail_send_enabled": True},
            )
        )

        self.assertEqual(result["reason"], "requires_approval")
        self.assertTrue(conversation.requires_approval)
        self.assertFalse(conversation.escalation_flag)
        send_gmail.assert_not_called()

    @patch("services.support_auto_dispatch.get_whatsapp_provider")
    def test_auto_dispatch_uses_whatsapp_text_when_window_active(self, provider_factory: Mock) -> None:
        provider = Mock(provider_name="meta")
        provider.send_text_message = AsyncMock(
            return_value={"status": "sent", "message_id": "wamid.text", "error": None, "provider": "meta"}
        )
        provider_factory.return_value = provider
        store = SimpleNamespace(session_manager=FakeWindowSessionManager(expired=False, session={"language_preference": "en"}))
        conversation = SimpleNamespace(customer_handle="+61412345678", draft_payload={})

        delivery, raw_payload = asyncio.run(
            _send_whatsapp_auto_reply(
                config=SimpleNamespace(
                    whatsapp_send_enabled=True,
                    whatsapp_access_token="token",
                    whatsapp_phone_number_id="phone-id",
                    whatsapp_graph_api_version="v23.0",
                ),
                store=store,  # type: ignore[arg-type]
                conversation=conversation,  # type: ignore[arg-type]
                conversation_id="conv-1",
                draft_text="Your order is on the way.",
                job_id="job-1",
            )
        )

        self.assertEqual(delivery["message_id"], "wamid.text")
        self.assertEqual(raw_payload["auto_dispatch"], True)
        self.assertEqual(raw_payload["send_mode"], "text")
        self.assertEqual(raw_payload["provider"], "meta")
        provider.send_text_message.assert_awaited_once()

    @patch("services.support_auto_dispatch.get_whatsapp_provider")
    def test_auto_dispatch_uses_whatsapp_template_when_window_expired(self, provider_factory: Mock) -> None:
        provider = Mock(provider_name="ycloud")
        provider.send_template_message = AsyncMock(
            return_value={"status": "sent", "message_id": "wamid.template", "error": None, "provider": "ycloud"}
        )
        provider_factory.return_value = provider
        store = SimpleNamespace(session_manager=FakeWindowSessionManager(expired=True, session={"language_preference": "ja"}))
        conversation = SimpleNamespace(
            customer_handle="+61412345678",
            draft_payload={"sentiment_analysis": {"language_detected": "ja"}},
        )

        delivery, raw_payload = asyncio.run(
            _send_whatsapp_auto_reply(
                config=SimpleNamespace(whatsapp_send_enabled=True),
                store=store,  # type: ignore[arg-type]
                conversation=conversation,  # type: ignore[arg-type]
                conversation_id="conv-1",
                draft_text="Your order is on the way.",
                job_id="job-1",
            )
        )

        self.assertEqual(delivery["message_id"], "wamid.template")
        self.assertEqual(raw_payload["send_mode"], "template")
        self.assertEqual(raw_payload["provider"], "ycloud")
        provider.send_template_message.assert_awaited_once()

    @patch("services.support_auto_dispatch.SessionLocal")
    @patch("services.support_auto_dispatch.send_gmail_reply_message")
    def test_auto_dispatch_does_not_send_when_approval_required(self, send_gmail: Mock, session_local: Mock) -> None:
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            channel="gmail",
            channel_thread_id="gmail-thread-1",
            customer_handle="maria@example.com",
            customer_handle_masked="ma***@example.com",
            draft_response=None,
            draft_payload=None,
            requires_approval=True,
            escalation_flag=False,
            status="processing",
        )
        session_local.return_value = FakeSessionContext(FakeSupportSession(conversation))

        result = asyncio.run(
            process_completed_support_job(
                job_id="job-1",
                inputs={"session_id": "conv-1"},
                result={
                    "ticket_id": "conv-1",
                    "drafted_response": "Please review this billing case.",
                    "requires_approval": True,
                    "escalation_flag": False,
                    "sentiment_analysis": {
                        "customer_tier": "STANDARD",
                        "sentiment_score": 0.1,
                        "intent_category": "BILLING_ISSUE",
                    },
                },
                config_context={"gmail_send_enabled": True},
            )
        )

        self.assertEqual(result["reason"], "requires_approval")
        send_gmail.assert_not_called()


if __name__ == "__main__":
    unittest.main()
