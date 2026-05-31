from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Protocol

import httpx
from pydantic import BaseModel, Field, field_validator

from runtime_config import load_runtime_config
from services.whatsapp_tmpl_mgr import WhatsAppTemplateManager
from tools.custom.whatsapp_tools import send_whatsapp_text_message

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover - requirements installs tenacity, this keeps imports degradable.
    def retry(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda fn: fn

    def retry_if_exception_type(*args: Any, **kwargs: Any) -> None:
        return None

    def stop_after_attempt(*args: Any, **kwargs: Any) -> None:
        return None

    def wait_exponential(*args: Any, **kwargs: Any) -> None:
        return None

logger = logging.getLogger(__name__)


class YCloudResponse(BaseModel):
    success: bool = Field(False)
    message_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    raw_data: dict[str, Any] | None = None

    @field_validator("success", mode="before")
    @classmethod
    def normalize_success(cls, value: Any) -> bool:
        return bool(value)


class WhatsAppProvider(Protocol):
    provider_name: str

    async def send_text_message(self, to: str | None, body: str) -> dict[str, Any]:
        ...

    async def send_template_message(
        self,
        to: str | None,
        template_name: str,
        language_code: str,
        parameters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ...

    async def check_template_status(self, name: str, language_code: str | None = None) -> str:
        ...

    async def submit_template_for_approval(
        self,
        name: str,
        category: str,
        body_text: str,
        lang: str = "en",
    ) -> str | None:
        ...

    async def send_media_message(
        self,
        to: str | None,
        media_type: Literal["image", "document", "video", "audio"],
        url: str,
        caption: str = "",
        filename: str | None = None,
    ) -> dict[str, Any]:
        ...

    async def send_interactive_message(
        self,
        to: str | None,
        body: str,
        buttons: list[dict[str, str]],
    ) -> dict[str, Any]:
        ...


class MetaCloudWhatsAppProvider:
    provider_name = "meta"

    def __init__(self, config: Any | None = None) -> None:
        self.config = config or load_runtime_config()
        self.template_manager = WhatsAppTemplateManager.from_config(self.config)

    async def send_text_message(self, to: str | None, body: str) -> dict[str, Any]:
        if not getattr(self.config, "whatsapp_access_token", None) or not getattr(self.config, "whatsapp_phone_number_id", None) or not to:
            return {
                "status": "missing_credentials",
                "provider": self.provider_name,
                "recipient": to,
                "message_id": None,
                "error": "WhatsApp access token, phone number id, and recipient are required.",
            }
        result = send_whatsapp_text_message(
            access_token=str(getattr(self.config, "whatsapp_access_token")),
            phone_number_id=str(getattr(self.config, "whatsapp_phone_number_id")),
            recipient=to,
            body=body,
            graph_api_version=str(getattr(self.config, "whatsapp_graph_api_version", "v23.0")),
        )
        result["provider"] = self.provider_name
        return result

    async def send_template_message(
        self,
        to: str | None,
        template_name: str,
        language_code: str,
        parameters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        result = await self.template_manager.send_template_message(
            to=to,
            template_name=template_name,
            language_code=language_code,
            parameters=parameters,
        )
        result["provider"] = self.provider_name
        return result

    async def check_template_status(self, name: str, language_code: str | None = None) -> str:
        return await self.template_manager.check_template_status(name)

    async def submit_template_for_approval(
        self,
        name: str,
        category: str,
        body_text: str,
        lang: str = "en",
    ) -> str | None:
        return await self.template_manager.submit_template_for_approval(name, category, body_text, lang)

    async def send_media_message(
        self,
        to: str | None,
        media_type: Literal["image", "document", "video", "audio"],
        url: str,
        caption: str = "",
        filename: str | None = None,
    ) -> dict[str, Any]:
        return self._unsupported(to, "media")

    async def send_interactive_message(
        self,
        to: str | None,
        body: str,
        buttons: list[dict[str, str]],
    ) -> dict[str, Any]:
        return self._unsupported(to, "interactive")

    def _unsupported(self, to: str | None, feature: str) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "provider": self.provider_name,
            "recipient": to,
            "message_id": None,
            "error": f"Meta provider adapter does not support {feature} messages in this workflow.",
        }


class YCloudWhatsAppProvider:
    provider_name = "ycloud"

    def __init__(
        self,
        config: Any | None = None,
        http_client_factory: Callable[..., Any] = httpx.AsyncClient,
    ) -> None:
        self.config = config or load_runtime_config()
        self.http_client_factory = http_client_factory
        self.api_key = getattr(self.config, "ycloud_api_key", None)
        self.sender = getattr(self.config, "ycloud_whatsapp_from", None)
        self.waba_id = getattr(self.config, "ycloud_waba_id", None)
        self.base_url = str(getattr(self.config, "ycloud_base_url", "https://api.ycloud.com/v2")).rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": str(self.api_key),
            "Content-Type": "application/json",
            "User-Agent": "CrossBorderAI/1.0 (YCLOUD-Adapter)",
        }

    async def send_text_message(self, to: str | None, body: str) -> dict[str, Any]:
        if not self.api_key or not self.sender or not to:
            return self._missing_credentials(to)
        payload = {
            "from": self.sender,
            "to": to,
            "type": "text",
            "text": {"body": body, "preview_url": False},
        }
        return await self._send_message(to, payload)

    async def send_template_message(
        self,
        to: str | None,
        template_name: str,
        language_code: str,
        parameters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key or not self.sender or not to:
            return self._missing_credentials(to)
        payload = {
            "from": self.sender,
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
                "components": [{"type": "body", "parameters": parameters or []}],
            },
        }
        return await self._send_message(to, payload)

    async def send_media_message(
        self,
        to: str | None,
        media_type: Literal["image", "document", "video", "audio"],
        url: str,
        caption: str = "",
        filename: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key or not self.sender or not to:
            return self._missing_credentials(to)
        media_payload: dict[str, Any] = {"link": url}
        if caption:
            media_payload["caption"] = caption
        if filename and media_type == "document":
            media_payload["filename"] = filename
        payload = {
            "from": self.sender,
            "to": to,
            "type": media_type,
            media_type: media_payload,
        }
        return await self._send_message(to, payload)

    async def send_interactive_message(
        self,
        to: str | None,
        body: str,
        buttons: list[dict[str, str]],
    ) -> dict[str, Any]:
        if not self.api_key or not self.sender or not to:
            return self._missing_credentials(to)
        if len(buttons) > 3:
            logger.warning("WhatsApp supports max 3 reply buttons. Truncating interactive payload.")
            buttons = buttons[:3]
        action_buttons = [
            {
                "type": "reply",
                "reply": {
                    "id": button.get("id") or f"btn_{index}",
                    "title": button["title"],
                },
            }
            for index, button in enumerate(buttons)
            if button.get("title")
        ]
        payload = {
            "from": self.sender,
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body},
                "action": {"buttons": action_buttons},
            },
        }
        return await self._send_message(to, payload)

    async def check_template_status(self, name: str, language_code: str | None = None) -> str:
        if not self.api_key or not self.waba_id:
            return "MISSING_CREDENTIALS"
        language = language_code or "en_US"
        try:
            response = await self._request(
                "GET",
                f"/whatsapp/templates/{self.waba_id}/{name}/{language}",
            )
            return str(
                (response.raw_data or {}).get("status")
                or (response.raw_data or {}).get("templateStatus")
                or "UNKNOWN"
            )
        except httpx.HTTPStatusError as exc:
            return "NOT_FOUND" if exc.response.status_code == 404 else f"FAILED:{exc.response.status_code}"
        except httpx.HTTPError:
            return "FAILED"

    async def submit_template_for_approval(
        self,
        name: str,
        category: str,
        body_text: str,
        lang: str = "en",
    ) -> str | None:
        if not self.api_key or not self.waba_id:
            return None
        payload = {
            "wabaId": self.waba_id,
            "name": name,
            "language": lang,
            "category": category,
            "components": [
                {"type": "BODY", "text": body_text},
                {"type": "FOOTER", "text": "Reply STOP to opt out"},
            ],
        }
        response = await self._request("POST", "/whatsapp/templates", payload)
        data = response.raw_data or {}
        return response.message_id or data.get("id") or data.get("templateId") or data.get("name")

    async def _send_message(self, to: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._request("POST", "/whatsapp/messages/sendDirectly", payload)
        except httpx.HTTPStatusError as exc:
            return {
                "status": "failed",
                "provider": self.provider_name,
                "recipient": to,
                "message_id": None,
                "error": f"{exc.response.status_code}: {exc.response.text}",
            }
        except httpx.HTTPError as exc:
            return {
                "status": "failed",
                "provider": self.provider_name,
                "recipient": to,
                "message_id": None,
                "error": str(exc),
            }
        return {
            "status": "sent" if response.success else "failed",
            "provider": self.provider_name,
            "recipient": to,
            "message_id": response.message_id,
            "error": response.error_message,
            "error_code": response.error_code,
            "raw_response": response.raw_data,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> YCloudResponse:
        url = f"{self.base_url}{endpoint}"
        async with self.http_client_factory(timeout=30.0) as client:
            response = await client.request(method, url, headers=self.headers, json=payload)
            response.raise_for_status()
        data = response.json()
        raw_data = data.get("data") if isinstance(data.get("data"), dict) else data
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        message_id = (
            (raw_data or {}).get("message_id")
            or (raw_data or {}).get("messageId")
            or (raw_data or {}).get("whatsappMessageId")
            or (raw_data or {}).get("id")
            or (raw_data or {}).get("wamid")
            or data.get("id")
            or data.get("messageId")
            or data.get("whatsappMessageId")
            or data.get("wamid")
        )
        success = data.get("success")
        if success is None:
            success = not error
        return YCloudResponse(
            success=success,
            message_id=message_id,
            error_code=error.get("code"),
            error_message=error.get("message"),
            raw_data=raw_data,
        )

    def _missing_credentials(self, to: str | None) -> dict[str, Any]:
        return {
            "status": "missing_credentials",
            "provider": self.provider_name,
            "recipient": to,
            "message_id": None,
            "error": "YCLOUD_API_KEY, YCLOUD_WHATSAPP_FROM, and recipient are required.",
        }


def get_whatsapp_provider(config: Any | None = None) -> WhatsAppProvider:
    config = config or load_runtime_config()
    provider = str(getattr(config, "whatsapp_provider", "ycloud") or "ycloud").lower()
    if provider == "meta":
        return MetaCloudWhatsAppProvider(config)
    return YCloudWhatsAppProvider(config)


async def send_rma_label_message(
    *,
    provider: WhatsAppProvider,
    to: str | None,
    label_url: str,
    order_id: str,
    language_code: str = "en",
    is_window_expired: bool = False,
) -> dict[str, Any]:
    if is_window_expired:
        return await provider.send_template_message(
            to=to,
            template_name="rma_return_label_doc",
            language_code=language_code,
            parameters=[
                {"type": "document", "document": {"link": label_url}},
                {"type": "text", "text": order_id},
            ],
        )
    return await provider.send_media_message(
        to=to,
        media_type="document",
        url=label_url,
        filename=f"RMA_{order_id}_Label.pdf",
        caption="Please print this label and attach it to your return package.",
    )
