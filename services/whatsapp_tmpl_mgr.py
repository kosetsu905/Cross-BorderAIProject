from __future__ import annotations

from typing import Any, Callable

import httpx

from runtime_config import load_runtime_config

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover - requirements installs tenacity, this keeps imports degradable.
    def retry(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda fn: fn

    def stop_after_attempt(*args: Any, **kwargs: Any) -> None:
        return None

    def wait_exponential(*args: Any, **kwargs: Any) -> None:
        return None


DEFAULT_APPROVED_TEMPLATES = {
    "en": {"name": "support_reengagement_en", "lang": "en_US"},
    "ja": {"name": "support_reengagement_ja", "lang": "ja"},
    "es": {"name": "support_reengagement_es", "lang": "es"},
    "default": {"name": "support_reengagement", "lang": "en_US"},
}


class WhatsAppTemplateManager:
    def __init__(
        self,
        *,
        access_token: str | None = None,
        phone_number_id: str | None = None,
        graph_api_version: str = "v23.0",
        approved_templates: dict[str, dict[str, str]] | None = None,
        http_client_factory: Callable[..., Any] = httpx.AsyncClient,
    ) -> None:
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.graph_api_version = graph_api_version or "v23.0"
        self.approved_templates = approved_templates or DEFAULT_APPROVED_TEMPLATES
        self.http_client_factory = http_client_factory

    @classmethod
    def from_config(cls, config: Any | None = None) -> "WhatsAppTemplateManager":
        config = config or load_runtime_config()
        return cls(
            access_token=getattr(config, "whatsapp_access_token", None),
            phone_number_id=getattr(config, "whatsapp_phone_number_id", None),
            graph_api_version=getattr(config, "whatsapp_graph_api_version", "v23.0"),
        )

    @property
    def base_url(self) -> str:
        return f"https://graph.facebook.com/{self.graph_api_version}"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def template_for_language(self, lang_code: str | None) -> dict[str, str]:
        normalized = (lang_code or "").split("-")[0].split("_")[0].lower()
        return dict(self.approved_templates.get(normalized) or self.approved_templates["default"])

    async def send_out_of_window_message(self, to: str | None, lang_code: str | None) -> dict[str, Any]:
        if not self.access_token or not self.phone_number_id or not to:
            return {
                "status": "missing_credentials",
                "recipient": to,
                "message_id": None,
                "error": "WhatsApp access token, phone number id, and recipient are required.",
            }

        template = self.template_for_language(lang_code)
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template["name"],
                "language": {"code": template["lang"]},
                "components": [{"type": "body", "parameters": []}],
            },
        }
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        try:
            async with self.http_client_factory(timeout=15) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "status": "failed",
                "recipient": to,
                "message_id": None,
                "error": f"{exc.response.status_code}: {exc.response.text}",
                "template": template,
            }
        except httpx.HTTPError as exc:
            return {
                "status": "failed",
                "recipient": to,
                "message_id": None,
                "error": str(exc),
                "template": template,
            }

        raw_response = response.json()
        messages = raw_response.get("messages") or []
        message_id = messages[0].get("id") if messages else None
        return {
            "status": "sent",
            "recipient": to,
            "message_id": message_id,
            "error": None,
            "template": template,
            "raw_response": raw_response,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def submit_template_for_approval(
        self,
        name: str,
        category: str,
        body_text: str,
        lang: str = "en",
    ) -> str | None:
        if not self.access_token or not self.phone_number_id:
            return None
        payload = {
            "name": name,
            "category": category,
            "language": lang,
            "components": [
                {"type": "BODY", "text": body_text},
                {"type": "FOOTER", "text": "Reply STOP to opt out"},
            ],
        }
        async with self.http_client_factory(timeout=15) as client:
            response = await client.post(
                f"{self.base_url}/{self.phone_number_id}/message_templates",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json().get("id")

    async def check_template_status(self, template_name: str) -> str:
        if not self.access_token or not self.phone_number_id:
            return "MISSING_CREDENTIALS"
        async with self.http_client_factory(timeout=15) as client:
            response = await client.get(
                f"{self.base_url}/{self.phone_number_id}/message_templates",
                headers=self.headers,
                params={"name": template_name},
            )
            response.raise_for_status()
        data = response.json().get("data") or []
        return str(data[0].get("status") or "NOT_FOUND") if data else "NOT_FOUND"
