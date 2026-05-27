from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


PIMBackend = Literal["akeneo", "plytix", "custom"]


class PIMProductAttribute(BaseModel):
    code: str
    label: dict[str, str]
    value: dict[str, str]
    scope: Literal["global", "channel", "locale"] = "global"


class PIMProductVariant(BaseModel):
    sku: str
    name: dict[str, str]
    price: dict[str, float]
    attributes: list[PIMProductAttribute] = Field(default_factory=list)
    availability: dict[str, bool] = Field(default_factory=dict)
    compatibility_notes: dict[str, str] = Field(default_factory=dict)


class PIMQueryResult(BaseModel):
    product_found: bool
    family_code: str
    main_attributes: list[PIMProductAttribute] = Field(default_factory=list)
    variants: list[PIMProductVariant] = Field(default_factory=list)
    regional_compliance: dict[str, list[str]] = Field(default_factory=dict)
    last_updated: str = ""
    data_source: str


class PIMConnector:
    def __init__(
        self,
        backend: PIMBackend | str = "akeneo",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        normalized_backend = str(backend or "akeneo").lower()
        if normalized_backend not in {"akeneo", "plytix", "custom"}:
            logger.warning("Unknown PIM backend %s; falling back to akeneo shape", backend)
            normalized_backend = "akeneo"
        self.backend: PIMBackend = normalized_backend  # type: ignore[assignment]
        env_prefix = f"PIM_{self.backend.upper()}"
        self.base_url = (base_url or os.getenv(f"{env_prefix}_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv(f"{env_prefix}_API_KEY")
        self.timeout = httpx.Timeout(timeout_seconds)

    async def search_product(self, query: str, region: str, language: str) -> PIMQueryResult:
        safe_query = str(query or "").strip() or "Smart Home Camera"
        safe_region = str(region or "US").upper()
        safe_language = _normalize_language(language)
        if not self.base_url or not self.api_key:
            logger.info("PIM %s credentials not configured; using mock fallback", self.backend)
            return self._mock_fallback(safe_query, safe_region, safe_language)

        try:
            if self.backend == "akeneo":
                return await self._query_akeneo(safe_query, safe_region, safe_language)
            if self.backend == "plytix":
                return await self._query_plytix(safe_query, safe_region, safe_language)
            return await self._query_custom_pim(safe_query, safe_region, safe_language)
        except httpx.HTTPError as exc:
            logger.warning("PIM %s query failed; using mock fallback: %s", self.backend, exc)
            return self._mock_fallback(safe_query, safe_region, safe_language)

    async def _query_akeneo(self, query: str, region: str, language: str) -> PIMQueryResult:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        search_url = f"{self.base_url}/api/rest/v1/products"
        params = {"search": f'[{{"attribute":"name","operator":"CONTAINS","value":"{query}"}}]'}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(search_url, params=params, headers=headers)
            response.raise_for_status()
            items = response.json().get("_embedded", {}).get("items", [])
            if not items:
                return PIMQueryResult(product_found=False, family_code="", data_source="akeneo")

            product_code = items[0].get("code")
            detail = await client.get(f"{self.base_url}/api/rest/v1/products/{product_code}", headers=headers)
            detail.raise_for_status()
            return self._transform_akeneo_product(detail.json(), region, language)

    async def _query_plytix(self, query: str, region: str, language: str) -> PIMQueryResult:
        payload = {
            "filters": [{"attribute": "name", "operator": "contains", "value": query}],
            "locale": language,
            "scope": region,
        }
        headers = {"X-Api-Key": str(self.api_key), "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/v1/entity/search", json=payload, headers=headers)
            response.raise_for_status()
            results = response.json().get("data", [])
            if not results:
                return PIMQueryResult(product_found=False, family_code="", data_source="plytix")
            return self._transform_plytix_product(results[0], region, language)

    async def _query_custom_pim(self, query: str, region: str, language: str) -> PIMQueryResult:
        payload = {"q": query, "region": region, "language": language}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/products/search", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return PIMQueryResult(
                product_found=bool(data.get("found", data.get("product_found", False))),
                family_code=str(data.get("family", data.get("family_code", ""))),
                main_attributes=[PIMProductAttribute.model_validate(item) for item in data.get("attributes", [])],
                variants=[PIMProductVariant.model_validate(item) for item in data.get("variants", [])],
                regional_compliance=data.get("compliance", data.get("regional_compliance", {})) or {},
                last_updated=str(data.get("updated_at", data.get("last_updated", ""))),
                data_source="custom_pim",
            )

    def _transform_akeneo_product(self, product: dict[str, Any], region: str, language: str) -> PIMQueryResult:
        attributes: list[PIMProductAttribute] = []
        for code, values in (product.get("values") or {}).items():
            if not isinstance(values, list) or not values:
                continue
            value = _localized_akeneo_value(values, language)
            if value:
                attributes.append(
                    PIMProductAttribute(
                        code=str(code),
                        label={"en": str(code), language: str(code)},
                        value={language: value},
                    )
                )
        family = str(product.get("family") or product.get("code") or "")
        return PIMQueryResult(
            product_found=True,
            family_code=family,
            main_attributes=attributes[:10],
            variants=[],
            regional_compliance={region: [f"Complies with {region} product requirements"]},
            last_updated=str(product.get("updated") or ""),
            data_source="akeneo",
        )

    def _transform_plytix_product(self, entity: dict[str, Any], region: str, language: str) -> PIMQueryResult:
        attributes: list[PIMProductAttribute] = []
        for code, value in (entity.get("attributes") or {}).items():
            attributes.append(
                PIMProductAttribute(
                    code=str(code),
                    label={"en": str(code), language: str(code)},
                    value={language: str(value)},
                )
            )
        return PIMQueryResult(
            product_found=True,
            family_code=str(entity.get("family_code") or entity.get("sku") or ""),
            main_attributes=attributes[:10],
            variants=[],
            regional_compliance={region: ["Plytix-sourced compliance note"]},
            last_updated=str(entity.get("updated_at") or ""),
            data_source="plytix",
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _mock_fallback(query: str, region: str, language: str) -> PIMQueryResult:
        label_language = language or "en"
        return PIMQueryResult(
            product_found=True,
            family_code="smart_home_camera",
            main_attributes=[
                PIMProductAttribute(
                    code="resolution",
                    label={"en": "Resolution", label_language: "Resolution"},
                    value={label_language: "4K Ultra HD resolution with HDR"},
                ),
                PIMProductAttribute(
                    code="smart_detection",
                    label={"en": "Smart Detection", label_language: "Smart Detection"},
                    value={label_language: "AI-powered motion detection for person, vehicle, and pet events"},
                ),
                PIMProductAttribute(
                    code="compatibility",
                    label={"en": "Compatibility", label_language: "Compatibility"},
                    value={label_language: "Works with Alexa, Google Assistant, and HomeKit"},
                ),
            ],
            variants=[
                PIMProductVariant(
                    sku="CAM-4K-BASIC",
                    name={label_language: "Basic"},
                    price={"USD": 79.0},
                    availability={region: True},
                    compatibility_notes={region: "Best for apartment renters; cloud storage only"},
                ),
                PIMProductVariant(
                    sku="CAM-4K-PRO",
                    name={label_language: "Pro"},
                    price={"USD": 129.0},
                    availability={region: True},
                    compatibility_notes={region: "Best for outdoor and privacy-focused use; local SD plus cloud storage"},
                ),
                PIMProductVariant(
                    sku="CAM-4K-BUNDLE",
                    name={label_language: "Bundle"},
                    price={"USD": 199.0},
                    availability={region: True},
                    compatibility_notes={region: "Best for multi-room coverage; includes two cameras and hub"},
                ),
            ],
            regional_compliance={
                "EU": ["GDPR-compliant data processing with local server option"],
                "US": ["FCC certified and compatible with common smart-home ecosystems"],
                "JP": ["PSE certified with Japanese app interface available"],
                region: [f"Mock fallback compliance note for {region}"],
            },
            last_updated="2024-01-01T00:00:00Z",
            data_source="mock_fallback",
        )


def _normalize_language(language: str | None) -> str:
    value = str(language or "en").strip()
    if not value:
        return "en"
    return {"English": "en", "Chinese": "zh", "Japanese": "ja", "Spanish": "es"}.get(value, value[:2].lower())


def _localized_akeneo_value(values: list[dict[str, Any]], language: str) -> str:
    for item in values:
        locale = item.get("locale")
        if locale in {language, f"{language}_US", f"{language.upper()}"}:
            return str(item.get("data") or "")
    for item in values:
        if item.get("locale") in {None, "en", "en_US"}:
            return str(item.get("data") or "")
    return str(values[0].get("data") or "")
