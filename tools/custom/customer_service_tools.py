from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any

from services.pim_connector import PIMConnector, PIMQueryResult
from services.intent_router import classify_intent

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

logger = logging.getLogger(__name__)


class PreSalesProductKnowledgeTool(BaseTool):
    name: str = "Pre-Sales Product Knowledge Base"
    description: str = (
        "Fetches verified product specs, compatibility info, use cases, and variant comparisons "
        "for pre-sales inquiries."
    )
    pim_backend: str = "akeneo"
    pim_base_url: str | None = None
    pim_api_key: str | None = None

    def _run(
        self,
        product_category: str,
        inquiry_keywords: list[str] | None = None,
        region: str = "US",
        language: str = "en",
    ) -> dict[str, Any]:
        logger.info("Pre-sales lookup: %s in %s", product_category, region)
        query = str(product_category or " ".join(inquiry_keywords or []) or "Smart Home Camera")
        pim_result = _run_coroutine_sync(
            PIMConnector(
                backend=self.pim_backend,
                base_url=self.pim_base_url,
                api_key=self.pim_api_key,
            ).search_product(query, region, language)
        )
        return _pim_result_to_pre_sales_context(pim_result, region, language)


def _pim_result_to_pre_sales_context(result: PIMQueryResult, region: str, language: str) -> dict[str, Any]:
    language_code = str(language or "en")[:2].lower()
    features = [
        next(iter(attribute.value.values()), "")
        for attribute in result.main_attributes
        if next(iter(attribute.value.values()), "")
    ]
    compatibility = {
        attribute.code: next(iter(attribute.value.values()), "")
        for attribute in result.main_attributes
        if attribute.code in {"compatibility", "wifi", "power", "app"}
    }
    variants: dict[str, dict[str, Any]] = {}
    for variant in result.variants:
        name = variant.name.get(language_code) or variant.name.get(language) or next(iter(variant.name.values()), variant.sku)
        note = variant.compatibility_notes.get(region) or next(iter(variant.compatibility_notes.values()), "")
        variants[str(name)] = {
            "sku": variant.sku,
            "price": variant.price,
            "in_stock": variant.availability.get(region),
            "best_for": note,
        }
    compliance_notes = result.regional_compliance.get(region) or result.regional_compliance.get(region.upper()) or []
    return {
        "product_found": result.product_found,
        "family_code": result.family_code,
        "verified_features": features,
        "compatibility_info": compatibility,
        "variant_options": variants,
        "regional_compliance": "; ".join(compliance_notes) if compliance_notes else "Standard global compliance",
        "confidence_level": 95 if result.product_found else 30,
        "last_updated": result.last_updated,
        "data_source": result.data_source,
    }


def _run_coroutine_sync(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coroutine)).result()


class OrderTrackingTool(BaseTool):
    name: str = "Order & Logistics Tracking System"
    description: str = "Verifies order existence, current order status, tracking info, and available modification options."

    def _run(self, order_id: str | None = None, customer_email: str | None = None, region: str = "US") -> dict[str, Any]:
        logger.info("Order lookup: %s in %s", order_id or customer_email, region)
        order_db = {
            "ORD-JP-2024-8842": {
                "status": "shipped",
                "carrier": "Yamato Transport",
                "tracking_number": "1234-5678-9012",
                "tracking_url": "https://track.yamato.co.jp/123456789012",
                "shipped_date": "2024-09-25",
                "estimated_delivery": "2024-09-28",
                "items": [{"sku": "CAM-4K-PRO", "qty": 1}],
                "modifiable": {"address": False, "cancel": False},
                "customs_status": "cleared" if region == "JP" else "pending",
            },
            "ORD-US-2024-3391": {
                "status": "processing",
                "carrier": "FedEx",
                "tracking_number": None,
                "tracking_url": None,
                "shipped_date": None,
                "estimated_delivery": "2024-09-30",
                "items": [{"sku": "THERMO-STEEL", "qty": 2}],
                "modifiable": {"address": True, "cancel": True},
                "customs_status": "domestic",
            },
        }
        order = order_db.get(order_id or "")
        if not order:
            return {
                "order_found": False,
                "error_message": "Order not found. Please verify the order ID or provide the purchase email.",
                "suggested_actions": ["Check order confirmation email", "Verify order ID format", "Contact human agent"],
                "data_source": "mock_order_db",
            }
        status = order["status"]
        return {
            "order_found": True,
            "order_id": order_id,
            "current_status": status,
            "status_explanation": self._explain_status(status, region),
            "tracking_info": {
                "carrier": order["carrier"],
                "tracking_number": order["tracking_number"],
                "tracking_url": order["tracking_url"],
                "last_update": order["shipped_date"] or "Pending shipment",
            } if order["tracking_number"] else None,
            "delivery_estimate": order["estimated_delivery"],
            "customs_info": {"status": order["customs_status"], "note": self._customs_note(order["customs_status"], region)},
            "available_actions": self._get_available_actions(order["modifiable"], status),
            "next_update_expected": self._next_update_timeline(status),
            "data_source": "mock_order_db",
        }

    def _explain_status(self, status: str, region: str) -> str:
        explanations = {
            "processing": f"Order confirmed and being prepared in our {region} warehouse",
            "shipped": "Package handed to carrier and in transit",
            "out_for_delivery": "With local courier and expected today",
            "delivered": "Successfully delivered",
            "delayed": "Unexpected delay; our operations team is monitoring it",
        }
        return explanations.get(status, f"Status: {status}")

    def _customs_note(self, customs_status: str, region: str) -> str:
        if customs_status == "cleared":
            return "Customs clearance is complete."
        if customs_status == "pending":
            return f"Cross-border customs review is in progress for {region}; typical clearance is 1-3 business days."
        return "Domestic shipment; no customs action required."

    def _get_available_actions(self, modifiable: dict[str, Any], status: str) -> list[str]:
        actions: list[str] = []
        if status == "processing" and modifiable.get("address"):
            actions.append("Update shipping address before shipment")
        if status == "processing" and modifiable.get("cancel"):
            actions.append("Cancel order for a full refund")
        if status == "shipped":
            actions.append("Set delivery preferences with carrier")
            actions.append("Request carrier pickup location redirect where available")
        actions.append("Contact human agent for special requests")
        return actions

    def _next_update_timeline(self, status: str) -> str:
        timelines = {
            "processing": "Next update when the order ships, typically within 24-48 hours",
            "shipped": "Next update on daily tracking sync or delivery confirmation",
            "out_for_delivery": "Next update on delivery confirmation today",
            "delivered": "No further updates expected",
        }
        return timelines.get(status, "Next update within 24 hours")


class IntentRouterTool(BaseTool):
    name: str = "Customer Service Intent Classifier & Router"
    description: str = "Classifies inquiries as pre-sales, order-fulfillment, or post-sales support."
    config_context: dict[str, Any] = {}
    classifier: Any | None = None

    def _run(
        self,
        inquiry_text: str,
        has_order_id: bool = False,
        customer_tier: str = "STANDARD",
        language: str = "en",
    ) -> dict[str, Any]:
        return classify_intent(
            inquiry_text=inquiry_text,
            has_order_id=has_order_id,
            customer_tier=customer_tier,
            language=language,
            config_context=self.config_context,
            classifier=self.classifier,
        )
