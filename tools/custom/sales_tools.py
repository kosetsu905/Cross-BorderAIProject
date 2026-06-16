import logging
from functools import lru_cache
from typing import Any

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

from tools.custom.commerce_api import CommerceApiConfig, fetch_commerce_metrics
from utils.tool_cache import cached_tool_call
from utils.tool_execution import AsyncToolExecutionMixin

logger = logging.getLogger(__name__)


class CRMFunnelTool(AsyncToolExecutionMixin, BaseTool):
    name: str = "CRM Funnel & Conversion Data Fetcher"
    description: str = (
        "Extracts funnel metrics, drop-off rates, and regional conversion data "
        "from CRM or e-commerce platforms."
    )
    crm_api_token: str | None = None
    shopify_store_domain: str | None = None
    shopify_admin_access_token: str | None = None
    shopify_api_version: str = "2025-07"
    amazon_sp_api_endpoint: str | None = None
    amazon_sp_api_access_token: str | None = None
    amazon_marketplace_ids: str | None = None
    tool_cache_context: dict[str, Any] | None = None

    def _run(self, product_category: str, target_markets: str) -> dict[str, Any]:
        config = CommerceApiConfig(
            shopify_store_domain=self.shopify_store_domain,
            shopify_admin_access_token=self.shopify_admin_access_token or self.crm_api_token,
            shopify_api_version=self.shopify_api_version,
            amazon_sp_api_endpoint=self.amazon_sp_api_endpoint,
            amazon_sp_api_access_token=self.amazon_sp_api_access_token or self.crm_api_token,
            amazon_marketplace_ids=self.amazon_marketplace_ids,
        )
        if not any(
            [
                config.shopify_admin_access_token and config.shopify_store_domain,
                config.amazon_sp_api_access_token and config.amazon_sp_api_endpoint,
            ]
        ):
            return self._dev_fallback(product_category, target_markets)

        regional_metrics: list[dict[str, Any]] = []
        provider_errors: list[str] = []
        for region in [item.strip() for item in target_markets.split(",") if item.strip()]:
            try:
                regional_metrics.append(
                    cached_tool_call(
                        self.tool_cache_context,
                        tool_name="Commerce Metrics Read",
                        tool_version="v1",
                        arguments={
                            "platform": "",
                            "region": region,
                            "date_range": "Last 60 Days",
                        },
                        provider_identity={
                            "provider": "commerce_api",
                            "shopify_store_domain": self.shopify_store_domain,
                            "shopify_api_version": self.shopify_api_version,
                            "amazon_sp_api_endpoint": self.amazon_sp_api_endpoint,
                            "amazon_marketplace_ids": self.amazon_marketplace_ids,
                        },
                        fetcher=lambda region=region: fetch_commerce_metrics(
                            config,
                            "",
                            region,
                            "Last 60 Days",
                        ),
                    )
                )
            except Exception as exc:
                provider_errors.append(f"{region}: {exc}")

        if not regional_metrics:
            fallback = self._dev_fallback(product_category, target_markets)
            fallback["provider_error"] = "; ".join(provider_errors)
            return fallback

        total_orders = sum((item.get("metrics") or {}).get("order_count", 0) for item in regional_metrics)
        cancelled_orders = sum(
            (item.get("metrics") or {}).get("cancelled_order_count", 0)
            for item in regional_metrics
        )
        cancellation_rate = cancelled_orders / total_orders if total_orders else 0

        return {
            "product": product_category,
            "markets": target_markets,
            "data_source": "external_commerce_orders_api",
            "confidence_level": "medium",
            "status": "live_provider",
            "regional_metrics": regional_metrics,
            "overall_conversion_rate": "not_available_from_orders_api",
            "top_drop_off": (
                f"Cancelled orders, {cancellation_rate:.1%}"
                if total_orders
                else "not_available_from_orders_api"
            ),
            "regional_gaps": {
                item["region"]: {
                    "order_count": (item.get("metrics") or {}).get("order_count"),
                    "cancelled_order_count": (item.get("metrics") or {}).get("cancelled_order_count"),
                    "note": "Derived from order statuses; full funnel requires analytics events.",
                }
                for item in regional_metrics
            },
            "assumption_notice": (
                "Sales funnel metrics are derived from commerce orders. Session, product-page, "
                "cart, checkout, and payment-step drop-offs require analytics or CRM events."
            ),
            "provider_errors": provider_errors,
        }

    @staticmethod
    @lru_cache(maxsize=16)
    def _dev_fallback(product_category: str, target_markets: str) -> dict[str, Any]:
        logger.info("Using CRM funnel development fallback")
        return {
            "product": product_category,
            "markets": target_markets,
            "data_source": "development_fallback_sample",
            "confidence_level": "low",
            "assumption_notice": (
                "This is sample fallback data, not a factual diagnosis of the user's business."
            ),
            "overall_conversion_rate": "2.8%",
            "top_drop_off": "Payment Gateway, 34%",
            "regional_gaps": {
                "EU": "High cart abandonment due to VAT display",
                "US": "Strong mobile conversion",
                "JP": "Payment method mismatch, needs local options",
            },
            "status": "dev_mode",
        }


class CROHeuristicsTool(BaseTool):
    name: str = "CRO Heuristics & A/B Test Recommender"
    description: str = (
        "Generates data-driven A/B test hypotheses and UX optimization "
        "recommendations for e-commerce funnels."
    )

    def _run(self, drop_off_point: str, region: str) -> dict[str, Any]:
        return {
            "hypothesis": f"Simplify {drop_off_point} for {region} users",
            "test_type": "A/B Multivariate",
            "expected_uplift": "8-12%",
            "implementation_effort": "Medium",
            "success_metric": "Checkout completion rate",
            "data_source": "heuristic_recommendation",
            "confidence_level": "medium",
            "status": "dev_mode",
        }


class PricingIntelTool(BaseTool):
    name: str = "Cross-Border Pricing & Margin Optimizer"
    description: str = (
        "Analyzes competitor pricing, regional purchasing power, and tax/VAT "
        "impacts to recommend optimal pricing strategies."
    )

    def _run(self, product_category: str, target_markets: str) -> dict[str, Any]:
        return {
            "category": product_category,
            "markets": target_markets,
            "data_source": "development_fallback_sample",
            "confidence_level": "low",
            "assumption_notice": (
                "This is illustrative fallback pricing guidance, not validated competitor intelligence."
            ),
            "recommended_adjustments": {
                "EU": "+5% to offset VAT and shipping",
                "JP": "Localized payment bundle discount",
                "US": "Maintain current pricing plus loyalty tier",
            },
            "margin_impact": "+3.2% overall projected",
            "status": "optimized",
        }
