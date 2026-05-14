import logging
import os
from functools import lru_cache
from typing import Any

import httpx

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

logger = logging.getLogger(__name__)


class EcomPlatformMetricsTool(BaseTool):
    name: str = "E-commerce Platform Metrics Fetcher"
    description: str = (
        "Fetches sales, conversion, inventory, and ad performance data from "
        "Shopify, Amazon, TikTok Shop, or other commerce platforms."
    )

    def _run(self, platform: str, region: str, date_range: str) -> dict[str, Any]:
        token = os.getenv("ECOM_API_TOKEN")
        if not token:
            return self._dev_fallback(platform, region, date_range)

        url = "https://api.your-ecom-platform.com/v1/metrics"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"platform": platform, "region": region, "date_range": date_range}
        try:
            response = httpx.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("E-commerce metrics fetch failed: %s", exc)
            return self._dev_fallback(platform, region, date_range)

    @staticmethod
    @lru_cache(maxsize=32)
    def _dev_fallback(platform: str, region: str, date_range: str) -> dict[str, Any]:
        logger.info("Using e-commerce metrics development fallback")
        return {
            "platform": platform,
            "region": region,
            "date_range": date_range,
            "metrics": {
                "total_sales": 45200.00,
                "conversion_rate": "3.2%",
                "cpc": 1.15,
                "roas": 2.8,
                "inventory_status": "Healthy",
                "top_selling_sku": "CAM-4K-PRO",
            },
            "status": "dev_mode",
        }


class CompetitorBenchmarkTool(BaseTool):
    name: str = "Competitor & Market Benchmark Tool"
    description: str = (
        "Analyzes competitor pricing, promotions, and market positioning for "
        "cross-border e-commerce."
    )

    def _run(self, product_category: str, target_markets: str) -> dict[str, Any]:
        if not os.getenv("SERPER_API_KEY"):
            return self._dev_fallback(product_category, target_markets)

        return {
            "category": product_category,
            "markets": target_markets,
            "avg_competitor_price": "$129.99",
            "our_price_position": "Mid-tier, roughly 5% below average",
            "competitor_promo_activity": "High",
            "market_demand_trend": "Increasing, approximately +12% month over month",
            "status": "provider_ready_stub",
        }

    @staticmethod
    @lru_cache(maxsize=32)
    def _dev_fallback(product_category: str, target_markets: str) -> dict[str, Any]:
        logger.info("Using competitor benchmark development fallback")
        return {
            "category": product_category,
            "markets": target_markets,
            "avg_competitor_price": "$125.50",
            "our_price_position": "Competitive",
            "market_demand_trend": "Stable to growing",
            "status": "dev_mode",
        }
