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
        region_key = region.strip().upper()
        sample_metrics = {
            "US": {
                "total_sales": 45200.00,
                "conversion_rate": "3.2%",
                "cpc": 1.15,
                "roas": 2.8,
                "inventory_status": "Healthy",
                "top_selling_sku": "CAM-4K-PRO",
            },
            "UK": {
                "total_sales": 31800.00,
                "conversion_rate": "2.7%",
                "cpc": 1.34,
                "roas": 2.3,
                "inventory_status": "Moderate",
                "top_selling_sku": "CAM-4K-LITE",
            },
            "GERMANY": {
                "total_sales": 28650.00,
                "conversion_rate": "2.4%",
                "cpc": 1.42,
                "roas": 2.1,
                "inventory_status": "Healthy",
                "top_selling_sku": "CAM-4K-PRO",
            },
            "JAPAN": {
                "total_sales": 24400.00,
                "conversion_rate": "2.1%",
                "cpc": 1.62,
                "roas": 1.9,
                "inventory_status": "Watch",
                "top_selling_sku": "CAM-MINI-JP",
            },
        }
        metrics = sample_metrics.get(
            region_key,
            {
                "total_sales": 15000.00,
                "conversion_rate": "2.0%",
                "cpc": 1.50,
                "roas": 2.0,
                "inventory_status": "Unknown",
                "top_selling_sku": "SAMPLE-SKU",
            },
        )
        return {
            "platform": platform,
            "region": region,
            "date_range": date_range,
            "data_source": "development_fallback_sample",
            "confidence_level": "low",
            "assumption_notice": (
                "This is sample fallback data, not factual platform performance."
            ),
            "metrics": metrics,
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
            "data_source": "provider_ready_stub",
            "confidence_level": "low",
            "assumption_notice": (
                "This is placeholder competitive intelligence until a real provider is connected."
            ),
            "status": "provider_ready_stub",
        }

    @staticmethod
    @lru_cache(maxsize=32)
    def _dev_fallback(product_category: str, target_markets: str) -> dict[str, Any]:
        logger.info("Using competitor benchmark development fallback")
        return {
            "category": product_category,
            "markets": target_markets,
            "data_source": "development_fallback_sample",
            "confidence_level": "low",
            "assumption_notice": (
                "This is sample fallback benchmark data, not validated market research."
            ),
            "avg_competitor_price": "$125.50",
            "our_price_position": "Competitive",
            "market_demand_trend": "Stable to growing",
            "status": "dev_mode",
        }
