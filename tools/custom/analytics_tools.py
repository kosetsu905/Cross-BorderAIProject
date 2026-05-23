import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from html import unescape
from typing import Any

import httpx

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

from tools.custom.commerce_api import CommerceApiConfig, fetch_commerce_metrics

logger = logging.getLogger(__name__)
SERPER_SEARCH_URL = "https://google.serper.dev/search"


class EcomPlatformMetricsTool(BaseTool):
    name: str = "E-commerce Platform Metrics Fetcher"
    description: str = (
        "Fetches sales, conversion, inventory, and ad performance data from "
        "Shopify, Amazon, TikTok Shop, or other commerce platforms."
    )
    ecom_api_token: str | None = None
    shopify_store_domain: str | None = None
    shopify_admin_access_token: str | None = None
    shopify_api_version: str = "2025-07"
    amazon_sp_api_endpoint: str | None = None
    amazon_sp_api_access_token: str | None = None
    amazon_marketplace_ids: str | None = None

    def _run(self, platform: str, region: str, date_range: str) -> dict[str, Any]:
        config = CommerceApiConfig(
            shopify_store_domain=self.shopify_store_domain,
            shopify_admin_access_token=self.shopify_admin_access_token or self.ecom_api_token,
            shopify_api_version=self.shopify_api_version,
            amazon_sp_api_endpoint=self.amazon_sp_api_endpoint,
            amazon_sp_api_access_token=self.amazon_sp_api_access_token or self.ecom_api_token,
            amazon_marketplace_ids=self.amazon_marketplace_ids,
        )
        if not any(
            [
                config.shopify_admin_access_token and config.shopify_store_domain,
                config.amazon_sp_api_access_token and config.amazon_sp_api_endpoint,
            ]
        ):
            return self._dev_fallback(platform, region, date_range)

        try:
            return fetch_commerce_metrics(config, platform, region, date_range)
        except Exception as exc:
            logger.warning("E-commerce metrics fetch failed: %s", exc)
            fallback = self._dev_fallback(platform, region, date_range)
            fallback["provider_error"] = str(exc)
            return fallback

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
    serper_api_key: str | None = None
    deep_read_enabled: bool = False
    deep_read_max_pages: int = 3
    deep_read_concurrency: int = 5
    deep_read_timeout_seconds: int = 10
    deep_read_max_chars: int = 4000

    def _run(self, product_category: str, target_markets: str) -> dict[str, Any]:
        if not self.serper_api_key:
            return self._dev_fallback(product_category, target_markets)

        try:
            return _fetch_serper_competitor_benchmarks(
                self.serper_api_key,
                product_category,
                target_markets,
                self.deep_read_enabled,
                self.deep_read_max_pages,
                self.deep_read_concurrency,
                self.deep_read_timeout_seconds,
                self.deep_read_max_chars,
            )
        except Exception as exc:
            logger.warning("Serper competitor benchmark lookup failed: %s", exc)
            fallback = self._dev_fallback(product_category, target_markets)
            fallback["provider_error"] = str(exc)
            return fallback

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


def _fetch_serper_competitor_benchmarks(
    serper_api_key: str,
    product_category: str,
    target_markets: str,
    deep_read_enabled: bool = False,
    deep_read_max_pages: int = 3,
    deep_read_concurrency: int = 5,
    deep_read_timeout_seconds: int = 10,
    deep_read_max_chars: int = 4000,
) -> dict[str, Any]:
    markets = _split_markets(target_markets)
    market_results = [
        _search_market_competitors(
            serper_api_key,
            product_category,
            market,
            deep_read_enabled,
            deep_read_max_pages,
            deep_read_concurrency,
            deep_read_timeout_seconds,
            deep_read_max_chars,
        )
        for market in markets
    ]
    sources = [
        source
        for result in market_results
        for source in result.get("sources", [])
    ]
    return {
        "category": product_category,
        "markets": target_markets,
        "data_source": "serper_deep_read" if deep_read_enabled else "serper_search",
        "confidence_level": "medium" if deep_read_enabled else "medium",
        "assumption_notice": (
            "Competitive insights are based on live Serper search results"
            + (" plus best-effort source page reads" if deep_read_enabled else " snippets")
            + " and should be verified before business decisions."
        ),
        "status": "live_search",
        "market_results": market_results,
        "source_urls": sources,
    }


def _search_market_competitors(
    serper_api_key: str,
    product_category: str,
    market: str,
    deep_read_enabled: bool,
    deep_read_max_pages: int,
    deep_read_concurrency: int,
    deep_read_timeout_seconds: int,
    deep_read_max_chars: int,
) -> dict[str, Any]:
    query = f"{product_category} competitors pricing promotions {market}"
    response = httpx.post(
        SERPER_SEARCH_URL,
        headers={
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": 5},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    organic_results = payload.get("organic") or []
    shopping_results = payload.get("shopping") or []
    sources = _extract_sources([*organic_results, *shopping_results])
    evidence = (
        _deep_read_sources(
            sources[: max(0, deep_read_max_pages)],
            deep_read_concurrency,
            deep_read_timeout_seconds,
            deep_read_max_chars,
        )
        if deep_read_enabled
        else []
    )
    return {
        "market": market,
        "query": query,
        "competitor_signals": [
            {
                "title": source["title"],
                "snippet": source["snippet"],
                "link": source["link"],
                "source_type": source["source_type"],
            }
            for source in sources
        ],
        "sources": sources,
        "deep_read_enabled": deep_read_enabled,
        "evidence": evidence,
    }


def _extract_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for result in results:
        title = str(result.get("title") or "").strip()
        link = str(result.get("link") or "").strip()
        snippet = str(result.get("snippet") or result.get("price") or "").strip()
        if not title and not link:
            continue
        sources.append(
            {
                "title": title,
                "link": link,
                "snippet": snippet,
                "source_type": "shopping" if result.get("price") else "organic",
            }
        )
    return sources[:5]


def _split_markets(target_markets: str) -> list[str]:
    markets = [market.strip() for market in target_markets.split(",") if market.strip()]
    if markets:
        return markets
    return [target_markets]


def _deep_read_sources(
    sources: list[dict[str, Any]],
    concurrency: int,
    timeout_seconds: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    if not sources:
        return []

    max_workers = max(1, min(concurrency, len(sources), 16))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {
            executor.submit(_read_source_page, source, timeout_seconds, max_chars): source
            for source in sources
            if source.get("link")
        }
        results: list[dict[str, Any]] = []
        for future in as_completed(future_to_source):
            results.append(future.result())
    return results


def _read_source_page(
    source: dict[str, Any],
    timeout_seconds: int,
    max_chars: int,
) -> dict[str, Any]:
    url = str(source.get("link") or "")
    base_payload = {
        "title": source.get("title", ""),
        "url": url,
        "search_snippet": source.get("snippet", ""),
        "source_type": source.get("source_type", "organic"),
    }
    try:
        response = httpx.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            },
            follow_redirects=True,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return {
                **base_payload,
                "read_status": "skipped",
                "reason": f"Unsupported content type: {content_type}",
            }

        text = _html_to_text(response.text)
        return {
            **base_payload,
            "read_status": "ok",
            "final_url": str(response.url),
            "page_excerpt": text[:max(0, max_chars)],
            "char_count": len(text),
        }
    except Exception as exc:
        return {
            **base_payload,
            "read_status": "failed",
            "reason": str(exc),
        }


def _html_to_text(value: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", value)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|li|h[1-6])>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()
