import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from html import unescape
from typing import Any
from urllib.parse import urlparse

import httpx

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

from tools.custom.commerce_api import CommerceApiConfig, fetch_commerce_metrics

logger = logging.getLogger(__name__)
SERPER_SEARCH_URL = "https://google.serper.dev/search"

MARKET_CANONICAL_NAMES = {
    "US": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "UNITED STATES": "United States",
    "UNITED STATES OF AMERICA": "United States",
    "AMERICA": "United States",
    "AU": "Australia",
    "AUSTRALIA": "Australia",
    "AUS": "Australia",
    "JP": "Japan",
    "JAPAN": "Japan",
    "UK": "United Kingdom",
    "GB": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom",
    "GERMANY": "Germany",
    "DE": "Germany",
    "EU": "European Union",
}

MARKET_CURRENCY_CODES = {
    "AUSTRALIA": "AUD",
    "JAPAN": "JPY",
    "UNITED STATES": "USD",
    "UNITED KINGDOM": "GBP",
    "GERMANY": "EUR",
    "EUROPEAN UNION": "EUR",
}

MARKET_ALIASES = {
    "Australia": ("Australia", "Australian", "AU market", "Aussie", "AUD", ".com.au", ".au"),
    "Japan": ("Japan", "Japanese", "JP market", "JPY", "yen", ".co.jp", ".jp"),
    "United States": (
        "United States",
        "USA",
        "U.S.",
        "US market",
        "America",
        "American",
        "USD",
    ),
    "United Kingdom": ("United Kingdom", "UK", "British", "GBP", ".co.uk"),
    "Germany": ("Germany", "German", "DE market", "EUR", ".de"),
    "European Union": ("European Union", "EU market", "Europe", "EUR"),
}

MARKET_PRICE_LANGUAGE_HINTS = {
    "Australia": ("driveaway", "drive-away", "on-road costs", "plus orc", "before on-road"),
}


def _canonical_market_name(market: str) -> str:
    raw_market = (market or "").strip()
    if not raw_market:
        return "Unknown"
    return MARKET_CANONICAL_NAMES.get(raw_market.upper(), raw_market)


def _currency_code_for_market(market: str, requested_currency: str | None = None) -> str:
    requested = (requested_currency or "").strip().upper()
    if re.fullmatch(r"[A-Z]{3}", requested):
        return requested
    canonical = _canonical_market_name(market).upper()
    return MARKET_CURRENCY_CODES.get(canonical, requested or "USD")


def _market_aliases(market: str) -> tuple[str, ...]:
    canonical = _canonical_market_name(market)
    aliases = MARKET_ALIASES.get(canonical, (canonical,))
    raw_market = (market or "").strip()
    if raw_market and raw_market not in aliases:
        return (*aliases, raw_market)
    return aliases


def _contains_market_term(text: str, term: str) -> bool:
    if not term:
        return False
    if term.startswith("."):
        return term.lower() in text.lower()
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def _market_mentions(text: str, market: str) -> bool:
    aliases = _market_aliases(market)
    if any(_contains_market_term(text, alias) for alias in aliases):
        return True
    return any(
        _contains_market_term(text, hint)
        for hint in MARKET_PRICE_LANGUAGE_HINTS.get(_canonical_market_name(market), ())
    )


def _source_text_for_market_detection(source: dict[str, Any]) -> str:
    parsed = urlparse(str(source.get("link") or source.get("url") or ""))
    host = parsed.netloc.lower()
    return " ".join(
        [
            str(source.get("title") or ""),
            str(source.get("snippet") or source.get("search_snippet") or ""),
            str(source.get("page_excerpt") or ""),
            host,
            str(source.get("link") or source.get("url") or ""),
        ]
    )


def _market_alignment(source: dict[str, Any], assigned_market: str) -> dict[str, Any]:
    canonical_market = _canonical_market_name(assigned_market)
    text = _source_text_for_market_detection(source)
    likely_markets = [
        market
        for market in MARKET_ALIASES
        if _market_mentions(text, market)
    ]
    same_market = canonical_market in likely_markets
    other_markets = [market for market in likely_markets if market != canonical_market]

    if same_market and not other_markets:
        alignment = "same_market"
    elif same_market and other_markets:
        alignment = "mixed_market"
    elif other_markets:
        alignment = "cross_market_reference"
    else:
        alignment = "unknown_market"

    return {
        "assigned_market": canonical_market,
        "market_currency_code": _currency_code_for_market(canonical_market),
        "market_alignment": alignment,
        "likely_markets": likely_markets,
    }


def _annotate_source_market(source: dict[str, Any], assigned_market: str) -> dict[str, Any]:
    return {
        **source,
        **_market_alignment(source, assigned_market),
    }


def _market_query_pack(
    product_category: str,
    market: str,
    date_range: str | None = None,
) -> list[dict[str, str]]:
    canonical_market = _canonical_market_name(market)
    date_text = (date_range or "recent").strip() or "recent"
    subject = str(product_category or "").strip()
    return [
        {
            "query_type": "sales",
            "query": f'"{subject}" sales "{canonical_market}" "{date_text}"',
        },
        {
            "query_type": "market_share",
            "query": f'"{subject}" market share "{canonical_market}"',
        },
        {
            "query_type": "pricing_competitors",
            "query": f'"{subject}" pricing competitors "{canonical_market}"',
        },
        {
            "query_type": "availability",
            "query": f'"{subject}" availability "{canonical_market}"',
        },
        {
            "query_type": "demand_trend",
            "query": f'"{subject}" reviews demand trend "{canonical_market}"',
        },
    ]


def _source_key(source: dict[str, Any]) -> str:
    raw_url = str(source.get("link") or source.get("url") or "").strip()
    if raw_url:
        parsed = urlparse(raw_url)
        normalized_path = parsed.path.rstrip("/")
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{normalized_path}".strip("/")
    return "|".join(
        [
            str(source.get("title") or "").strip().lower(),
            str(source.get("snippet") or source.get("search_snippet") or "").strip().lower(),
        ]
    )


def _domain_for_source(source: dict[str, Any]) -> str:
    raw_url = str(source.get("link") or source.get("url") or "").strip()
    return urlparse(raw_url).netloc.lower() if raw_url else ""


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_sources: list[dict[str, Any]] = []
    for source in sources:
        key = _source_key(source)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_sources.append(source)
    return unique_sources


def _assign_source_ids_to_market_results(
    market_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_ids: dict[str, str] = {}
    next_index = 1
    for result in market_results:
        for source in result.get("sources", []):
            key = _source_key(source)
            if key not in source_ids:
                source_ids[key] = f"S{next_index}"
                next_index += 1
            source["source_id"] = source_ids[key]
        result["competitor_signals"] = [
            {
                "source_id": source.get("source_id", ""),
                "title": source.get("title", ""),
                "snippet": source.get("snippet", ""),
                "link": source.get("link", ""),
                "source_type": source.get("source_type", ""),
                "market_alignment": source.get("market_alignment", "unknown_market"),
                "likely_markets": source.get("likely_markets", []),
                "query_type": source.get("query_type", ""),
            }
            for source in result.get("sources", [])
        ]
    return market_results


def _attach_deep_read_evidence(
    market_results: list[dict[str, Any]],
    deep_read_max_pages: int,
    deep_read_concurrency: int,
    deep_read_timeout_seconds: int,
    deep_read_max_chars: int,
) -> list[dict[str, Any]]:
    for result in market_results:
        evidence = _deep_read_sources(
            result.get("sources", [])[: max(0, deep_read_max_pages)],
            deep_read_concurrency,
            deep_read_timeout_seconds,
            deep_read_max_chars,
        )
        result["evidence"] = evidence
        evidence_by_id = {
            item.get("source_id"): item
            for item in evidence
            if item.get("source_id")
        }
        for source in result.get("sources", []):
            evidence_item = evidence_by_id.get(source.get("source_id"))
            if not evidence_item:
                continue
            source["read_status"] = evidence_item.get("read_status", "snippet_only")
            if evidence_item.get("page_excerpt"):
                source["page_excerpt"] = evidence_item["page_excerpt"]
            if evidence_item.get("final_url"):
                source["final_url"] = evidence_item["final_url"]
    return market_results


def _source_reliability(source: dict[str, Any]) -> str:
    alignment = source.get("market_alignment")
    read_status = source.get("read_status", "snippet_only")
    if alignment == "same_market" and read_status == "ok":
        return "high"
    if alignment in {"same_market", "mixed_market"}:
        return "medium"
    if alignment == "cross_market_reference":
        return "low_for_assigned_market"
    return "source_needs_validation"


def _source_bibliography_from_market_results(
    market_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bibliography: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in market_results:
        for source in result.get("sources", []):
            source_id = str(source.get("source_id") or "")
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            key_points = [
                value
                for value in [
                    str(source.get("snippet") or "").strip(),
                    str(source.get("page_excerpt") or "").strip()[:600],
                ]
                if value
            ]
            bibliography.append(
                {
                    "source_id": source_id,
                    "title": source.get("title", ""),
                    "url": source.get("link") or source.get("url") or "",
                    "domain": _domain_for_source(source),
                    "market": source.get("assigned_market") or result.get("market", ""),
                    "source_type": source.get("source_type", ""),
                    "read_status": source.get("read_status", "snippet_only"),
                    "market_alignment": source.get("market_alignment", "unknown_market"),
                    "currency_context": source.get("market_currency_code") or "source_unspecified",
                    "key_points": key_points,
                    "reliability": _source_reliability(source),
                }
            )
    return bibliography


def _public_market_fact_candidates(
    market_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for result in market_results:
        market = str(result.get("market") or "")
        for source in result.get("sources", []):
            snippet = str(source.get("snippet") or "").strip()
            source_id = str(source.get("source_id") or "")
            if not snippet or not source_id:
                continue
            candidates.append(
                {
                    "market": market,
                    "fact_type": source.get("query_type") or "source_signal",
                    "statement": snippet,
                    "value": "not_extracted",
                    "time_period": "source_unspecified",
                    "source_ids": [source_id],
                    "confidence": "medium" if source.get("market_alignment") == "same_market" else "low",
                }
            )
    return candidates


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
    successful_live_fetch_count: int = 0
    last_metrics_status: str | None = None

    def _run(
        self,
        platform: str,
        region: str,
        date_range: str,
        currency: str | None = None,
    ) -> dict[str, Any]:
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
            object.__setattr__(self, "last_metrics_status", "unavailable")
            return self._metrics_unavailable(
                platform,
                region,
                date_range,
                currency,
                "No Shopify or Amazon commerce provider credentials are configured.",
            )

        try:
            result = fetch_commerce_metrics(config, platform, region, date_range)
            object.__setattr__(self, "successful_live_fetch_count", self.successful_live_fetch_count + 1)
            object.__setattr__(self, "last_metrics_status", "live_provider")
            return result
        except Exception as exc:
            logger.warning("E-commerce metrics fetch failed: %s", exc)
            object.__setattr__(self, "last_metrics_status", "unavailable")
            return self._metrics_unavailable(
                platform,
                region,
                date_range,
                currency,
                "Configured commerce provider call failed; KPI metrics are unavailable.",
                provider_error=str(exc),
            )

    @staticmethod
    @lru_cache(maxsize=32)
    def _metrics_unavailable(
        platform: str,
        region: str,
        date_range: str,
        currency: str | None = None,
        reason: str = "Real platform metrics are unavailable.",
        provider_error: str | None = None,
    ) -> dict[str, Any]:
        logger.info("E-commerce platform metrics unavailable")
        currency_code = _currency_code_for_market(region, currency)
        payload = {
            "platform": platform,
            "region": region,
            "date_range": date_range,
            "currency": currency_code,
            "data_source": "metrics_unavailable",
            "confidence_level": "none",
            "assumption_notice": reason,
            "metrics": {},
            "status": "unavailable",
        }
        if provider_error:
            payload["provider_error"] = provider_error
        return payload


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

    def _run(
        self,
        product_category: str,
        target_markets: str,
        date_range: str | None = None,
    ) -> dict[str, Any]:
        if not self.serper_api_key:
            return self._dev_fallback(product_category, target_markets)

        try:
            return _fetch_serper_competitor_benchmarks(
                self.serper_api_key,
                product_category,
                target_markets,
                date_range,
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
    date_range: str | None = None,
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
            date_range,
            deep_read_enabled,
            deep_read_max_pages,
            deep_read_concurrency,
            deep_read_timeout_seconds,
            deep_read_max_chars,
        )
        for market in markets
    ]
    market_results = _assign_source_ids_to_market_results(market_results)
    if deep_read_enabled:
        market_results = _attach_deep_read_evidence(
            market_results,
            deep_read_max_pages,
            deep_read_concurrency,
            deep_read_timeout_seconds,
            deep_read_max_chars,
        )
    sources = [
        source
        for result in market_results
        for source in result.get("sources", [])
    ]
    source_bibliography = _source_bibliography_from_market_results(market_results)
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
        "source_bibliography": source_bibliography,
        "public_market_fact_candidates": _public_market_fact_candidates(market_results),
        "source_selection_rule": (
            "Use source_bibliography.source_id values for public_market_facts, source_evidence, "
            "and market_verdicts. Keep public web facts separate from internal/platform KPIs."
        ),
    }


def _search_market_competitors(
    serper_api_key: str,
    product_category: str,
    market: str,
    date_range: str | None,
    deep_read_enabled: bool,
    deep_read_max_pages: int,
    deep_read_concurrency: int,
    deep_read_timeout_seconds: int,
    deep_read_max_chars: int,
) -> dict[str, Any]:
    canonical_market = _canonical_market_name(market)
    query_pack = _market_query_pack(product_category, canonical_market, date_range)
    sources: list[dict[str, Any]] = []
    for query_def in query_pack:
        response = httpx.post(
            SERPER_SEARCH_URL,
            headers={
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json",
            },
            json={"q": query_def["query"], "num": 5},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        organic_results = payload.get("organic") or []
        shopping_results = payload.get("shopping") or []
        sources.extend(
            {
                **_annotate_source_market(source, canonical_market),
                "query_type": query_def["query_type"],
                "query": query_def["query"],
                "read_status": "snippet_only",
            }
            for source in _extract_sources([*organic_results, *shopping_results])
        )
    sources = _dedupe_sources(sources)
    return {
        "market": canonical_market,
        "requested_market": market,
        "market_currency_code": _currency_code_for_market(canonical_market),
        "market_aliases": list(_market_aliases(canonical_market)),
        "query_pack": query_pack,
        "source_selection_rule": (
            "Use same_market sources for this market's claims. Do not assign "
            "cross_market_reference sources to this market except as explicit "
            "availability or source-coverage caveats."
        ),
        "competitor_signals": [],
        "sources": sources,
        "deep_read_enabled": deep_read_enabled,
        "evidence": [],
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
    assigned_market = str(source.get("assigned_market") or "")
    base_payload = {
        "source_id": source.get("source_id", ""),
        "title": source.get("title", ""),
        "url": url,
        "search_snippet": source.get("snippet", ""),
        "source_type": source.get("source_type", "organic"),
        "query_type": source.get("query_type", ""),
        "query": source.get("query", ""),
        "assigned_market": source.get("assigned_market", ""),
        "market_currency_code": source.get("market_currency_code", ""),
        "market_alignment": source.get("market_alignment", "unknown_market"),
        "likely_markets": source.get("likely_markets", []),
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
        page_excerpt = text[:max(0, max_chars)]
        if assigned_market:
            base_payload.update(
                _market_alignment(
                    {
                        **source,
                        "url": str(response.url),
                        "page_excerpt": page_excerpt,
                    },
                    assigned_market,
                )
            )
        return {
            **base_payload,
            "read_status": "ok",
            "final_url": str(response.url),
            "page_excerpt": page_excerpt,
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
