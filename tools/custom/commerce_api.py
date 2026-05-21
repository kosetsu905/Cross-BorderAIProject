from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx


REGION_TO_COUNTRY = {
    "US": "US",
    "UNITED STATES": "US",
    "UK": "GB",
    "UNITED KINGDOM": "GB",
    "GB": "GB",
    "DE": "DE",
    "GERMANY": "DE",
    "JP": "JP",
    "JAPAN": "JP",
    "EU": "EU",
}


@dataclass(frozen=True)
class CommerceApiConfig:
    shopify_store_domain: str | None = None
    shopify_admin_access_token: str | None = None
    shopify_api_version: str = "2025-07"
    amazon_sp_api_endpoint: str | None = None
    amazon_sp_api_access_token: str | None = None
    amazon_marketplace_ids: str | None = None


def created_after_from_range(date_range: str) -> str:
    text = (date_range or "").lower()
    days = 30
    for candidate in (7, 14, 30, 60, 90, 180, 365):
        if str(candidate) in text:
            days = candidate
            break
    return (datetime.now(UTC) - timedelta(days=days)).isoformat().replace("+00:00", "Z")


def normalize_region(region: str) -> str:
    return REGION_TO_COUNTRY.get(region.strip().upper(), region.strip().upper())


def parse_marketplace_ids(raw_marketplace_ids: str | None) -> list[str]:
    return [
        item.strip()
        for item in (raw_marketplace_ids or "").split(",")
        if item.strip()
    ]


def summarize_shopify_orders(
    orders: list[dict[str, Any]],
    platform: str,
    region: str,
    date_range: str,
) -> dict[str, Any]:
    country = normalize_region(region)
    filtered_orders = [
        order
        for order in orders
        if country == "EU"
        or (order.get("shipping_address") or {}).get("country_code") == country
        or (order.get("billing_address") or {}).get("country_code") == country
    ]
    if not filtered_orders:
        filtered_orders = orders

    total_sales = sum(float(order.get("total_price") or 0) for order in filtered_orders)
    cancelled = sum(1 for order in filtered_orders if order.get("cancelled_at"))
    paid = sum(1 for order in filtered_orders if order.get("financial_status") in {"paid", "partially_paid"})
    sku_counts: dict[str, int] = {}
    for order in filtered_orders:
        for item in order.get("line_items") or []:
            sku = item.get("sku") or item.get("title") or "UNKNOWN"
            sku_counts[sku] = sku_counts.get(sku, 0) + int(item.get("quantity") or 0)

    top_sku = max(sku_counts, key=sku_counts.get) if sku_counts else "not_available"
    currency = filtered_orders[0].get("currency") if filtered_orders else "not_available"

    return {
        "platform": platform or "Shopify",
        "region": region,
        "date_range": date_range,
        "data_source": "external_shopify_admin_api",
        "confidence_level": "high",
        "status": "live_provider",
        "metrics": {
            "total_sales": round(total_sales, 2),
            "currency": currency,
            "order_count": len(filtered_orders),
            "paid_order_count": paid,
            "cancelled_order_count": cancelled,
            "conversion_rate": "not_available_from_orders_api",
            "cpc": "not_available_from_orders_api",
            "roas": "not_available_from_orders_api",
            "inventory_status": "not_available_from_orders_api",
            "top_selling_sku": top_sku,
        },
        "assumption_notice": (
            "Metrics are derived from Shopify Admin orders data. Conversion rate, CPC, ROAS, "
            "and inventory require additional analytics/ad/inventory APIs."
        ),
    }


def summarize_amazon_orders(
    orders: list[dict[str, Any]],
    platform: str,
    region: str,
    date_range: str,
) -> dict[str, Any]:
    total_sales = 0.0
    currency = "not_available"
    shipped = 0
    cancelled = 0
    for order in orders:
        amount = (order.get("OrderTotal") or {}).get("Amount")
        if amount is not None:
            total_sales += float(amount)
        currency = (order.get("OrderTotal") or {}).get("CurrencyCode") or currency
        status = order.get("OrderStatus")
        if status == "Shipped":
            shipped += 1
        if status == "Canceled":
            cancelled += 1

    return {
        "platform": platform or "Amazon",
        "region": region,
        "date_range": date_range,
        "data_source": "external_amazon_sp_api",
        "confidence_level": "medium",
        "status": "live_provider",
        "metrics": {
            "total_sales": round(total_sales, 2),
            "currency": currency,
            "order_count": len(orders),
            "shipped_order_count": shipped,
            "cancelled_order_count": cancelled,
            "conversion_rate": "not_available_from_orders_api",
            "cpc": "not_available_from_orders_api",
            "roas": "not_available_from_orders_api",
            "inventory_status": "not_available_from_orders_api",
            "top_selling_sku": "not_available_from_orders_api",
        },
        "assumption_notice": (
            "Metrics are derived from Amazon SP-API Orders data. Direct SP-API calls normally "
            "require AWS SigV4 signing; use an SP-API-compatible signed proxy or signed client."
        ),
    }


def fetch_shopify_metrics(
    config: CommerceApiConfig,
    platform: str,
    region: str,
    date_range: str,
) -> dict[str, Any]:
    if not config.shopify_store_domain or not config.shopify_admin_access_token:
        raise ValueError("Shopify store domain and admin access token are required.")

    domain = config.shopify_store_domain.replace("https://", "").replace("http://", "").strip("/")
    url = f"https://{domain}/admin/api/{config.shopify_api_version}/orders.json"
    headers = {"X-Shopify-Access-Token": config.shopify_admin_access_token}
    params = {
        "status": "any",
        "limit": 250,
        "created_at_min": created_after_from_range(date_range),
    }
    response = httpx.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    return summarize_shopify_orders(response.json().get("orders", []), platform, region, date_range)


def fetch_amazon_metrics(
    config: CommerceApiConfig,
    platform: str,
    region: str,
    date_range: str,
) -> dict[str, Any]:
    marketplace_ids = parse_marketplace_ids(config.amazon_marketplace_ids)
    if not config.amazon_sp_api_endpoint or not config.amazon_sp_api_access_token or not marketplace_ids:
        raise ValueError("Amazon SP-API endpoint, access token, and marketplace IDs are required.")

    url = f"{config.amazon_sp_api_endpoint.rstrip('/')}/orders/v0/orders"
    headers = {"x-amz-access-token": config.amazon_sp_api_access_token}
    params = {
        "CreatedAfter": created_after_from_range(date_range),
        "MarketplaceIds": ",".join(marketplace_ids),
        "MaxResultsPerPage": 100,
    }
    response = httpx.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json().get("payload", {})
    return summarize_amazon_orders(payload.get("Orders", []), platform, region, date_range)


def fetch_commerce_metrics(
    config: CommerceApiConfig,
    platform: str,
    region: str,
    date_range: str,
) -> dict[str, Any]:
    platform_key = (platform or "").lower()
    if "amazon" in platform_key:
        return fetch_amazon_metrics(config, platform, region, date_range)
    if "shopify" in platform_key:
        return fetch_shopify_metrics(config, platform, region, date_range)
    if config.shopify_store_domain and config.shopify_admin_access_token:
        return fetch_shopify_metrics(config, platform or "Shopify", region, date_range)
    if config.amazon_sp_api_endpoint and config.amazon_sp_api_access_token:
        return fetch_amazon_metrics(config, platform or "Amazon", region, date_range)
    raise ValueError("No commerce provider credentials are configured.")
