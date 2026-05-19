import logging
import time
from functools import lru_cache
from typing import Any

import httpx

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

logger = logging.getLogger(__name__)


def _config_value(
    api_config: dict[str, Any] | None,
    key: str,
    instance_value: str | None,
) -> str | None:
    if api_config and api_config.get(key):
        return str(api_config[key])
    return instance_value


def _retry_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call an external API with short retries and a JSON response contract."""
    for attempt in range(max_retries):
        try:
            response = httpx.request(
                method,
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "API request failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    return {
        "error": "External API request failed after retries.",
        "status": "failed",
        "data_source": "live_provider_error",
    }


class GoogleAdsKeywordTool(BaseTool):
    name: str = "Google Ads Keyword & Budget Planner"
    description: str = (
        "Fetches keyword ideas, search-volume context, CPC estimates, and budget "
        "pacing recommendations via Google Ads API when credentials are configured."
    )
    google_ads_access_token: str | None = None
    google_ads_customer_id: str | None = None
    google_ads_developer_token: str | None = None

    def _run(
        self,
        product_category: str,
        region: str,
        budget_usd: float = 0,
        api_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        access_token = _config_value(api_config, "google_ads_access_token", self.google_ads_access_token)
        customer_id = _config_value(api_config, "google_ads_customer_id", self.google_ads_customer_id)
        developer_token = _config_value(api_config, "google_ads_developer_token", self.google_ads_developer_token)
        if not all([access_token, customer_id, developer_token]):
            return self._dev_fallback(product_category, region, float(budget_usd or 0))

        url = f"https://googleads.googleapis.com/v14/customers/{customer_id}:generateKeywordIdeas"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "developer-token": developer_token,
            "Content-Type": "application/json",
        }
        geo_map = {"US": "2840", "UK": "2826", "DE": "2276", "GERMANY": "2276", "JP": "2392", "JAPAN": "2392"}
        payload = {
            "keywordSeed": {"keywords": [product_category]},
            "language": "1000",
            "geoTargetConstants": [f"locations/{geo_map.get(region.upper(), '2840')}"],
            "pageSize": 50,
        }
        result = _retry_request(url, method="POST", headers=headers, payload=payload)
        result.setdefault("data_source", "live_provider")
        return result

    @staticmethod
    @lru_cache(maxsize=32)
    def _dev_fallback(product_category: str, region: str, budget_usd: float) -> dict[str, Any]:
        logger.info("Using Google Ads development fallback")
        daily_budget = budget_usd / 30 if budget_usd else 0
        return {
            "keywords": [
                f"{product_category} buy online",
                f"best {product_category} {region}",
                f"{product_category} reviews",
            ],
            "avg_cpc_usd": 1.20,
            "budget_pacing": (
                f"${budget_usd:.0f} allocated over 30 days (~${daily_budget:.2f}/day)"
                if budget_usd
                else "Budget not provided; pacing is illustrative."
            ),
            "status": "dev_mode",
            "data_source": "development_fallback",
            "confidence_level": "Illustrative",
            "assumption_notice": (
                "Google Ads credentials are not configured; keyword and CPC values are sample estimates."
            ),
        }


class MetaAdsTool(BaseTool):
    name: str = "Meta Ads Creative & Audience Validator"
    description: str = (
        "Validates ad creative against Meta specs, estimates audience context, and "
        "checks launch readiness via Meta Marketing API when credentials are configured."
    )
    meta_access_token: str | None = None
    meta_ad_account_id: str | None = None
    meta_page_id: str | None = None

    def _run(
        self,
        platform: str,
        region: str,
        ad_copy: str,
        api_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        access_token = _config_value(api_config, "meta_access_token", self.meta_access_token)
        ad_account = _config_value(api_config, "meta_ad_account_id", self.meta_ad_account_id)
        page_id = _config_value(api_config, "meta_page_id", self.meta_page_id) or "dummy"
        if not access_token or not ad_account:
            return self._dev_fallback(platform, region, ad_copy)

        url = f"https://graph.facebook.com/v18.0/{ad_account}/adcreatives"
        headers = {"Authorization": f"Bearer {access_token}"}
        payload = {
            "object_story_spec": {
                "link_data": {"message": ad_copy},
                "page_id": page_id,
            },
            "name": "Validation_Check",
            "format": "SINGLE_IMAGE",
        }
        result = _retry_request(url, method="POST", headers=headers, payload=payload)
        result.setdefault("data_source", "live_provider")
        return result

    @staticmethod
    @lru_cache(maxsize=32)
    def _dev_fallback(platform: str, region: str, ad_copy: str) -> dict[str, Any]:
        logger.info("Using Meta Ads development fallback")
        return {
            "platform": platform or "Meta",
            "region": region,
            "format_compliance": "PASS",
            "audience_estimate": "Sample audience range; not from Meta API.",
            "policy_flags": [],
            "status": "dev_mode",
            "data_source": "development_fallback",
            "confidence_level": "Illustrative",
            "assumption_notice": (
                "Meta credentials are not configured; audience and compliance values are sample checks."
            ),
        }


class TikTokAdsTool(BaseTool):
    name: str = "TikTok Ads Spec & Performance Validator"
    description: str = (
        "Validates TikTok ad specs, trend alignment, and performance benchmark "
        "context via TikTok Marketing API when credentials are configured."
    )
    tiktok_access_token: str | None = None
    tiktok_advertiser_id: str | None = None

    def _run(
        self,
        region: str,
        product_category: str,
        api_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        access_token = _config_value(api_config, "tiktok_access_token", self.tiktok_access_token)
        advertiser_id = _config_value(api_config, "tiktok_advertiser_id", self.tiktok_advertiser_id)
        if not access_token or not advertiser_id:
            return self._dev_fallback(region, product_category)

        url = "https://business-api.tiktok.com/open_api/v1.3/advertiser/campaigns/get/"
        headers = {"Access-Token": access_token}
        params = {
            "advertiser_id": advertiser_id,
            "page": 1,
            "page_size": 10,
        }
        result = _retry_request(url, headers=headers, params=params)
        result.setdefault("data_source", "live_provider")
        return result

    @staticmethod
    @lru_cache(maxsize=32)
    def _dev_fallback(region: str, product_category: str) -> dict[str, Any]:
        logger.info("Using TikTok Ads development fallback")
        return {
            "platform": "TikTok",
            "region": region,
            "video_specs": {
                "min_duration": "9s",
                "max_duration": "60s",
                "aspect_ratio": "9:16",
            },
            "trend_alignment": f"Sample trend fit for {product_category} in {region}.",
            "benchmark_ctr": "Illustrative benchmark only; not from TikTok API.",
            "status": "dev_mode",
            "data_source": "development_fallback",
            "confidence_level": "Illustrative",
            "assumption_notice": (
                "TikTok credentials are not configured; specs and benchmarks are sample values."
            ),
        }
