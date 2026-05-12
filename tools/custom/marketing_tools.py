from typing import Any

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool


class PlatformAdSpecsTool(BaseTool):
    name: str = "Platform Ad Specifications Fetcher"
    description: str = (
        "Retrieves ad format limits, character constraints, and policy guidelines "
        "for major ad platforms."
    )

    def _run(self, platform: str, region: str | None = None) -> dict[str, Any]:
        specs = {
            "meta": {
                "headline_limit": 40,
                "text_limit": 125,
                "image_ratio": "1.91:1",
                "policy_notes": "Avoid absolute claims and follow Meta Ad Policies.",
            },
            "google": {
                "headline_limit": 30,
                "description_limit": 90,
                "policy_notes": "Trademark rules apply and destination pages must match claims.",
            },
            "tiktok": {
                "text_limit": 100,
                "video_length": "9-60s",
                "policy_notes": "Avoid unrealistic health or wealth claims and use clear CTAs.",
            },
            "amazon": {
                "title_limit": 200,
                "bullet_points": 5,
                "policy_notes": "Avoid promotional language in titles and follow regional rules.",
            },
        }
        return specs.get(
            platform.lower(),
            {"error": "Platform not supported", "platform": platform, "region": region},
        )


class KeywordResearchTool(BaseTool):
    name: str = "E-commerce Keyword Research Tool"
    description: str = (
        "Generates high-intent keyword ideas and rough demand estimates for "
        "cross-border e-commerce campaigns."
    )

    def _run(self, product_category: str, region: str) -> dict[str, Any]:
        return {
            "primary_keywords": [
                f"{product_category} buy online",
                f"best {product_category}",
            ],
            "long_tail_keywords": [
                f"affordable {product_category} for {region}",
                f"{product_category} free shipping",
            ],
            "search_volume_estimate": "Medium to High",
            "cpc_estimate": "$0.50 - $2.10",
            "status": "dev_mode",
        }


class ComplianceCheckerTool(BaseTool):
    name: str = "Ad Compliance & Policy Validator"
    description: str = (
        "Validates ad copy against regional regulations and platform advertising policies."
    )

    def _run(self, ad_text: str, platform: str, region: str) -> dict[str, Any]:
        flags: list[str] = []
        normalized_text = ad_text.lower()
        normalized_region = region.lower()

        restricted_claims = {"free", "guaranteed", "#1", "best ever", "cure"}
        if any(term in normalized_text for term in restricted_claims):
            flags.append("Contains restricted superlatives or absolute claims.")

        if normalized_region in {"eu", "uk", "germany", "france"} and "privacy" not in normalized_text:
            flags.append("GDPR market: add privacy/data disclaimer if collecting lead data.")

        return {
            "compliant": not flags,
            "platform": platform,
            "region": region,
            "flags": flags or ["No issues detected."],
            "recommendations": (
                "Replace absolute claims with qualified wording and include privacy "
                "policy links for lead capture campaigns."
            ),
        }
