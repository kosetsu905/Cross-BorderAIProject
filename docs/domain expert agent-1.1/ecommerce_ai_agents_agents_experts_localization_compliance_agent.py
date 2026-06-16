--- ecommerce_ai_agents/agents/experts/localization_compliance_agent.py (原始)


+++ ecommerce_ai_agents/agents/experts/localization_compliance_agent.py (修改后)
"""
Localization & Compliance Agent
Domain Expert Agent for Cross-Border E-commerce

This agent acts as the "Gatekeeper" ensuring all content, marketing campaigns,
and business operations comply with local cultural norms, legal regulations,
and platform-specific policies before execution.

Features:
- Cultural Adaptation Check (taboos, tone, units)
- Regulatory Compliance (advertising laws, GDPR/PIPL, product restrictions)
- Platform Policy Alignment (TikTok, Amazon, Google Ads, etc.)
- Multi-language Context Validation
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re


class LocalizationComplianceAgent:
    """
    Domain Expert Agent for Localization and Compliance.

    This agent validates content against:
    1. Cultural norms and taboos
    2. Legal regulations by country
    3. Platform-specific advertising policies
    4. Language context and appropriateness
    """

    def __init__(self):
        self.name = "LocalizationComplianceAgent"
        self.version = "1.0.0"

        # Initialize knowledge bases
        self.cultural_rules = self._load_cultural_rules()
        self.regulatory_rules = self._load_regulatory_rules()
        self.platform_policies = self._load_platform_policies()
        self.language_patterns = self._load_language_patterns()

    def _load_cultural_rules(self) -> Dict[str, Any]:
        """Load cultural rules and taboos by region/country."""
        return {
            "CN": {
                "taboos": ["4", "white flowers", "clocks", "green hat"],
                "preferred_colors": ["red", "gold"],
                "date_format": "YYYY-MM-DD",
                "currency": "CNY",
                "tone": "respectful, formal",
                "festivals": ["Spring Festival", "Mid-Autumn Festival", "Double 11"]
            },
            "US": {
                "taboos": ["political sensitivity", "racial stereotypes"],
                "preferred_colors": ["blue", "red"],
                "date_format": "MM/DD/YYYY",
                "currency": "USD",
                "tone": "friendly, direct",
                "festivals": ["Black Friday", "Cyber Monday", "Christmas"]
            },
            "EU": {
                "taboos": ["excessive claims", "privacy violations"],
                "preferred_colors": ["blue", "green"],
                "date_format": "DD/MM/YYYY",
                "currency": "EUR",
                "tone": "professional, privacy-conscious",
                "festivals": ["Christmas", "Summer Sales"]
            },
            "JP": {
                "taboos": ["4", "9", "direct confrontation", "gifts in sets of 4"],
                "preferred_colors": ["white", "red"],
                "date_format": "YYYY/MM/DD",
                "currency": "JPY",
                "tone": "polite, humble",
                "festivals": ["Golden Week", "Obon"]
            },
            "SEA": {
                "taboos": ["head touching", "left hand usage", "religious insensitivity"],
                "preferred_colors": ["varies by country"],
                "date_format": "DD/MM/YYYY",
                "currency": "varies",
                "tone": "respectful, community-focused",
                "festivals": ["Ramadan", "Songkran", "Tet"]
            }
        }

    def _load_regulatory_rules(self) -> Dict[str, Any]:
        """Load regulatory compliance rules by region."""
        return {
            "CN": {
                "advertising_law": "Advertising Law of PRC",
                "prohibited_claims": ["best", "number one", "cure", "guaranteed"],
                "data_privacy": "PIPL (Personal Information Protection Law)",
                "product_restrictions": ["tobacco ads", "medical claims without approval"],
                "required_disclaimers": ["广告", "promotional"]
            },
            "US": {
                "advertising_law": "FTC Guidelines",
                "prohibited_claims": ["false medical claims", "deceptive pricing"],
                "data_privacy": "CCPA (California), COPPA",
                "product_restrictions": ["alcohol to minors", "unsubstantiated health claims"],
                "required_disclaimers": ["Ad", "Sponsored", "#ad"]
            },
            "EU": {
                "advertising_law": "Unfair Commercial Practices Directive",
                "prohibited_claims": ["misleading environmental claims", "hidden costs"],
                "data_privacy": "GDPR",
                "product_restrictions": ["comparison without consent", "aggressive selling"],
                "required_disclaimers": ["Advertisement", "Paid partnership"]
            },
            "GLOBAL": {
                "general": ["no hate speech", "no discrimination", "no illegal products"]
            }
        }

    def _load_platform_policies(self) -> Dict[str, Any]:
        """Load platform-specific advertising policies."""
        return {
            "tiktok": {
                "max_video_length": 60,
                "text_overlay_limit": "20% of screen",
                "prohibited_content": ["misinformation", "dangerous acts", "adult content"],
                "music_restrictions": "copyrighted music requires license",
                "hashtag_rules": "no misleading hashtags"
            },
            "facebook": {
                "text_image_ratio": "20% rule (relaxed but still relevant)",
                "prohibited_content": ["before/after images", "personal attributes"],
                "targeting_restrictions": ["housing", "credit", "employment"],
                "landing_page_quality": "must match ad content"
            },
            "google_ads": {
                "character_limits": {"headline": 30, "description": 90},
                "prohibited_content": ["counterfeit goods", "bypass systems"],
                "quality_score_factors": ["relevance", "CTR", "landing page experience"]
            },
            "amazon": {
                "title_length": 200,
                "bullet_points": 5,
                "prohibited_content": ["subjective claims", "promotional text in listing"],
                "image_requirements": "white background, 1000x1000px minimum"
            }
        }

    def _load_language_patterns(self) -> Dict[str, Any]:
        """Load language-specific patterns and validation rules."""
        return {
            "zh-CN": {
                "encoding": "UTF-8",
                "common_errors": ["simplified/traditional mix", "incorrect measure words"],
                "formality_levels": ["casual", "formal", "honorific"]
            },
            "en-US": {
                "encoding": "UTF-8",
                "common_errors": ["spelling (color/colour)", "date format confusion"],
                "formality_levels": ["casual", "professional", "academic"]
            },
            "ja-JP": {
                "encoding": "UTF-8",
                "common_errors": ["keigo misuse", "kanji reading errors"],
                "formality_levels": ["casual", "polite", "humble", "honorific"]
            },
            "ko-KR": {
                "encoding": "UTF-8",
                "common_errors": ["honorific levels", "spacing rules"],
                "formality_levels": ["casual", "polite", "formal"]
            }
        }

    def validate_content(
        self,
        content: str,
        target_market: str,
        content_type: str = "marketing",
        platform: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of content for a specific market.

        Args:
            content: The content to validate (text, ad copy, product description, etc.)
            target_market: Target country/region code (e.g., 'US', 'CN', 'EU')
            content_type: Type of content ('marketing', 'product', 'support', 'email')
            platform: Optional platform name ('tiktok', 'facebook', 'google_ads', 'amazon')
            language: Optional language code ('zh-CN', 'en-US', etc.)

        Returns:
            Dictionary with validation results, issues found, and recommendations
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "target_market": target_market,
            "content_type": content_type,
            "platform": platform,
            "language": language,
            "overall_status": "PASS",
            "risk_level": "LOW",
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "compliance_score": 100
        }

        # Run all validation checks
        cultural_result = self._check_cultural_compliance(content, target_market)
        regulatory_result = self._check_regulatory_compliance(content, target_market)
        platform_result = self._check_platform_compliance(content, platform) if platform else {"status": "SKIP"}
        language_result = self._check_language_context(content, language, target_market) if language else {"status": "SKIP"}

        # Aggregate results
        all_checks = [cultural_result, regulatory_result, platform_result, language_result]

        for check in all_checks:
            if check.get("status") == "FAIL":
                results["overall_status"] = "FAIL"
                results["risk_level"] = "HIGH"
                results["issues"].extend(check.get("issues", []))
                results["compliance_score"] -= check.get("penalty", 20)
            elif check.get("status") == "WARNING":
                if results["risk_level"] != "HIGH":
                    results["risk_level"] = "MEDIUM"
                results["warnings"].extend(check.get("warnings", []))
                results["compliance_score"] -= check.get("penalty", 5)

            results["recommendations"].extend(check.get("recommendations", []))

        results["compliance_score"] = max(0, results["compliance_score"])

        return results

    def _check_cultural_compliance(self, content: str, target_market: str) -> Dict[str, Any]:
        """Check content against cultural norms and taboos."""
        result = {"status": "PASS", "issues": [], "warnings": [], "recommendations": [], "penalty": 0}

        market_rules = self.cultural_rules.get(target_market, {})
        if not market_rules:
            result["warnings"].append(f"No specific cultural rules found for market: {target_market}")
            return result

        content_lower = content.lower()

        # Check for taboos
        for taboo in market_rules.get("taboos", []):
            if taboo.lower() in content_lower:
                result["status"] = "FAIL"
                result["issues"].append({
                    "type": "CULTURAL_TABOO",
                    "severity": "HIGH",
                    "detail": f"Content contains culturally sensitive element: '{taboo}'",
                    "market": target_market
                })
                result["penalty"] += 25

        # Check color symbolism (if mentioned)
        preferred_colors = market_rules.get("preferred_colors", [])
        # This is a simplified check; real implementation would use NLP

        # Check tone appropriateness
        expected_tone = market_rules.get("tone", "")
        if expected_tone:
            result["recommendations"].append(
                f"Ensure content tone aligns with local expectations: {expected_tone}"
            )

        # Festival timing check
        festivals = market_rules.get("festivals", [])
        current_month = datetime.now().month
        # Simplified festival check - would need full calendar in production

        return result

    def _check_regulatory_compliance(self, content: str, target_market: str) -> Dict[str, Any]:
        """Check content against legal and regulatory requirements."""
        result = {"status": "PASS", "issues": [], "warnings": [], "recommendations": [], "penalty": 0}

        # Get market-specific rules
        market_rules = self.regulatory_rules.get(target_market, {})
        global_rules = self.regulatory_rules.get("GLOBAL", {})

        # Combine rules
        prohibited_claims = market_rules.get("prohibited_claims", []) + \
                           global_rules.get("prohibited_claims", [])
        required_disclaimers = market_rules.get("required_disclaimers", [])

        content_lower = content.lower()

        # Check for prohibited claims
        for claim in prohibited_claims:
            if claim.lower() in content_lower:
                result["status"] = "FAIL"
                result["issues"].append({
                    "type": "REGULATORY_VIOLATION",
                    "severity": "CRITICAL",
                    "detail": f"Content contains prohibited claim: '{claim}'",
                    "regulation": market_rules.get("advertising_law", "Local Advertising Law"),
                    "market": target_market
                })
                result["penalty"] += 30

        # Check for required disclaimers
        has_disclaimer = any(
            disclaimer.lower() in content_lower
            for disclaimer in required_disclaimers
        )
        if required_disclaimers and not has_disclaimer:
            result["status"] = "WARNING" if result["status"] == "PASS" else result["status"]
            result["warnings"].append({
                "type": "MISSING_DISCLAIMER",
                "severity": "MEDIUM",
                "detail": f"Content may require disclaimer. Recommended: {required_disclaimers}",
                "market": target_market
            })
            result["penalty"] += 10
            result["recommendations"].append(
                f"Add appropriate disclaimer: {', '.join(required_disclaimers)}"
            )

        # Data privacy reminder
        data_privacy = market_rules.get("data_privacy", "")
        if data_privacy:
            result["recommendations"].append(
                f"Ensure compliance with {data_privacy} when collecting user data"
            )

        return result

    def _check_platform_compliance(self, content: str, platform: str) -> Dict[str, Any]:
        """Check content against platform-specific policies."""
        result = {"status": "PASS", "issues": [], "warnings": [], "recommendations": [], "penalty": 0}

        platform = platform.lower()
        platform_rules = self.platform_policies.get(platform, {})

        if not platform_rules:
            result["warnings"].append(f"No specific rules found for platform: {platform}")
            return result

        # Check character limits for platforms that have them
        if platform == "google_ads":
            char_limits = platform_rules.get("character_limits", {})
            # This would need structured input (headline, description) in real implementation
            result["recommendations"].append(
                f"Google Ads limits: Headline={char_limits.get('headline', 'N/A')} chars, "
                f"Description={char_limits.get('description', 'N/A')} chars"
            )

        # Check prohibited content
        prohibited = platform_rules.get("prohibited_content", [])
        content_lower = content.lower()

        for item in prohibited:
            if item.lower() in content_lower:
                result["status"] = "FAIL"
                result["issues"].append({
                    "type": "PLATFORM_POLICY_VIOLATION",
                    "severity": "HIGH",
                    "detail": f"Content may violate {platform} policy: '{item}'",
                    "platform": platform
                })
                result["penalty"] += 25

        # Platform-specific recommendations
        if platform == "tiktok":
            result["recommendations"].append(
                "TikTok: Keep videos engaging in first 3 seconds, use trending audio"
            )
        elif platform == "facebook":
            result["recommendations"].append(
                "Facebook: Avoid before/after images and personal attribute targeting"
            )
        elif platform == "amazon":
            result["recommendations"].append(
                "Amazon: Use white background images, avoid promotional text in titles"
            )

        return result

    def _check_language_context(self, content: str, language: str, target_market: str) -> Dict[str, Any]:
        """Check language-specific context and appropriateness."""
        result = {"status": "PASS", "issues": [], "warnings": [], "recommendations": [], "penalty": 0}

        lang_patterns = self.language_patterns.get(language, {})

        if not lang_patterns:
            return result

        # Check for common errors
        common_errors = lang_patterns.get("common_errors", [])

        # Basic encoding check
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            result["status"] = "FAIL"
            result["issues"].append({
                "type": "ENCODING_ERROR",
                "severity": "CRITICAL",
                "detail": "Content contains invalid characters for specified language",
                "language": language
            })
            result["penalty"] += 20
            return result

        # Formality recommendation
        formality_levels = lang_patterns.get("formality_levels", [])
        if formality_levels:
            result["recommendations"].append(
                f"Available formality levels for {language}: {', '.join(formality_levels)}. "
                f"Choose based on target audience."
            )

        # Market-language alignment check
        expected_languages = {
            "CN": ["zh-CN"],
            "US": ["en-US"],
            "JP": ["ja-JP"],
            "KR": ["ko-KR"],
            "DE": ["de-DE"],
            "FR": ["fr-FR"]
        }

        expected = expected_languages.get(target_market, [])
        if expected and language not in expected:
            result["warnings"].append({
                "type": "LANGUAGE_MARKET_MISMATCH",
                "severity": "LOW",
                "detail": f"Language {language} may not be optimal for market {target_market}. "
                         f"Expected: {expected}",
            })
            result["penalty"] += 5

        return result

    def generate_localized_version(
        self,
        content: str,
        source_market: str,
        target_market: str,
        content_type: str = "marketing"
    ) -> Dict[str, Any]:
        """
        Generate recommendations for localizing content from one market to another.

        Note: This provides structural recommendations. Actual translation should be
        done by specialized translation models or human translators.

        Args:
            content: Original content
            source_market: Source market code
            target_market: Target market code
            content_type: Type of content

        Returns:
            Localization recommendations and adaptation guidelines
        """
        source_rules = self.cultural_rules.get(source_market, {})
        target_rules = self.cultural_rules.get(target_market, {})

        recommendations = {
            "source_market": source_market,
            "target_market": target_market,
            "original_preview": content[:100] + "..." if len(content) > 100 else content,
            "adaptation_guidelines": [],
            "cultural_adjustments": [],
            "format_changes": [],
            "legal_considerations": []
        }

        # Tone adjustment
        source_tone = source_rules.get("tone", "neutral")
        target_tone = target_rules.get("tone", "neutral")
        if source_tone != target_tone:
            recommendations["adaptation_guidelines"].append(
                f"Adjust tone from '{source_tone}' to '{target_tone}'"
            )

        # Date format conversion
        source_date_fmt = source_rules.get("date_format", "YYYY-MM-DD")
        target_date_fmt = target_rules.get("date_format", "YYYY-MM-DD")
        if source_date_fmt != target_date_fmt:
            recommendations["format_changes"].append(
                f"Convert date format from {source_date_fmt} to {target_date_fmt}"
            )

        # Currency conversion reminder
        source_currency = source_rules.get("currency", "USD")
        target_currency = target_rules.get("currency", "USD")
        if source_currency != target_currency:
            recommendations["format_changes"].append(
                f"Convert prices from {source_currency} to {target_currency}"
            )

        # Cultural elements to remove/add
        source_taboos = set(source_rules.get("taboos", []))
        target_taboos = set(target_rules.get("taboos", []))

        if target_taboos - source_taboos:
            recommendations["cultural_adjustments"].append(
                f"Avoid these elements in {target_market}: {list(target_taboos - source_taboos)}"
            )

        # Festival alignment
        source_festivals = source_rules.get("festivals", [])
        target_festivals = target_rules.get("festivals", [])
        recommendations["cultural_adjustments"].append(
            f"Replace {source_market} festivals {source_festivals} with "
            f"{target_market} equivalents: {target_festivals}"
        )

        # Legal considerations
        target_regulations = self.regulatory_rules.get(target_market, {})
        if target_regulations:
            recommendations["legal_considerations"].append(
                f"Comply with {target_regulations.get('advertising_law', 'local advertising laws')}"
            )
            if target_regulations.get("required_disclaimers"):
                recommendations["legal_considerations"].append(
                    f"Add required disclaimers: {target_regulations['required_disclaimers']}"
                )

        return recommendations

    def batch_validate(
        self,
        contents: List[Dict[str, Any]],
        default_market: str
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple content items in batch.

        Args:
            contents: List of content dictionaries with keys:
                     - content (str): The content text
                     - target_market (str, optional): Override default market
                     - content_type (str, optional): Type of content
                     - platform (str, optional): Target platform
                     - language (str, optional): Content language
            default_market: Default target market if not specified per item

        Returns:
            List of validation results for each content item
        """
        results = []

        for item in contents:
            content = item.get("content", "")
            target_market = item.get("target_market", default_market)
            content_type = item.get("content_type", "marketing")
            platform = item.get("platform")
            language = item.get("language")

            result = self.validate_content(
                content=content,
                target_market=target_market,
                content_type=content_type,
                platform=platform,
                language=language
            )
            results.append(result)

        return results


# Integration with CrawAI Framework
def create_localization_agent():
    """Factory function to create a Localization & Compliance Agent instance."""
    return LocalizationComplianceAgent()


if __name__ == "__main__":
    # Demo usage
    agent = create_localization_agent()

    # Test content for US market
    test_content_us = """
    🎉 BEST product ever! Guaranteed to cure all your problems!
    Buy now and get number one results! Contact us at midnight on 4/4/2024.
    #ad
    """

    print("=" * 80)
    print("Testing US Market Compliance")
    print("=" * 80)
    result_us = agent.validate_content(
        content=test_content_us,
        target_market="US",
        content_type="marketing",
        platform="facebook",
        language="en-US"
    )

    print(f"Overall Status: {result_us['overall_status']}")
    print(f"Risk Level: {result_us['risk_level']}")
    print(f"Compliance Score: {result_us['compliance_score']}/100")
    print(f"\nIssues Found: {len(result_us['issues'])}")
    for issue in result_us['issues']:
        print(f"  ❌ [{issue['severity']}] {issue['detail']}")

    print(f"\nWarnings: {len(result_us['warnings'])}")
    for warning in result_us['warnings']:
        print(f"  ⚠️  [{warning['severity']}] {warning['detail']}")

    print(f"\nRecommendations:")
    for rec in result_us['recommendations']:
        print(f"  💡 {rec}")

    # Test content for China market
    test_content_cn = """
    🎉 最好的产品！保证治愈所有问题！
    立即购买，获得第一名的结果！联系我们在4月4日午夜。
    广告
    """

    print("\n" + "=" * 80)
    print("Testing China Market Compliance")
    print("=" * 80)
    result_cn = agent.validate_content(
        content=test_content_cn,
        target_market="CN",
        content_type="marketing",
        platform="wechat",
        language="zh-CN"
    )

    print(f"Overall Status: {result_cn['overall_status']}")
    print(f"Risk Level: {result_cn['risk_level']}")
    print(f"Compliance Score: {result_cn['compliance_score']}/100")
    print(f"\nIssues Found: {len(result_cn['issues'])}")
    for issue in result_cn['issues']:
        print(f"  ❌ [{issue['severity']}] {issue['detail']}")

    print(f"\nRecommendations:")
    for rec in result_cn['recommendations']:
        print(f"  💡 {rec}")

    # Test localization recommendations
    print("\n" + "=" * 80)
    print("Localization Recommendations: US → Japan")
    print("=" * 80)
    localization_rec = agent.generate_localized_version(
        content="Get the best deal this Black Friday! Limited time offer!",
        source_market="US",
        target_market="JP",
        content_type="marketing"
    )

    print(f"Source: {localization_rec['source_market']} → Target: {localization_rec['target_market']}")
    print("\nAdaptation Guidelines:")
    for guideline in localization_rec['adaptation_guidelines']:
        print(f"  • {guideline}")

    print("\nCultural Adjustments:")
    for adj in localization_rec['cultural_adjustments']:
        print(f"  • {adj}")

    print("\nFormat Changes:")
    for fmt in localization_rec['format_changes']:
        print(f"  • {fmt}")

    print("\nLegal Considerations:")
    for legal in localization_rec['legal_considerations']:
        print(f"  • {legal}")