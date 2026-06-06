import unittest

from crews.analytics_crew import _apply_provider_status
from tools.custom.analytics_tools import (
    EcomPlatformMetricsTool,
    _annotate_source_market,
    _assign_source_ids_to_market_results,
    _canonical_market_name,
    _currency_code_for_market,
    _market_query_pack,
    _public_market_fact_candidates,
    _source_bibliography_from_market_results,
)


class AnalyticsToolCurrencyAndMarketTests(unittest.TestCase):
    def test_platform_metrics_without_credentials_returns_empty_metrics(self) -> None:
        result = EcomPlatformMetricsTool()._run("Shopify", "Australia", "Last 30 Days", "AUD")

        self.assertEqual(result["currency"], "AUD")
        self.assertEqual(result["data_source"], "metrics_unavailable")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["metrics"], {})
        self.assertNotIn("total_sales", result["metrics"])
        self.assertNotIn("conversion_rate", result["metrics"])
        self.assertNotIn("roas", result["metrics"])

    def test_provider_status_clears_regional_kpis_without_live_metrics(self) -> None:
        result = {
            "regional_kpis": [
                {
                    "region": "Australia",
                    "sales_volume": "$45,000.00",
                    "currency_code": "AUD",
                    "conversion_rate": "2.0%",
                    "roas": "2.0",
                }
            ],
            "data_quality_notes": [],
            "public_market_facts": [
                {
                    "market": "Australia",
                    "fact_type": "sales",
                    "statement": "Public-source sales signal.",
                    "value": "not_extracted",
                    "time_period": "source_unspecified",
                    "source_ids": ["S1"],
                    "confidence": "medium",
                }
            ],
        }

        normalized = _apply_provider_status(
            result,
            {"serper_api_key": "serper-key"},
            {"currency": "AUD"},
            has_live_platform_metrics=False,
        )

        self.assertEqual(normalized["regional_kpis"], [])
        self.assertEqual(len(normalized["public_market_facts"]), 1)
        self.assertTrue(
            any("regional_kpis is intentionally empty" in note for note in normalized["data_quality_notes"])
        )
        self.assertTrue(
            any("Platform KPIs are unavailable" in assumption for assumption in normalized["assumptions"])
        )

    def test_america_normalizes_to_united_states_currency(self) -> None:
        self.assertEqual(_canonical_market_name("America"), "United States")
        self.assertEqual(_currency_code_for_market("America"), "USD")

    def test_market_query_pack_is_generic(self) -> None:
        query_pack = _market_query_pack("Wireless Headphones", "Germany", "Last 30 Days")

        self.assertEqual(
            {item["query_type"] for item in query_pack},
            {"sales", "market_share", "pricing_competitors", "availability", "demand_trend"},
        )
        self.assertTrue(all("Wireless Headphones" in item["query"] for item in query_pack))
        self.assertTrue(all("Germany" in item["query"] for item in query_pack))
        self.assertTrue(any("Last 30 Days" in item["query"] for item in query_pack))

    def test_source_bibliography_uses_stable_deduped_ids(self) -> None:
        source = _annotate_source_market(
            {
                "title": "Market source",
                "link": "https://example.com/report?utm_source=test",
                "snippet": "Australia same-market sales and pricing signal.",
                "source_type": "organic",
                "query_type": "sales",
                "read_status": "snippet_only",
            },
            "Australia",
        )
        duplicate = {**source, "query_type": "pricing_competitors"}
        other = _annotate_source_market(
            {
                "title": "Second source",
                "link": "https://example.com/second",
                "snippet": "Additional demand signal.",
                "source_type": "organic",
                "query_type": "demand_trend",
                "read_status": "snippet_only",
            },
            "Australia",
        )
        market_results = _assign_source_ids_to_market_results(
            [{"market": "Australia", "sources": [source, duplicate, other]}]
        )

        self.assertEqual(market_results[0]["sources"][0]["source_id"], "S1")
        self.assertEqual(market_results[0]["sources"][1]["source_id"], "S1")
        self.assertEqual(market_results[0]["sources"][2]["source_id"], "S2")

        bibliography = _source_bibliography_from_market_results(market_results)

        self.assertEqual([item["source_id"] for item in bibliography], ["S1", "S2"])
        self.assertEqual(bibliography[0]["domain"], "example.com")
        self.assertEqual(bibliography[0]["market"], "Australia")

    def test_public_market_fact_candidates_keep_source_context(self) -> None:
        source = _annotate_source_market(
            {
                "title": "Market source",
                "link": "https://example.com/report",
                "snippet": "Australia public-source market fact.",
                "source_type": "organic",
                "query_type": "sales",
                "read_status": "snippet_only",
            },
            "Australia",
        )
        market_results = _assign_source_ids_to_market_results(
            [{"market": "Australia", "sources": [source]}]
        )

        candidates = _public_market_fact_candidates(market_results)

        self.assertEqual(candidates[0]["market"], "Australia")
        self.assertEqual(candidates[0]["fact_type"], "sales")
        self.assertEqual(candidates[0]["source_ids"], ["S1"])
        self.assertEqual(candidates[0]["time_period"], "source_unspecified")
        self.assertEqual(candidates[0]["confidence"], "medium")

    def test_australia_source_is_cross_market_for_united_states(self) -> None:
        source = {
            "title": "BYD Sealion 7 Australia driveaway price",
            "link": "https://www.carsales.com.au/editorial/details/byd-sealion-7-2025-review-148683/",
            "snippet": "BYD Sealion 7 starts from $54,990 plus on-road costs.",
            "source_type": "organic",
        }

        annotated = _annotate_source_market(source, "America")

        self.assertEqual(annotated["assigned_market"], "United States")
        self.assertEqual(annotated["market_alignment"], "cross_market_reference")
        self.assertIn("Australia", annotated["likely_markets"])

    def test_united_states_source_is_same_market(self) -> None:
        source = {
            "title": "BYD Sealion availability in the United States",
            "link": "https://example.com/us/byd-sealion-availability",
            "snippet": "United States market availability and pricing status.",
            "source_type": "organic",
        }

        annotated = _annotate_source_market(source, "America")

        self.assertEqual(annotated["assigned_market"], "United States")
        self.assertEqual(annotated["market_alignment"], "same_market")


if __name__ == "__main__":
    unittest.main()
