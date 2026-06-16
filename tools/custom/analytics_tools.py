import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from html import unescape
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

from tools.custom.commerce_api import CommerceApiConfig, fetch_commerce_metrics
from utils.tool_cache import cached_tool_call
from utils.tool_execution import AsyncToolExecutionMixin

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


def _safe_json_payload(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace(",", "").replace("$", "").replace("%", "")
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _text_payload_notice(value: Any, label: str) -> list[str]:
    if isinstance(value, str) and value.strip() and _safe_json_payload(value) is None:
        return [f"{label} was provided as unstructured text; provide JSON metrics for computed analytics."]
    return []


class AdvancedAttributionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_metrics: dict[str, Any] | list[dict[str, Any]] | str | None = Field(
        None,
        description="Channel-level metrics as JSON or structured data.",
    )
    historical_metrics: dict[str, Any] | str | None = Field(
        None,
        description="Optional experiment, treatment/control, ROI, or historical performance metrics.",
    )


class AdvancedAttributionTool(BaseTool):
    name: str = "Advanced Attribution & Causal Inference Tool"
    description: str = (
        "Computes channel contribution shares and optional causal lift from provided "
        "channel_metrics or historical_metrics. It returns insufficient_data instead "
        "of sample values when inputs are missing."
    )
    args_schema: type[BaseModel] = AdvancedAttributionInput

    def _run(
        self,
        channel_metrics: dict[str, Any] | list[dict[str, Any]] | str | None = None,
        historical_metrics: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        rows = _channel_metric_rows(channel_metrics)
        notes = _text_payload_notice(channel_metrics, "channel_metrics")
        if not rows:
            return {
                "status": "insufficient_data",
                "confidence_level": "none",
                "method": "requires_channel_metrics",
                "channel_contributions": [],
                "did_incremental_lift_pct": None,
                "true_roi": None,
                "budget_recommendations": [],
                "data_quality_notes": [
                    "No parseable channel_metrics were provided; attribution was not estimated.",
                    *notes,
                ],
            }

        total_value = sum(row["value"] for row in rows)
        channel_count = len(rows)
        contributions = [
            {
                "channel": row["channel"],
                "contribution_pct": round((row["value"] / total_value) * 100, 2),
                "basis": row["basis"],
            }
            for row in rows
            if total_value > 0
        ]
        average_share = 100 / channel_count if channel_count else 0
        recommendations = [
            {
                "channel": item["channel"],
                "recommendation": (
                    "consider_incremental_budget_test"
                    if item["contribution_pct"] >= average_share
                    else "hold_or_reduce_pending_incrementality_test"
                ),
                "rationale": (
                    f"Observed contribution share is {item['contribution_pct']}% "
                    f"versus an even-share baseline of {round(average_share, 2)}%."
                ),
            }
            for item in contributions
        ]
        lift_pct, true_roi, causal_notes = _causal_metrics_from_history(historical_metrics)
        return {
            "status": "computed",
            "confidence_level": "medium" if lift_pct is not None or true_roi is not None else "low",
            "method": "deterministic_share_of_observed_channel_metric",
            "channel_contributions": contributions,
            "did_incremental_lift_pct": lift_pct,
            "true_roi": true_roi,
            "budget_recommendations": recommendations,
            "data_quality_notes": [
                "Attribution uses provided channel metrics only; validate with incrementality testing before budget moves.",
                *causal_notes,
                *notes,
            ],
        }


def _channel_metric_rows(channel_metrics: Any) -> list[dict[str, Any]]:
    payload = _safe_json_payload(channel_metrics)
    if isinstance(payload, dict):
        if isinstance(payload.get("channels"), list):
            candidates = payload["channels"]
        elif isinstance(payload.get("channel_metrics"), list):
            candidates = payload["channel_metrics"]
        else:
            candidates = [
                {"channel": key, **value}
                if isinstance(value, dict)
                else {"channel": key, "value": value}
                for key, value in payload.items()
            ]
    elif isinstance(payload, list):
        candidates = payload
    else:
        return []

    rows: list[dict[str, Any]] = []
    metric_keys = (
        "attributed_revenue",
        "revenue",
        "sales",
        "conversions",
        "orders",
        "value",
        "spend",
    )
    for item in candidates:
        if not isinstance(item, dict):
            continue
        channel = str(
            item.get("channel")
            or item.get("platform")
            or item.get("source")
            or item.get("name")
            or ""
        ).strip()
        if not channel:
            continue
        for metric_key in metric_keys:
            metric_value = _coerce_float(item.get(metric_key))
            if metric_value is not None and metric_value > 0:
                rows.append(
                    {
                        "channel": channel,
                        "value": metric_value,
                        "basis": metric_key,
                    }
                )
                break
    return rows


def _causal_metrics_from_history(
    historical_metrics: dict[str, Any] | str | None,
) -> tuple[float | None, float | None, list[str]]:
    payload = _safe_json_payload(historical_metrics)
    notes = _text_payload_notice(historical_metrics, "historical_metrics")
    if not isinstance(payload, dict):
        return None, None, [
            "No parseable historical_metrics were provided for DiD lift or true ROI.",
            *notes,
        ]

    treatment = _coerce_float(payload.get("treatment_conversion_rate") or payload.get("treatment_rate"))
    control = _coerce_float(payload.get("control_conversion_rate") or payload.get("control_rate"))
    lift_pct = None
    if treatment is not None and control not in (None, 0):
        lift_pct = round(((treatment - float(control)) / float(control)) * 100, 2)

    incremental_revenue = _coerce_float(payload.get("incremental_revenue"))
    incremental_cost = _coerce_float(payload.get("incremental_cost") or payload.get("incremental_spend"))
    true_roi = None
    if incremental_revenue is not None and incremental_cost not in (None, 0):
        true_roi = round(incremental_revenue / float(incremental_cost), 2)

    missing_notes: list[str] = []
    if lift_pct is None:
        missing_notes.append("DiD lift requires treatment and non-zero control conversion rates.")
    if true_roi is None:
        missing_notes.append("True ROI requires incremental_revenue and non-zero incremental_cost.")
    return lift_pct, true_roi, [*missing_notes, *notes]


class GlobalMacroRiskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_markets: str = Field(..., min_length=1)
    base_currency: str = Field("USD", min_length=1)
    macro_signals: dict[str, Any] | str | None = Field(
        None,
        description="Optional FX, tariff, policy, or margin signals as JSON.",
    )


class GlobalMacroRiskTool(BaseTool):
    name: str = "Global Macro & Risk Fusion Tool"
    description: str = (
        "Normalizes provided macro risk signals such as FX rates, tariff alerts, "
        "margin impact, and mitigation notes. It does not fabricate rates or policy changes."
    )
    args_schema: type[BaseModel] = GlobalMacroRiskInput

    def _run(
        self,
        target_markets: str,
        base_currency: str = "USD",
        macro_signals: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        payload = _safe_json_payload(macro_signals)
        notes = _text_payload_notice(macro_signals, "macro_signals")
        if not isinstance(payload, dict):
            return {
                "status": "insufficient_data",
                "confidence_level": "none",
                "base_currency": (base_currency or "USD").upper(),
                "target_markets": target_markets,
                "fx_rates": [],
                "tariff_alerts": [],
                "margin_impact_pct": None,
                "risk_level": "unknown",
                "strategic_recommendations": [],
                "data_quality_notes": [
                    "No parseable macro_signals were provided; FX, tariff, and margin risk were not estimated.",
                    *notes,
                ],
            }

        risk_quantification = payload.get("risk_quantification")
        risk_payload = risk_quantification if isinstance(risk_quantification, dict) else {}
        margin_impact = _coerce_float(
            payload.get("margin_impact_pct")
            or payload.get("margin_decline_pct")
            or risk_payload.get("margin_impact_pct")
            or risk_payload.get("margin_decline_pct")
        )
        risk_level = str(
            payload.get("risk_level")
            or risk_payload.get("risk_level")
            or "unknown"
        )
        return {
            "status": "computed",
            "confidence_level": "medium",
            "base_currency": (base_currency or "USD").upper(),
            "target_markets": target_markets,
            "fx_rates": _fx_rate_items(payload.get("fx_rates")),
            "tariff_alerts": _tariff_alert_items(payload.get("tariff_alerts")),
            "margin_impact_pct": margin_impact,
            "risk_level": risk_level,
            "strategic_recommendations": _string_list(
                payload.get("strategic_recommendations") or payload.get("strategic_advice")
            ),
            "data_quality_notes": [
                "Macro risk uses provided macro_signals only; connect live FX and policy feeds for production alerts.",
                *notes,
            ],
        }


def _fx_rate_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [
            {"pair": str(pair), "rate": _coerce_float(rate), "source": "provided_macro_signals"}
            for pair, rate in value.items()
        ]
    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "pair": str(item.get("pair") or item.get("currency_pair") or ""),
                    "rate": _coerce_float(item.get("rate")),
                    "source": str(item.get("source") or "provided_macro_signals"),
                }
            )
        return [item for item in items if item["pair"]]
    return []


def _tariff_alert_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    alerts: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        alerts.append(
            {
                "region": str(item.get("region") or item.get("market") or "unknown"),
                "policy_change": str(item.get("policy_change") or item.get("policy") or "not_specified"),
                "effective": str(item.get("effective") or item.get("effective_date") or "source_unspecified"),
                "source": str(item.get("source") or "provided_macro_signals"),
            }
        )
    return alerts


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


class PredictiveAnomalyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_category: str = Field(..., min_length=1)
    historical_metrics: dict[str, Any] | list[dict[str, Any]] | str | None = Field(
        None,
        description="Daily or period metrics as JSON or structured data.",
    )


class PredictiveAnomalyTool(BaseTool):
    name: str = "Predictive Forecast & Anomaly Detection Tool"
    description: str = (
        "Produces a simple 14-period forecast and anomaly notes only from provided "
        "historical_metrics. It returns insufficient_data when history is absent."
    )
    args_schema: type[BaseModel] = PredictiveAnomalyInput

    def _run(
        self,
        product_category: str,
        historical_metrics: dict[str, Any] | list[dict[str, Any]] | str | None = None,
    ) -> dict[str, Any]:
        payload = _safe_json_payload(historical_metrics)
        notes = _text_payload_notice(historical_metrics, "historical_metrics")
        series = _metric_series(payload)
        if len(series) < 3:
            return {
                "status": "insufficient_data",
                "confidence_level": "none",
                "product_category": product_category,
                "forecast_14d": [],
                "anomalies": _provided_anomalies(payload),
                "model_confidence": "none",
                "data_quality_notes": [
                    "At least three historical metric points are required for a conservative forecast.",
                    *notes,
                ],
            }

        values = [item["value"] for item in series]
        window = values[-min(7, len(values)) :]
        forecast_value = round(sum(window) / len(window), 2)
        forecast = [
            {
                "period": f"horizon_day_{index}",
                "predicted_value": forecast_value,
                "basis": "moving_average_of_provided_history",
            }
            for index in range(1, 15)
        ]
        anomalies = _provided_anomalies(payload) or _detected_anomalies(series)
        return {
            "status": "computed",
            "confidence_level": "low",
            "product_category": product_category,
            "forecast_14d": forecast,
            "anomalies": anomalies,
            "model_confidence": "low; deterministic moving average over provided data",
            "data_quality_notes": [
                "Forecast is a lightweight deterministic estimate, not a fitted production model.",
                *notes,
            ],
        }


def _metric_series(payload: Any) -> list[dict[str, Any]]:
    candidates: Any
    if isinstance(payload, dict):
        for key in ("daily_sales", "sales_history", "time_series", "metrics", "history"):
            if isinstance(payload.get(key), list):
                candidates = payload[key]
                break
        else:
            candidates = []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []

    series: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        value = None
        for key in ("units", "predicted_units", "sales", "revenue", "orders", "value"):
            value = _coerce_float(item.get(key))
            if value is not None:
                break
        if value is None:
            continue
        series.append(
            {
                "period": str(item.get("date") or item.get("period") or f"point_{index}"),
                "value": value,
            }
        )
    return series


def _provided_anomalies(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("anomalies"), list):
        return []
    anomalies: list[dict[str, Any]] = []
    for item in payload["anomalies"]:
        if not isinstance(item, dict):
            continue
        anomalies.append(
            {
                "period": str(item.get("date") or item.get("period") or "source_unspecified"),
                "metric": str(item.get("metric") or "unknown"),
                "severity": str(item.get("severity") or "unclassified"),
                "description": str(item.get("root_cause") or item.get("description") or "provided anomaly"),
            }
        )
    return anomalies


def _detected_anomalies(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [item["value"] for item in series]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_dev = variance**0.5
    if std_dev == 0:
        return []
    anomalies: list[dict[str, Any]] = []
    for item in series:
        z_score = (item["value"] - mean_value) / std_dev
        if abs(z_score) >= 2:
            anomalies.append(
                {
                    "period": item["period"],
                    "metric": "provided_history_value",
                    "severity": "high" if abs(z_score) >= 3 else "medium",
                    "description": f"Value is {round(z_score, 2)} standard deviations from provided-history mean.",
                }
            )
    return anomalies


class ChatBIPreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_query: str | None = Field(None, description="Natural-language analytics question.")
    db_schema_context: str | None = Field(
        None,
        description="Optional schema hint for the SQL preview. No database connection is used.",
    )


class ChatBISQLPreviewTool(BaseTool):
    name: str = "ChatBI SQL Preview Tool"
    description: str = (
        "Classifies a natural-language analytics query and returns a fixed safe SQL "
        "preview. It never connects to or queries a database."
    )
    args_schema: type[BaseModel] = ChatBIPreviewInput

    def _run(
        self,
        user_query: str | None = None,
        db_schema_context: str | None = None,
    ) -> dict[str, Any]:
        query = (user_query or "").strip()
        if not query:
            return {
                "status": "skipped",
                "intent": "not_provided",
                "generated_sql": "",
                "business_insight": "No ChatBI query was provided.",
                "safety_notes": ["ChatBI preview skipped because user_query is empty."],
            }

        normalized = query.lower()
        if "conversion" in normalized:
            intent = "conversion_performance_query"
        elif "sales" in normalized or "revenue" in normalized:
            intent = "sales_performance_query"
        else:
            intent = "general_analytics_query"

        generated_sql = (
            "SELECT region, SUM(total_sales) AS total_sales, "
            "AVG(conversion_rate) AS avg_conversion_rate "
            "FROM analytics_metrics "
            "WHERE metric_date >= CURRENT_DATE - INTERVAL '30 days' "
            "GROUP BY region "
            "ORDER BY total_sales DESC;"
        )
        schema_note = (
            " Schema hint was provided for reviewer context."
            if db_schema_context and db_schema_context.strip()
            else ""
        )
        return {
            "status": "preview_only",
            "intent": intent,
            "generated_sql": generated_sql,
            "business_insight": (
                "This is a SQL preview for analyst review only; connect an approved "
                "warehouse and validate table names before execution."
            ),
            "safety_notes": [
                "No database connection was opened.",
                "The SQL template does not interpolate raw user text.",
                f"Original query retained for audit: {query[:200]}.{schema_note}",
            ],
        }


class ClosedLoopAutomationPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_stock_forecast: bool = False
    conversion_anomaly: bool = False
    macro_risk: bool = False
    critical_alert: bool = False
    sku: str | None = None
    forecasted_demand: int | None = Field(None, ge=0)
    campaign_id: str | None = None
    price_adjustment: str | None = None
    alert_message: str | None = None


class ClosedLoopAutomationPlanTool(BaseTool):
    name: str = "Closed-Loop Automation Dry-Run Planner"
    description: str = (
        "Creates dry-run automation plans for inventory, campaign, pricing, and "
        "alert actions. It never calls external write APIs."
    )
    args_schema: type[BaseModel] = ClosedLoopAutomationPlanInput

    def _run(
        self,
        low_stock_forecast: bool = False,
        conversion_anomaly: bool = False,
        macro_risk: bool = False,
        critical_alert: bool = False,
        sku: str | None = None,
        forecasted_demand: int | None = None,
        campaign_id: str | None = None,
        price_adjustment: str | None = None,
        alert_message: str | None = None,
    ) -> dict[str, Any]:
        actions: list[dict[str, Any]] = []
        actions.extend(_low_stock_actions(low_stock_forecast, sku, forecasted_demand))
        actions.extend(_conversion_anomaly_actions(conversion_anomaly, campaign_id))
        actions.extend(_macro_risk_actions(macro_risk, sku, price_adjustment))
        actions.extend(_critical_alert_actions(critical_alert, alert_message))

        planned_count = sum(1 for action in actions if action["status"] == "planned")
        insufficient_count = sum(1 for action in actions if action["status"] == "insufficient_data")
        status = (
            "planned"
            if planned_count
            else "insufficient_data"
            if insufficient_count
            else "skipped"
        )
        return {
            "status": status,
            "execution_mode": "dry_run",
            "actions": actions,
            "data_quality_notes": [
                "Dry-run only: no Amazon, TikTok, AliExpress, Shopify, ERP, or Slack write API was called.",
                "Require human approval and platform-specific credential validation before enabling live actions.",
            ],
        }


def _automation_action(
    *,
    action_type: str,
    platform: str,
    target: str,
    status: str,
    reason: str,
    required_credentials: list[str],
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "platform": platform,
        "target": target,
        "status": status,
        "execution_mode": "dry_run",
        "reason": reason,
        "required_credentials": required_credentials,
    }


def _low_stock_actions(
    enabled: bool,
    sku: str | None,
    forecasted_demand: int | None,
) -> list[dict[str, Any]]:
    if not enabled:
        return [
            _automation_action(
                action_type="LOW_STOCK_WORKFLOW",
                platform="ERP/Shopify/TikTok Shop",
                target="not_applicable",
                status="skipped",
                reason="low_stock_forecast trigger is false.",
                required_credentials=[],
            )
        ]
    if not sku or not forecasted_demand:
        return [
            _automation_action(
                action_type="LOW_STOCK_WORKFLOW",
                platform="ERP/Shopify/TikTok Shop",
                target=sku or "missing_sku",
                status="insufficient_data",
                reason="sku and positive forecasted_demand are required.",
                required_credentials=[],
            )
        ]
    return [
        _automation_action(
            action_type="ERP_PURCHASE_ORDER_CREATE",
            platform="ERP",
            target=sku,
            status="planned",
            reason=f"Forecasted demand is {forecasted_demand}; prepare purchase order review.",
            required_credentials=["erp_api_endpoint", "erp_api_token"],
        ),
        _automation_action(
            action_type="SHOPIFY_INVENTORY_SYNC",
            platform="Shopify",
            target=sku,
            status="planned",
            reason="Dry-run inventory sync candidate based on provided low-stock trigger.",
            required_credentials=["shopify_store_domain", "shopify_admin_access_token"],
        ),
        _automation_action(
            action_type="TIKTOK_STOCK_SYNC",
            platform="TikTok Shop",
            target=sku,
            status="planned",
            reason="Dry-run TikTok Shop stock sync candidate based on provided low-stock trigger.",
            required_credentials=["tiktok_access_token"],
        ),
    ]


def _conversion_anomaly_actions(enabled: bool, campaign_id: str | None) -> list[dict[str, Any]]:
    if not enabled:
        return [
            _automation_action(
                action_type="CONVERSION_ANOMALY_WORKFLOW",
                platform="Amazon/Shopify",
                target="not_applicable",
                status="skipped",
                reason="conversion_anomaly trigger is false.",
                required_credentials=[],
            )
        ]
    if not campaign_id:
        return [
            _automation_action(
                action_type="CONVERSION_ANOMALY_WORKFLOW",
                platform="Amazon/Shopify",
                target="missing_campaign_id",
                status="insufficient_data",
                reason="campaign_id is required before planning campaign or product actions.",
                required_credentials=[],
            )
        ]
    return [
        _automation_action(
            action_type="AMAZON_AD_PAUSE_REVIEW",
            platform="Amazon Ads",
            target=campaign_id,
            status="planned",
            reason="Dry-run ad pause review for provided conversion anomaly trigger.",
            required_credentials=["amazon_sp_api_endpoint", "amazon_sp_api_access_token"],
        ),
        _automation_action(
            action_type="SHOPIFY_PRODUCT_STATUS_REVIEW",
            platform="Shopify",
            target=campaign_id,
            status="planned",
            reason="Dry-run product visibility review for provided conversion anomaly trigger.",
            required_credentials=["shopify_store_domain", "shopify_admin_access_token"],
        ),
    ]


def _macro_risk_actions(
    enabled: bool,
    sku: str | None,
    price_adjustment: str | None,
) -> list[dict[str, Any]]:
    if not enabled:
        return [
            _automation_action(
                action_type="MACRO_RISK_PRICING_WORKFLOW",
                platform="Amazon/AliExpress",
                target="not_applicable",
                status="skipped",
                reason="macro_risk trigger is false.",
                required_credentials=[],
            )
        ]
    if not sku or not price_adjustment:
        return [
            _automation_action(
                action_type="MACRO_RISK_PRICING_WORKFLOW",
                platform="Amazon/AliExpress",
                target=sku or "missing_sku",
                status="insufficient_data",
                reason="sku and price_adjustment are required before pricing actions can be planned.",
                required_credentials=[],
            )
        ]
    return [
        _automation_action(
            action_type="AMAZON_DYNAMIC_PRICING_REVIEW",
            platform="Amazon",
            target=sku,
            status="planned",
            reason=f"Dry-run pricing review for requested adjustment {price_adjustment}.",
            required_credentials=["amazon_sp_api_endpoint", "amazon_sp_api_access_token"],
        ),
        _automation_action(
            action_type="ALIEXPRESS_PRICE_UPDATE_REVIEW",
            platform="AliExpress",
            target=sku,
            status="planned",
            reason=f"Dry-run price update review for requested adjustment {price_adjustment}.",
            required_credentials=["aliexpress_access_token"],
        ),
    ]


def _critical_alert_actions(enabled: bool, alert_message: str | None) -> list[dict[str, Any]]:
    if not enabled:
        return [
            _automation_action(
                action_type="CRITICAL_ALERT_WORKFLOW",
                platform="Slack",
                target="not_applicable",
                status="skipped",
                reason="critical_alert trigger is false.",
                required_credentials=[],
            )
        ]
    if not alert_message:
        return [
            _automation_action(
                action_type="CRITICAL_ALERT_WORKFLOW",
                platform="Slack",
                target="#growth-ops",
                status="insufficient_data",
                reason="alert_message is required before alert escalation can be planned.",
                required_credentials=[],
            )
        ]
    return [
        _automation_action(
            action_type="SLACK_ESCALATION_REVIEW",
            platform="Slack",
            target="#growth-ops",
            status="planned",
            reason=f"Dry-run escalation message prepared: {alert_message[:160]}",
            required_credentials=["slack_webhook_url"],
        )
    ]


class EcomPlatformMetricsTool(AsyncToolExecutionMixin, BaseTool):
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
    tool_cache_context: dict[str, Any] | None = None

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
            result = cached_tool_call(
                self.tool_cache_context,
                tool_name="Commerce Metrics Read",
                tool_version="v1",
                arguments={
                    "platform": platform,
                    "region": region,
                    "date_range": date_range,
                    "currency": currency,
                },
                provider_identity={
                    "provider": "commerce_api",
                    "shopify_store_domain": self.shopify_store_domain,
                    "shopify_api_version": self.shopify_api_version,
                    "amazon_sp_api_endpoint": self.amazon_sp_api_endpoint,
                    "amazon_marketplace_ids": self.amazon_marketplace_ids,
                },
                fetcher=lambda: fetch_commerce_metrics(config, platform, region, date_range),
            )
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


class CompetitorBenchmarkTool(AsyncToolExecutionMixin, BaseTool):
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
    tool_cache_context: dict[str, Any] | None = None

    def _run(
        self,
        product_category: str,
        target_markets: str,
        date_range: str | None = None,
    ) -> dict[str, Any]:
        if not self.serper_api_key:
            return self._dev_fallback(product_category, target_markets)

        try:
            return cached_tool_call(
                self.tool_cache_context,
                tool_name=self.name,
                tool_version="v1",
                arguments={
                    "product_category": product_category,
                    "target_markets": target_markets,
                    "date_range": date_range,
                    "deep_read_enabled": self.deep_read_enabled,
                    "deep_read_max_pages": self.deep_read_max_pages,
                    "deep_read_concurrency": self.deep_read_concurrency,
                    "deep_read_timeout_seconds": self.deep_read_timeout_seconds,
                    "deep_read_max_chars": self.deep_read_max_chars,
                },
                provider_identity={
                    "provider": "serper",
                    "search_type": "competitor_benchmark",
                    "deep_read_enabled": self.deep_read_enabled,
                },
                fetcher=lambda: _fetch_serper_competitor_benchmarks(
                    self.serper_api_key or "",
                    product_category,
                    target_markets,
                    date_range,
                    self.deep_read_enabled,
                    self.deep_read_max_pages,
                    self.deep_read_concurrency,
                    self.deep_read_timeout_seconds,
                    self.deep_read_max_chars,
                ),
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
