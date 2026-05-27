from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from services.language_detector import LanguageDetector

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool


class SentimentIntentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_label: Literal["ANGRY", "FRUSTRATED", "NEUTRAL", "SATISFIED", "DELIGHTED"]
    intent_category: Literal[
        "RMA_REQUEST",
        "SHIPPING_INQUIRY",
        "PRODUCT_USAGE",
        "BILLING_ISSUE",
        "VIP_ESCALATION",
        "GENERAL",
    ]
    customer_tier: Literal["VIP", "PREMIUM", "STANDARD", "NEW"]
    urgency_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    language_detected: str
    requires_human_handoff: bool


class RMAValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    eligible_for_return: bool
    eligibility_reason: str
    return_window_days: int
    days_since_delivery: int
    item_condition_accepted: list[str]
    restocking_fee_pct: float | None = Field(..., ge=0, le=100)
    return_shipping_responsibility: Literal["CUSTOMER", "BRAND", "SHARED"]


class LogisticsIntegrationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carrier: Literal["ShipStation", "EasyPost", "DHL", "FedEx", "Local_Post"]
    prepaid_label_url: str
    tracking_number: str
    estimated_refund_days: int
    warehouse_inbound_notification: str
    return_instructions_localized: str


NEGATIVE_KEYWORDS = {
    "angry": 0.35,
    "disappointed": 0.3,
    "complaint": 0.35,
    "terrible": 0.45,
    "unacceptable": 0.55,
    "waste": 0.35,
    "never again": 0.55,
    "damaged": 0.35,
    "broken": 0.35,
    "refund": 0.2,
    "chargeback": 0.55,
    "legal": 0.5,
    "lawsuit": 0.6,
}
POSITIVE_KEYWORDS = {
    "thanks": 0.15,
    "thank you": 0.15,
    "great": 0.25,
    "excellent": 0.35,
    "love": 0.25,
    "happy": 0.2,
}
VIP_DOMAINS = ("@enterprise.com", "@vip-client.com", "@corporate.net")
RETURN_INTENT_KEYWORDS = ("return", "refund", "exchange", "rma")
PRODUCT_DAMAGE_KEYWORDS = ("damaged", "defective", "broken", "does not work", "not working")
PACKAGING_DAMAGE_KEYWORDS = ("package", "packaging", "box", "parcel")
MINOR_DAMAGE_QUALIFIERS = ("slightly", "minor", "small", "cosmetic", "outer")
SHIPPING_KEYWORDS = ("shipping", "delivery", "track", "tracking", "where is", "delayed")
PRODUCT_USAGE_KEYWORDS = ("how to", "setup", "install", "use", "manual", "instructions")
BILLING_KEYWORDS = ("charge", "billing", "invoice", "payment")
ESCALATION_KEYWORDS = ("manager", "supervisor", "legal", "chargeback")
DEFECT_KEYWORDS = ("damaged", "defective", "broken", "wrong item", "incorrect item")
USED_OR_OPENED_KEYWORDS = ("used", "worn", "opened", "seal broken", "removed tag")
CHANGE_OF_MIND_KEYWORDS = ("don't like", "do not like", "changed my mind", "change of mind", "wrong size", "wrong color")
HYGIENE_SENSITIVE_KEYWORDS = ("earring", "ear ring", "earrings", "piercing", "underwear", "swimwear")


def detect_language(text: str, fallback: str | None = None) -> str:
    return LanguageDetector.detect(text, fallback=fallback or "en")


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_minor_packaging_issue(text: str) -> bool:
    return (
        _has_any(text, PACKAGING_DAMAGE_KEYWORDS)
        and _has_any(text, PRODUCT_DAMAGE_KEYWORDS)
        and _has_any(text, MINOR_DAMAGE_QUALIFIERS)
        and not _has_any(text, RETURN_INTENT_KEYWORDS)
    )


def _classify_intent(text: str) -> str:
    if _has_any(text, RETURN_INTENT_KEYWORDS):
        return "RMA_REQUEST"
    if _is_minor_packaging_issue(text):
        return "GENERAL"
    if _has_any(text, PRODUCT_DAMAGE_KEYWORDS):
        return "RMA_REQUEST"
    if _has_any(text, SHIPPING_KEYWORDS):
        return "SHIPPING_INQUIRY"
    if _has_any(text, PRODUCT_USAGE_KEYWORDS):
        return "PRODUCT_USAGE"
    if _has_any(text, BILLING_KEYWORDS):
        return "BILLING_ISSUE"
    if _has_any(text, ESCALATION_KEYWORDS):
        return "VIP_ESCALATION"
    return "GENERAL"


def _is_defect_reason(text: str) -> bool:
    return _has_any(text, DEFECT_KEYWORDS)


def _is_used_or_opened(text: str) -> bool:
    return _has_any(text, USED_OR_OPENED_KEYWORDS)


def _is_hygiene_sensitive_item(text: str, item_sku: str) -> bool:
    return _has_any(f"{text} {item_sku.lower()}", HYGIENE_SENSITIVE_KEYWORDS)


def analyze_sentiment_intent(
    inquiry_text: str,
    customer_email: str,
    order_history: dict[str, Any] | None = None,
    detected_language: str | None = None,
) -> dict[str, Any]:
    text_lower = inquiry_text.lower()
    negative_weight = sum(weight for keyword, weight in NEGATIVE_KEYWORDS.items() if keyword in text_lower)
    positive_weight = sum(weight for keyword, weight in POSITIVE_KEYWORDS.items() if keyword in text_lower)
    sentiment_score = max(-1.0, min(1.0, 0.35 + positive_weight - negative_weight))
    intent = _classify_intent(text_lower)

    order_history = order_history or {}
    customer_email_lower = customer_email.lower()
    if any(domain in customer_email_lower for domain in VIP_DOMAINS) or float(order_history.get("lifetime_value") or 0) > 5000:
        tier = "VIP"
    elif int(order_history.get("order_count") or 0) >= 5:
        tier = "PREMIUM"
    elif int(order_history.get("order_count") or 0) >= 1:
        tier = "STANDARD"
    else:
        tier = "NEW"

    requires_handoff = sentiment_score < -0.6 and tier in {"VIP", "PREMIUM"}
    if requires_handoff:
        urgency = "CRITICAL"
    elif sentiment_score < -0.25 or intent in {"RMA_REQUEST", "VIP_ESCALATION"}:
        urgency = "HIGH"
    elif intent in {"SHIPPING_INQUIRY", "PRODUCT_USAGE", "BILLING_ISSUE"}:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    if sentiment_score < -0.6:
        label = "ANGRY"
    elif sentiment_score < -0.2:
        label = "FRUSTRATED"
    elif sentiment_score < 0.2:
        label = "NEUTRAL"
    elif sentiment_score < 0.7:
        label = "SATISFIED"
    else:
        label = "DELIGHTED"

    return SentimentIntentAnalysis(
        sentiment_score=round(sentiment_score, 2),
        sentiment_label=label,
        intent_category=intent,
        customer_tier=tier,
        urgency_level=urgency,
        language_detected=detect_language(inquiry_text, detected_language),
        requires_human_handoff=requires_handoff,
    ).model_dump()


def process_rma_request(
    order_id: str,
    item_sku: str,
    return_reason: str,
    detected_language: str,
    order_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    order_history = order_history or {}
    delivered_days_ago = int(order_history.get("days_since_delivery") or 10)
    item_condition = str(order_history.get("item_condition") or "unopened").lower()
    region = str(order_history.get("region") or _infer_region(order_id, detected_language)).upper()
    return_window = 14 if region in {"EU", "DE", "FR"} else 30
    eligible_conditions = ["unopened", "defective", "damaged"]
    reason_text = return_reason.lower()
    defect_reason = _is_defect_reason(reason_text)
    used_or_opened = _is_used_or_opened(reason_text)
    hygiene_sensitive = _is_hygiene_sensitive_item(reason_text, item_sku)

    if defect_reason:
        effective_condition = "damaged"
    elif used_or_opened:
        effective_condition = "used"
    else:
        effective_condition = item_condition

    if delivered_days_ago > return_window:
        eligible = False
        reason = f"Return window expired: {delivered_days_ago} days since delivery exceeds {return_window} days."
    elif hygiene_sensitive and used_or_opened and not defect_reason:
        eligible = False
        reason = "Hygiene-sensitive items such as earrings are not eligible for change-of-mind returns after opening or use unless defective."
    elif used_or_opened and _has_any(reason_text, CHANGE_OF_MIND_KEYWORDS) and not defect_reason:
        eligible = False
        reason = "Used or opened items are not eligible for change-of-mind returns under the standard return policy."
    else:
        eligible = effective_condition in eligible_conditions
        reason = (
            "Within return window and item condition is accepted."
            if eligible
            else f"Item condition '{effective_condition}' is not eligible under the standard return policy."
        )

    shipping_responsibility = "BRAND" if effective_condition in {"defective", "damaged"} else "CUSTOMER"
    rma = RMAValidationResult(
        order_id=order_id,
        eligible_for_return=eligible,
        eligibility_reason=reason,
        return_window_days=return_window,
        days_since_delivery=delivered_days_ago,
        item_condition_accepted=eligible_conditions,
        restocking_fee_pct=0.0 if shipping_responsibility == "BRAND" else 15.0,
        return_shipping_responsibility=shipping_responsibility,
    ).model_dump()

    if not eligible:
        return {"rma_validation": rma, "logistics_output": None}

    carrier = _carrier_for_region(region)
    refund_days = 5 if shipping_responsibility == "BRAND" else 10
    logistics = LogisticsIntegrationOutput(
        carrier=carrier,
        prepaid_label_url=f"https://labels.example.local/{carrier.lower()}/return/{order_id}_{item_sku}.pdf",
        tracking_number=f"RTN{re.sub(r'[^A-Za-z0-9]', '', order_id)}{item_sku[:3].upper()}",
        estimated_refund_days=refund_days,
        warehouse_inbound_notification=(
            f"ASN-{datetime.now(UTC).strftime('%Y%m%d')}-{order_id} sent to WMS; "
            f"expected inbound receipt in {refund_days} days."
        ),
        return_instructions_localized=_localized_return_instructions(detected_language, refund_days),
    ).model_dump()
    return {"rma_validation": rma, "logistics_output": logistics}


def _infer_region(order_id: str, detected_language: str) -> str:
    upper_order = order_id.upper()
    if upper_order.startswith(("EU", "DE", "FR")) or detected_language in {"de", "fr"}:
        return "EU"
    if upper_order.startswith("JP") or detected_language == "ja":
        return "JP"
    return "US"


def _carrier_for_region(region: str) -> Literal["ShipStation", "EasyPost", "DHL", "FedEx", "Local_Post"]:
    if region in {"EU", "DE", "FR"}:
        return "DHL"
    if region == "JP":
        return "Local_Post"
    if region == "US":
        return "EasyPost"
    return "ShipStation"


def _localized_return_instructions(language: str, estimated_refund_days: int) -> str:
    templates = {
        "en": "Print the prepaid return label, pack the item securely, and drop it off with the listed carrier. Refund review begins after warehouse receipt.",
        "es": "Imprima la etiqueta de devolucion prepagada, embale el articulo de forma segura y entreguelo al transportista indicado. La revision del reembolso empieza al recibirse en almacen.",
        "de": "Drucken Sie das vorausbezahlte Ruecksendeetikett aus, verpacken Sie den Artikel sicher und geben Sie ihn beim angegebenen Versanddienst ab.",
        "fr": "Imprimez l'etiquette de retour prepayee, emballez l'article avec soin et deposez-le aupres du transporteur indique.",
        "ja": "Prepaid return label wo insatsu shi, shohin wo anzen ni konpo shite, shitei no haiso gyosha ni watashite kudasai.",
        "zh": "请打印预付退货标签，妥善包装商品，并交给指定承运商。仓库签收后将开始退款审核。",
    }
    base = templates.get(language, templates["en"])
    return f"{base} Estimated refund processing time: {estimated_refund_days} days after receipt."


class SentimentIntentGradingTool(BaseTool):
    name: str = "Sentiment & Intent Grading Engine"
    description: str = (
        "Analyzes customer inquiry text for sentiment score, intent classification, "
        "customer tier, detected language, urgency, and human handoff recommendation."
    )

    def _run(
        self,
        inquiry_text: str,
        customer_email: str,
        order_history: dict[str, Any] | None = None,
        detected_language: str | None = None,
    ) -> dict[str, Any]:
        return analyze_sentiment_intent(
            inquiry_text=inquiry_text,
            customer_email=customer_email,
            order_history=order_history,
            detected_language=detected_language,
        )


class RMAAutomationTool(BaseTool):
    name: str = "RMA Policy Validation & Logistics Automation Tool"
    description: str = (
        "Validates return eligibility with local policy rules and simulates prepaid "
        "return label generation plus warehouse inbound notification."
    )

    def _run(
        self,
        order_id: str,
        item_sku: str,
        return_reason: str,
        detected_language: str = "en",
        order_history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return process_rma_request(
            order_id=order_id,
            item_sku=item_sku,
            return_reason=return_reason,
            detected_language=detected_language,
            order_history=order_history,
        )
