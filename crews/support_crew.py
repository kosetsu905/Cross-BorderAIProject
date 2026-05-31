from pathlib import Path
from typing import Any
import json
import re

import httpx
import yaml
from crewai import Agent, Crew, Task
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from services.language_detector import LanguageDetector
from tools.custom.gmail_tools import resolve_gmail_access_token, send_gmail_message
from tools.custom.support_automation_tools import (
    LogisticsIntegrationOutput,
    RMAAutomationTool,
    RMAValidationResult,
    SentimentIntentAnalysis,
    SentimentIntentGradingTool,
    analyze_sentiment_intent,
    process_rma_request,
)
from tools.custom.customer_service_tools import (
    IntentRouterTool,
    OrderTrackingTool,
    PreSalesProductKnowledgeTool,
)
from tools.custom.support_rag_tools import (
    SupportKnowledgeSearchTool,
    extract_catalog_product_offer,
    load_knowledge_chunks,
    search_knowledge_base,
)
from tools.custom.support_search_tools import build_support_external_search_tools
from utils.crew_result import serialize_crew_result
from utils.llm_config import (
    build_llm,
    llm_api_key,
    llm_chat_completions_url,
    llm_model_name,
    llm_reasoning_compat_params,
)
from utils.project_intelligence import augment_agents_config
from utils.usage_tracking import INTERNAL_USAGE_KEY
from utils.workflow_progress import attach_task_progress

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "support"
PRICE_VALUE_RE = re.compile(r"\$\s*\d+(?:\.\d{1,2})?")
RETURN_LABEL_URL_RE = re.compile(r"https?://[^\s)\]]*(?:labels\.example\.local|/return/)[^\s)\]]*", re.IGNORECASE)
RMA_UNSUPPORTED_CLAIM_RE = re.compile(
    r"\b(?:prepaid\s+return\s+label|return\s+label|tracking\s+number|"
    r"we\s+can\s+proceed\s+with\s+the\s+return|proceed\s+with\s+the\s+return|"
    r"initiate\s+a\s+refund|refund\s+to\s+your\s+original\s+payment|"
    r"drop\s+off\s+the\s+package|pack\s+the\s+item|pack\s+the\s+earrings)\b",
    re.IGNORECASE,
)
TRACKING_IDENTIFIER_RE = re.compile(r"(?<![A-Z0-9])(?:[A-Z]{1,4}\d{6,}|\d{8,})(?![A-Z0-9])", re.IGNORECASE)
TRACKING_NO_RE = re.compile(r"\bTracking\s+No\.?\s*([A-Z0-9-]+)", re.IGNORECASE)
REFERENCE_NO_RE = re.compile(r"\bReference\s+No\.?\s*([A-Z0-9-]+)", re.IGNORECASE)
LAST_STATUS_RE = re.compile(r"\bLast\s+Status(?!\s+Date)\s+(.+?)(?=\s+Booking\s+Date|\s+Shipment\s+Details|$)", re.IGNORECASE)
LAST_STATUS_DATE_RE = re.compile(r"\bLast\s+Status\s+Date\s+(.+?)(?=\s+Reference\s+No\.?|\s+Last\s+Status\b|$)", re.IGNORECASE)
BOOKING_DATE_RE = re.compile(r"\bBooking\s+Date\s+(.+?)(?=\s+Shipment\s+Details|\s+Destination\b|$)", re.IGNORECASE)
ORIGIN_DESTINATION_RE = re.compile(r"\bOrigin\s+(.+?)\s+Destination\s+(.+?)(?=\s+Pincode|\s+No\.\s+of\s+pieces|$)", re.IGNORECASE)
PIECES_SERVICE_RE = re.compile(r"\bNo\.\s+of\s+pieces\s+([0-9]+)\s+Service\s+Type\s+(.+?)(?=\s+Package\s+contents|$)", re.IGNORECASE)
PACKAGE_CONTENTS_RE = re.compile(r"\bPackage\s+contents\s*([A-Za-z0-9 /_-]+?)(?=\s+Receiver\s+Details|\s+Receiver\s+Name|$)", re.IGNORECASE)
RECEIVER_NAME_RE = re.compile(r"\bReceiver\s+Name\s+(.+?)(?=\s+Relationship|\s+Phone|\s+Shipment\s+Tracking\s+History|$)", re.IGNORECASE)
TRACKING_EVENT_RE = re.compile(
    r"(?P<date>[A-Z][a-z]{2},\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+'?\d{2})\|"
    r"(?P<time>[0-9]{1,2}:[0-9]{2}\s+Hrs)\s+"
    r"(?P<activity>.+?)\s+"
    r"(?P<location>[A-Z][A-Za-z (),.'/-]+?)"
    r"(?=\s+[A-Z][a-z]{2},\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+'?\d{2}\||$)",
    re.IGNORECASE,
)
SUPPORT_QA_MODE_ADAPTIVE_FAST = "adaptive_fast"
ADAPTIVE_FAST_QA_FLAG = "LLM_QA_SKIPPED_ADAPTIVE_FAST"
SUPPORT_HARD_COMPLIANCE_FLAGS = {
    "LOW_ROUTING_CONFIDENCE",
    "HUMAN_HANDOFF",
    "HUMAN_HANDOFF_REQUIRED",
    "POLICY_GAP",
    "BILLING_DISPUTE",
    "UNSAFE_RESPONSE",
    "VIP_REVIEW",
    "NEGATIVE_SENTIMENT",
}


class EmailDeliveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Gmail delivery status")
    recipient: str | None = Field(..., description="Recipient email address")
    message_id: str | None = Field(..., description="Gmail message id when sent")
    error: str | None = Field(..., description="Delivery error when sending fails")


class OutboundPayloadPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str = Field(..., description="Outbound channel")
    to: str | None = Field(..., description="Masked or provider-safe recipient reference")
    type: str = Field(..., description="Outbound message type, such as text")
    body: str = Field(..., description="Draft response body preview")


class SupportTicketOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(..., description="Support ticket identifier")
    customer_email: str = Field(..., description="Customer email or safe placeholder")
    phone_number: str | None = Field(..., description="Optional customer phone number")
    ticket_summary: str = Field(..., description="Brief summary of the customer inquiry")
    sentiment_analysis: SentimentIntentAnalysis = Field(
        ..., description="Sentiment, intent, customer tier, urgency, language, and handoff result"
    )
    rma_validation: RMAValidationResult | None = Field(
        ..., description="Return eligibility result when the inquiry is an RMA request, otherwise null"
    )
    logistics_output: LogisticsIntegrationOutput | None = Field(
        ..., description="Simulated return label and WMS notification when RMA is eligible, otherwise null"
    )
    escalation_flag: bool = Field(..., description="True when human support handoff is required")
    resolution_steps: list[str] = Field(
        ..., description="Step-by-step resolution or troubleshooting guide"
    )
    drafted_response: str = Field(
        ..., description="Final, polished response ready to send to the customer"
    )
    internal_notes: str = Field(..., description="Internal notes for support agents")
    qa_notes: str = Field(..., description="Quality assurance feedback and compliance checks")
    compliance_tags: list[str] = Field(..., description="Compliance and workflow handling tags")
    recommended_follow_up: str = Field(
        ..., description="Suggested next steps or follow-up timing"
    )
    channel_recommended_action: str = Field(
        "approve_send", description="Recommended channel action, such as approve_send, human_handoff, or draft_only"
    )
    outbound_channel: str = Field("email", description="Channel where the response should be sent")
    outbound_payload_preview: OutboundPayloadPreview = Field(
        ...,
        description="Provider-safe preview payload for the outbound channel",
    )
    requires_approval: bool = Field(True, description="True when an agent must approve before sending")
    email_delivery: EmailDeliveryResult | None = Field(
        ..., description="Gmail delivery result, or null before post-processing"
    )


class CustomerServiceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Unique conversation identifier")
    customer_context: "CustomerContextOutput" = Field(..., description="Customer name, tier, language, and channel")
    detected_language: str = Field("en", description="Detected ISO language code for the latest customer message")
    language_plan: str = Field("English", description="Human-readable response language for CrewAI prompts")
    detected_intent: str = Field(..., description="pre_sales | order_fulfillment | post_sales_support")
    routing_confidence: float = Field(..., ge=0, le=1)
    pre_sales_response: "PreSalesResponseOutput | None" = None
    order_response: "OrderResponseOutput | None" = None
    support_response: "SupportResponseOutput | None" = None
    final_response: str = Field(..., description="Polished, channel-adapted customer response")
    qa_status: str = Field(..., description="APPROVED | REVIEW_REQUIRED | REJECTED")
    compliance_flags: list[str] = Field(default_factory=list)
    recommended_follow_up: str = Field(..., description="When/how to follow up if needed")
    escalation_needed: bool = Field(..., description="True if a human agent should take over")
    data_sources: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class CustomerContextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Customer"
    tier: str = "STANDARD"
    language: str = "English"
    channel: str = "email"


class KeyValueOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str


class VariantOptionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sku: str | None = None
    price: str | None = None
    in_stock: bool | None = None
    best_for: str | None = None


class KnowledgeResultOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float | None = None
    source: str
    heading: str
    content: str


class CatalogProductOfferOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "not_found"
    product_found: bool | None = None
    product_name: str | None = None
    unit_price: str | None = None
    carton_quantity: str | None = None
    carton_size: str | None = None
    carton_weight: str | None = None
    single_product_size: str | None = None
    single_product_weight: str | None = None
    discount_policy: str | None = None
    source: str | None = None
    heading: str | None = None
    evidence: list[str] = Field(default_factory=list)
    data_source: str | None = None


class PreSalesResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_found: bool | None = None
    family_code: str | None = None
    product_recommendation: str | None = None
    feature_explanation: str | None = None
    comparison_summary: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    verified_features: list[str] = Field(default_factory=list)
    compatibility_info: list[KeyValueOutput] = Field(default_factory=list)
    variant_options: list[VariantOptionOutput] = Field(default_factory=list)
    regional_compliance: str | None = None
    confidence_level: float | None = None
    requires_human_review: bool | None = None
    last_updated: str | None = None
    data_source: str | None = None
    knowledge_data_source: str | None = None
    catalog_knowledge_results: list[KnowledgeResultOutput] = Field(default_factory=list)
    catalog_product_offer: CatalogProductOfferOutput | None = None
    pricing_guardrails: list[str] = Field(default_factory=list)


class TrackingInfoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carrier: str | None = None
    tracking_number: str | None = None
    tracking_url: str | None = None
    last_update: str | None = None


class CustomsInfoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = None
    note: str | None = None


class TrackingEventOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str | None = None
    time: str | None = None
    activity: str
    location: str | None = None


class LocalTrackingRecordOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_number: str | None = None
    reference_number: str | None = None
    last_status: str | None = None
    last_status_date: str | None = None
    booking_date: str | None = None
    origin: str | None = None
    destination: str | None = None
    service_type: str | None = None
    pieces: int | None = None
    package_contents: str | None = None
    receiver_name: str | None = None
    tracking_history: list[TrackingEventOutput] = Field(default_factory=list)
    source: str | None = None
    data_source: str = "local_tracking_pdf"


class OrderResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_found: bool | None = None
    order_verified: bool | None = None
    tracking_record_found: bool | None = None
    tracking_lookup_status: str | None = None
    tracking_lookup_query: list[str] = Field(default_factory=list)
    order_id: str | None = None
    current_status: str | None = None
    status_explanation: str | None = None
    tracking_info: TrackingInfoOutput | None = None
    delivery_estimate: str | None = None
    customs_info: CustomsInfoOutput | None = None
    available_actions: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    escalation_needed: bool | None = None
    next_update_expected: str | None = None
    next_update_timeline: str | None = None
    error_message: str | None = None
    data_source: str | None = None
    knowledge_data_source: str | None = None
    order_knowledge_results: list[KnowledgeResultOutput] = Field(default_factory=list)
    local_tracking_record: LocalTrackingRecordOutput | None = None


class SupportResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_reference: str | None = None
    sentiment_analysis: str | None = None
    rma_validation: str | None = None
    logistics_output: str | None = None
    resolution_steps: list[str] = Field(default_factory=list)
    internal_notes: str | None = None
    qa_notes: str | None = None


CustomerServiceOutput.model_rebuild()
CUSTOMER_SERVICE_OUTPUT_FIELDS = set(CustomerServiceOutput.model_fields)


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_support_tools(config_context: dict[str, Any]) -> list[Any]:
    return [
        SupportKnowledgeSearchTool(
            knowledge_dir=config_context.get("support_knowledge_dir")
            or str(BASE_DIR / "docs" / "knowledge_base")
        ),
        *build_support_external_search_tools("post_sales_support", config_context),
    ]


def _fallback_customer_email(inputs: dict[str, Any]) -> str:
    if inputs.get("customer_email"):
        return str(inputs["customer_email"])
    person = str(inputs.get("person") or "customer").strip().lower()
    customer = str(inputs.get("customer") or "unknown").strip().lower()
    safe_local = "".join(character for character in person.replace(" ", ".") if character.isalnum() or character == ".")
    safe_domain = "".join(character for character in customer.replace(" ", "") if character.isalnum())
    return f"{safe_local or 'customer'}@{safe_domain or 'unknown'}.local"


def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    normalized["ticket_id"] = normalized.get("ticket_id") or "TKT-PENDING"
    normalized["customer_email"] = _fallback_customer_email(normalized)
    normalized["phone_number"] = normalized.get("phone_number")
    normalized["inquiry_text"] = normalized.get("inquiry_text") or normalized.get("inquiry") or ""
    normalized["return_reason"] = normalized.get("return_reason") or normalized["inquiry_text"]
    normalized["order_id"] = normalized.get("order_id") or "ORDER-NOT-PROVIDED"
    normalized["item_sku"] = normalized.get("item_sku") or "SKU-NOT-PROVIDED"
    normalized["order_history"] = normalized.get("order_history") or {}
    normalized["channel"] = normalized.get("channel") or "email"
    normalized["session_id"] = normalized.get("session_id") or "unknown"
    normalized["channel_thread_id"] = normalized.get("channel_thread_id")
    normalized["channel_message_id"] = normalized.get("channel_message_id")
    normalized["sender_profile"] = normalized.get("sender_profile") or {}
    normalized["attachments"] = normalized.get("attachments") or []
    normalized["conversation_history"] = normalized.get("conversation_history") or []
    normalized["customer_tier"] = normalized.get("customer_tier") or _customer_tier_from_history(normalized.get("order_history"))
    normalized["region"] = normalized.get("region") or _region_from_inputs(normalized)
    normalized["product_category"] = normalized.get("product_category") or normalized.get("product_category_hint") or "Smart Home Camera"
    normalized["use_case_if_provided"] = normalized.get("use_case_if_provided") or normalized.get("use_case") or "not provided"
    normalized["order_id_if_provided"] = normalized.get("order_id_if_provided") or (
        normalized.get("order_id") if normalized.get("order_id") != "ORDER-NOT-PROVIDED" else "not provided"
    )
    detected_language = normalized.get("detected_language") or LanguageDetector.detect(normalized["inquiry_text"])
    normalized["detected_language"] = detected_language
    normalized["language_plan"] = normalized.get("language_plan") or LanguageDetector.get_crewai_language_plan(
        detected_language
    )
    return normalized


def _customer_tier_from_history(order_history: dict[str, Any] | None) -> str:
    order_history = order_history or {}
    if float(order_history.get("lifetime_value") or 0) > 5000:
        return "VIP"
    if int(order_history.get("order_count") or 0) >= 5:
        return "PREMIUM"
    if int(order_history.get("order_count") or 0) >= 1:
        return "STANDARD"
    return "NEW"


def _region_from_inputs(inputs: dict[str, Any]) -> str:
    order_history = inputs.get("order_history") if isinstance(inputs.get("order_history"), dict) else {}
    region = inputs.get("region") or order_history.get("region")
    if region:
        return str(region).upper()
    language = str(inputs.get("detected_language") or "").lower()
    return {"ja": "JP", "zh": "CN", "de": "DE", "fr": "FR", "es": "ES"}.get(language, "US")


def _build_automation_context(inputs: dict[str, Any]) -> dict[str, Any]:
    sentiment = analyze_sentiment_intent(
        inquiry_text=inputs["inquiry_text"],
        customer_email=inputs["customer_email"],
        order_history=inputs.get("order_history"),
        detected_language=inputs.get("detected_language"),
    )
    rma_payload: dict[str, Any] = {"rma_validation": None, "logistics_output": None}
    if sentiment["intent_category"] == "RMA_REQUEST":
        rma_payload = process_rma_request(
            order_id=inputs["order_id"],
            item_sku=inputs["item_sku"],
            return_reason=inputs["return_reason"],
            detected_language=sentiment["language_detected"],
            order_history=inputs.get("order_history"),
        )

    automation_context = {
        "sentiment_analysis": sentiment,
        "rma_validation": rma_payload.get("rma_validation"),
        "logistics_output": rma_payload.get("logistics_output"),
        "escalation_flag": sentiment["requires_human_handoff"],
        "compliance_tags": _compliance_tags(sentiment, rma_payload),
    }
    return automation_context


def _compliance_tags(sentiment: dict[str, Any], rma_payload: dict[str, Any]) -> list[str]:
    tags = ["GDPR_COMPLIANT_REVIEW", "CCPA_OPT_OUT_AVAILABLE"]
    if sentiment["requires_human_handoff"]:
        tags.append("HUMAN_HANDOFF_REQUIRED")
    if sentiment["intent_category"] == "RMA_REQUEST":
        tags.append("RMA_POLICY_CHECKED")
    if rma_payload.get("logistics_output"):
        tags.append("SIMULATED_LOGISTICS_LABEL")
    return tags


def _attach_automation_context(result: dict[str, Any], context: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.setdefault("ticket_id", inputs["ticket_id"])
    normalized.setdefault("customer_email", inputs["customer_email"])
    normalized["phone_number"] = normalized.get("phone_number") or inputs.get("phone_number")
    normalized["sentiment_analysis"] = normalized.get("sentiment_analysis") or context["sentiment_analysis"]
    normalized["rma_validation"] = normalized.get("rma_validation") or context["rma_validation"]
    normalized["logistics_output"] = normalized.get("logistics_output") or context["logistics_output"]
    normalized["escalation_flag"] = bool(context["escalation_flag"])
    normalized["compliance_tags"] = normalized.get("compliance_tags") or context["compliance_tags"]
    normalized["outbound_channel"] = normalized.get("outbound_channel") or inputs.get("channel") or "email"
    normalized["requires_approval"] = _requires_human_approval(normalized, context)
    normalized["channel_recommended_action"] = normalized.get("channel_recommended_action") or (
        "human_handoff" if normalized["escalation_flag"] else "approve_send"
    )
    normalized["outbound_payload_preview"] = normalized.get("outbound_payload_preview") or {
        "channel": normalized["outbound_channel"],
        "to": inputs.get("channel_thread_id") or inputs.get("customer_email"),
        "type": "text",
        "body": normalized.get("drafted_response") or "",
    }
    if not normalized.get("internal_notes"):
        handoff = "Human handoff required." if context["escalation_flag"] else "No human handoff required."
        normalized["internal_notes"] = f"{handoff} Automation context generated before CrewAI response drafting."
    return normalized


def _requires_human_approval(result: dict[str, Any], context: dict[str, Any]) -> bool:
    sentiment = context["sentiment_analysis"]
    rma = context.get("rma_validation")
    forced_review = bool(
        context["escalation_flag"]
        or sentiment.get("customer_tier") in {"VIP", "PREMIUM"}
        or float(sentiment.get("sentiment_score") or 0) < -0.25
        or sentiment.get("intent_category") == "BILLING_ISSUE"
        or (rma and not rma.get("eligible_for_return", True))
    )
    if forced_review:
        return True
    if "requires_approval" in result and result.get("requires_approval") is not None:
        return bool(result.get("requires_approval"))
    return bool(
        result.get("channel_recommended_action") in {"approve_send", "draft_only", "human_handoff"}
    )


def _bool_config(config_context: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config_context.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _email_delivery_status(
    result: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    recipient = result.get("customer_email")
    if result.get("outbound_channel") and result.get("outbound_channel") != "email":
        return {
            "status": "skipped_channel",
            "recipient": recipient,
            "message_id": None,
            "error": None,
        }
    if result.get("escalation_flag"):
        return {
            "status": "skipped_escalation",
            "recipient": recipient,
            "message_id": None,
            "error": None,
        }

    if not _bool_config(config_context, "gmail_send_enabled"):
        return {
            "status": "disabled",
            "recipient": recipient,
            "message_id": None,
            "error": None,
        }

    access_token = resolve_gmail_access_token(
        access_token=config_context.get("gmail_access_token"),
        client_id=config_context.get("gmail_client_id"),
        client_secret=config_context.get("gmail_client_secret"),
        refresh_token=config_context.get("gmail_refresh_token"),
    )
    sender = config_context.get("gmail_sender_email")
    if not access_token or not sender or not recipient:
        return {
            "status": "missing_credentials",
            "recipient": recipient,
            "message_id": None,
            "error": "Gmail access token or refresh-token credentials, sender email, and customer email are required.",
        }

    return send_gmail_message(
        access_token=str(access_token),
        sender=str(sender),
        recipient=str(recipient),
        subject=f"Re: Support ticket {result.get('ticket_id') or 'TKT-PENDING'}",
        body=str(result.get("drafted_response") or ""),
    )


def _memory_enabled(config_context: dict[str, Any]) -> bool:
    return bool(config_context.get("crewai_memory_enabled"))


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_support_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for Customer Service orchestration."""
    config_context = config_context or {}
    normalized_inputs = _normalize_inputs(inputs)
    automation_context = _build_automation_context(normalized_inputs)
    router_tool = IntentRouterTool(config_context=config_context)
    router_result = router_tool._run(
        inquiry_text=normalized_inputs["inquiry_text"],
        has_order_id=bool(normalized_inputs.get("order_id") and normalized_inputs.get("order_id") != "ORDER-NOT-PROVIDED"),
        customer_tier=normalized_inputs.get("customer_tier", "STANDARD"),
        language=normalized_inputs.get("detected_language", "en"),
    )
    if router_result["requires_human_review"]:
        return _handoff_customer_service_output(normalized_inputs, automation_context, router_result)
    product_hint = (router_result.get("context_enrichment") or {}).get("product_category_hint")
    if product_hint and normalized_inputs.get("product_category") == "Smart Home Camera":
        normalized_inputs["product_category"] = product_hint

    pre_sales_tool = _build_pre_sales_tool(config_context)
    pre_sales_context = pre_sales_tool._run(
        product_category=normalized_inputs["product_category"],
        inquiry_keywords=str(normalized_inputs["inquiry_text"]).split(),
        region=normalized_inputs["region"],
        language=normalized_inputs["detected_language"],
    )
    _attach_catalog_knowledge_context(pre_sales_context, normalized_inputs, config_context)
    order_context = OrderTrackingTool()._run(
        order_id=None if normalized_inputs["order_id"] == "ORDER-NOT-PROVIDED" else normalized_inputs["order_id"],
        customer_email=normalized_inputs.get("customer_email"),
        region=normalized_inputs["region"],
    )
    _attach_order_knowledge_context(order_context, normalized_inputs, config_context)
    crew_inputs = {
        **normalized_inputs,
        "response_type": router_result["detected_intent"],
        "routing_confidence": router_result["confidence_score"],
        "pre_sales_context": pre_sales_context,
        "order_context": order_context,
        "sentiment_analysis": automation_context["sentiment_analysis"],
        "rma_validation": automation_context["rma_validation"],
        "logistics_output": automation_context["logistics_output"],
        "escalation_flag": automation_context["escalation_flag"],
        "compliance_tags": automation_context["compliance_tags"],
        "language_detected": automation_context["sentiment_analysis"]["language_detected"],
        "language_plan": normalized_inputs["language_plan"],
        "channel": normalized_inputs["channel"],
        "session_id": normalized_inputs["session_id"],
        "channel_thread_id": normalized_inputs.get("channel_thread_id"),
        "channel_message_id": normalized_inputs.get("channel_message_id"),
        "sender_profile": normalized_inputs.get("sender_profile"),
        "attachments": normalized_inputs.get("attachments"),
        "conversation_history": normalized_inputs.get("conversation_history"),
    }

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(agents_config, workflow='support')
    tasks_config = _load_yaml_config("tasks.yaml")
    llm = build_llm(config_context)

    pre_sales_agent = Agent(
        config=agents_config["pre_sales_specialist"],
        llm=llm,
        tools=[pre_sales_tool, *build_support_external_search_tools("pre_sales", config_context)],
        allow_delegation=False,
    )
    order_agent = Agent(
        config=agents_config["order_fulfillment_specialist"],
        llm=llm,
        tools=[OrderTrackingTool(), *build_support_external_search_tools("order_fulfillment", config_context)],
        allow_delegation=False,
    )
    support_agent = Agent(
        config=agents_config["senior_support_agent"],
        llm=llm,
        tools=[RMAAutomationTool(), *_build_support_tools(config_context)],
        allow_delegation=True,
    )
    qa_agent = Agent(
        config=agents_config["support_qa_specialist"],
        llm=llm,
        allow_delegation=False,
    )

    task_by_intent = {
        "pre_sales": ("pre_sales_consultation", pre_sales_agent),
        "order_fulfillment": ("order_status_handling", order_agent),
        "post_sales_support": ("inquiry_resolution", support_agent),
    }
    stage_task_name, stage_agent = task_by_intent[router_result["detected_intent"]]
    skip_llm_qa = _should_skip_support_llm_qa(
        inputs=normalized_inputs,
        automation_context=automation_context,
        router_result=router_result,
        pre_sales_context=pre_sales_context,
        config_context=config_context,
    )
    stage_task = Task(
        config=tasks_config[stage_task_name],
        agent=stage_agent,
    )
    tasks = [stage_task]
    agents = [stage_agent]
    task_names = [stage_task_name]
    if not skip_llm_qa:
        qa_task = Task(
            config=tasks_config["quality_assurance_review"],
            agent=qa_agent,
            context=[stage_task],
            output_pydantic=CustomerServiceOutput,
        )
        tasks.append(qa_task)
        agents.append(qa_agent)
        task_names.append("quality_assurance_review")
    attach_task_progress(config_context, "support", tasks, task_names)

    support_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=False,
        memory=_memory_enabled(config_context),
    )

    result = _serialize_crew_result(support_crew.kickoff(inputs=crew_inputs))
    if skip_llm_qa:
        result = _apply_adaptive_fast_qa_defaults(result, automation_context, router_result)
    return _attach_customer_service_context(
        result=result,
        inputs=normalized_inputs,
        automation_context=automation_context,
        router_result=router_result,
        pre_sales_context=pre_sales_context,
        order_context=order_context,
        config_context=config_context,
    )


def _should_skip_support_llm_qa(
    *,
    inputs: dict[str, Any],
    automation_context: dict[str, Any],
    router_result: dict[str, Any],
    pre_sales_context: dict[str, Any],
    config_context: dict[str, Any],
) -> bool:
    if str(config_context.get("support_qa_mode") or "full_llm").lower() != SUPPORT_QA_MODE_ADAPTIVE_FAST:
        return False
    if float(router_result.get("confidence_score") or 0) < 0.85:
        return False

    detected_intent = str(router_result.get("detected_intent") or "")
    sentiment = automation_context.get("sentiment_analysis") or {}
    if automation_context.get("escalation_flag") or sentiment.get("requires_human_handoff"):
        return False
    if sentiment.get("customer_tier") in {"VIP", "PREMIUM"}:
        return False
    if float(sentiment.get("sentiment_score") or 0) < -0.25:
        return False
    if sentiment.get("sentiment_label") in {"ANGRY", "FRUSTRATED"}:
        return False
    if sentiment.get("intent_category") == "BILLING_ISSUE":
        return False
    if inputs.get("attachments"):
        return False

    compliance_flags = {str(flag).upper() for flag in automation_context.get("compliance_tags") or []}
    if SUPPORT_HARD_COMPLIANCE_FLAGS.intersection(compliance_flags):
        return False

    if detected_intent == "post_sales_support":
        return True
    if detected_intent == "pre_sales":
        return _has_verified_pre_sales_context(pre_sales_context)
    return False


def _has_verified_pre_sales_context(pre_sales_context: dict[str, Any]) -> bool:
    offer = pre_sales_context.get("catalog_product_offer")
    if isinstance(offer, dict) and offer.get("status") == "found":
        return True
    if pre_sales_context.get("catalog_knowledge_results"):
        return True

    data_source = str(pre_sales_context.get("data_source") or "").lower()
    if data_source and data_source != "mock_fallback":
        return bool(pre_sales_context.get("product_found"))
    return False


def _apply_adaptive_fast_qa_defaults(
    result: dict[str, Any],
    automation_context: dict[str, Any],
    router_result: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(result)
    normalized["qa_status"] = _adaptive_fast_qa_status(automation_context, router_result)
    normalized.setdefault("recommended_follow_up", "Follow up if the customer replies.")
    normalized.setdefault("escalation_needed", False)

    flags = list(normalized.get("compliance_flags") or automation_context.get("compliance_tags") or [])
    if ADAPTIVE_FAST_QA_FLAG not in flags:
        flags.append(ADAPTIVE_FAST_QA_FLAG)
    normalized["compliance_flags"] = flags

    assumptions = list(normalized.get("assumptions") or [])
    assumptions.append(
        f"Low-risk {router_result.get('detected_intent')} LLM QA was skipped by SUPPORT_QA_MODE=adaptive_fast."
    )
    normalized["assumptions"] = assumptions
    return normalized


def _adaptive_fast_qa_status(automation_context: dict[str, Any], router_result: dict[str, Any]) -> str:
    if router_result.get("detected_intent") == "pre_sales":
        return "APPROVED"
    sentiment = automation_context.get("sentiment_analysis") or {}
    rma = automation_context.get("rma_validation")
    if sentiment.get("intent_category") == "RMA_REQUEST":
        if not isinstance(rma, dict) or not rma.get("eligible_for_return"):
            return "REVIEW_REQUIRED"
        if automation_context.get("logistics_output") is None:
            return "REVIEW_REQUIRED"
    return "APPROVED"


def _build_pre_sales_tool(config_context: dict[str, Any]) -> PreSalesProductKnowledgeTool:
    backend = str(config_context.get("pim_backend") or "akeneo").lower()
    return PreSalesProductKnowledgeTool(
        pim_backend=backend,
        pim_base_url=config_context.get(f"pim_{backend}_base_url"),
        pim_api_key=config_context.get(f"pim_{backend}_api_key"),
    )


def _attach_catalog_knowledge_context(
    pre_sales_context: dict[str, Any],
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    knowledge_dir = config_context.get("support_knowledge_dir") or str(BASE_DIR / "docs" / "knowledge_base")
    query = " ".join(
        item
        for item in [
            str(inputs.get("product_category") or ""),
            str(inputs.get("inquiry_text") or ""),
            str(inputs.get("use_case_if_provided") or ""),
        ]
        if item.strip()
    )
    results = search_knowledge_base(query, knowledge_dir=knowledge_dir, top_k=3)
    if results:
        pre_sales_context["catalog_knowledge_results"] = results
        pre_sales_context["knowledge_data_source"] = "local_vector_knowledge_base"
    if pre_sales_context.get("data_source") == "mock_fallback":
        pre_sales_context["verified_features"] = []
        pre_sales_context["compatibility_info"] = {}
        pre_sales_context["variant_options"] = {}
        pre_sales_context["mock_fallback_notice"] = (
            "Mock fallback data is not customer-facing evidence; use catalog or real PIM facts only."
        )
    offer = extract_catalog_product_offer(query, knowledge_dir=knowledge_dir)
    if offer.get("status") == "found":
        pre_sales_context["catalog_product_offer"] = offer
        pre_sales_context["knowledge_data_source"] = "local_pdf_catalog"
        pre_sales_context["pricing_guardrails"] = [
            "Quote only the catalog unit_price when a price is provided in catalog_product_offer.",
            "Do not invent discount tiers, reduced prices, wholesale rates, or negotiated offers.",
            "If the customer asks for a discount, state that discounts require sales approval and offer to have sales review the request.",
            "Treat catalog wording such as 'Please contact me for a discount' as a review path, not as approval to quote a discount.",
            "Do not use mock_fallback variants, SKUs, features, compatibility claims, or prices as customer-facing facts.",
            "If only catalog packaging and price facts are available, say detailed feature specifications are not listed in the catalog.",
        ]
    return pre_sales_context


def _attach_order_knowledge_context(
    order_context: dict[str, Any],
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> dict[str, Any]:
    knowledge_dir = config_context.get("support_knowledge_dir") or str(BASE_DIR / "docs" / "knowledge_base")
    identifiers = _tracking_identifiers_from_inputs(inputs)
    if identifiers:
        order_context["tracking_lookup_query"] = sorted(identifiers)
        order_context["tracking_lookup_status"] = "not_found"
        order_context["tracking_record_found"] = False
    query = " ".join(
        item
        for item in [
            str(inputs.get("order_id") or ""),
            str(inputs.get("order_id_if_provided") or ""),
            str(inputs.get("customer_email") or ""),
            str(inputs.get("inquiry_text") or ""),
            " ".join(
                str(attachment.get("filename") or "")
                for attachment in inputs.get("attachments", [])
                if isinstance(attachment, dict)
            ),
        ]
        if item.strip() and item.strip() != "ORDER-NOT-PROVIDED"
    )
    if not query.strip():
        return order_context

    results = search_knowledge_base(query, knowledge_dir=knowledge_dir, top_k=5)
    candidate_order_results = [
        result
        for result in results
        if _looks_like_order_knowledge_result(result)
    ]
    local_record = _extract_local_tracking_record(candidate_order_results, inputs)
    if identifiers and not local_record:
        exact_result = _find_exact_tracking_result(identifiers, knowledge_dir)
        if exact_result:
            candidate_order_results = [exact_result]
            local_record = _parse_local_tracking_record(
                str(exact_result.get("content") or ""),
                str(exact_result.get("source") or ""),
            )
    order_results = candidate_order_results if not identifiers else []
    if local_record:
        order_results = [
            result
            for result in candidate_order_results
            if str(result.get("source") or "") == str(local_record.get("source") or "")
        ] or candidate_order_results[:1]
    if not order_results:
        return order_context

    order_context["order_knowledge_results"] = order_results
    order_context["knowledge_data_source"] = "local_order_knowledge_base"
    if local_record:
        order_context["tracking_record_found"] = True
        order_context["tracking_lookup_status"] = "found"
        order_context["local_tracking_record"] = local_record
        order_context["current_status"] = local_record.get("last_status") or order_context.get("current_status")
        order_context["status_explanation"] = _local_tracking_status_explanation(local_record)
        order_context["tracking_info"] = {
            "carrier": "DTDC",
            "tracking_number": local_record.get("tracking_number"),
            "tracking_url": None,
            "last_update": local_record.get("last_status_date"),
        }
        order_context["delivery_estimate"] = local_record.get("last_status_date")
        order_context["available_actions"] = [
            "Share the local tracking record details with the customer.",
            "Escalate to operations only if the customer disputes the delivery scan or needs proof of delivery.",
        ]
    if not order_context.get("order_found"):
        order_context["data_source"] = "local_order_knowledge_base"
        if local_record:
            order_context["error_message"] = None
            order_context["suggested_actions"] = [
                "Use the local tracking document facts as the customer-facing answer.",
                "Mention that the tracking record is available even if it is not linked to an internal order record.",
            ]
        else:
            order_context["status_explanation"] = (
                "Order details were not found in the commerce tracking system, "
                "but related local order/tracking knowledge base documents were found."
            )
            order_context["suggested_actions"] = [
                "Use the local order/tracking document facts in the customer reply.",
                "Escalate to operations if any delivery status, address, or carrier detail remains unclear.",
            ]
    return order_context


def _find_exact_tracking_result(identifiers: set[str], knowledge_dir: str) -> dict[str, Any] | None:
    if not identifiers:
        return None
    chunks = load_knowledge_chunks(str(Path(knowledge_dir)))
    for chunk in chunks:
        if not str(chunk.source).lower().endswith(".pdf"):
            continue
        record = _parse_local_tracking_record(chunk.content, chunk.source)
        if not record:
            continue
        record_identifiers = {
            str(record.get("tracking_number") or "").upper(),
            str(record.get("reference_number") or "").upper(),
        }
        if identifiers.intersection(record_identifiers):
            return {
                "score": 1.0,
                "source": chunk.source,
                "heading": chunk.heading,
                "content": chunk.content,
            }
    return None


def _looks_like_order_knowledge_result(result: dict[str, Any]) -> bool:
    source = str(result.get("source") or "").lower()
    heading = str(result.get("heading") or "").lower()
    content = str(result.get("content") or "").lower()
    haystack = f"{source}\n{heading}\n{content}"
    order_terms = {
        "order",
        "tracking",
        "shipment",
        "shipping",
        "delivery",
        "carrier",
        "parcel",
        "package",
        "logistics",
        "consignee",
    }
    return source.endswith(".pdf") and any(term in haystack for term in order_terms)


def _extract_local_tracking_record(
    order_results: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> dict[str, Any] | None:
    identifiers = _tracking_identifiers_from_inputs(inputs)
    if not identifiers:
        return None
    for result in order_results:
        content = str(result.get("content") or "")
        record = _parse_local_tracking_record(content, str(result.get("source") or ""))
        if not record:
            continue
        record_identifiers = {
            str(record.get("tracking_number") or "").upper(),
            str(record.get("reference_number") or "").upper(),
        }
        if identifiers and not identifiers.intersection(record_identifiers):
            continue
        return record
    return None


def _tracking_identifiers_from_inputs(inputs: dict[str, Any]) -> set[str]:
    fields = [
        inputs.get("order_id"),
        inputs.get("order_id_if_provided"),
        inputs.get("inquiry_text"),
        inputs.get("inquiry"),
    ]
    identifiers: set[str] = set()
    for field in fields:
        text = str(field or "")
        if not text or text == "ORDER-NOT-PROVIDED":
            continue
        identifiers.update(match.group(0).upper() for match in TRACKING_IDENTIFIER_RE.finditer(text))
    return identifiers


def _parse_local_tracking_record(content: str, source: str | None = None) -> dict[str, Any] | None:
    text = _single_line(content)
    tracking_number = _first_match(TRACKING_NO_RE, text)
    reference_number = _first_match(REFERENCE_NO_RE, text)
    if not tracking_number and not reference_number:
        return None
    pieces = _first_match(PIECES_SERVICE_RE, text, group=1)
    return {
        "tracking_number": tracking_number,
        "reference_number": reference_number,
        "last_status": _clean_tracking_value(_first_match(LAST_STATUS_RE, text)),
        "last_status_date": _normalize_tracking_date(_first_match(LAST_STATUS_DATE_RE, text)),
        "booking_date": _normalize_tracking_date(_first_match(BOOKING_DATE_RE, text)),
        "origin": _clean_tracking_value(_first_match(ORIGIN_DESTINATION_RE, text, group=1)),
        "destination": _clean_tracking_value(_first_match(ORIGIN_DESTINATION_RE, text, group=2)),
        "service_type": _clean_tracking_value(_first_match(PIECES_SERVICE_RE, text, group=2)),
        "pieces": int(pieces) if pieces and pieces.isdigit() else None,
        "package_contents": _clean_tracking_value(_first_match(PACKAGE_CONTENTS_RE, text)),
        "receiver_name": _clean_tracking_value(_first_match(RECEIVER_NAME_RE, text)),
        "tracking_history": _tracking_events(text),
        "source": source,
        "data_source": "local_tracking_pdf",
    }


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _first_match(pattern: re.Pattern[str], text: str, group: int = 1) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return str(match.group(group)).strip()


def _clean_tracking_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" .|")
    return cleaned or None


def _normalize_tracking_date(value: str | None) -> str | None:
    cleaned = _clean_tracking_value(value)
    if not cleaned:
        return None
    match = re.match(r"(.+?)\s+([A-Za-z]+)'?(\d{2})$", cleaned)
    if not match:
        return cleaned
    day, month, year = match.groups()
    return f"{day} {month.title()} 20{year}"


def _tracking_events(text: str) -> list[dict[str, str | None]]:
    events: list[dict[str, str | None]] = []
    for match in TRACKING_EVENT_RE.finditer(text):
        activity, location = _split_tracking_activity_location(
            f"{match.group('activity')} {match.group('location')}"
        )
        if not activity:
            continue
        events.append(
            {
                "date": _normalize_tracking_date(match.group("date")),
                "time": _clean_tracking_value(match.group("time")),
                "activity": activity,
                "location": location,
            }
        )
    return events


def _split_tracking_activity_location(value: str) -> tuple[str | None, str | None]:
    cleaned = _clean_tracking_value(value)
    if not cleaned:
        return None, None
    known_activities = [
        "Successfully Delivered",
        "Out For Delivery",
        "Received At Facility",
        "Processed & Forwarded To Hub",
        "Received At Hub",
        "Processed & Forwarded To Facility",
        "Booked At Facility",
    ]
    for activity in known_activities:
        if cleaned.lower().startswith(activity.lower()):
            location = _clean_tracking_value(cleaned[len(activity) :])
            return activity, location
    parts = cleaned.split(" ", 1)
    return parts[0], parts[1] if len(parts) > 1 else None


def _local_tracking_status_explanation(record: dict[str, Any]) -> str:
    tracking_number = record.get("tracking_number") or "the provided tracking number"
    status = record.get("last_status") or "a tracking update"
    status_date = record.get("last_status_date")
    destination = record.get("destination")
    details = [f"Local tracking record {tracking_number} shows {status}"]
    if destination:
        details.append(f"for destination {destination}")
    if status_date:
        details.append(f"with last status date {status_date}")
    return " ".join(details) + "."


def _customer_context(inputs: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(inputs.get("customer") or "Customer"),
        "tier": str(inputs.get("customer_tier") or "STANDARD"),
        "language": str(inputs.get("language_plan") or "English"),
        "channel": str(inputs.get("channel") or "email"),
    }


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _key_value_outputs(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return []
    return [{"key": str(key), "value": _json_text(item) or ""} for key, item in value.items()]


def _variant_outputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    variants: list[dict[str, Any]] = []
    for name, details in value.items():
        item = details if isinstance(details, dict) else {}
        variants.append(
            {
                "name": str(name),
                "sku": item.get("sku"),
                "price": _json_text(item.get("price")),
                "in_stock": item.get("in_stock") if isinstance(item.get("in_stock"), bool) else None,
                "best_for": item.get("best_for"),
            }
        )
    return variants


def _knowledge_outputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    results: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "score": float(item["score"]) if item.get("score") is not None else None,
                "source": str(item.get("source") or ""),
                "heading": str(item.get("heading") or ""),
                "content": str(item.get("content") or ""),
            }
        )
    return results


def _tracking_event_outputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        activity = str(item.get("activity") or "").strip()
        if not activity:
            continue
        events.append(
            {
                "date": item.get("date"),
                "time": item.get("time"),
                "activity": activity,
                "location": item.get("location"),
            }
        )
    return events


def _local_tracking_record_output(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    pieces = value.get("pieces")
    return {
        "tracking_number": value.get("tracking_number"),
        "reference_number": value.get("reference_number"),
        "last_status": value.get("last_status"),
        "last_status_date": value.get("last_status_date"),
        "booking_date": value.get("booking_date"),
        "origin": value.get("origin"),
        "destination": value.get("destination"),
        "service_type": value.get("service_type"),
        "pieces": int(pieces) if isinstance(pieces, (int, str)) and str(pieces).isdigit() else None,
        "package_contents": value.get("package_contents"),
        "receiver_name": value.get("receiver_name"),
        "tracking_history": _tracking_event_outputs(value.get("tracking_history")),
        "source": value.get("source"),
        "data_source": str(value.get("data_source") or "local_tracking_pdf"),
    }


def _apply_authoritative_pre_sales_context(
    pre_sales_response: dict[str, Any] | None,
    pre_sales_context: dict[str, Any],
) -> None:
    if not isinstance(pre_sales_response, dict):
        return
    offer = _catalog_offer_output(pre_sales_context.get("catalog_product_offer"))
    if offer and offer.get("status") == "found":
        pre_sales_response["product_found"] = True
        pre_sales_response["requires_human_review"] = False
        pre_sales_response["catalog_product_offer"] = offer
        pre_sales_response["knowledge_data_source"] = pre_sales_context.get("knowledge_data_source") or "local_pdf_catalog"
        pre_sales_response["pricing_guardrails"] = _string_list(pre_sales_context.get("pricing_guardrails"))
        if pre_sales_response.get("data_source") in {None, "mock_fallback", "Product catalog"}:
            pre_sales_response["variant_options"] = []
            pre_sales_response["verified_features"] = []
            pre_sales_response["compatibility_info"] = []


def _catalog_offer_output(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "status": str(value.get("status") or "not_found"),
        "product_found": value.get("product_found") if isinstance(value.get("product_found"), bool) else None,
        "product_name": value.get("product_name"),
        "unit_price": value.get("unit_price"),
        "carton_quantity": value.get("carton_quantity"),
        "carton_size": value.get("carton_size"),
        "carton_weight": value.get("carton_weight"),
        "single_product_size": value.get("single_product_size"),
        "single_product_weight": value.get("single_product_weight"),
        "discount_policy": value.get("discount_policy"),
        "source": value.get("source"),
        "heading": value.get("heading"),
        "evidence": _string_list(value.get("evidence")),
        "data_source": value.get("data_source"),
    }


def _normalize_pre_sales_response(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"feature_explanation": str(value)}
    use_customer_facing_specs = value.get("data_source") != "mock_fallback" or isinstance(value.get("catalog_product_offer"), dict)
    return {
        "product_found": value.get("product_found") if isinstance(value.get("product_found"), bool) else None,
        "family_code": value.get("family_code"),
        "product_recommendation": value.get("product_recommendation"),
        "feature_explanation": value.get("feature_explanation"),
        "comparison_summary": value.get("comparison_summary"),
        "next_steps": _string_list(value.get("next_steps")),
        "verified_features": _string_list(value.get("verified_features")) if use_customer_facing_specs else [],
        "compatibility_info": _key_value_outputs(value.get("compatibility_info")) if use_customer_facing_specs else [],
        "variant_options": _variant_outputs(value.get("variant_options")) if value.get("data_source") != "mock_fallback" else [],
        "regional_compliance": _json_text(value.get("regional_compliance")),
        "confidence_level": float(value["confidence_level"]) if value.get("confidence_level") is not None else None,
        "requires_human_review": value.get("requires_human_review")
        if isinstance(value.get("requires_human_review"), bool)
        else None,
        "last_updated": value.get("last_updated"),
        "data_source": value.get("data_source"),
        "knowledge_data_source": value.get("knowledge_data_source"),
        "catalog_knowledge_results": _knowledge_outputs(value.get("catalog_knowledge_results")),
        "catalog_product_offer": _catalog_offer_output(value.get("catalog_product_offer")),
        "pricing_guardrails": _string_list(value.get("pricing_guardrails")),
    }


def _normalize_order_response(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"status_explanation": str(value)}
    tracking = value.get("tracking_info") if isinstance(value.get("tracking_info"), dict) else None
    customs = value.get("customs_info") if isinstance(value.get("customs_info"), dict) else None
    return {
        "order_found": value.get("order_found") if isinstance(value.get("order_found"), bool) else None,
        "order_verified": value.get("order_verified") if isinstance(value.get("order_verified"), bool) else None,
        "tracking_record_found": value.get("tracking_record_found")
        if isinstance(value.get("tracking_record_found"), bool)
        else None,
        "tracking_lookup_status": value.get("tracking_lookup_status"),
        "tracking_lookup_query": _string_list(value.get("tracking_lookup_query")),
        "order_id": value.get("order_id"),
        "current_status": value.get("current_status"),
        "status_explanation": value.get("status_explanation"),
        "tracking_info": tracking,
        "delivery_estimate": value.get("delivery_estimate"),
        "customs_info": customs,
        "available_actions": _string_list(value.get("available_actions")),
        "suggested_actions": _string_list(value.get("suggested_actions")),
        "escalation_needed": value.get("escalation_needed") if isinstance(value.get("escalation_needed"), bool) else None,
        "next_update_expected": value.get("next_update_expected"),
        "next_update_timeline": value.get("next_update_timeline"),
        "error_message": value.get("error_message"),
        "data_source": value.get("data_source"),
        "knowledge_data_source": value.get("knowledge_data_source"),
        "order_knowledge_results": _knowledge_outputs(value.get("order_knowledge_results")),
        "local_tracking_record": _local_tracking_record_output(value.get("local_tracking_record")),
    }


def _normalize_support_response(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"policy_reference": str(value)}
    return {
        "policy_reference": value.get("policy_reference"),
        "sentiment_analysis": _json_text(value.get("sentiment_analysis")),
        "rma_validation": _json_text(value.get("rma_validation")),
        "logistics_output": _json_text(value.get("logistics_output")),
        "resolution_steps": _string_list(value.get("resolution_steps")),
        "internal_notes": value.get("internal_notes"),
        "qa_notes": value.get("qa_notes"),
    }


def _handoff_customer_service_output(
    inputs: dict[str, Any],
    automation_context: dict[str, Any],
    router_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": inputs.get("session_id", "unknown"),
        "customer_context": _customer_context(inputs),
        "detected_language": str(inputs.get("detected_language") or "en"),
        "language_plan": str(inputs.get("language_plan") or "English"),
        "detected_intent": router_result["detected_intent"],
        "routing_confidence": router_result["confidence_score"],
        "pre_sales_response": None,
        "order_response": None,
        "support_response": None,
        "final_response": (
            "Thank you for reaching out. To make sure we handle this correctly, "
            "I'm connecting you with a support specialist who can review the details."
        ),
        "qa_status": "REVIEW_REQUIRED",
        "compliance_flags": ["LOW_ROUTING_CONFIDENCE", *automation_context.get("compliance_tags", [])],
        "recommended_follow_up": "Human agent should respond within 15 minutes.",
        "escalation_needed": True,
        "data_sources": ["intent_router"],
        "assumptions": [
            "Routing confidence below automatic handling threshold after hybrid routing and optional fallback."
        ],
    }


def _attach_customer_service_context(
    *,
    result: dict[str, Any],
    inputs: dict[str, Any],
    automation_context: dict[str, Any],
    router_result: dict[str, Any],
    pre_sales_context: dict[str, Any],
    order_context: dict[str, Any],
    config_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_context = config_context or {}
    normalized = _expand_raw_json_result(dict(result))
    usage_metrics = normalized.pop(INTERNAL_USAGE_KEY, None)
    intent = router_result["detected_intent"]
    final_response = (
        normalized.get("final_response")
        or normalized.get("drafted_response")
        or normalized.get("response")
        or normalized.get("body")
        or normalized.get("raw")
        or "Thank you for reaching out. We are reviewing your request and will follow up shortly."
    )
    normalized["session_id"] = str(normalized.get("session_id") or inputs.get("session_id") or "unknown")
    normalized["customer_context"] = normalized.get("customer_context") or _customer_context(inputs)
    normalized["detected_language"] = str(normalized.get("detected_language") or inputs.get("detected_language") or "en")
    normalized["language_plan"] = str(
        normalized.get("language_plan")
        or inputs.get("language_plan")
        or LanguageDetector.get_crewai_language_plan(normalized["detected_language"])
    )
    if isinstance(normalized["customer_context"], dict):
        normalized["customer_context"]["language"] = normalized["language_plan"]
    normalized["detected_intent"] = normalized.get("detected_intent") or intent
    normalized["routing_confidence"] = float(normalized.get("routing_confidence") or router_result["confidence_score"])
    normalized["final_response"] = str(final_response)
    normalized["qa_status"] = normalized.get("qa_status") or (
        "REVIEW_REQUIRED" if normalized["routing_confidence"] < 0.75 or automation_context["escalation_flag"] else "APPROVED"
    )
    normalized["compliance_flags"] = normalized.get("compliance_flags") or automation_context.get("compliance_tags", [])
    normalized["recommended_follow_up"] = normalized.get("recommended_follow_up") or "Follow up if the customer replies."
    normalized["data_sources"] = normalized.get("data_sources") or _data_sources_for_intent(intent, pre_sales_context, order_context)
    normalized["assumptions"] = normalized.get("assumptions") or []
    normalized["pre_sales_response"] = _normalize_pre_sales_response(
        normalized.get("pre_sales_response")
        or (
            _pre_sales_response_payload(normalized, pre_sales_context)
            if intent == "pre_sales"
            else None
        )
    )
    if intent == "pre_sales":
        _apply_authoritative_pre_sales_context(normalized["pre_sales_response"], pre_sales_context)
    if intent == "pre_sales":
        normalized["final_response"] = _guard_pre_sales_catalog_pricing(
            normalized["final_response"],
            normalized["pre_sales_response"],
            inputs,
            config_context,
        )
        if normalized["final_response"] != str(final_response):
            flags = list(normalized.get("compliance_flags") or [])
            if "UNVERIFIED_PRODUCT_FACT_REWRITTEN" not in flags:
                flags.append("UNVERIFIED_PRODUCT_FACT_REWRITTEN")
            normalized["compliance_flags"] = flags
        if _is_low_risk_pre_sales_output(normalized, automation_context):
            normalized["qa_status"] = "APPROVED"
            if isinstance(normalized["pre_sales_response"], dict):
                normalized["pre_sales_response"]["requires_human_review"] = False
    normalized["order_response"] = _normalize_order_response(
        normalized.get("order_response") or (order_context if intent == "order_fulfillment" else None)
    )
    if intent == "order_fulfillment":
        _apply_authoritative_order_context(normalized["order_response"], order_context)
        guarded_order_response = _guard_order_tracking_response(
            normalized["final_response"],
            normalized["order_response"],
            inputs,
            config_context,
        )
        if guarded_order_response != normalized["final_response"]:
            normalized["final_response"] = guarded_order_response
            flags = list(normalized.get("compliance_flags") or [])
            if "LOCAL_TRACKING_RECORD_REWRITTEN" not in flags:
                flags.append("LOCAL_TRACKING_RECORD_REWRITTEN")
            normalized["compliance_flags"] = flags
    normalized["support_response"] = _normalize_support_response(
        normalized.get("support_response")
        or (
            _post_sales_support_response_payload(normalized, automation_context)
            if intent == "post_sales_support"
            else None
        )
    )
    if intent == "post_sales_support":
        guarded_support_response = _guard_post_sales_rma_response(
            normalized["final_response"],
            automation_context.get("rma_validation"),
            automation_context.get("logistics_output"),
            inputs,
            config_context,
        )
        if guarded_support_response != normalized["final_response"]:
            normalized["final_response"] = guarded_support_response
            flags = list(normalized.get("compliance_flags") or [])
            if "RMA_POLICY_RESPONSE_REWRITTEN" not in flags:
                flags.append("RMA_POLICY_RESPONSE_REWRITTEN")
            normalized["compliance_flags"] = flags
    normalized["escalation_needed"] = _customer_service_escalation_needed(
        normalized,
        intent,
        automation_context,
    )
    normalized = _strip_customer_service_extra_fields(normalized)
    validated = _validate_customer_service_output(normalized)
    if usage_metrics:
        validated[INTERNAL_USAGE_KEY] = usage_metrics
    return validated


def _pre_sales_response_payload(
    normalized: dict[str, Any],
    pre_sales_context: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(pre_sales_context)
    for key in (
        "product_recommendation",
        "feature_explanation",
        "comparison_summary",
        "next_steps",
        "confidence_level",
        "requires_human_review",
    ):
        value = normalized.get(key)
        if value not in (None, "", []):
            payload[key] = value
    return payload


def _post_sales_support_response_payload(
    normalized: dict[str, Any],
    automation_context: dict[str, Any],
) -> dict[str, Any]:
    stage_notes = _post_sales_stage_notes(normalized)
    return {
        "policy_reference": normalized.get("policy_reference")
        or "Return/RMA policy, warranty eligibility, and logistics guidance checked.",
        "sentiment_analysis": automation_context["sentiment_analysis"],
        "rma_validation": automation_context["rma_validation"],
        "logistics_output": automation_context["logistics_output"],
        "resolution_steps": _string_list(normalized.get("resolution_steps")),
        "internal_notes": stage_notes or normalized.get("internal_notes"),
        "qa_notes": normalized.get("qa_notes"),
    }


def _post_sales_stage_notes(normalized: dict[str, Any]) -> str | None:
    notes = []
    for key, label in (
        ("response_type", "response_type"),
        ("issue_category", "issue_category"),
        ("compensation_offered", "compensation_offered"),
        ("follow_up_required", "follow_up_required"),
    ):
        value = normalized.get(key)
        if value not in (None, "", []):
            notes.append(f"{label}={value}")
    return "; ".join(notes) if notes else None


def _strip_customer_service_extra_fields(normalized: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in normalized.items()
        if key in CUSTOMER_SERVICE_OUTPUT_FIELDS
    }


def _validate_customer_service_output(normalized: dict[str, Any]) -> dict[str, Any]:
    try:
        return CustomerServiceOutput.model_validate(normalized).model_dump()
    except ValidationError as exc:
        field_errors = [
            f"{'.'.join(str(part) for part in error.get('loc', ()))}:{error.get('type')}"
            for error in exc.errors(include_url=False)
        ]
        raise ValueError(
            "CustomerServiceOutput validation failed for fields "
            f"{field_errors}; payload_keys={sorted(normalized)}"
        ) from exc


def _expand_raw_json_result(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("raw")
    if not isinstance(raw, str):
        return result
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return result
    if not isinstance(parsed, dict):
        return result
    return {**parsed, **{key: value for key, value in result.items() if key != "raw"}}


def _apply_authoritative_order_context(
    order_response: dict[str, Any] | None,
    order_context: dict[str, Any],
) -> None:
    if not isinstance(order_response, dict):
        return
    if order_context.get("tracking_lookup_status"):
        order_response["tracking_lookup_status"] = order_context.get("tracking_lookup_status")
    if order_context.get("tracking_lookup_query"):
        order_response["tracking_lookup_query"] = _string_list(order_context.get("tracking_lookup_query"))
    if order_context.get("tracking_record_found") is False:
        order_response["tracking_record_found"] = False
        order_response["local_tracking_record"] = None
    local_record = _local_tracking_record_output(order_context.get("local_tracking_record"))
    if not local_record:
        return
    order_response["tracking_record_found"] = True
    order_response["local_tracking_record"] = local_record
    order_response["knowledge_data_source"] = order_context.get("knowledge_data_source") or "local_order_knowledge_base"
    order_response["order_knowledge_results"] = _knowledge_outputs(order_context.get("order_knowledge_results"))
    order_response["current_status"] = local_record.get("last_status") or order_response.get("current_status")
    order_response["status_explanation"] = _local_tracking_status_explanation(local_record)
    order_response["tracking_info"] = order_response.get("tracking_info") or {
        "carrier": "DTDC",
        "tracking_number": local_record.get("tracking_number"),
        "tracking_url": None,
        "last_update": local_record.get("last_status_date"),
    }
    order_response["delivery_estimate"] = local_record.get("last_status_date") or order_response.get("delivery_estimate")
    order_response["error_message"] = None


def _guard_order_tracking_response(
    final_response: str,
    order_response: dict[str, Any] | None,
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    if not isinstance(order_response, dict):
        return final_response
    lookup_query = {item.upper() for item in _string_list(order_response.get("tracking_lookup_query"))}
    response_identifiers = {match.group(0).upper() for match in TRACKING_IDENTIFIER_RE.finditer(final_response)}
    response_mentions_wrong_identifier = bool(lookup_query and (response_identifiers - lookup_query))
    response_language_mismatch = _response_language_mismatch(final_response, inputs)
    if order_response.get("tracking_record_found") is False:
        if (
            str(order_response.get("tracking_lookup_status") or "") == "not_found"
            and (response_mentions_wrong_identifier or not lookup_query.issubset(response_identifiers) or response_language_mismatch)
        ):
            return _safe_tracking_not_found_response(
                tracking_queries=sorted(lookup_query),
                inputs=inputs,
                config_context=config_context,
            )
        return final_response
    if not order_response.get("tracking_record_found"):
        return final_response
    local_record = order_response.get("local_tracking_record")
    if not isinstance(local_record, dict):
        return final_response
    missing_tracking_number = bool(
        local_record.get("tracking_number") and str(local_record["tracking_number"]) not in final_response
    )
    missing_reference_number = bool(
        local_record.get("reference_number") and str(local_record["reference_number"]) not in final_response
    )
    missing_status = bool(local_record.get("last_status") and str(local_record["last_status"]) not in final_response)
    frames_as_missing_order = bool(
        re.search(
            r"\b(?:could not|couldn't|cannot|can't|unable to|not able to)\s+find\s+(?:an?\s+)?order\b|"
            r"\border\s+(?:was\s+)?not\s+found\b|"
            r"\bno\s+order\s+(?:was\s+)?(?:found|associated)\b",
            final_response,
            flags=re.IGNORECASE,
        )
    )
    if not (
        missing_tracking_number
        or missing_reference_number
        or missing_status
        or frames_as_missing_order
        or response_mentions_wrong_identifier
        or response_language_mismatch
    ):
        return final_response
    return _safe_local_tracking_response(local_record, inputs, config_context)


def _guard_post_sales_rma_response(
    final_response: str,
    rma_validation: dict[str, Any] | None,
    logistics_output: dict[str, Any] | None,
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    if not isinstance(rma_validation, dict):
        if logistics_output is None and _contains_unsupported_rma_claim(final_response):
            return _safe_rma_pending_details_response(inputs, config_context)
        return final_response

    if rma_validation.get("eligible_for_return") is False:
        return _safe_rma_denial_response(rma_validation, inputs, config_context)

    if logistics_output is None and _contains_unsupported_rma_claim(final_response):
        return _safe_rma_pending_details_response(inputs, config_context)

    if RETURN_LABEL_URL_RE.search(final_response):
        return _safe_rma_pending_details_response(inputs, config_context)

    return final_response


def _contains_unsupported_rma_claim(response: str) -> bool:
    return bool(RETURN_LABEL_URL_RE.search(response) or RMA_UNSUPPORTED_CLAIM_RE.search(response))


def _safe_rma_denial_response(
    rma_validation: dict[str, Any],
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    reason = str(
        rma_validation.get("eligibility_reason")
        or "The item is not eligible under our return policy."
    )
    response = (
        "Hello,\n\n"
        "Thank you for reaching out. I checked our return policy for this request.\n\n"
        f"We are not able to approve a return or refund for this item based on the details provided. {reason} "
        "Our policy requires returned items to be unused and unworn, and hygiene-sensitive or intimate items "
        "cannot be returned after they have been opened or worn unless they are faulty, damaged, or incorrect.\n\n"
        "Because this return is not eligible, I cannot provide a return label or refund approval. If the item is "
        "faulty, damaged, or not the item you ordered, please reply with the order number and clear photos so a "
        "support lead can review it.\n\n"
        "Best regards,\n"
        "Customer Service Team"
    )
    return _localize_rma_safe_response(
        response=response,
        inputs=inputs,
        structured_facts={"rma_validation": rma_validation, "response_type": "rma_denial"},
        config_context=config_context,
    )


def _safe_rma_pending_details_response(
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    response = (
        "Hello,\n\n"
        "Thank you for reaching out. We need to verify the order details and return eligibility before any return "
        "label, carrier details, or refund approval can be provided.\n\n"
        "Please reply with your order number, delivery date, item condition, and the reason for the return. If the "
        "item is faulty, damaged, or incorrect, please also include clear photos.\n\n"
        "Best regards,\n"
        "Customer Service Team"
    )
    return _localize_rma_safe_response(
        response=response,
        inputs=inputs,
        structured_facts={"response_type": "rma_pending_details"},
        config_context=config_context,
    )


def _localize_rma_safe_response(
    *,
    response: str,
    inputs: dict[str, Any],
    structured_facts: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    rewritten = _rewrite_response_language(
        response=response,
        inputs=inputs,
        structured_facts=structured_facts,
        config_context=config_context,
    )
    return rewritten or response


def _response_language_mismatch(response: str, inputs: dict[str, Any]) -> bool:
    target_language = str(inputs.get("language_plan") or "").lower()
    if not target_language or target_language == "english":
        return False
    detected = LanguageDetector.detect(response)
    expected = _normalize_language_code(str(inputs.get("detected_language") or ""))
    return bool(expected and detected != expected)


def _safe_local_tracking_response(
    record: dict[str, Any],
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    lines = [
        "Hello,",
        "",
        "Thank you for reaching out. I found the tracking record in our local tracking document.",
        "",
        "Here are the details I found:",
        *_local_tracking_fact_lines(record),
    ]
    history_lines = _local_tracking_history_lines(record)
    if history_lines:
        lines.extend(["", "Tracking history:", *history_lines])
    lines.extend(
        [
            "",
            "This tracking record is available locally, though it may not be linked to an internal order record. "
            "If you need proof of delivery or want us to investigate a delivery dispute, please reply and we can escalate it to operations.",
            "",
            "Best regards,",
            "Customer Service Team",
        ]
    )
    response = "\n".join(lines)
    return _localize_order_safe_response(
        response=response,
        inputs=inputs,
        structured_facts={"tracking_record": record, "response_type": "tracking_found"},
        config_context=config_context,
        fallback_factory=lambda language: _deterministic_tracking_found_response(record, language),
    )


def _safe_tracking_not_found_response(
    *,
    tracking_queries: list[str],
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    query_text = ", ".join(tracking_queries) or str(inputs.get("order_id_if_provided") or "the provided tracking number")
    response = (
        "Hello,\n\n"
        f"Thank you for reaching out. I could not find a local tracking record that exactly matches {query_text}.\n\n"
        "Please double-check the tracking number or share the reference number, order ID, or purchase email, "
        "and we will check again.\n\n"
        "Best regards,\n"
        "Customer Service Team"
    )
    return _localize_order_safe_response(
        response=response,
        inputs=inputs,
        structured_facts={"tracking_queries": tracking_queries, "response_type": "tracking_not_found"},
        config_context=config_context,
        fallback_factory=lambda language: _deterministic_tracking_not_found_response(tracking_queries, language),
    )


def _localize_order_safe_response(
    *,
    response: str,
    inputs: dict[str, Any],
    structured_facts: dict[str, Any],
    config_context: dict[str, Any],
    fallback_factory: Any,
) -> str:
    rewritten = _rewrite_response_language(
        response=response,
        inputs=inputs,
        structured_facts=structured_facts,
        config_context=config_context,
    )
    if rewritten:
        return rewritten
    language = str(inputs.get("language_plan") or LanguageDetector.get_crewai_language_plan(str(inputs.get("detected_language") or "en")))
    return str(fallback_factory(language))


def _deterministic_tracking_found_response(record: dict[str, Any], language: str) -> str:
    if str(language).lower() != "japanese":
        return _deterministic_tracking_found_response_en(record)
    fact_lines = _local_tracking_fact_lines(record)
    history_lines = _local_tracking_history_lines(record)
    translated_labels = {
        "Tracking No.": "追跡番号",
        "Reference No.": "参照番号",
        "Last Status": "最新ステータス",
        "Last Status Date": "最新ステータス日",
        "Booking Date": "受付日",
        "Origin": "発送元",
        "Destination": "宛先",
        "Pieces": "個数",
        "Service Type": "サービス種別",
        "Package contents": "荷物内容",
        "Receiver Name": "受取人名",
    }
    localized_facts = []
    for line in fact_lines:
        label, _, value = line.removeprefix("- ").partition(": ")
        localized_facts.append(f"- {translated_labels.get(label, label)}: {value}")
    lines = [
        "こんにちは。",
        "",
        "お問い合わせありがとうございます。社内の追跡資料で一致する追跡記録を確認しました。",
        "",
        "確認できた情報は以下の通りです。",
        *localized_facts,
    ]
    if history_lines:
        lines.extend(["", "追跡履歴:", *history_lines])
    lines.extend(
        [
            "",
            "この追跡記録は社内資料で確認できていますが、社内注文レコードとは紐づいていない場合があります。配達証明や調査が必要な場合は、このメールにご返信ください。",
            "",
            "よろしくお願いいたします。",
            "カスタマーサービスチーム",
        ]
    )
    return "\n".join(lines)


def _deterministic_tracking_found_response_en(record: dict[str, Any]) -> str:
    lines = [
        "Hello,",
        "",
        "Thank you for reaching out. I found the tracking record in our local tracking document.",
        "",
        "Here are the details I found:",
        *_local_tracking_fact_lines(record),
    ]
    history_lines = _local_tracking_history_lines(record)
    if history_lines:
        lines.extend(["", "Tracking history:", *history_lines])
    lines.extend(
        [
            "",
            "This tracking record is available locally, though it may not be linked to an internal order record. If you need proof of delivery or want us to investigate a delivery dispute, please reply and we can escalate it to operations.",
            "",
            "Best regards,",
            "Customer Service Team",
        ]
    )
    return "\n".join(lines)


def _deterministic_tracking_not_found_response(tracking_queries: list[str], language: str) -> str:
    query_text = ", ".join(tracking_queries) or "the provided tracking number"
    if str(language).lower() == "japanese":
        return (
            "こんにちは。\n\n"
            f"お問い合わせありがとうございます。追跡番号 {query_text} と完全に一致する社内追跡記録は見つかりませんでした。\n\n"
            "追跡番号をご確認いただくか、参照番号、注文番号、または購入時のメールアドレスをお知らせください。再度確認いたします。\n\n"
            "よろしくお願いいたします。\n"
            "カスタマーサービスチーム"
        )
    return (
        "Hello,\n\n"
        f"Thank you for reaching out. I could not find a local tracking record that exactly matches {query_text}.\n\n"
        "Please double-check the tracking number or share the reference number, order ID, or purchase email, and we will check again.\n\n"
        "Best regards,\n"
        "Customer Service Team"
    )


def _local_tracking_fact_lines(record: dict[str, Any]) -> list[str]:
    fields = [
        ("Tracking No.", record.get("tracking_number")),
        ("Reference No.", record.get("reference_number")),
        ("Last Status", record.get("last_status")),
        ("Last Status Date", record.get("last_status_date")),
        ("Booking Date", record.get("booking_date")),
        ("Origin", record.get("origin")),
        ("Destination", record.get("destination")),
        ("Pieces", record.get("pieces")),
        ("Service Type", record.get("service_type")),
        ("Package contents", record.get("package_contents")),
        ("Receiver Name", record.get("receiver_name")),
    ]
    return [f"- {label}: {value}" for label, value in fields if value not in {None, ""}]


def _local_tracking_history_lines(record: dict[str, Any]) -> list[str]:
    history = record.get("tracking_history")
    if not isinstance(history, list):
        return []
    lines: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        date_time = " ".join(str(value) for value in [item.get("date"), item.get("time")] if value)
        location = f" - {item['location']}" if item.get("location") else ""
        lines.append(f"- {date_time}: {item.get('activity')}{location}".strip())
    return lines


def _customer_service_escalation_needed(
    normalized: dict[str, Any],
    intent: str,
    automation_context: dict[str, Any],
) -> bool:
    if automation_context["escalation_flag"]:
        return True
    qa_status = str(normalized.get("qa_status") or "").upper()
    confidence = float(normalized.get("routing_confidence") or 0)
    compliance_flags = {str(flag).upper() for flag in normalized.get("compliance_flags") or []}
    hard_flags = {
        "LOW_ROUTING_CONFIDENCE",
        "HUMAN_HANDOFF",
        "POLICY_GAP",
        "BILLING_DISPUTE",
        "UNSAFE_RESPONSE",
        "VIP_REVIEW",
        "NEGATIVE_SENTIMENT",
    }
    if qa_status == "REJECTED" or hard_flags.intersection(compliance_flags):
        return True
    if (
        intent == "pre_sales"
        and _is_low_risk_pre_sales_output(normalized, automation_context)
    ):
        return False
    return bool(normalized.get("escalation_needed"))


def _is_low_risk_pre_sales_output(
    normalized: dict[str, Any],
    automation_context: dict[str, Any],
) -> bool:
    if str(normalized.get("detected_intent") or "").lower() != "pre_sales":
        return False
    if automation_context["escalation_flag"]:
        return False
    if float(normalized.get("routing_confidence") or 0) < 0.75 or not normalized.get("final_response"):
        return False
    compliance_flags = {str(flag).upper() for flag in normalized.get("compliance_flags") or []}
    hard_flags = {
        "LOW_ROUTING_CONFIDENCE",
        "HUMAN_HANDOFF",
        "POLICY_GAP",
        "BILLING_DISPUTE",
        "UNSAFE_RESPONSE",
        "VIP_REVIEW",
        "NEGATIVE_SENTIMENT",
    }
    if hard_flags.intersection(compliance_flags):
        return False
    pre_sales = normalized.get("pre_sales_response")
    if not isinstance(pre_sales, dict):
        return False
    return True


def _guard_pre_sales_catalog_pricing(
    final_response: str,
    pre_sales_response: dict[str, Any] | None,
    inputs: dict[str, Any],
    config_context: dict[str, Any],
) -> str:
    if not pre_sales_response:
        return final_response
    offer = pre_sales_response.get("catalog_product_offer")
    if not isinstance(offer, dict) or offer.get("status") != "found" or not offer.get("unit_price"):
        return final_response

    unit_price = str(offer["unit_price"])
    quoted_prices = {match.replace(" ", "") for match in PRICE_VALUE_RE.findall(final_response)}
    has_unverified_price = bool(quoted_prices - {unit_price})
    missing_verified_price = unit_price not in quoted_prices
    missing_catalog_facts = any(
        str(value) not in final_response
        for value in (
            offer.get("carton_quantity"),
            offer.get("carton_size"),
            offer.get("carton_weight"),
            offer.get("single_product_size"),
            offer.get("single_product_weight"),
        )
        if value
    )
    looks_like_discount_table = bool(re.search(r"\bbuy\s+\d+\b|\b\d+\s*:\s*\$", final_response, flags=re.I))
    has_unverified_sku = bool(re.search(r"\bSKU\s*:?\s*[A-Z0-9][A-Z0-9_-]{2,}\b|\b[A-Z]{2,}-[A-Z0-9_-]{3,}\b", final_response))
    has_variant_section = bool(re.search(r"\bvariants?\s+(?:available|options?)\b|\bbasic\s+headset\b|\bpro\s+headset\b", final_response, flags=re.I))
    has_unverified_feature = bool(re.search(r"\bnoise isolation\b|\bnoise cancellation\b", final_response, flags=re.I))
    understates_catalog_specs = bool(re.search(r"detailed specifications are not available|detailed feature specifications are not listed", final_response, flags=re.I))
    if not (
        has_unverified_price
        or missing_verified_price
        or missing_catalog_facts
        or looks_like_discount_table
        or has_unverified_sku
        or has_variant_section
        or has_unverified_feature
        or understates_catalog_specs
    ):
        return final_response

    facts_payload = _catalog_facts_payload(offer, inputs)
    safe_draft = _safe_catalog_response_draft(
        channel=str(inputs.get("channel") or "email").lower(),
        facts=facts_payload,
    )
    rewritten = _rewrite_response_language(
        response=safe_draft,
        inputs=inputs,
        structured_facts=facts_payload,
        config_context=config_context,
    )
    return rewritten or safe_draft


def _catalog_facts_payload(offer: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_name": _catalog_display_name(offer, inputs),
        "unit_price": offer.get("unit_price"),
        "carton_quantity": offer.get("carton_quantity"),
        "carton_size": offer.get("carton_size"),
        "carton_weight": offer.get("carton_weight"),
        "single_product_size": offer.get("single_product_size"),
        "single_product_weight": offer.get("single_product_weight"),
        "missing_specs_note": (
            "The catalog lists packaging/specification facts but does not list battery life, audio performance, "
            "Bluetooth version, variants, or SKU options for this matched item."
        ),
        "discount_policy": "Discount requests require sales approval before quoting any reduced price.",
    }


def _catalog_display_name(offer: dict[str, Any], inputs: dict[str, Any]) -> str:
    product_category = _clean_catalog_product_name(str(inputs.get("product_category") or ""))
    inquiry_name = _clean_catalog_product_name(
        str(_product_name_from_inquiry(str(inputs.get("inquiry_text") or inputs.get("inquiry") or "")) or "")
    )
    offer_name = _clean_catalog_product_name(str(offer.get("product_name") or ""))
    if offer_name and product_category and _is_more_specific_catalog_name(offer_name, product_category):
        return offer_name
    for cleaned in (product_category, inquiry_name, offer_name):
        if cleaned:
            return cleaned
    return "the matched product"


def _product_name_from_inquiry(inquiry_text: str) -> str | None:
    match = re.search(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*){1,5}\b", inquiry_text)
    return match.group(0) if match else None


def _clean_catalog_product_name(value: str) -> str:
    cleaned = re.sub(r"(?:1|one)\s+carton\s+contains?\s*:?\s*[0-9]+\s*(?:pcs|pcs\.|pieces)?", "", value, flags=re.I)
    cleaned = re.sub(r"please\s+contact\s+me\s+for\s+a\s+discount\.?", "", cleaned, flags=re.I)
    cleaned = PRICE_VALUE_RE.sub("", cleaned)
    cleaned = re.split(r"\bbox\s+gauge\b|\bweight\b|\bsingle\s+product\b", cleaned, maxsplit=1, flags=re.I)[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" /,.-")
    return cleaned


def _is_more_specific_catalog_name(candidate: str, baseline: str) -> bool:
    if re.search(r"\d", baseline):
        return False
    baseline_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", baseline)}
    candidate_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", candidate)}
    extra_tokens = candidate_tokens - baseline_tokens
    return bool(baseline_tokens and baseline_tokens.issubset(candidate_tokens) and any(re.search(r"\d", token) for token in extra_tokens))


def _safe_catalog_response_draft(*, channel: str, facts: dict[str, Any]) -> str:
    fact_lines = [
        f"current catalog unit price: {facts['unit_price']}",
        *(f"{label}: {facts[key]}" for key, label in (
            ("carton_quantity", "carton quantity"),
            ("carton_size", "carton size"),
            ("carton_weight", "carton weight"),
            ("single_product_size", "single product size"),
            ("single_product_weight", "single product weight"),
        ) if facts.get(key)),
    ]
    details_note = str(facts["missing_specs_note"])
    if channel in {"whatsapp", "webchat", "social"}:
        return (
            f"Thanks for your interest. I found a catalog match: {facts['product_name']}. "
            f"The verified catalog details are: {'; '.join(fact_lines)}. {details_note} "
            "You can reply to confirm this is the item you want, and sales can help with the purchase "
            "or review any discount request."
        )
    return (
        f"Hello,\n\n"
        f"Thank you for your interest. I found a catalog match: {facts['product_name']}.\n\n"
        "The verified catalog details I have are:\n"
        + "\n".join(f"- {fact}" for fact in fact_lines)
        + f"\n\n{details_note} You can reply to confirm this is the item you want, and our sales team can help with the purchase"
        " or review any discount request.\n\nBest,\nCustomer Service Team"
    )


def _rewrite_response_language(
    *,
    response: str,
    inputs: dict[str, Any],
    structured_facts: dict[str, Any],
    config_context: dict[str, Any],
) -> str | None:
    target_language = str(inputs.get("language_plan") or LanguageDetector.get_crewai_language_plan(str(inputs.get("detected_language") or "en")))
    rewriter = config_context.get("response_language_rewriter")
    if callable(rewriter):
        rewritten = rewriter(
            response=response,
            target_language=target_language,
            detected_language=str(inputs.get("detected_language") or "en"),
            channel=str(inputs.get("channel") or "email"),
            inquiry_text=str(inputs.get("inquiry_text") or inputs.get("inquiry") or ""),
            structured_facts=structured_facts,
        )
        return str(rewritten).strip() if rewritten else None

    api_key = llm_api_key(config_context)
    if not api_key:
        return None
    try:
        payload = {
            "model": llm_model_name(config_context),
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the customer-service response in the target language only. "
                        "Preserve product model names, prices, quantities, sizes, and weights exactly. "
                        "Do not add facts, discounts, SKUs, variants, or claims not present in structured_facts. "
                        "Return JSON with a single key: response."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "target_language": target_language,
                            "detected_language": inputs.get("detected_language"),
                            "channel": inputs.get("channel"),
                            "customer_inquiry": inputs.get("inquiry_text") or inputs.get("inquiry"),
                            "structured_facts": structured_facts,
                            "response_to_rewrite": response,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            **llm_reasoning_compat_params(config_context),
        }
        llm_response = httpx.post(
            llm_chat_completions_url(config_context),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        llm_response.raise_for_status()
        content = llm_response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        rewritten = parsed.get("response")
        return str(rewritten).strip() if rewritten else None
    except Exception:
        return None


def _normalize_language_code(language: str | None) -> str:
    if not language:
        return "en"
    normalized = str(language).strip().lower().replace("_", "-")
    language_names = {value.lower(): key for key, value in LanguageDetector.SUPPORTED_MAP.items()}
    if normalized in language_names:
        return language_names[normalized]
    return normalized.split("-", 1)[0] or "en"


def _data_sources_for_intent(intent: str, pre_sales_context: dict[str, Any], order_context: dict[str, Any]) -> list[str]:
    if intent == "pre_sales":
        sources = [str(pre_sales_context.get("data_source") or "product_knowledge")]
        if pre_sales_context.get("knowledge_data_source"):
            sources.append(str(pre_sales_context["knowledge_data_source"]))
        return list(dict.fromkeys(sources))
    if intent == "order_fulfillment":
        return [str(order_context.get("data_source") or "order_tracking")]
    return ["support_knowledge_base", "support_automation_context"]
