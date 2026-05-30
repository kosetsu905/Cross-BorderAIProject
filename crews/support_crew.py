from pathlib import Path
from typing import Any
import json
import re

import httpx
import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

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
    search_knowledge_base,
)
from utils.crew_result import serialize_crew_result
from utils.usage_tracking import INTERNAL_USAGE_KEY
from utils.workflow_progress import attach_task_progress
from utils.project_intelligence import augment_agents_config

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "support"
PRICE_VALUE_RE = re.compile(r"\$\s*\d+(?:\.\d{1,2})?")


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


class OrderResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_found: bool | None = None
    order_verified: bool | None = None
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


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_support_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [
        SupportKnowledgeSearchTool(
            knowledge_dir=config_context.get("support_knowledge_dir")
            or str(BASE_DIR / "docs" / "knowledge_base")
        ),
        ScrapeWebsiteTool(),
    ]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


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

    pre_sales_agent = Agent(
        config=agents_config["pre_sales_specialist"],
        tools=[pre_sales_tool, *_build_search_tools(config_context)],
        allow_delegation=False,
    )
    order_agent = Agent(
        config=agents_config["order_fulfillment_specialist"],
        tools=[OrderTrackingTool()],
        allow_delegation=False,
    )
    support_agent = Agent(
        config=agents_config["senior_support_agent"],
        tools=[RMAAutomationTool(), *_build_support_tools(config_context)],
        allow_delegation=True,
    )
    qa_agent = Agent(
        config=agents_config["support_qa_specialist"],
        allow_delegation=False,
    )

    task_by_intent = {
        "pre_sales": ("pre_sales_consultation", pre_sales_agent),
        "order_fulfillment": ("order_status_handling", order_agent),
        "post_sales_support": ("inquiry_resolution", support_agent),
    }
    stage_task_name, stage_agent = task_by_intent[router_result["detected_intent"]]
    stage_task = Task(
        config=tasks_config[stage_task_name],
        agent=stage_agent,
    )
    qa_task = Task(
        config=tasks_config["quality_assurance_review"],
        agent=qa_agent,
        context=[stage_task],
        output_pydantic=CustomerServiceOutput,
    )
    tasks = [stage_task, qa_task]
    attach_task_progress(config_context, "support", tasks, [stage_task_name, "quality_assurance_review"])

    support_crew = Crew(
        agents=[stage_agent, qa_agent],
        tasks=tasks,
        verbose=False,
        memory=_memory_enabled(config_context),
    )

    result = _serialize_crew_result(support_crew.kickoff(inputs=crew_inputs))
    return _attach_customer_service_context(
        result=result,
        inputs=normalized_inputs,
        automation_context=automation_context,
        router_result=router_result,
        pre_sales_context=pre_sales_context,
        order_context=order_context,
        config_context=config_context,
    )


def _build_search_tools(config_context: dict[str, Any]) -> list[Any]:
    tools: list[Any] = [ScrapeWebsiteTool()]
    if config_context.get("serper_api_key"):
        tools.insert(0, SerperDevTool())
    return tools


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
    normalized = dict(result)
    usage_metrics = normalized.pop(INTERNAL_USAGE_KEY, None)
    intent = router_result["detected_intent"]
    final_response = (
        normalized.get("final_response")
        or normalized.get("drafted_response")
        or normalized.get("response")
        or normalized.get("body")
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
        normalized.get("pre_sales_response") or (pre_sales_context if intent == "pre_sales" else None)
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
    normalized["support_response"] = _normalize_support_response(
        normalized.get("support_response")
        or (
            {
                "policy_reference": "Return/RMA policy, warranty eligibility, and logistics guidance checked.",
                "sentiment_analysis": automation_context["sentiment_analysis"],
                "rma_validation": automation_context["rma_validation"],
                "logistics_output": automation_context["logistics_output"],
            }
            if intent == "post_sales_support"
            else None
        )
    )
    normalized["escalation_needed"] = _customer_service_escalation_needed(
        normalized,
        intent,
        automation_context,
    )
    validated = CustomerServiceOutput.model_validate(normalized).model_dump()
    if usage_metrics:
        validated[INTERNAL_USAGE_KEY] = usage_metrics
    return validated


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

    api_key = config_context.get("openai_api_key")
    if not api_key:
        return None
    try:
        payload = {
            "model": config_context.get("openai_model_name") or "gpt-4o-mini",
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
        }
        llm_response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
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
