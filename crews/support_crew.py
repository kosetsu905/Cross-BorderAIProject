from pathlib import Path
from typing import Any

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
from tools.custom.support_rag_tools import SupportKnowledgeSearchTool
from utils.crew_result import serialize_crew_result
from utils.workflow_progress import attach_task_progress

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "support"


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
    customer_context: dict[str, str] = Field(..., description="Customer name, tier, language, and channel")
    detected_intent: str = Field(..., description="pre_sales | order_fulfillment | post_sales_support")
    routing_confidence: float = Field(..., ge=0, le=1)
    pre_sales_response: dict[str, Any] | None = None
    order_response: dict[str, Any] | None = None
    support_response: dict[str, Any] | None = None
    final_response: str = Field(..., description="Polished, channel-adapted customer response")
    qa_status: str = Field(..., description="APPROVED | REVIEW_REQUIRED | REJECTED")
    compliance_flags: list[str] = Field(default_factory=list)
    recommended_follow_up: str = Field(..., description="When/how to follow up if needed")
    escalation_needed: bool = Field(..., description="True if a human agent should take over")
    data_sources: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


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
    router_tool = IntentRouterTool()
    router_result = router_tool._run(
        inquiry_text=normalized_inputs["inquiry_text"],
        has_order_id=bool(normalized_inputs.get("order_id") and normalized_inputs.get("order_id") != "ORDER-NOT-PROVIDED"),
        customer_tier=normalized_inputs.get("customer_tier", "STANDARD"),
    )
    if router_result["requires_human_review"]:
        return _handoff_customer_service_output(normalized_inputs, automation_context, router_result)

    pre_sales_tool = _build_pre_sales_tool(config_context)
    pre_sales_context = pre_sales_tool._run(
        product_category=normalized_inputs["product_category"],
        inquiry_keywords=str(normalized_inputs["inquiry_text"]).split(),
        region=normalized_inputs["region"],
        language=normalized_inputs["detected_language"],
    )
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


def _customer_context(inputs: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(inputs.get("customer") or "Customer"),
        "tier": str(inputs.get("customer_tier") or "STANDARD"),
        "language": str(inputs.get("language_plan") or "English"),
        "channel": str(inputs.get("channel") or "email"),
    }


def _handoff_customer_service_output(
    inputs: dict[str, Any],
    automation_context: dict[str, Any],
    router_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": inputs.get("session_id", "unknown"),
        "customer_context": _customer_context(inputs),
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
        "assumptions": ["Routing confidence below automatic handling threshold."],
    }


def _attach_customer_service_context(
    *,
    result: dict[str, Any],
    inputs: dict[str, Any],
    automation_context: dict[str, Any],
    router_result: dict[str, Any],
    pre_sales_context: dict[str, Any],
    order_context: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(result)
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
    normalized["detected_intent"] = normalized.get("detected_intent") or intent
    normalized["routing_confidence"] = float(normalized.get("routing_confidence") or router_result["confidence_score"])
    normalized["final_response"] = str(final_response)
    normalized["qa_status"] = normalized.get("qa_status") or (
        "REVIEW_REQUIRED" if normalized["routing_confidence"] < 0.75 or automation_context["escalation_flag"] else "APPROVED"
    )
    normalized["compliance_flags"] = normalized.get("compliance_flags") or automation_context.get("compliance_tags", [])
    normalized["recommended_follow_up"] = normalized.get("recommended_follow_up") or "Follow up if the customer replies."
    normalized["escalation_needed"] = bool(
        normalized.get("escalation_needed")
        or automation_context["escalation_flag"]
        or normalized["qa_status"] in {"REVIEW_REQUIRED", "REJECTED"}
    )
    normalized["data_sources"] = normalized.get("data_sources") or _data_sources_for_intent(intent, pre_sales_context, order_context)
    normalized["assumptions"] = normalized.get("assumptions") or []
    normalized.setdefault("pre_sales_response", pre_sales_context if intent == "pre_sales" else None)
    normalized.setdefault("order_response", order_context if intent == "order_fulfillment" else None)
    normalized.setdefault(
        "support_response",
        {
            "policy_reference": "Return/RMA policy, warranty eligibility, and logistics guidance checked.",
            "sentiment_analysis": automation_context["sentiment_analysis"],
            "rma_validation": automation_context["rma_validation"],
            "logistics_output": automation_context["logistics_output"],
        } if intent == "post_sales_support" else None,
    )
    return CustomerServiceOutput.model_validate(normalized).model_dump()


def _data_sources_for_intent(intent: str, pre_sales_context: dict[str, Any], order_context: dict[str, Any]) -> list[str]:
    if intent == "pre_sales":
        return [str(pre_sales_context.get("data_source") or "product_knowledge")]
    if intent == "order_fulfillment":
        return [str(order_context.get("data_source") or "order_tracking")]
    return ["support_knowledge_base", "support_automation_context"]
