from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.support_automation_tools import (
    LogisticsIntegrationOutput,
    RMAAutomationTool,
    RMAValidationResult,
    SentimentIntentAnalysis,
    SentimentIntentGradingTool,
    analyze_sentiment_intent,
    process_rma_request,
)
from tools.custom.support_rag_tools import SupportKnowledgeSearchTool
from utils.crew_result import serialize_crew_result
from utils.workflow_progress import attach_task_progress

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "support"


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
    return normalized


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
    if not normalized.get("internal_notes"):
        handoff = "Human handoff required." if context["escalation_flag"] else "No human handoff required."
        normalized["internal_notes"] = f"{handoff} Automation context generated before CrewAI response drafting."
    return normalized


def _memory_enabled(config_context: dict[str, Any]) -> bool:
    return bool(config_context.get("crewai_memory_enabled"))


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    return serialize_crew_result(result)


def run_support_crew(inputs: dict[str, Any], config_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    config_context = config_context or {}
    normalized_inputs = _normalize_inputs(inputs)
    automation_context = _build_automation_context(normalized_inputs)
    crew_inputs = {
        **normalized_inputs,
        "sentiment_analysis": automation_context["sentiment_analysis"],
        "rma_validation": automation_context["rma_validation"],
        "logistics_output": automation_context["logistics_output"],
        "escalation_flag": automation_context["escalation_flag"],
        "compliance_tags": automation_context["compliance_tags"],
        "language_detected": automation_context["sentiment_analysis"]["language_detected"],
    }

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    triage_agent = Agent(
        config=agents_config["support_triage_specialist"],
        tools=[SentimentIntentGradingTool()],
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

    triage_task = Task(
        config=tasks_config["triage_and_routing"],
        agent=triage_agent,
    )
    resolution_task = Task(
        config=tasks_config["inquiry_resolution"],
        agent=support_agent,
        context=[triage_task],
    )
    qa_task = Task(
        config=tasks_config["quality_assurance_review"],
        agent=qa_agent,
        context=[triage_task, resolution_task],
        output_pydantic=SupportTicketOutput,
    )
    tasks = [triage_task, resolution_task, qa_task]
    attach_task_progress(config_context, "support", tasks, list(tasks_config.keys()))

    support_crew = Crew(
        agents=[triage_agent, support_agent, qa_agent],
        tasks=tasks,
        verbose=False,
        memory=_memory_enabled(config_context),
    )

    result = _serialize_crew_result(support_crew.kickoff(inputs=crew_inputs))
    return _attach_automation_context(result, automation_context, normalized_inputs)
