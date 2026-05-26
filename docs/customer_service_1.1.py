# customer_support_agent.py
# 📍 Path: /workspace/ecommerce_ai_agents/workflows/customer_support_workflow.py

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool, SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 📦 1. Pydantic Structured Output Models (v2 Compatible)
# =============================================================================
class SentimentIntentAnalysis(BaseModel):
    """情绪与意图分级分析结果"""
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="VADER-like sentiment score: -1.0 (very negative) to +1.0 (very positive)")
    sentiment_label: Literal["ANGRY", "FRUSTRATED", "NEUTRAL", "SATISFIED", "DELIGHTED"]
    intent_category: Literal["RMA_REQUEST", "SHIPPING_INQUIRY", "PRODUCT_USAGE", "BILLING_ISSUE", "VIP_ESCALATION", "GENERAL"]
    customer_tier: Literal["VIP", "PREMIUM", "STANDARD", "NEW"]
    urgency_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    language_detected: str = Field(..., description="ISO 639-1 language code detected from inquiry")
    requires_human_handoff: bool = Field(..., description="True if sentiment < -0.6 AND customer_tier in [VIP, PREMIUM]")

class RMAValidationResult(BaseModel):
    """退换货政策校验结果"""
    order_id: str
    eligible_for_return: bool
    eligibility_reason: str
    return_window_days: int
    days_since_delivery: int
    item_condition_accepted: List[str]
    restocking_fee_pct: Optional[float] = Field(None, ge=0, le=100)
    return_shipping_responsibility: Literal["CUSTOMER", "BRAND", "SHARED"]

class LogisticsIntegrationOutput(BaseModel):
    """物流对接与面单生成结果"""
    carrier: Literal["ShipStation", "EasyPost", "DHL", "FedEx", "Local_Post"]
    prepaid_label_url: str = Field(..., description="Simulated URL to downloadable prepaid return label PDF")
    tracking_number: str
    estimated_refund_days: int
    warehouse_inbound_notification: str = Field(..., description="Confirmation that WMS received inbound ASN")
    return_instructions_localized: str = Field(..., description="Return instructions in customer's detected language")

class ResponseTemplate(BaseModel):
    """多语言回复模板"""
    language: str
    subject: str
    body: str
    tone: Literal["EMPATHETIC_APOLOGETIC", "PROFESSIONAL_HELPFUL", "CONCISE_INFORMATIVE"]
    includes_compensation: bool
    compensation_details: Optional[Dict[str, Any]] = None

class SupportTicketResolution(BaseModel):
    """完整工单处理输出"""
    ticket_id: str
    customer_email: str
    sentiment_analysis: SentimentIntentAnalysis
    rma_validation: Optional[RMAValidationResult] = None
    logistics_output: Optional[LogisticsIntegrationOutput] = None
    generated_response: ResponseTemplate
    escalation_flag: bool
    internal_notes: str
    recommended_follow_up_hours: int
    compliance_tags: List[str]  # e.g., ["GDPR_COMPLIANT", "CCPA_OPT_OUT_AVAILABLE"]

# =============================================================================
# 🛠️ 2. Advanced Custom Tools
# =============================================================================
class SentimentIntentGradingTool(BaseTool):
    name: str = "Sentiment & Intent Grading Engine"
    description: str = "Analyzes customer inquiry text for sentiment score, intent classification, customer tier detection, and handoff recommendation."

    def _run(self, inquiry_text: str, customer_email: str, order_history: Optional[Dict] = None) -> dict:
        logger.info(f"Running sentiment & intent analysis for inquiry from {customer_email}...")
        
        # 🔧 Production Hook: Replace with fine-tuned multilingual sentiment model (e.g., cardiffnlp/twitter-xlm-roberta)
        # + intent classifier trained on support ticket taxonomy
        
        # Simple heuristic simulation (expand with real NLP in production)
        negative_keywords = ["angry", "disappointed", "refund", "complaint", "terrible", "waste", "never again", "unacceptable"]
        vip_domains = ["@enterprise.com", "@vip-client.com", "@corporate.net"]
        
        # Sentiment scoring (mock VADER-like)
        text_lower = inquiry_text.lower()
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        sentiment_score = max(-1.0, min(1.0, 0.5 - (neg_count * 0.25)))  # Simplified
        
        # Intent classification
        if any(kw in text_lower for kw in ["return", "refund", "exchange", "RMA"]):
            intent = "RMA_REQUEST"
        elif any(kw in text_lower for kw in ["shipping", "delivery", "track", "where is"]):
            intent = "SHIPPING_INQUIRY"
        elif any(kw in text_lower for kw in ["how to", "setup", "install", "use", "manual"]):
            intent = "PRODUCT_USAGE"
        elif any(kw in text_lower for kw in ["charge", "billing", "invoice", "payment"]):
            intent = "BILLING_ISSUE"
        else:
            intent = "GENERAL"
        
        # Customer tier detection
        if any(domain in customer_email for domain in vip_domains) or (order_history and order_history.get("lifetime_value", 0) > 5000):
            tier = "VIP"
        elif order_history and order_history.get("order_count", 0) >= 5:
            tier = "PREMIUM"
        elif order_history and order_history.get("order_count", 0) >= 1:
            tier = "STANDARD"
        else:
            tier = "NEW"
        
        # Language detection (mock)
        lang_map = {
            r'[\u3040-\u309f\u30a0-\u30ff]': 'ja',
            r'[\u4e00-\u9fff]': 'zh',
            r'ñ|á|é|í|ó|ú|¿|¡': 'es',
            r'ä|ö|ü|ß': 'de',
            r'ç|é|à|è': 'fr'
        }
        detected_lang = "en"  # default
        for pattern, lang in lang_map.items():
            if re.search(pattern, inquiry_text):
                detected_lang = lang
                break
        
        # Urgency & handoff logic
        if sentiment_score < -0.6 and tier in ["VIP", "PREMIUM"]:
            urgency = "CRITICAL"
            handoff = True
        elif sentiment_score < -0.3 or intent == "RMA_REQUEST":
            urgency = "HIGH"
            handoff = False
        else:
            urgency = "MEDIUM" if intent in ["SHIPPING_INQUIRY", "PRODUCT_USAGE"] else "LOW"
            handoff = False
        
        # Sentiment label mapping
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
        
        return {
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_label": label,
            "intent_category": intent,
            "customer_tier": tier,
            "urgency_level": urgency,
            "language_detected": detected_lang,
            "requires_human_handoff": handoff
        }

class RMAAutomationTool(BaseTool):
    name: str = "RMA Policy Validation & Logistics Automation Tool"
    description: str = "Validates return eligibility against policy, generates prepaid return labels via logistics APIs, and triggers warehouse inbound notifications."

    def _run(self, order_id: str, customer_email: str, item_sku: str, return_reason: str, detected_language: str) -> dict:
        logger.info(f"Processing RMA request for order {order_id}, item {item_sku}...")
        
        # 🔧 Production Hook: Integrate with:
        # - Order DB/API for delivery date & item condition
        # - ShipStation/EasyPost API for label generation
        # - WMS (NetSuite/Odoo) webhook for inbound ASN
        
        # Mock order data lookup
        order_data = {
            "order_id": order_id,
            "delivery_date": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            "item_condition": "unopened",  # or "opened", "damaged"
            "original_price": 89.99,
            "region": "EU" if "de" in detected_language or "fr" in detected_language else "US"
        }
        
        # Policy validation logic
        delivery_date = datetime.fromisoformat(order_data["delivery_date"].replace("Z", "+00:00"))
        days_since_delivery = (datetime.utcnow() - delivery_date).days
        return_window = 30 if order_data["region"] == "US" else 14  # EU has shorter window
        
        eligible = days_since_delivery <= return_window and order_data["item_condition"] in ["unopened", "defective"]
        
        rma_result = {
            "order_id": order_id,
            "eligible_for_return": eligible,
            "eligibility_reason": "Within return window & item condition accepted" if eligible else f"Return window expired ({days_since_delivery} > {return_window} days)" if days_since_delivery > return_window else f"Item condition '{order_data['item_condition']}' not eligible",
            "return_window_days": return_window,
            "days_since_delivery": days_since_delivery,
            "item_condition_accepted": ["unopened", "defective"],
            "restocking_fee_pct": 0.0 if order_data["item_condition"] == "defective" else 15.0,
            "return_shipping_responsibility": "BRAND" if order_data["item_condition"] == "defective" else "CUSTOMER"
        }
        
        if not eligible:
            return {"rma_validation": rma_result, "logistics_output": None}
        
        # Logistics integration (mock ShipStation/EasyPost)
        carriers = {"US": "EasyPost", "EU": "DHL", "JP": "Local_Post"}
        carrier = carriers.get(order_data["region"], "ShipStation")
        
        logistics_output = {
            "carrier": carrier,
            "prepaid_label_url": f"https://labels.{carrier.lower()}.com/return/{order_id}_{item_sku}.pdf",
            "tracking_number": f"RTN{order_id.replace('-', '')}{item_sku[:3].upper()}",
            "estimated_refund_days": 5 if order_data["item_condition"] == "defective" else 10,
            "warehouse_inbound_notification": f"✅ ASN-2024-{order_id} sent to WMS | Expected receipt: +{logistics_output['estimated_refund_days'] if 'logistics_output' in locals() else 7} days",
            "return_instructions_localized": self._get_localized_instructions(detected_language, order_data["region"], rma_result)
        }
        
        return {
            "rma_validation": rma_result,
            "logistics_output": logistics_output
        }
    
    def _get_localized_instructions(self, lang: str, region: str, rma: dict) -> str:
        """Generate return instructions in customer's language"""
        templates = {
            "en": f"Print your prepaid label, pack the item securely, and drop off at any {rma['return_shipping_responsibility'] == 'BRAND' and 'carrier' or 'authorized'} location. Refund processed within {rma['return_window_days']} days of receipt.",
            "ja": f"事前払い戻しラベルを印刷し、商品を丁寧に梱包して、指定の配送拠点にお持ち込みください。商品到着後{rma['return_window_days']}日以内に返金処理いたします。",
            "es": f"Imprima su etiqueta prepagada, empaque el artículo de forma segura y entréguelo en cualquier punto de recogida autorizado. El reembolso se procesará en {rma['return_window_days']} días tras la recepción.",
            "de": f"Drucken Sie Ihr vorausbezahltes Etikett aus, verpacken Sie den Artikel sicher und geben Sie ihn bei einer autorisierten Annahmestelle ab. Die Rückerstattung erfolgt innerhalb von {rma['return_window_days']} Tagen nach Eingang.",
            "fr": f"Imprimez votre étiquette prépayée, emballez l'article en toute sécurité et déposez-le dans un point de dépôt autorisé. Le remboursement sera traité sous {rma['return_window_days']} jours après réception."
        }
        return templates.get(lang, templates["en"])

class MultilingualResponseGeneratorTool(BaseTool):
    name: str = "Multilingual Response Template Generator"
    description: str = "Generates culturally appropriate, tone-matched support responses in the customer's detected language, with optional compensation logic."

    def _run(self, sentiment: SentimentIntentAnalysis, rma_result: Optional[Dict], inquiry_summary: str) -> dict:
        logger.info(f"Generating {sentiment.language_detected} response for intent: {sentiment.intent_category}...")
        
        # Tone mapping
        if sentiment.sentiment_score < -0.4:
            tone = "EMPATHETIC_APOLOGETIC"
            include_comp = sentiment.customer_tier in ["VIP", "PREMIUM"] and sentiment.sentiment_score < -0.6
        elif sentiment.intent_category == "RMA_REQUEST":
            tone = "PROFESSIONAL_HELPFUL"
            include_comp = False
        else:
            tone = "CONCISE_INFORMATIVE"
            include_comp = False
        
        # Compensation logic
        compensation = None
        if include_comp:
            compensation = {
                "type": "STORE_CREDIT",
                "amount_pct": 15 if sentiment.customer_tier == "VIP" else 10,
                "expiry_days": 90,
                "auto_applied": True
            }
        
        # Response templates by language & intent
        templates = {
            "en": {
                "RMA_REQUEST": {
                    "subject": "Your Return Request for Order {{order_id}}",
                    "body": "Thank you for reaching out. {{#eligible}}We've generated a prepaid return label for your item. {{label_instructions}}{{/eligible}}{{^eligible}}Unfortunately, your order is outside our {{window}}-day return window. {{alternative_options}}{{/eligible}} {{#compensation}}As a gesture of goodwill, we've added {{amount}}% store credit to your account.{{/compensation}}"
                },
                "SHIPPING_INQUIRY": {
                    "subject": "Update on Your Order {{order_id}}",
                    "body": "Your order is currently {{status}}. Expected delivery: {{date}}. Track anytime: {{tracking_link}}. {{#compensation}}We apologize for the delay and have added {{amount}}% credit to your account.{{/compensation}}"
                }
            },
            "ja": {
                "RMA_REQUEST": {
                    "subject": "ご注文 {{order_id}} の返品リクエストについて",
                    "body": "お問い合わせいただきありがとうございます。{{#eligible}}返品用事前払いラベルを発行いたしました。{{label_instructions}}{{/eligible}}{{^eligible}}誠に恐れ入りますが、ご注文は返品期限（{{window}} 日）を過ぎています。{{alternative_options}}{{/eligible}} {{#compensation}}お詫びとして、アカウントに{{amount}}% のストアクレジットを追加させていただきました。{{/compensation}}"
                }
            },
            "es": {
                "RMA_REQUEST": {
                    "subject": "Solicitud de devolución para el pedido {{order_id}}",
                    "body": "Gracias por contactarnos. {{#eligible}}Hemos generado una etiqueta de devolución prepagada para su artículo. {{label_instructions}}{{/eligible}}{{^eligible}}Lamentablemente, su pedido está fuera de nuestra ventana de devolución de {{window}} días. {{alternative_options}}{{/eligible}} {{#compensation}}Como gesto de buena voluntad, hemos añadido {{amount}}% de crédito en su cuenta.{{/compensation}}"
                }
            }
        }
        
        # Simple template rendering (mock)
        lang = sentiment.language_detected
        intent = sentiment.intent_category
        base_template = templates.get(lang, templates["en"]).get(intent, templates["en"]["RMA_REQUEST"])
        
        # Placeholder replacement
        body = base_template["body"]
        if rma_result:
            body = body.replace("{{order_id}}", rma_result["order_id"])
            body = body.replace("{{eligible}}", "✓" if rma_result["eligible_for_return"] else "")
            body = body.replace("{{window}}", str(rma_result["return_window_days"]))
            if rma_result["eligible_for_return"] and "logistics_output" in str(rma_result):
                body = body.replace("{{label_instructions}}", f"Download label: {rma_result.get('logistics_output', {}).get('prepaid_label_url', '#')}")
            else:
                body = body.replace("{{alternative_options}}", "Please contact us for alternative solutions.")
        if compensation:
            body = body.replace("{{amount}}", str(compensation["amount_pct"]))
        
        # Clean up unused placeholders
        body = re.sub(r'\{\{[^}]+\}\}', '', body)
        
        return {
            "language": lang,
            "subject": base_template["subject"].replace("{{order_id}}", rma_result["order_id"] if rma_result else "XXXX"),
            "body": body.strip(),
            "tone": tone,
            "includes_compensation": include_comp,
            "compensation_details": compensation
        }

# =============================================================================
# 🧠 3. Agents, Tasks & Crew Assembly
# =============================================================================
def build_customer_support_crew():
    # Tools
    sentiment_tool = SentimentIntentGradingTool()
    rma_tool = RMAAutomationTool()
    response_tool = MultilingualResponseGeneratorTool()
    serper_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()
    web_search_tool = WebsiteSearchTool()

    # Agents
    triage_specialist = Agent(
        role="Customer Sentiment & Intent Triage Specialist",
        goal="Analyze incoming inquiries for sentiment, intent, customer tier, and escalation requirements.",
        backstory="""You are a support operations expert trained in emotional intelligence and customer segmentation. 
        You quickly identify high-risk interactions requiring human attention while routing routine queries to automation.""",
        tools=[sentiment_tool],
        verbose=True
    )

    rma_automation_agent = Agent(
        role="RMA Policy & Logistics Automation Specialist",
        goal="Validate return eligibility, generate prepaid labels, and synchronize with warehouse systems.",
        backstory="""You are a reverse logistics coordinator who ensures seamless return experiences. 
        You balance policy compliance with customer satisfaction, automating approvals where possible 
        and flagging exceptions for human review.""",
        tools=[rma_tool, scrape_tool],
        verbose=True
    )

    multilingual_responder = Agent(
        role="Multilingual Support Response Generator",
        goal="Craft culturally appropriate, tone-matched responses in the customer's native language.",
        backstory="""You are a global customer communications specialist fluent in 10+ languages. 
        You adapt brand voice to local norms, ensuring every response feels personal and respectful 
        while maintaining efficiency through smart templating.""",
        tools=[response_tool, web_search_tool],
        verbose=True
    )

    qa_compliance_agent = Agent(
        role="Support Quality & Compliance Auditor",
        goal="Review all auto-generated responses for accuracy, brand alignment, and regulatory compliance.",
        backstory="""You are a risk mitigation specialist ensuring every customer interaction meets 
        legal, brand, and quality standards before delivery. You catch edge cases automation might miss.""",
        tools=[serper_tool],
        verbose=True
    )

    # Tasks
    t1_triage = Task(
        description="""Analyze inquiry from {customer_email}: "{inquiry_text}". 
        Determine sentiment score, intent category, customer tier, urgency, detected language, and handoff requirement.
        Order history context: {order_history}""",
        expected_output="SentimentIntentAnalysis with all grading fields populated.",
        agent=triage_specialist
    )

    t2_rma_process = Task(
        description="""If intent is RMA_REQUEST, validate return eligibility for order {order_id}, item {item_sku}.
        Generate prepaid label and warehouse notification if eligible. Use detected language: {language_detected}.""",
        expected_output="RMAValidationResult + LogisticsIntegrationOutput (or null if not RMA/ineligible).",
        agent=rma_automation_agent,
        context=[t1_triage]
    )

    t3_generate_response = Task(
        description="""Generate a support response in {language_detected} based on sentiment analysis and RMA outcome.
        Apply appropriate tone (empathetic/professional/concise). Include compensation if criteria met.""",
        expected_output="ResponseTemplate with localized subject, body, tone, and compensation details.",
        agent=multilingual_responder,
        context=[t1_triage, t2_rma_process]
    )

    t4_qa_finalize = Task(
        description="""Review the complete ticket resolution: verify sentiment grading accuracy, RMA policy compliance, 
        response tone/language appropriateness, and GDPR/CCPA compliance tags. Add internal notes for human agents if escalated.""",
        expected_output="Final SupportTicketResolution with all fields validated and compliance tags attached.",
        agent=qa_compliance_agent,
        context=[t1_triage, t2_rma_process, t3_generate_response],
        output_pydantic=SupportTicketResolution
    )

    crew = Crew(
        agents=[triage_specialist, rma_automation_agent, multilingual_responder, qa_compliance_agent],
        tasks=[t1_triage, t2_rma_process, t3_generate_response, t4_qa_finalize],
        verbose=True,
        memory=True,
        process="sequential"
    )
    return crew

# =============================================================================
# 🚀 4. Orchestration & Validation Runner
# =============================================================================
def run_customer_support_crew(inputs: dict) -> dict:
    crew = build_customer_support_crew()
    result = crew.kickoff(inputs=inputs)
    
    # CrewAI v0.30+ returns TaskOutput with pydantic attribute
    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic.model_dump()
    return json.loads(result.raw) if isinstance(result.raw, str) else result.raw

def validate_support_outputs(report: dict):
    print("\n" + "="*70)
    print("✅ CUSTOMER SUPPORT AGENT - CORE FEATURE VALIDATION")
    print("="*70)
    
    sentiment = report.get("sentiment_analysis", {})
    rma = report.get("rma_validation")
    logistics = report.get("logistics_output")
    response = report.get("generated_response", {})
    
    checks = [
        # 😊 Sentiment & Intent Grading
        ("1. Sentiment & Intent Grading", "sentiment_score" in sentiment),
        ("   ├─ Sentiment Score Range", -1.0 <= sentiment.get("sentiment_score", -999) <= 1.0),
        ("   ├─ Intent Classification", sentiment.get("intent_category") in ["RMA_REQUEST", "SHIPPING_INQUIRY", "PRODUCT_USAGE", "BILLING_ISSUE", "VIP_ESCALATION", "GENERAL"]),
        ("   ├─ Customer Tier Detection", sentiment.get("customer_tier") in ["VIP", "PREMIUM", "STANDARD", "NEW"]),
        ("   ├─ Language Detection", len(sentiment.get("language_detected", "")) == 2),  # ISO 639-1
        ("   └─ VIP Anger Handoff Logic", sentiment.get("requires_human_handoff") == (sentiment.get("sentiment_score", 0) < -0.6 and sentiment.get("customer_tier") in ["VIP", "PREMIUM"])),
        
        # 📦 RMA Automation
        ("2. RMA Policy Validation", rma is not None if sentiment.get("intent_category") == "RMA_REQUEST" else True),
        ("   ├─ Eligibility Logic", "eligible_for_return" in rma if rma else True),
        ("   ├─ Return Window Calculation", "days_since_delivery" in rma and "return_window_days" in rma if rma else True),
        ("   └─ Restocking Fee Logic", "restocking_fee_pct" in rma if rma else True),
        
        # 🚚 Logistics Integration
        ("3. Logistics Automation", logistics is not None if (rma and rma.get("eligible_for_return")) else True),
        ("   ├─ Carrier Selection", logistics.get("carrier") in ["ShipStation", "EasyPost", "DHL", "FedEx", "Local_Post"] if logistics else True),
        ("   ├─ Prepaid Label URL Generated", "prepaid_label_url" in logistics if logistics else True),
        ("   ├─ Warehouse ASN Notification", "warehouse_inbound_notification" in logistics if logistics else True),
        ("   └─ Localized Return Instructions", len(logistics.get("return_instructions_localized", "")) > 20 if logistics else True),
        
        # 💬 Multilingual Response
        ("4. Response Generation", "body" in response),
        ("   ├─ Language Match", response.get("language") == sentiment.get("language_detected")),
        ("   ├─ Tone Appropriateness", response.get("tone") in ["EMPATHETIC_APOLOGETIC", "PROFESSIONAL_HELPFUL", "CONCISE_INFORMATIVE"]),
        ("   ├─ Compensation Logic", (response.get("includes_compensation") == (sentiment.get("sentiment_score", 0) < -0.6 and sentiment.get("customer_tier") in ["VIP", "PREMIUM"]))),
        ("   └─ Compliance Tags", len(report.get("compliance_tags", [])) >= 1),
        
        # 🎯 End-to-End Flow
        ("5. Escalation Flag Accuracy", report.get("escalation_flag") == sentiment.get("requires_human_handoff")),
        ("   ├─ Follow-up Recommendation", report.get("recommended_follow_up_hours", -1) >= 0),
        ("   └─ Internal Notes for Handoff", len(report.get("internal_notes", "")) > 10 if report.get("escalation_flag") else True)
    ]

    passed = 0
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
        if status: passed += 1

    print(f"\n📊 Validation Score: {passed}/{len(checks)} Checks Passed")
    print("="*70)
    
    # Print demo output snippet
    print(f"\n💬 Generated Response Preview ({response.get('language', 'en').upper()}):")
    print(f"   Subject: {response.get('subject', 'N/A')}")
    print(f"   Body: {response.get('body', 'N/A')[:150]}...")
    if response.get("includes_compensation"):
        print(f"   🎁 Compensation: {response.get('compensation_details', {})}")
    
    if logistics:
        print(f"\n🚚 Logistics Output:")
        print(f"   Carrier: {logistics.get('carrier')}")
        print(f"   Label: {logistics.get('prepaid_label_url', '')[:60]}...")
        print(f"   WMS Notification: {logistics.get('warehouse_inbound_notification', '')}")
    
    return passed == len(checks)

if __name__ == "__main__":
    # 🧪 Test Case 1: VIP Customer, Angry, RMA Request (Japanese)
    test_inputs_vip = {
        "ticket_id": "TKT-JP-VIP-001",
        "customer_email": "tanaka@enterprise.com",
        "inquiry_text": "注文した商品が破損していました。非常に失望しています。すぐに返品手続きをお願いします。",  # "The item I ordered arrived damaged. Very disappointed. Please process return immediately."
        "order_id": "JP-2024-8842",
        "item_sku": "CAM-4K-PRO",
        "order_history": {
            "lifetime_value": 8500.00,
            "order_count": 12,
            "last_order_date": "2024-09-15"
        }
    }
    
    # 🧪 Test Case 2: Standard Customer, Neutral, Shipping Inquiry (Spanish)
    test_inputs_standard = {
        "ticket_id": "TKT-ES-STD-042",
        "customer_email": "maria.garcia@email.com",
        "inquiry_text": "¿Dónde está mi pedido? El número de seguimiento no funciona.",  # "Where is my order? The tracking number doesn't work."
        "order_id": "ES-2024-3391",
        "item_sku": "THERMO-STEEL",
        "order_history": {
            "lifetime_value": 210.00,
            "order_count": 2,
            "last_order_date": "2024-09-20"
        }
    }

    print("🚀 Starting Customer Support Crew Execution...")
    print("🧪 Test Case: VIP Japanese Customer - Damaged Item RMA Request")
    print("-"*70)
    
    report = run_customer_support_crew(test_inputs_vip)
    is_valid = validate_support_outputs(report)
    
    if is_valid:
        print("\n💡 All core features successfully integrated and validated.")
        print("📦 Ready for FastAPI integration via `orchestrator.register_crew(WorkflowType.SUPPORT, run_customer_support_crew)`")
        print("\n✨ Production Integration Tips:")
        print("   • Connect sentiment_tool._run() to fine-tuned XLM-RoBERTa for multilingual sentiment")
        print("   • Replace RMAAutomationTool logistics calls with ShipStation/EasyPost SDK")
        print("   • Pipe generated_response to Zendesk/Intercom API with language routing")
        print("   • Use escalation_flag to trigger Slack/Teams alert to human support queue")
    else:
        print("\n⚠️ Some validation checks failed. Review logs for mismatches.")