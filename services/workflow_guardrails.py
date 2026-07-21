from __future__ import annotations

import base64
import hashlib
import importlib
import ipaddress
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MethodType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import yaml

from models import WorkflowType
from runtime_config import LLMProfileConfig, load_runtime_config
from utils.observability import guardrail_span, set_span_attributes
from utils.support_drafts import customer_facing_draft_text

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "guardrails.yaml"
MODEL_CACHE_DIR = BASE_DIR / ".cache"
GUARDRAIL_REVIEW_FLAG = "GUARDRAIL_REVIEW_REQUIRED"
GUARDRAIL_HIGH_RISK_FLAG = "GUARDRAIL_HIGH_RISK"
PROMPT_INJECTION_VALIDATOR_ID = "prompt_injection"
PROMPT_INJECTION_CACHE_PREFIX = "workflow-guardrails:prompt-injection:v1"
PROMPT_INJECTION_MAX_CHARS = 8000
LOCAL_DETECTOR_VERSION = "guardrail-local-v2"
SEMANTIC_DETECTOR_VERSION = "qwen-semantic-v1"
SEMANTIC_VALIDATOR_IDS = {
    "forbidden_terms",
    "prompt_injection",
    "provenance_llm",
    "toxic_language",
}
RAW_LOCAL_VALIDATOR_IDS = {"detect_pii", "secrets_present"}
SUPPORT_PROMPT_TEXT_KEYS: tuple[str, ...] = (
    "inquiry_text",
    "inquiry",
    "message",
    "text",
    "prompt",
)
CONTENT_PROMPT_TEXT_KEYS: tuple[str, ...] = (
    "subject",
    "product_category",
    "product_features",
    "brand_voice",
    "brand_name",
    "primary_keywords",
)

SECRET_REDACTION_RE_LIST: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bsk-lf-[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:[A-Za-z0-9_.-]+[_-])?(?:api[_-]?key|client[_-]?secret|webhook[_-]?secret|secret|password|token|access[_-]?token|refresh[_-]?token)"
        r"\s*(?::|=|\bis\b)\s*['\"]?(?!\[?(?:REDACTED|SECRET)\]?\b|YOUR_[A-Z0-9_]+\b|\$\{)[^'\"\s,;]{8,}",
        re.IGNORECASE,
    ),
    re.compile(r"\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
    re.compile(
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:/@]+:[^\s/@]{6,}@",
        re.IGNORECASE,
    ),
    re.compile(r"\bs\s*k\s*-\s*(?:[A-Za-z0-9_-]\s*){8,}\b", re.IGNORECASE),
    re.compile(r"\btoken\s+parts?\s*:\s*[^\n,;]{12,}", re.IGNORECASE),
)
EMAIL_ADDRESS_REDACTOR = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_NUMBER_REDACTOR = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
IP_ADDRESS_REDACTOR = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
PAYMENT_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
BASE64_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])"
)
BUSINESS_IDENTIFIER_RE = re.compile(
    r"\b(?:business|channel\s+thread|conversation|isbn|job|message|order|product|reference|session|sku|ticket|tracking)"
    r"(?:\s+(?:id|no\.?|number))?\s*[:#=-]?\s*([A-Z0-9][A-Z0-9._/-]{3,})",
    re.IGNORECASE,
)
UUID_TOKEN_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
ENV_VALUE_RE = re.compile(r"^\$\{([A-Z0-9_]+)(?::([^}]*))?\}$")
PRESERVED_IDENTIFIER_FIELDS = {
    "channel_thread_id",
    "conversation_id",
    "family_code",
    "job_id",
    "message_id",
    "order_id",
    "order_number",
    "product_id",
    "product_sku",
    "reference",
    "reference_number",
    "session_id",
    "sku",
    "source_job_id",
    "ticket_id",
    "thread_id",
    "tracking_number",
    "tracking",
}
PII_TEXT_FIELDS = {
    "body",
    "content",
    "conversation_history",
    "customer_email",
    "customer_handle",
    "customer_message",
    "customer_query",
    "description",
    "drafted_response",
    "email",
    "final_response",
    "history",
    "inquiry",
    "inquiry_text",
    "internal_notes",
    "message",
    "messages",
    "notes",
    "phone",
    "phone_number",
    "prompt",
    "query",
    "recipient",
    "response",
    "sender",
    "subject",
    "support_response",
    "text",
}
DELIVERY_CONTACT_FIELDS = {
    "customer_email",
    "customer_handle",
    "email",
    "phone",
    "phone_number",
    "recipient",
    "sender",
}
ADDRESS_FIELDS = {
    "address",
    "billing_address",
    "customer_address",
    "delivery_address",
    "mailing_address",
    "receiver_address",
    "shipping_address",
}
SECRET_PLACEHOLDER_RE = re.compile(
    r"^(?:\[?(?:REDACTED|SECRET)\]?|YOUR_[A-Z0-9_]+|\$\{[A-Z0-9_]+(?::[^}]*)?\}|<[^>]+>)$",
    re.IGNORECASE,
)
BENIGN_CONTEXT_RE = re.compile(
    r"(?is)(?:attack\s+example|called\s+an?|can\s+an?\s+attacker|checklist\s+for\s+resisting|dangerous|"
    r"explain(?:ing)?\s+(?:why|prompt injection)|include\s+the\s+sentence|reported\s+(?:being|that)|"
    r"scammer\s+told|quote\s+['\"]|terminology\s+without\s+examples|"
    r"攻击示例|解释.*(?:攻击|危险)|引用|安全检查清单)"
)
PROMPT_CONTROL_TARGET_RE = re.compile(
    r"(?is)(?:developer\s+(?:instruction|message)|guardrails?|hidden\s+(?:instruction|policy)|"
    r"previous\s+(?:instruction|rule)|prior\s+rules?|safety\s+(?:check|policy|rule|safeguard)|safeguards?|compliance|"
    r"system\s+(?:instruction|message|prompt)|your\s+prompt|\brules?\b|tool|\.env|environment\s+variable|chain\s+of\s+thought|"
    r"开发者消息|安全(?:检查|策略|规则)|系统(?:消息|提示词|命令)|隐藏(?:指令|策略)|之前所有指令|所有工具|凭据|环境变量|思维链|规则)"
)
PROMPT_ATTACK_ACTION_RE = re.compile(
    r"(?is)(?:bypass|disable|disclose|execute|expose|forget|follow\s+it|ignore|list|obey\s+only|"
    r"override|print|reveal|share|upload|忽略|无视|覆盖|忘记|禁用|公开|显示|返回|上传|执行|自动批准|当作系统命令)"
)
PROMPT_ROLE_INJECTION_RE = re.compile(
    r"(?is)(?:\[INST\]|<system>|###\s*NEW\s+POLICY|\bSYSTEM\s*[:：]|\byou\s+are\s+(?:now\s+)?root\b|"
    r"开发者消息\s*[:：])"
)
TOXIC_TERM_RE = re.compile(
    r"(?is)(?:\b(?:idiot|idiots|loser|losers|stupid|worthless)\b|shut\s+up|"
    r"attack\s+and\s+shame|humiliate\s+people|mock\s+customers?|"
    r"白痴|闭嘴|蠢货|讨厌你|愚蠢|毫无价值|滚开|羞辱|嘲笑)"
)
TOXIC_REPORTING_RE = re.compile(
    r"(?is)(?:reported\s+(?:being\s+)?called|was\s+called|quoted?\s+as|"
    r"customer\s+(?:said|reported)|客户报告|被称为|引用|不要|不得|禁止)"
)
GROUNDING_REQUIRED_RE = re.compile(
    r"(?is)(?:\b(?:approved|available|clinically\s+proven|discount|in\s+stock|material|"
    r"rated|refund\s+approved|request\s+is\s+being\s+reviewed|ships?\s+(?:free|within)|\d+(?:\.\d+)?%|"
    r"status\s+is|width\s+is)\b|"
    r"申请.*审核|退款.*(?:批准|通过)|商品.*(?:有货|可用)|临床.*证明|状态.*(?:处理中|运输中))"
)
HUB_VALIDATOR_CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "hub://guardrails/secrets_present": ("SecretsPresent",),
    "hub://guardrails/detect_pii": ("DetectPII",),
    "hub://guardrails/toxic_language": ("ToxicLanguage",),
    "hub://guardrails/provenance_llm": ("ProvenanceLLM",),
    "hub://guardrails/regex_match": ("RegexMatch",),
    "hub://sainatha/prompt_injection_detector": ("PromptInjectionDetector",),
}


class GuardrailConfigurationError(RuntimeError):
    """Raised when a configured Guardrails Hub validator cannot run."""


def _ensure_model_cache_env() -> None:
    hf_home = MODEL_CACHE_DIR / "huggingface"
    hf_hub_cache = hf_home / "hub"
    sentence_transformers_home = MODEL_CACHE_DIR / "sentence-transformers"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(sentence_transformers_home))
    for cache_dir in (hf_home, hf_hub_cache, sentence_transformers_home):
        cache_dir.mkdir(parents=True, exist_ok=True)


class GuardrailStage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    ACTION = "action"
    TOOL = "tool"


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    MONITOR = "monitor"
    MASK = "mask"
    REVIEW_REQUIRED = "review_required"
    BLOCK = "block"


class GuardrailSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_RANK: dict[str, int] = {
    GuardrailSeverity.NONE.value: 0,
    GuardrailSeverity.LOW.value: 1,
    GuardrailSeverity.MEDIUM.value: 2,
    GuardrailSeverity.HIGH.value: 3,
    GuardrailSeverity.CRITICAL.value: 4,
}
ACTION_RANK: dict[GuardrailAction, int] = {
    GuardrailAction.ALLOW: 0,
    GuardrailAction.MONITOR: 1,
    GuardrailAction.MASK: 2,
    GuardrailAction.REVIEW_REQUIRED: 3,
    GuardrailAction.BLOCK: 4,
}


class GuardrailFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str
    validator: str
    severity: GuardrailSeverity = GuardrailSeverity.HIGH
    message: str
    evidence_masked: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectorSignal(BaseModel):
    """Non-persistent detector evidence used by the policy aggregator."""

    model_config = ConfigDict(extra="forbid", strict=True)

    policy_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    language: str = Field(default="unknown", min_length=2, max_length=16)
    entity_type: str | None = None
    evidence_hash: str = Field(..., min_length=12, max_length=64)
    detector_version: str = Field(..., min_length=1)


class RedactionResult(BaseModel):
    """Sanitized payload plus safe aggregate redaction metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sanitized_payload: Any
    redaction_counts: dict[str, int] = Field(default_factory=dict)
    complete: bool


class _SemanticLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    policy_id: str
    unsafe: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason_code: str = Field(default="semantic_risk", min_length=1, max_length=80)


class _SemanticAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    labels: list[_SemanticLabel] = Field(default_factory=list)


class GuardrailDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_type: str
    stage: GuardrailStage
    action: GuardrailAction
    severity: GuardrailSeverity
    findings: list[GuardrailFinding] = Field(default_factory=list)
    sanitized_payload: Any | None = None
    human_override_allowed: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.action in {
            GuardrailAction.ALLOW,
            GuardrailAction.MONITOR,
            GuardrailAction.MASK,
        }


@dataclass
class _PayloadEvaluation:
    findings: list[GuardrailFinding]
    skipped_validators: list[dict[str, Any]]
    signals: list[DetectorSignal] = field(default_factory=list)
    redaction: RedactionResult | None = None
    degraded: bool = False


class _CachedValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _payload_to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, default=str)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def mask_text(text: str) -> str:
    masked = _mask_secrets(str(text))
    masked = EMAIL_ADDRESS_REDACTOR.sub(
        lambda match: _mask_email(match.group(0)), masked
    )
    masked = _mask_ip_addresses(masked)
    masked = _mask_payment_cards(masked)
    masked = _mask_phone_numbers_preserving_business_ids(masked)
    return masked


def _mask_secrets(text: str) -> str:
    masked = str(text)
    for pattern in SECRET_REDACTION_RE_LIST:
        masked = pattern.sub("[SECRET]", masked)
    masked = _mask_encoded_secrets(masked)
    return masked


def _mask_encoded_secrets(text: str) -> str:
    if not re.search(
        r"(?i)\b(?:credential|secret|token|decode|凭据|密钥|令牌)\b", text
    ):
        return text

    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            padded = candidate + "=" * (-len(candidate) % 4)
            decoded = base64.b64decode(padded, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return candidate
        if _looks_like_secret_material(decoded):
            return "[SECRET]"
        return candidate

    return BASE64_CANDIDATE_RE.sub(replace, text)


def _looks_like_secret_material(value: str) -> bool:
    normalized = value.strip()
    if not normalized or SECRET_PLACEHOLDER_RE.fullmatch(normalized):
        return False
    return bool(
        re.search(
            r"(?i)(?:\bsk-[A-Za-z0-9_-]{8,}|password|secret|token|credential|api[_-]?key)",
            normalized,
        )
    )


def _mask_ip_addresses(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            return candidate
        return "[IP_ADDRESS]"

    return IP_ADDRESS_REDACTOR.sub(replace, text)


def _mask_payment_cards(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = "".join(character for character in candidate if character.isdigit())
        if _span_is_preserved_business_identifier(
            text, match.span(0)
        ) or not _passes_luhn(digits):
            return candidate
        return f"[PAYMENT_CARD:{digits[-4:]}]"

    return PAYMENT_CARD_CANDIDATE_RE.sub(replace, text)


def _passes_luhn(digits: str) -> bool:
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _mask_phone_numbers_preserving_business_ids(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if _span_is_preserved_business_identifier(text, match.span(0)):
            return match.group(0)
        return _mask_phone(match.group(0))

    return PHONE_NUMBER_REDACTOR.sub(replace, text)


def sanitize_payload(payload: Any, field_name: str | None = None) -> Any:
    if isinstance(payload, str):
        normalized_field = str(field_name or "").lower()
        secret_masked = _mask_secrets(payload)
        if normalized_field in ADDRESS_FIELDS and secret_masked.strip():
            return "[ADDRESS]"
        if _is_preserved_identifier_field(normalized_field):
            return secret_masked
        return mask_text(secret_masked)
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                sanitized[key_text] = _mask_sensitive_value(value)
            else:
                sanitized[key_text] = sanitize_payload(value, key_text)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(item, field_name) for item in payload]
    if isinstance(payload, tuple):
        return [sanitize_payload(item, field_name) for item in payload]
    return payload


def redact_payload(payload: Any) -> RedactionResult:
    """Recursively redact every string and return safe aggregate metadata."""
    sanitized = sanitize_payload(payload)
    serialized = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, default=str)
    redaction_counts = {
        "address": serialized.count("[ADDRESS]"),
        "email": len(re.findall(r"[\w.+-]{0,2}\*{3}@[\w.-]+", serialized)),
        "ip_address": serialized.count("[IP_ADDRESS]"),
        "payment_card": serialized.count("[PAYMENT_CARD:"),
        "phone": serialized.count("[PHONE"),
        "secret": serialized.count("[SECRET]"),
    }
    return RedactionResult(
        sanitized_payload=sanitized,
        redaction_counts={
            key: value for key, value in redaction_counts.items() if value
        },
        complete=not _contains_unredacted_private_data(sanitized),
    )


def _contains_unredacted_private_data(
    payload: Any, field_name: str | None = None
) -> bool:
    if isinstance(payload, str):
        normalized_field = str(field_name or "").lower()
        if normalized_field in ADDRESS_FIELDS and payload not in {"", "[ADDRESS]"}:
            return True
        if _is_preserved_identifier_field(normalized_field):
            return False
        if EMAIL_ADDRESS_REDACTOR.search(payload) or _contains_valid_ip(payload):
            return True
        if any(
            _passes_luhn(
                "".join(
                    character for character in match.group(0) if character.isdigit()
                )
            )
            and not _span_is_preserved_business_identifier(payload, match.span(0))
            for match in PAYMENT_CARD_CANDIDATE_RE.finditer(payload)
        ):
            return True
        if _contains_unmasked_phone(payload) or _contains_secret_material(payload):
            return True
        return False
    if isinstance(payload, dict):
        return any(
            _contains_unredacted_private_data(value, str(key))
            for key, value in payload.items()
        )
    if isinstance(payload, (list, tuple)):
        return any(
            _contains_unredacted_private_data(item, field_name) for item in payload
        )
    return False


def _contains_valid_ip(text: str) -> bool:
    for match in IP_ADDRESS_REDACTOR.finditer(text):
        try:
            ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        return True
    return False


def _contains_unmasked_phone(text: str) -> bool:
    for match in PHONE_NUMBER_REDACTOR.finditer(text):
        if _span_is_preserved_business_identifier(text, match.span(0)):
            continue
        return True
    return False


def _span_is_preserved_business_identifier(
    text: str,
    span: tuple[int, int],
) -> bool:
    protected_spans = [match.span(1) for match in BUSINESS_IDENTIFIER_RE.finditer(text)]
    protected_spans.extend(match.span(0) for match in UUID_TOKEN_RE.finditer(text))
    start, end = span
    return any(
        start < protected_end and end > protected_start
        for protected_start, protected_end in protected_spans
    )


def _contains_secret_material(text: str) -> bool:
    return _mask_secrets(text) != text


def mask_delivery_contacts(payload: Any, field_name: str | None = None) -> Any:
    if isinstance(payload, str):
        normalized_field = str(field_name or "").lower()
        if normalized_field in DELIVERY_CONTACT_FIELDS:
            return mask_text(payload)
        return payload
    if isinstance(payload, dict):
        return {
            str(key): (
                _mask_sensitive_value(value)
                if _is_sensitive_key(str(key))
                else mask_delivery_contacts(value, str(key))
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [mask_delivery_contacts(item, field_name) for item in payload]
    if isinstance(payload, tuple):
        return [mask_delivery_contacts(item, field_name) for item in payload]
    return payload


def support_provenance_claim(result: dict[str, Any]) -> str:
    return str(customer_facing_draft_text(result) or "").strip()


def support_provenance_grounding_context(result: dict[str, Any]) -> list[str]:
    sources: list[str] = []

    def add_text(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in sources:
            sources.append(text[:2000])

    def add_knowledge_results(value: Any) -> None:
        if not isinstance(value, list):
            return
        for item in value:
            if isinstance(item, dict):
                add_text(item.get("content"))

    pre_sales = result.get("pre_sales_response")
    if isinstance(pre_sales, dict):
        add_knowledge_results(pre_sales.get("catalog_knowledge_results"))
        offer = pre_sales.get("catalog_product_offer")
        if isinstance(offer, dict):
            for evidence in offer.get("evidence") or []:
                add_text(evidence)

    order = result.get("order_response")
    if isinstance(order, dict):
        add_knowledge_results(order.get("order_knowledge_results"))
        tracking_record = order.get("local_tracking_record")
        if isinstance(tracking_record, dict):
            safe_tracking_record = dict(tracking_record)
            safe_tracking_record.pop("receiver_name", None)
            add_text(
                json.dumps(
                    sanitize_payload(safe_tracking_record),
                    ensure_ascii=False,
                    default=str,
                )
            )

    return sources[:20]


def decision_event_payload(decision: GuardrailDecision) -> dict[str, Any]:
    dumped = decision.model_dump(mode="json", exclude={"sanitized_payload"})
    dumped["finding_count"] = len(decision.findings)
    dumped["policies"] = sorted({finding.policy_id for finding in decision.findings})
    return _json_safe(dumped)


def decision_result_payload(decision: GuardrailDecision) -> dict[str, Any]:
    return _json_safe(decision.model_dump(mode="json", exclude={"sanitized_payload"}))


def has_high_risk_guardrail(payload: Any) -> bool:
    decision = _guardrail_payload_from_result(payload)
    if not isinstance(decision, dict):
        return False
    action = str(decision.get("action") or "").lower()
    return action in {"review_required", "block"}


def guardrail_requires_override(payload: Any) -> bool:
    return has_high_risk_guardrail(payload)


def apply_support_guardrail_result(
    result: dict[str, Any], decision: GuardrailDecision
) -> dict[str, Any]:
    result = dict(result)
    result["guardrail_decision"] = decision_result_payload(decision)
    result["guardrail_action"] = decision.action.value
    result["guardrail_findings"] = [
        finding.model_dump(mode="json") for finding in decision.findings
    ]
    if decision.action in {GuardrailAction.REVIEW_REQUIRED, GuardrailAction.BLOCK}:
        result["qa_status"] = "REVIEW_REQUIRED"
        result["requires_approval"] = True
        flags = [str(flag) for flag in result.get("compliance_flags") or []]
        for flag in (GUARDRAIL_REVIEW_FLAG, GUARDRAIL_HIGH_RISK_FLAG):
            if flag not in flags:
                flags.append(flag)
        result["compliance_flags"] = flags
    return result


def apply_output_guardrail_result(
    workflow_type: WorkflowType | str,
    result: dict[str, Any],
    decision: GuardrailDecision,
) -> dict[str, Any]:
    workflow_value = _workflow_value(workflow_type)
    sanitized = (
        decision.sanitized_payload
        if isinstance(decision.sanitized_payload, dict)
        else result
    )
    output = dict(sanitized)
    output["guardrail_decision"] = decision_result_payload(decision)
    output["guardrail_action"] = decision.action.value
    output["guardrail_findings"] = [
        finding.model_dump(mode="json") for finding in decision.findings
    ]
    if workflow_value == WorkflowType.SUPPORT.value:
        return apply_support_guardrail_result(output, decision)
    return output


class WorkflowGuardrailService:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        _ensure_model_cache_env()
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self._embed_model: Any | None = None
        self._validator_cache: dict[str, Any] = {}
        self._redis_clients: dict[str, Any] = {}
        self._validate_config()

    def evaluate_input(
        self,
        workflow_type: WorkflowType | str,
        inputs: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        config_context = context or {}
        with guardrail_span(
            "guardrail_input_evaluated",
            workflow_type=workflow_value,
            stage=GuardrailStage.INPUT.value,
            job_id=_safe_trace_identifier((context or {}).get("job_id")),
            conversation_id=_conversation_trace_id(inputs, context),
            config_context=config_context,
        ):
            evaluation = self._evaluate_payload(
                workflow_value,
                GuardrailStage.INPUT,
                inputs,
                context=context,
            )
            action = self._policy_action(
                workflow_value, GuardrailStage.INPUT, evaluation.findings
            )
            redaction = evaluation.redaction or redact_payload(inputs)
            decision = self._decision(
                workflow_value,
                GuardrailStage.INPUT,
                action,
                evaluation.findings,
                sanitized_payload=redaction.sanitized_payload,
                metadata=self._evaluation_metadata(evaluation, redaction),
            )
            _record_guardrail_observability(
                decision, config_context, _conversation_trace_id(inputs, context)
            )
            return decision

    def evaluate_output(
        self,
        workflow_type: WorkflowType | str,
        result: dict[str, Any],
        grounding_context: list[str] | None = None,
        query_function: Any | None = None,
        embed_function: Any | None = None,
        config_context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        context = {
            **(config_context or {}),
            "embed_function": embed_function,
            "grounding_context": grounding_context or [],
            "query_function": query_function,
        }
        with guardrail_span(
            "guardrail_output_evaluated",
            workflow_type=workflow_value,
            stage=GuardrailStage.OUTPUT.value,
            job_id=_safe_trace_identifier(context.get("job_id")),
            conversation_id=_conversation_trace_id(result, context),
            config_context=context,
        ):
            evaluation = self._evaluate_payload(
                workflow_value,
                GuardrailStage.OUTPUT,
                result,
                context=context,
            )
            action = self._policy_action(
                workflow_value, GuardrailStage.OUTPUT, evaluation.findings
            )
            redaction = evaluation.redaction or redact_payload(result)
            decision = self._decision(
                workflow_value,
                GuardrailStage.OUTPUT,
                action,
                evaluation.findings,
                sanitized_payload=redaction.sanitized_payload,
                metadata={
                    "grounding_context_available": bool(grounding_context),
                    "deferred_validators": [
                        str(item.get("id") or item.get("hub") or "")
                        for item in self._validators_for(
                            workflow_value,
                            GuardrailStage.OUTPUT,
                            execution="async",
                        )
                    ],
                    **self._evaluation_metadata(evaluation, redaction),
                },
            )
            _record_guardrail_observability(
                decision, context, _conversation_trace_id(result, context)
            )
            return decision

    def evaluate_provenance(
        self,
        workflow_type: WorkflowType | str,
        claim: str,
        *,
        grounding_context: list[str] | None = None,
        query_function: Any | None = None,
        embed_function: Any | None = None,
        config_context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        sanitized_sources = [
            str(sanitize_payload(source))
            for source in (grounding_context or [])
            if str(source or "").strip()
        ]
        context = {
            **(config_context or {}),
            "embed_function": embed_function,
            "grounding_context": sanitized_sources,
            "query_function": query_function,
        }
        validators = [
            item
            for item in self._validators_for(
                workflow_value,
                GuardrailStage.OUTPUT,
                execution="async",
            )
            if str(item.get("id") or "") == "provenance_llm"
        ]
        with guardrail_span(
            "guardrail_provenance_evaluated",
            workflow_type=workflow_value,
            stage=GuardrailStage.OUTPUT.value,
            job_id=_safe_trace_identifier(context.get("job_id")),
            conversation_id=_conversation_trace_id(context, context),
            config_context=context,
            attributes={"guardrail_execution": "async", "guardrail_advisory": True},
        ):
            redaction = redact_payload(str(claim or ""))
            evaluation = self._evaluate_provenance_payload(
                workflow_value,
                GuardrailStage.OUTPUT,
                str(redaction.sanitized_payload or ""),
                validators,
                context,
            )
            action = self._policy_action(
                workflow_value,
                GuardrailStage.OUTPUT,
                evaluation.findings,
            )
            decision = self._decision(
                workflow_value,
                GuardrailStage.OUTPUT,
                action,
                evaluation.findings,
                sanitized_payload=redaction.sanitized_payload,
                metadata={
                    "advisory": True,
                    "execution": "async",
                    "grounding_context_available": bool(grounding_context),
                    **self._evaluation_metadata(evaluation, redaction),
                },
            )
            _record_guardrail_observability(
                decision, context, _conversation_trace_id(context, context)
            )
            return decision

    def evaluate_action(
        self,
        workflow_type: WorkflowType | str,
        action_type: str,
        payload: dict[str, Any],
        config_context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        context = {**(config_context or {}), "action_type": action_type}
        with guardrail_span(
            "guardrail_action_evaluated",
            workflow_type=workflow_value,
            stage=GuardrailStage.ACTION.value,
            job_id=_safe_trace_identifier(context.get("job_id")),
            conversation_id=_conversation_trace_id(payload, context),
            config_context=context,
            attributes={"action_type": action_type},
        ):
            evaluation = self._evaluate_payload(
                workflow_value,
                GuardrailStage.ACTION,
                payload,
                action_type=action_type,
                context=context,
            )
            action = self._policy_action(
                workflow_value,
                GuardrailStage.ACTION,
                evaluation.findings,
                action_type=action_type,
            )
            if evaluation.degraded and self._qwen_fail_closed_for_action(action_type):
                action = _max_action(action, GuardrailAction.REVIEW_REQUIRED)
            redaction = evaluation.redaction or redact_payload(payload)
            decision = self._decision(
                workflow_value,
                GuardrailStage.ACTION,
                action,
                evaluation.findings,
                sanitized_payload=redaction.sanitized_payload,
                metadata={
                    "action_type": action_type,
                    **self._evaluation_metadata(evaluation, redaction),
                },
            )
            _record_guardrail_observability(
                decision, context, _conversation_trace_id(payload, context)
            )
            return decision

    def _evaluate_payload(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        payload: Any,
        *,
        action_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> _PayloadEvaluation:
        if self._mode() == "off":
            return _PayloadEvaluation(
                findings=[],
                skipped_validators=[],
                redaction=redact_payload(payload),
            )
        config_context = context or {}
        validators = self._validators_for(workflow_type, stage, action_type)
        redaction = redact_payload(payload)
        local_signals = self._local_detector_signals(
            workflow_type,
            stage,
            payload,
            validators,
            grounding_context=config_context.get("grounding_context") or [],
        )
        qwen_signals, qwen_degraded = self._qwen_semantic_signals(
            workflow_type,
            stage,
            redaction.sanitized_payload,
            validators,
            config_context,
            action_type=action_type,
        )
        raw_validators = [
            item
            for item in validators
            if str(item.get("id") or "") in RAW_LOCAL_VALIDATOR_IDS
        ]
        semantic_validators = [
            item
            for item in validators
            if str(item.get("id") or "") not in RAW_LOCAL_VALIDATOR_IDS
        ]
        semantic_hub_validators = [
            item
            for item in semantic_validators
            if config_context.get("workflow_guardrails_hub_enabled") is True
            or item.get("hub_runtime", True)
            or str(item.get("id") or "")
            == str(getattr(self, "_fault_validator_id", ""))
            or (
                str(item.get("id") or "") == PROMPT_INJECTION_VALIDATOR_ID
                and hasattr(self, "_validation_passed")
            )
        ]
        if config_context.get("workflow_guardrails_hub_enabled") is False:
            raw_hub = _PayloadEvaluation(findings=[], skipped_validators=[])
            semantic_hub = _PayloadEvaluation(findings=[], skipped_validators=[])
        else:
            raw_hub = self._guardrails_ai_findings(
                workflow_type,
                stage,
                payload,
                raw_validators,
                {
                    **config_context,
                    "workflow_guardrails_native_tracing_enabled": False,
                },
            )
            semantic_hub = self._guardrails_ai_findings(
                workflow_type,
                stage,
                redaction.sanitized_payload,
                semantic_hub_validators,
                config_context,
            )
        signals = _dedupe_signals([*local_signals, *qwen_signals])
        findings = self._aggregate_findings(
            workflow_type,
            stage,
            payload,
            validators,
            signals,
            [*raw_hub.findings, *semantic_hub.findings],
            grounding_context=config_context.get("grounding_context") or [],
        )
        return _PayloadEvaluation(
            findings=findings,
            skipped_validators=[
                *raw_hub.skipped_validators,
                *semantic_hub.skipped_validators,
            ],
            signals=signals,
            redaction=redaction,
            degraded=qwen_degraded,
        )

    def _evaluate_provenance_payload(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        claim: str,
        validators: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> _PayloadEvaluation:
        local_signals = self._local_detector_signals(
            workflow_type,
            stage,
            claim,
            validators,
            grounding_context=context.get("grounding_context") or [],
        )
        qwen_signals, qwen_degraded = self._qwen_semantic_signals(
            workflow_type,
            stage,
            claim,
            validators,
            context,
        )
        hub = (
            _PayloadEvaluation(findings=[], skipped_validators=[])
            if context.get("workflow_guardrails_hub_enabled") is False
            else self._guardrails_ai_findings(
                workflow_type,
                stage,
                claim,
                validators,
                context,
            )
        )
        signals = _dedupe_signals([*local_signals, *qwen_signals])
        findings = self._aggregate_findings(
            workflow_type,
            stage,
            claim,
            validators,
            signals,
            hub.findings,
            grounding_context=context.get("grounding_context") or [],
        )
        return _PayloadEvaluation(
            findings=findings,
            skipped_validators=hub.skipped_validators,
            signals=signals,
            redaction=redact_payload(claim),
            degraded=qwen_degraded,
        )

    @staticmethod
    def _evaluation_metadata(
        evaluation: _PayloadEvaluation,
        redaction: RedactionResult,
    ) -> dict[str, Any]:
        return {
            "degraded": evaluation.degraded,
            "detector_versions": sorted(
                {signal.detector_version for signal in evaluation.signals}
            ),
            "redaction_complete": redaction.complete,
            "redaction_counts": redaction.redaction_counts,
            "signal_count": len(evaluation.signals),
            "skipped_validators": evaluation.skipped_validators,
        }

    def _local_detector_signals(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        payload: Any,
        validators: list[dict[str, Any]],
        *,
        grounding_context: list[str],
    ) -> list[DetectorSignal]:
        configured_ids = {str(item.get("id") or "") for item in validators}
        signals: list[DetectorSignal] = []
        if "secrets_present" in configured_ids:
            signals.extend(_secret_detector_signals(payload))
        if "detect_pii" in configured_ids:
            signals.extend(_pii_detector_signals(payload))

        semantic_text = _semantic_validation_text(workflow_type, payload)
        if "prompt_injection" in configured_ids:
            prompt_text = _prompt_injection_text(workflow_type, payload)
            signal = _prompt_injection_signal(prompt_text)
            if signal is not None:
                signals.append(signal)
        if "toxic_language" in configured_ids:
            signal = _toxicity_signal(semantic_text)
            if signal is not None:
                signals.append(signal)
        if "forbidden_terms" in configured_ids:
            signal = _forbidden_terms_signal(semantic_text, workflow_type)
            if signal is not None:
                signals.append(signal)
        if "provenance_llm" in configured_ids:
            signal = _missing_grounding_signal(semantic_text, grounding_context)
            if signal is not None:
                signals.append(signal)
        return _dedupe_signals(signals)

    def _qwen_semantic_signals(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        payload: Any,
        validators: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        action_type: str | None = None,
    ) -> tuple[list[DetectorSignal], bool]:
        qwen_config = self._qwen_config()
        if not qwen_config.get("enabled", False):
            return [], False
        if context.get("workflow_guardrails_semantic_enabled") is False:
            return [], False
        validator_ids = sorted(
            {
                str(item.get("id") or "")
                for item in validators
                if str(item.get("id") or "") in SEMANTIC_VALIDATOR_IDS
            }
        )
        semantic_text = _semantic_validation_text(workflow_type, payload)
        if not validator_ids or not semantic_text.strip():
            return [], False

        profile_name = str(
            context.get("workflow_guardrails_semantic_model")
            or qwen_config.get("profile")
            or "openrouter_qwen37"
        ).strip()
        runtime_context = _guardrails_runtime_context(context)
        profiles = runtime_context.get("llm_profiles")
        if not isinstance(profiles, dict) or profile_name not in profiles:
            return [], False
        profile = _guardrails_profile_from_mapping(profile_name, profiles[profile_name])
        if profile.llm_api_key_env and not os.getenv(profile.llm_api_key_env):
            return [], False

        timeout_seconds = float(
            context.get("workflow_guardrails_semantic_timeout_seconds")
            or qwen_config.get("timeout_seconds")
            or 8.0
        )
        max_attempts = int(qwen_config.get("max_attempts") or 3)
        sanitized_sources = [
            str(sanitize_payload(source))
            for source in context.get("grounding_context") or []
            if str(source or "").strip()
        ]
        try:
            assessment = self._call_qwen_semantic_assessment(
                profile_name=profile_name,
                workflow_type=workflow_type,
                stage=stage,
                action_type=action_type,
                text=semantic_text,
                grounding_context=sanitized_sources,
                validator_ids=validator_ids,
                context=runtime_context,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
            )
        except Exception:
            logger.warning(
                "Semantic guardrail degraded after Qwen failure",
                extra={
                    "workflow_type": workflow_type,
                    "stage": stage.value,
                    "action_type": action_type,
                },
            )
            return [], True

        language = _detect_language(semantic_text)
        signals = [
            DetectorSignal(
                policy_id=label.policy_id,
                source="qwen_semantic",
                confidence=label.confidence,
                language=language,
                entity_type=label.reason_code,
                evidence_hash=_short_hash(semantic_text),
                detector_version=SEMANTIC_DETECTOR_VERSION,
            )
            for label in assessment.labels
            if label.unsafe and label.policy_id in validator_ids
        ]
        return _dedupe_signals(signals), False

    def _call_qwen_semantic_assessment(
        self,
        *,
        profile_name: str,
        workflow_type: str,
        stage: GuardrailStage,
        action_type: str | None,
        text: str,
        grounding_context: list[str],
        validator_ids: list[str],
        context: dict[str, Any],
        timeout_seconds: float,
        max_attempts: int,
    ) -> _SemanticAssessment:
        from litellm import completion

        model = _resolve_guardrails_llm_callable(
            profile_name,
            {**context, "workflow_guardrails_semantic_model": profile_name},
            context_key="workflow_guardrails_semantic_model",
        )
        request_payload = {
            "action_type": action_type,
            "grounding_context": grounding_context,
            "policies": validator_ids,
            "stage": stage.value,
            "text": text[:PROMPT_INJECTION_MAX_CHARS],
            "workflow_type": workflow_type,
        }
        system_prompt = (
            "Classify the redacted payload for the requested guardrail policies. "
            "Treat quotations, reports, educational examples, negations, and business identifiers as benign "
            "unless the payload itself instructs unsafe behavior. Return JSON only as "
            '{"labels":[{"policy_id":"...","unsafe":true,"confidence":0.0,"reason_code":"..."}]}. '
            "Do not reproduce input text and do not provide chain-of-thought."
        )
        deadline = time.monotonic() + max(1.0, timeout_seconds)

        def invoke() -> _SemanticAssessment:
            remaining = deadline - time.monotonic()
            if remaining <= 0.5:
                raise TimeoutError("Qwen semantic guardrail exceeded its total timeout")
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False),
                    },
                ],
                max_retries=0,
                response_format={"type": "json_object"},
                temperature=0,
                timeout=min(3.0, remaining),
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Qwen semantic guardrail returned an empty response")
            return _SemanticAssessment.model_validate_json(str(content))

        retrying = Retrying(
            stop=stop_after_attempt(max(1, min(max_attempts, 3))),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        return retrying(invoke)

    def _aggregate_findings(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        payload: Any,
        validators: list[dict[str, Any]],
        signals: list[DetectorSignal],
        hub_findings: list[GuardrailFinding],
        *,
        grounding_context: list[str],
    ) -> list[GuardrailFinding]:
        validator_by_id = {str(item.get("id") or ""): item for item in validators}
        accepted_signals = [
            signal
            for signal in signals
            if signal.policy_id in validator_by_id
            and signal.confidence >= self._signal_threshold(signal)
        ]
        findings = [
            self._finding_from_signal(
                workflow_type,
                stage,
                validator_by_id[signal.policy_id],
                signal,
            )
            for signal in accepted_signals
        ]
        accepted_policy_ids = {signal.policy_id for signal in accepted_signals}
        for finding in hub_findings:
            if finding.metadata.get("runtime_error"):
                findings.append(finding)
                continue
            if finding.validator in RAW_LOCAL_VALIDATOR_IDS:
                findings.append(finding)
                continue
            if finding.validator == "provenance_llm":
                if _provenance_hub_finding_allowed(payload, grounding_context):
                    findings.append(finding)
                continue
            if finding.validator in accepted_policy_ids:
                continue
            # Semantic Hub validators are retained as telemetry signals only. A
            # calibrated local or Qwen signal is required to create a finding.
        return _dedupe_findings(findings)

    def _signal_threshold(self, signal: DetectorSignal) -> float:
        if signal.source == "local_rule":
            return 0.90
        pipeline = self.config.get("detector_pipeline")
        semantic = pipeline.get("semantic") if isinstance(pipeline, dict) else None
        thresholds = semantic.get("thresholds") if isinstance(semantic, dict) else None
        policy_threshold = (
            thresholds.get(signal.policy_id) if isinstance(thresholds, dict) else None
        )
        if isinstance(policy_threshold, dict):
            value = policy_threshold.get(
                signal.language, policy_threshold.get("default")
            )
        else:
            value = policy_threshold
        try:
            return float(value if value is not None else 0.85)
        except (TypeError, ValueError):
            return 0.85

    def _finding_from_signal(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        validator_config: dict[str, Any],
        signal: DetectorSignal,
    ) -> GuardrailFinding:
        severity = _severity_from_text(str(validator_config.get("severity") or "high"))
        if signal.policy_id == "detect_pii" and signal.entity_type == "payment_card":
            severity = GuardrailSeverity.HIGH
        entity = str(signal.entity_type or signal.policy_id).upper()
        return GuardrailFinding(
            policy_id=f"{workflow_type}.{stage.value}.{signal.policy_id}",
            validator=signal.policy_id,
            severity=severity,
            message=f"{signal.policy_id} detected by calibrated guardrail pipeline.",
            evidence_masked=f"[{entity}]"[:500],
            metadata={
                "confidence": signal.confidence,
                "detector_version": signal.detector_version,
                "evidence_hash": signal.evidence_hash,
                "language": signal.language,
                "source": signal.source,
            },
        )

    def _qwen_config(self) -> dict[str, Any]:
        pipeline = self.config.get("detector_pipeline")
        semantic = pipeline.get("semantic") if isinstance(pipeline, dict) else None
        qwen = semantic.get("qwen") if isinstance(semantic, dict) else None
        return _resolve_config_value(dict(qwen)) if isinstance(qwen, dict) else {}

    def _qwen_fail_closed_for_action(self, action_type: str) -> bool:
        configured = self._qwen_config().get("fail_closed_actions") or []
        return action_type in {str(item) for item in configured}

    def _guardrails_ai_findings(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        payload: Any,
        validators: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> _PayloadEvaluation:
        guardrails_validators = [item for item in validators if item.get("hub")]
        if not guardrails_validators:
            return _PayloadEvaluation(findings=[], skipped_validators=[])
        try:
            from guardrails import Guard
        except Exception as exc:
            raise GuardrailConfigurationError(
                "guardrails-ai is required for workflow guardrails"
            ) from exc
        _configure_guardrails_native_tracing(context)

        findings: list[GuardrailFinding] = []
        skipped_validators: list[dict[str, Any]] = []
        default_text = _payload_to_text(payload)
        for validator_config in guardrails_validators:
            validator_id = str(
                validator_config.get("id")
                or validator_config.get("hub")
                or "guardrails_ai"
            )
            skip_reason = self._skip_reason(validator_config, context)
            if skip_reason:
                skipped_validators.append(
                    {
                        "id": validator_id,
                        "hub": validator_config.get("hub"),
                        "reason": skip_reason,
                        "status": "not_applicable",
                    }
                )
                continue
            validation_text = self._validation_text(
                workflow_type,
                validator_id,
                payload,
                default_text,
            )
            cache_key = None
            if validator_id == PROMPT_INJECTION_VALIDATOR_ID:
                cache_key = self._prompt_injection_cache_key(
                    validation_text, validator_config, context
                )
                cached = self._read_prompt_injection_cache(cache_key, context)
                if cached is not None:
                    if not cached.validation_passed:
                        failure_reason = (
                            "\n".join(cached.failure_reasons)
                            or "Guardrails validation failed."
                        )
                        findings.append(
                            self._finding(
                                workflow_type,
                                stage,
                                validator_config,
                                validator_id,
                                failure_reason,
                                failure_reason,
                                metadata={
                                    "cache_hit": True,
                                    "hub": validator_config.get("hub"),
                                    "source": "guardrails_ai",
                                },
                            )
                        )
                    continue
            validator = self._build_hub_validator(validator_config, context)
            try:
                guard = Guard()
                guard.configure(allow_metrics_collection=False)
                outcome = guard.use(validator).validate(
                    validation_text,
                    metadata=self._metadata_for_validator(validator_config, context),
                )
            except Exception as exc:
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        validator_config,
                        validator_id,
                        f"Guardrails validator failed: {type(exc).__name__}",
                        type(exc).__name__,
                        metadata={
                            "runtime_error": True,
                            "runtime_error_type": type(exc).__name__,
                        },
                    )
                )
                continue
            validation_passed = bool(getattr(outcome, "validation_passed", False))
            failure_reasons = (
                [] if validation_passed else _validation_failure_reasons(outcome)
            )
            if cache_key:
                self._write_prompt_injection_cache(
                    cache_key,
                    _CachedValidationResult(
                        validation_passed=validation_passed,
                        failure_reasons=failure_reasons,
                    ),
                    context,
                )
            if not validation_passed:
                failure_reason = "\n".join(failure_reasons)
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        validator_config,
                        validator_id,
                        failure_reason or "Guardrails validation failed.",
                        failure_reason or "Guardrails validation failed.",
                        metadata={
                            "hub": validator_config.get("hub"),
                            "source": "guardrails_ai",
                        },
                    )
                )
        return _PayloadEvaluation(
            findings=_dedupe_findings(findings), skipped_validators=skipped_validators
        )

    def _build_hub_validator(
        self, validator_config: dict[str, Any], context: dict[str, Any] | None = None
    ) -> Any:
        validator_id = str(
            validator_config.get("id") or validator_config.get("hub") or ""
        )
        hub_uri = str(validator_config.get("hub") or validator_id)
        class_names = _class_names_for_validator(validator_config)
        if not class_names:
            raise GuardrailConfigurationError(
                f"Guardrails Hub validator class is not configured for '{hub_uri}'"
            )
        kwargs = self._validator_kwargs(validator_config, context or {})
        cache_key = _validator_cache_key(validator_config, kwargs)
        cached = self._validator_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            import guardrails.hub as hub
        except Exception as exc:
            raise GuardrailConfigurationError(
                "guardrails.hub is required for workflow guardrails"
            ) from exc
        klass = self._import_hub_validator(hub, hub_uri, class_names)
        try:
            validator = klass(**kwargs)
        except TypeError as exc:
            raise GuardrailConfigurationError(
                f"Guardrails Hub validator '{hub_uri}' could not be constructed with configured args"
            ) from exc
        if validator_id == PROMPT_INJECTION_VALIDATOR_ID:
            _configure_prompt_injection_transport(
                validator,
                _prompt_injection_timeout_seconds(context or {}),
            )
        self._validator_cache[cache_key] = validator
        return validator

    def _validation_text(
        self,
        workflow_type: str,
        validator_id: str,
        payload: Any,
        default_text: str,
    ) -> str:
        if validator_id != PROMPT_INJECTION_VALIDATOR_ID:
            return default_text
        return _prompt_injection_text(workflow_type, payload)

    def _prompt_injection_cache_key(
        self,
        text: str,
        validator_config: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        normalized = _normalize_prompt_injection_text(text)
        model = str(
            context.get("workflow_guardrails_prompt_injection_model")
            or os.getenv("WORKFLOW_GUARDRAILS_PROMPT_INJECTION_MODEL")
            or "openai_gpt4o_mini"
        )
        threshold = str((validator_config.get("args") or {}).get("threshold", "0.8"))
        digest = hashlib.sha256(
            f"{model}\n{threshold}\n{normalized}".encode("utf-8", errors="ignore")
        ).hexdigest()
        return f"{PROMPT_INJECTION_CACHE_PREFIX}:{digest}"

    def _read_prompt_injection_cache(
        self,
        cache_key: str,
        context: dict[str, Any],
    ) -> _CachedValidationResult | None:
        client = self._prompt_injection_redis_client(context)
        if client is None:
            return None
        try:
            raw = client.get(cache_key)
            if not raw:
                return None
            return _CachedValidationResult.model_validate_json(raw)
        except Exception:
            logger.debug("Prompt-injection cache read failed", exc_info=True)
            return None

    def _write_prompt_injection_cache(
        self,
        cache_key: str,
        result: _CachedValidationResult,
        context: dict[str, Any],
    ) -> None:
        client = self._prompt_injection_redis_client(context)
        if client is None:
            return
        ttl_seconds = _prompt_injection_cache_ttl_seconds(context)
        if ttl_seconds <= 0:
            return
        try:
            client.setex(cache_key, ttl_seconds, result.model_dump_json())
        except Exception:
            logger.debug("Prompt-injection cache write failed", exc_info=True)

    def _prompt_injection_redis_client(self, context: dict[str, Any]) -> Any | None:
        redis_url = str(
            context.get("tool_cache_redis_url")
            or os.getenv("TOOL_CACHE_REDIS_URL")
            or os.getenv("CELERY_BROKER_URL")
            or os.getenv("REDIS_URL")
            or ""
        ).strip()
        if not redis_url:
            return None
        cached = self._redis_clients.get(redis_url)
        if cached is not None:
            return cached
        try:
            from redis import Redis

            client = Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
        except Exception:
            logger.debug("Prompt-injection Redis cache is unavailable", exc_info=True)
            return None
        self._redis_clients[redis_url] = client
        return client

    def _import_hub_validator(
        self, hub: Any, validator_id: str, class_names: tuple[str, ...]
    ) -> Any:
        for name in class_names:
            try:
                candidate = getattr(hub, name, None)
            except Exception as exc:
                raise GuardrailConfigurationError(
                    f"Guardrails Hub validator '{validator_id}' could not be imported from registry"
                ) from exc
            if candidate is not None:
                return candidate
            for module_name in _module_names_for_validator(validator_id):
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    continue
                except Exception as exc:
                    raise GuardrailConfigurationError(
                        f"Guardrails Hub validator '{validator_id}' could not be imported from package"
                    ) from exc
                candidate = getattr(module, name, None)
                if candidate is not None:
                    return candidate
        install_hint = (
            f"guardrails hub install {validator_id}"
            if validator_id.startswith("hub://")
            else "guardrails hub list"
        )
        raise GuardrailConfigurationError(
            f"Guardrails Hub validator '{validator_id}' is not installed or does not expose {class_names}. "
            f"Run: {install_hint}"
        )

    def _validator_kwargs(
        self, validator_config: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        defaults = (
            self.config.get("defaults")
            if isinstance(self.config.get("defaults"), dict)
            else {}
        )
        validator_id = str(validator_config.get("id") or "")
        kwargs = dict(validator_config.get("args") or {})
        for key in (
            "llm_callable",
            "match_type",
            "max_tokens",
            "regex",
            "threshold",
            "top_k",
            "validation_method",
        ):
            if key in validator_config:
                kwargs.setdefault(key, validator_config[key])
        if validator_id == "detect_pii" and "entities" in validator_config:
            kwargs.setdefault(
                "pii_entities", list(validator_config.get("entities") or [])
            )
        if "forbidden_terms" in validator_config:
            kwargs.setdefault(
                "regex",
                _negative_regex_from_terms(
                    list(validator_config.get("forbidden_terms") or [])
                ),
            )
            kwargs.setdefault("match_type", "fullmatch")
        kwargs.setdefault(
            "on_fail",
            str(validator_config.get("on_fail") or defaults.get("on_fail") or "noop"),
        )
        kwargs = _resolve_config_value(kwargs)
        if "llm_callable" in kwargs:
            context_key = (
                "workflow_guardrails_prompt_injection_model"
                if validator_id == PROMPT_INJECTION_VALIDATOR_ID
                else "workflow_guardrails_model"
            )
            kwargs["llm_callable"] = _resolve_guardrails_llm_callable(
                str(kwargs["llm_callable"]),
                context or {},
                context_key=context_key,
            )
        return kwargs

    def _metadata_for_validator(
        self, validator_config: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        if validator_config.get("id") == "provenance_llm":
            query_function = context.get("query_function")
            if query_function is not None:
                return {"query_function": query_function}
            sources = context.get("grounding_context") or []
            embed_function = (
                context.get("embed_function") or self._default_embed_function()
            )
            metadata = {"embed_function": embed_function, "sources": sources}
            if "pass_on_invalid" in validator_config:
                metadata["pass_on_invalid"] = bool(
                    validator_config.get("pass_on_invalid")
                )
            return metadata
        return _resolve_config_value(dict(validator_config.get("metadata") or {}))

    def _default_embed_function(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise GuardrailConfigurationError(
                "ProvenanceLLM requires sentence-transformers when no query_function or embed_function is provided. "
                "Run: .\\.venv\\Scripts\\python.exe -m pip install sentence-transformers"
            ) from exc
        if self._embed_model is None:
            model_name = os.getenv(
                "WORKFLOW_GUARDRAILS_EMBED_MODEL", "paraphrase-MiniLM-L6-v2"
            )
            self._embed_model = SentenceTransformer(model_name)

        def embed(value: str | list[str]) -> Any:
            return self._embed_model.encode(value, normalize_embeddings=True)

        return embed

    def _skip_reason(
        self, validator_config: dict[str, Any], context: dict[str, Any]
    ) -> str | None:
        if validator_config.get(
            "enabled_when"
        ) == "grounding_context_available" and not context.get("grounding_context"):
            return "grounding_context_unavailable"
        return None

    def _policy_action(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        findings: list[GuardrailFinding],
        *,
        action_type: str | None = None,
    ) -> GuardrailAction:
        if not findings:
            return GuardrailAction.ALLOW
        if self._mode() == "off":
            return GuardrailAction.ALLOW
        if stage == GuardrailStage.ACTION:
            return self._action_policy_action(workflow_type, findings, action_type)
        action = (
            GuardrailAction.MONITOR
            if self._mode() == "monitor"
            else GuardrailAction.REVIEW_REQUIRED
        )
        stage_config = self._stage_config(workflow_type, stage)
        validators = (
            list(stage_config.get("validators") or [])
            if isinstance(stage_config, dict)
            else []
        )
        policy_by_id = {
            str(item.get("id") or item.get("builtin") or ""): item
            for item in validators
        }
        for finding in findings:
            validator_config = policy_by_id.get(finding.validator)
            if finding.metadata.get("runtime_error"):
                configured = GuardrailAction.REVIEW_REQUIRED
            else:
                configured = _action_from_text(
                    str((validator_config or {}).get("policy") or "monitor")
                )
            action = _max_action(action, configured)
        return action

    def _action_policy_action(
        self,
        workflow_type: str,
        findings: list[GuardrailFinding],
        action_type: str | None,
    ) -> GuardrailAction:
        stage_config = self._stage_config(workflow_type, GuardrailStage.ACTION)
        action_config = (
            stage_config.get(action_type or "")
            if isinstance(stage_config, dict)
            else None
        )
        if not isinstance(action_config, dict):
            severity = _max_severity(findings)
            if severity in {GuardrailSeverity.HIGH, GuardrailSeverity.CRITICAL}:
                return GuardrailAction.BLOCK
            return GuardrailAction.REVIEW_REQUIRED
        policy_actions = (
            action_config.get("policies")
            if isinstance(action_config.get("policies"), dict)
            else {}
        )
        selected = GuardrailAction.MONITOR
        for finding in findings:
            configured = _action_from_text(
                str(
                    policy_actions.get(finding.validator)
                    or action_config.get(finding.severity.value)
                    or action_config.get("high")
                    or "monitor"
                )
            )
            selected = _max_action(selected, configured)
        return selected

    def _decision(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        action: GuardrailAction,
        findings: list[GuardrailFinding],
        *,
        sanitized_payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        severity = _max_severity(findings)
        return GuardrailDecision(
            workflow_type=workflow_type,
            stage=stage,
            action=action,
            severity=severity,
            findings=findings,
            sanitized_payload=sanitized_payload,
            human_override_allowed=action != GuardrailAction.BLOCK,
            metadata={
                "mode": self._mode(),
                "config_version": self.config.get("config_version"),
                **(metadata or {}),
            },
        )

    def _finding(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        validator_config: dict[str, Any],
        validator: str,
        message: str,
        evidence: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> GuardrailFinding:
        severity = _severity_from_text(str(validator_config.get("severity") or "high"))
        return GuardrailFinding(
            policy_id=f"{workflow_type}.{stage.value}.{validator}",
            validator=validator,
            severity=severity,
            message=mask_text(message)[:500],
            evidence_masked=mask_text(str(evidence))[:500],
            metadata={
                "evidence_hash": _short_hash(str(evidence)),
                **(metadata or {}),
            },
        )

    def _validators_for(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        action_type: str | None = None,
        *,
        execution: str = "sync",
    ) -> list[dict[str, Any]]:
        stage_config = self._stage_config(workflow_type, stage)
        if stage == GuardrailStage.ACTION:
            output_config = self._stage_config(workflow_type, GuardrailStage.OUTPUT)
            validators = (
                list(output_config.get("validators") or [])
                if isinstance(output_config, dict)
                else []
            )
            return [
                item
                for item in validators
                if _validator_execution(item) == "sync"
                and _action_from_text(str(item.get("policy") or "monitor"))
                in {GuardrailAction.REVIEW_REQUIRED, GuardrailAction.BLOCK}
            ]
        validators = (
            list(stage_config.get("validators") or [])
            if isinstance(stage_config, dict)
            else []
        )
        return [item for item in validators if _validator_execution(item) == execution]

    def _stage_config(
        self, workflow_type: str, stage: GuardrailStage
    ) -> dict[str, Any]:
        guards = self.config.get("guards") or {}
        workflow_config = guards.get(workflow_type) or {}
        stage_config = workflow_config.get(stage.value) or {}
        return stage_config if isinstance(stage_config, dict) else {}

    def validate_runtime(self, *, smoke_toxic: bool = False) -> None:
        try:
            from guardrails import Guard
        except Exception as exc:
            raise GuardrailConfigurationError(
                "guardrails-ai is required for workflow guardrails"
            ) from exc
        for validator_config in self._configured_validator_configs():
            validator = self._build_hub_validator(validator_config)
            if smoke_toxic and validator_config.get("id") == "toxic_language":
                try:
                    guard = Guard()
                    guard.configure(allow_metrics_collection=False)
                    guard.use(validator).validate(
                        "Please keep the reply professional and respectful."
                    )
                except Exception as exc:
                    raise GuardrailConfigurationError(
                        "hub://guardrails/toxic_language installed but failed smoke validation. "
                        "Try reinstalling CPU torch and the validator, then rerun this script."
                    ) from exc

    def _validate_config(self) -> None:
        if self._mode() == "off":
            return
        for validator_config in self._configured_validator_configs():
            validator_id = str(
                validator_config.get("id") or validator_config.get("hub") or ""
            )
            if validator_config.get("builtin"):
                raise GuardrailConfigurationError(
                    f"Workflow guardrail '{validator_id}' uses builtin detection. Use a hub:// validator instead."
                )
            if not validator_config.get("hub"):
                raise GuardrailConfigurationError(
                    f"Workflow guardrail '{validator_id}' must configure a hub:// validator"
                )
            if validator_config.get("enable_ai_runtime") is False:
                raise GuardrailConfigurationError(
                    f"Workflow guardrail '{validator_id}' disables Hub runtime. Install the validator instead."
                )
            if _validator_execution(validator_config) not in {"sync", "async"}:
                raise GuardrailConfigurationError(
                    f"Workflow guardrail '{validator_id}' has invalid execution mode. Use sync or async."
                )
            if str(validator_config.get("id") or "") == "forbidden_terms":
                args = (
                    validator_config.get("args")
                    if isinstance(validator_config.get("args"), dict)
                    else {}
                )
                if not args.get("regex") and not validator_config.get(
                    "forbidden_terms"
                ):
                    raise GuardrailConfigurationError(
                        "forbidden_terms must configure regex or forbidden_terms"
                    )

    def _configured_validator_configs(self) -> list[dict[str, Any]]:
        validators: list[dict[str, Any]] = []
        guards = (
            self.config.get("guards")
            if isinstance(self.config.get("guards"), dict)
            else {}
        )
        for workflow_config in guards.values():
            if not isinstance(workflow_config, dict):
                continue
            for stage_name in (GuardrailStage.INPUT.value, GuardrailStage.OUTPUT.value):
                stage_config = workflow_config.get(stage_name)
                if not isinstance(stage_config, dict):
                    continue
                for item in stage_config.get("validators") or []:
                    if isinstance(item, dict):
                        validators.append(item)
        return validators

    def _mode(self) -> str:
        return str(self.config.get("mode") or "monitor").lower()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Guardrail config must be an object: {path}")
        return raw


def _guardrail_payload_from_result(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    direct = payload.get("guardrail_decision")
    if isinstance(direct, dict):
        return direct
    draft = payload.get("draft_payload")
    if isinstance(draft, dict) and isinstance(draft.get("guardrail_decision"), dict):
        return draft["guardrail_decision"]
    return None


def _workflow_value(workflow_type: WorkflowType | str) -> str:
    return (
        workflow_type.value
        if isinstance(workflow_type, WorkflowType)
        else str(workflow_type)
    )


def _record_guardrail_observability(
    decision: GuardrailDecision,
    config_context: dict[str, Any] | None,
    conversation_id: str | None,
) -> None:
    set_span_attributes(
        {
            "conversation_id": conversation_id,
            "guardrail_action": decision.action.value,
            "guardrail_severity": decision.severity.value,
            "guardrail_stage": decision.stage.value,
            "guardrail_finding_count": len(decision.findings),
        },
        config_context=config_context,
    )


def _conversation_trace_id(
    payload: Any, context: dict[str, Any] | None = None
) -> str | None:
    for source in (payload, context, (context or {}).get("metadata")):
        if not isinstance(source, dict):
            continue
        for key in ("conversation_id", "session_id", "ticket_id"):
            trace_id = _safe_trace_identifier(source.get(key))
            if trace_id:
                return trace_id
    return None


def _safe_trace_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        uuid.UUID(text)
        return text
    except ValueError:
        pass
    if EMAIL_ADDRESS_REDACTOR.search(text) or PHONE_NUMBER_REDACTOR.search(text):
        return None
    if not re.fullmatch(r"[\w:.-]{1,128}", text):
        return None
    return text


def _is_preserved_identifier_field(key: str) -> bool:
    normalized = str(key or "").lower()
    return normalized in PRESERVED_IDENTIFIER_FIELDS or normalized.endswith("_id")


def _is_sensitive_key(key: str) -> bool:
    return bool(
        re.search(
            r"(api[_-]?key|authorization|bearer|client[_-]?secret|credential|password|refresh[_-]?token|secret|token)",
            key,
            re.IGNORECASE,
        )
    )


def _mask_sensitive_value(value: Any) -> str:
    return "[SECRET]" if value not in (None, "") else ""


def _mask_email(value: str) -> str:
    local, _, domain = value.partition("@")
    return f"{local[:2]}***@{domain}" if domain else "[EMAIL]"


def _mask_phone(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) <= 4:
        return "[PHONE]"
    return f"[PHONE:{digits[-4:]}]"


def _severity_from_text(value: str) -> GuardrailSeverity:
    normalized = value.lower()
    if normalized in GuardrailSeverity._value2member_map_:
        return GuardrailSeverity(normalized)
    return GuardrailSeverity.HIGH


def _action_from_text(value: str) -> GuardrailAction:
    normalized = value.lower()
    if normalized in GuardrailAction._value2member_map_:
        return GuardrailAction(normalized)
    return GuardrailAction.MONITOR


def _validator_execution(validator_config: dict[str, Any]) -> str:
    return str(validator_config.get("execution") or "sync").strip().lower()


def _max_severity(findings: list[GuardrailFinding]) -> GuardrailSeverity:
    if not findings:
        return GuardrailSeverity.NONE
    return max(
        (finding.severity for finding in findings),
        key=lambda item: SEVERITY_RANK[item.value],
    )


def _max_action(left: GuardrailAction, right: GuardrailAction) -> GuardrailAction:
    return left if ACTION_RANK[left] >= ACTION_RANK[right] else right


def _dedupe_findings(findings: list[GuardrailFinding]) -> list[GuardrailFinding]:
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[GuardrailFinding] = []
    for finding in findings:
        key = (finding.policy_id, finding.validator, finding.evidence_masked)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _dedupe_signals(signals: list[DetectorSignal]) -> list[DetectorSignal]:
    selected: dict[tuple[str, str, str | None, str], DetectorSignal] = {}
    for signal in signals:
        key = (
            signal.policy_id,
            signal.source,
            signal.entity_type,
            signal.evidence_hash,
        )
        current = selected.get(key)
        if current is None or signal.confidence > current.confidence:
            selected[key] = signal
    return list(selected.values())


def _walk_payload_strings(
    payload: Any,
    field_name: str | None = None,
) -> list[tuple[str | None, str]]:
    if isinstance(payload, str):
        return [(field_name, payload)]
    if isinstance(payload, dict):
        values: list[tuple[str | None, str]] = []
        for key, value in payload.items():
            values.extend(_walk_payload_strings(value, str(key)))
        return values
    if isinstance(payload, (list, tuple)):
        values = []
        for item in payload:
            values.extend(_walk_payload_strings(item, field_name))
        return values
    return []


def _secret_detector_signals(payload: Any) -> list[DetectorSignal]:
    signals: list[DetectorSignal] = []
    for field_name, text in _walk_payload_strings(payload):
        normalized = text.strip()
        if not normalized or normalized in {"[SECRET]", "[REDACTED]"}:
            continue
        field_secret = bool(field_name and _is_sensitive_key(field_name))
        placeholder = bool(SECRET_PLACEHOLDER_RE.fullmatch(normalized))
        redacted = _mask_secrets(text)
        if placeholder or (redacted == text and not field_secret):
            continue
        signals.append(
            DetectorSignal(
                policy_id="secrets_present",
                source="local_rule",
                confidence=0.99,
                language=_detect_language(text),
                entity_type="secret",
                evidence_hash=_short_hash(text),
                detector_version=LOCAL_DETECTOR_VERSION,
            )
        )
    return _dedupe_signals(signals)


def _pii_detector_signals(payload: Any) -> list[DetectorSignal]:
    signals: list[DetectorSignal] = []
    for field_name, text in _walk_payload_strings(payload):
        normalized_field = str(field_name or "").lower()
        if _is_preserved_identifier_field(normalized_field):
            continue
        pii_text = _mask_secrets(text)
        entity_types: set[str] = set()
        if normalized_field in ADDRESS_FIELDS and pii_text.strip() not in {
            "",
            "[ADDRESS]",
        }:
            entity_types.add("address")
        if EMAIL_ADDRESS_REDACTOR.search(pii_text):
            entity_types.add("email")
        if _contains_valid_ip(pii_text):
            entity_types.add("ip_address")
        if any(
            _passes_luhn(
                "".join(
                    character for character in match.group(0) if character.isdigit()
                )
            )
            and not _span_is_preserved_business_identifier(pii_text, match.span(0))
            for match in PAYMENT_CARD_CANDIDATE_RE.finditer(pii_text)
        ):
            entity_types.add("payment_card")
        if _contains_unmasked_phone(pii_text):
            entity_types.add("phone")
        for entity_type in entity_types:
            signals.append(
                DetectorSignal(
                    policy_id="detect_pii",
                    source="local_rule",
                    confidence=0.98,
                    language=_detect_language(text),
                    entity_type=entity_type,
                    evidence_hash=_short_hash(f"{entity_type}:{text}"),
                    detector_version=LOCAL_DETECTOR_VERSION,
                )
            )
    return _dedupe_signals(signals)


def _semantic_validation_text(workflow_type: str, payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    values = [
        value for _field, value in _walk_payload_strings(payload) if value.strip()
    ]
    return "\n".join(values)[:PROMPT_INJECTION_MAX_CHARS]


def _prompt_injection_signal(text: str) -> DetectorSignal | None:
    if not text.strip() or BENIGN_CONTEXT_RE.search(text):
        return None
    candidates = [text, _normalize_obfuscated_instruction(text)]
    candidates.extend(_decoded_instruction_candidates(text))
    unsafe = any(
        PROMPT_ROLE_INJECTION_RE.search(candidate)
        or (
            PROMPT_ATTACK_ACTION_RE.search(candidate)
            and PROMPT_CONTROL_TARGET_RE.search(candidate)
        )
        for candidate in candidates
    )
    if not unsafe:
        return None
    return DetectorSignal(
        policy_id="prompt_injection",
        source="local_rule",
        confidence=0.97,
        language=_detect_language(text),
        entity_type="instruction_override",
        evidence_hash=_short_hash(text),
        detector_version=LOCAL_DETECTOR_VERSION,
    )


def _normalize_obfuscated_instruction(text: str) -> str:
    translated = str(text).translate(str.maketrans({"0": "o", "1": "i"}))
    translated = re.sub(
        r"(?i)\bi[.\s_-]*g[.\s_-]*n[.\s_-]*o[.\s_-]*r[.\s_-]*e\b", "ignore", translated
    )
    return " ".join(translated.split())


def _decoded_instruction_candidates(text: str) -> list[str]:
    decoded_values: list[str] = []
    for match in BASE64_CANDIDATE_RE.finditer(text):
        candidate = match.group(0)
        try:
            padded = candidate + "=" * (-len(candidate) % 4)
            decoded = base64.b64decode(padded, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if decoded.strip():
            decoded_values.append(_normalize_obfuscated_instruction(decoded))
    return decoded_values


def _toxicity_signal(text: str) -> DetectorSignal | None:
    if not text.strip() or TOXIC_REPORTING_RE.search(text):
        return None
    if not TOXIC_TERM_RE.search(text):
        return None
    return DetectorSignal(
        policy_id="toxic_language",
        source="local_rule",
        confidence=0.97,
        language=_detect_language(text),
        entity_type="abusive_language",
        evidence_hash=_short_hash(text),
        detector_version=LOCAL_DETECTOR_VERSION,
    )


def _forbidden_terms_signal(text: str, workflow_type: str) -> DetectorSignal | None:
    if not text.strip() or _forbidden_hard_negative(text):
        return None
    support_patterns = (
        r"(?is)\binternal\s+discount\s+code\b",
        r"(?is)\bguaranteed\s+(?:a\s+)?refund\s+without\s+approval\b",
        r"(?:仅供员工.*促销(?:标识|代码)|没有(?:经理)?审核.*(?:一定|保证).*退款)",
    )
    content_patterns = (
        r"(?is)\bunverified\s+medical\s+cure\b",
        r"(?is)\bguaranteed\s+income\b",
        r"(?:没有证据.*(?:治疗|治愈).*(?:所有|患者)|确保.*每个人.*(?:赚钱|收入))",
    )
    patterns = (
        support_patterns
        if workflow_type == WorkflowType.SUPPORT.value
        else content_patterns
    )
    if not any(re.search(pattern, text) for pattern in patterns):
        return None
    return DetectorSignal(
        policy_id="forbidden_terms",
        source="local_rule",
        confidence=0.98,
        language=_detect_language(text),
        entity_type="restricted_claim",
        evidence_hash=_short_hash(text),
        detector_version=LOCAL_DETECTOR_VERSION,
    )


def _forbidden_hard_negative(text: str) -> bool:
    return bool(
        re.search(
            r"(?is)(?:cannot\s+provide\s+the\s+internal\s+discount\s+code|"
            r"explain\s+why.*dangerous|no\s+outcome\s+is\s+guaranteed|"
            r"not\s+guaranteed\s+without\s+approval|never\s+promise|"
            r"without\s+cure\s+claims?|refund\s+eligibility\s+requires\s+approval|"
            r"不保证|不能提供|未经审核不能|没有保证)",
            text,
        )
    )


def _missing_grounding_signal(
    claim: str,
    grounding_context: list[str],
) -> DetectorSignal | None:
    if (
        not claim.strip()
        or not GROUNDING_REQUIRED_RE.search(claim)
        or any(str(source or "").strip() for source in grounding_context)
    ):
        return None
    return DetectorSignal(
        policy_id="provenance_llm",
        source="local_rule",
        confidence=0.99,
        language=_detect_language(claim),
        entity_type="missing_grounding",
        evidence_hash=_short_hash(claim),
        detector_version=LOCAL_DETECTOR_VERSION,
    )


def _provenance_hub_finding_allowed(payload: Any, grounding_context: list[str]) -> bool:
    claim = _semantic_validation_text("", payload).strip()
    sources = [
        str(source or "").strip()
        for source in grounding_context
        if str(source or "").strip()
    ]
    if not claim:
        return False
    if not sources:
        return True
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    source_number_sets = [
        set(re.findall(r"\b\d+(?:\.\d+)?\b", source)) for source in sources
    ]
    if (
        claim_numbers
        and source_number_sets
        and all(numbers == claim_numbers for numbers in source_number_sets)
    ):
        return False
    return True


def _detect_language(text: str) -> str:
    return "zh" if re.search(r"[\u3400-\u9fff]", text) else "en"


def _class_names_for_validator(validator_config: dict[str, Any]) -> tuple[str, ...]:
    configured = validator_config.get("class_name")
    if configured:
        return (str(configured),)
    hub_uri = str(validator_config.get("hub") or "")
    return HUB_VALIDATOR_CLASS_NAMES.get(hub_uri, ())


def _module_names_for_validator(validator_id: str) -> tuple[str, ...]:
    if not validator_id.startswith("hub://"):
        return ()
    parts = validator_id.removeprefix("hub://").split("/")
    owner = parts[-2].replace("-", "_") if len(parts) >= 2 else "guardrails"
    package_name = parts[-1].replace("-", "_")
    module_names = (f"{owner}_grhub_{package_name}", f"guardrails_grhub_{package_name}")
    if validator_id == "hub://guardrails/toxic_language":
        return (*module_names, "validator")
    return module_names


def _validator_cache_key(
    validator_config: dict[str, Any], kwargs: dict[str, Any]
) -> str:
    payload = {
        "class_name": validator_config.get("class_name"),
        "hub": validator_config.get("hub"),
        "id": validator_config.get("id"),
        "kwargs": _json_safe(kwargs),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _negative_regex_from_terms(terms: list[Any]) -> str:
    escaped_terms = [re.escape(str(term)) for term in terms if str(term)]
    if not escaped_terms:
        raise GuardrailConfigurationError(
            "forbidden_terms must contain at least one non-empty term"
        )
    return rf"(?is)^(?!.*(?:{'|'.join(escaped_terms)})).*$"


def _prompt_injection_text(workflow_type: str, payload: Any) -> str:
    if isinstance(payload, str):
        return _normalize_prompt_injection_text(payload)[:PROMPT_INJECTION_MAX_CHARS]
    if not isinstance(payload, dict):
        return ""

    values: list[str] = []
    if workflow_type == WorkflowType.SUPPORT.value:
        for key in SUPPORT_PROMPT_TEXT_KEYS:
            value = payload.get(key)
            if value not in (None, ""):
                values.append(str(value))
        history = payload.get("conversation_history") or payload.get("history")
        if isinstance(history, list):
            for item in history[-4:]:
                if not isinstance(item, dict):
                    continue
                direction = str(item.get("direction") or item.get("role") or "").lower()
                if direction not in {"", "inbound", "user"}:
                    continue
                value = item.get("text") or item.get("content")
                if value not in (None, ""):
                    values.append(str(value))
    elif workflow_type == WorkflowType.CONTENT.value:
        for key in CONTENT_PROMPT_TEXT_KEYS:
            value = payload.get(key)
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                values.extend(str(item) for item in value if str(item).strip())
            else:
                values.append(str(value))
    else:
        for key in ("prompt", "message", "text", "subject"):
            value = payload.get(key)
            if value not in (None, ""):
                values.append(str(value))

    return _normalize_prompt_injection_text("\n".join(values))[
        :PROMPT_INJECTION_MAX_CHARS
    ]


def _normalize_prompt_injection_text(text: str) -> str:
    return " ".join(str(text).split())


def _prompt_injection_timeout_seconds(context: dict[str, Any]) -> float:
    value = context.get("workflow_guardrails_prompt_injection_timeout_seconds")
    if value in (None, ""):
        value = os.getenv("WORKFLOW_GUARDRAILS_PROMPT_INJECTION_TIMEOUT_SECONDS", "5")
    try:
        return max(0.1, float(value))
    except (TypeError, ValueError):
        return 5.0


def _prompt_injection_cache_ttl_seconds(context: dict[str, Any]) -> int:
    value = context.get("workflow_guardrails_prompt_injection_cache_ttl_seconds")
    if value in (None, ""):
        value = os.getenv(
            "WORKFLOW_GUARDRAILS_PROMPT_INJECTION_CACHE_TTL_SECONDS", "86400"
        )
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 86400


def _configure_prompt_injection_transport(
    validator: Any, timeout_seconds: float
) -> None:
    if getattr(validator, "_workflow_timeout_seconds", None) == timeout_seconds:
        return

    def get_llm_response(instance: Any, prompt: str) -> str:
        from guardrails.stores.context import get_call_kwarg
        from litellm import completion, get_llm_provider

        kwargs: dict[str, Any] = {
            "max_retries": 0,
            "timeout": timeout_seconds,
        }
        _model, provider, *_rest = get_llm_provider(instance.llm_callable)
        if provider == "openai":
            kwargs["api_key"] = get_call_kwarg("api_key") or os.environ.get(
                "OPENAI_API_KEY"
            )
        try:
            response = completion(
                model=instance.llm_callable,
                messages=[{"content": prompt, "role": "user"}],
                **kwargs,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError(
                    "Prompt injection evaluator returned an empty response"
                )
            return str(content).strip(" .").lower().strip()
        except Exception as exc:
            raise RuntimeError("Prompt injection evaluator request failed") from exc

    validator.get_llm_response = MethodType(get_llm_response, validator)
    validator._workflow_timeout_seconds = timeout_seconds


def _configure_guardrails_native_tracing(context: dict[str, Any]) -> None:
    """Control Guardrails' local OTel spans without enabling Hub metrics collection."""
    try:
        from guardrails.settings import settings

        value = context.get("workflow_guardrails_native_tracing_enabled")
        if value in (None, ""):
            value = os.getenv("WORKFLOW_GUARDRAILS_NATIVE_TRACING_ENABLED", "false")
        enabled = (
            value
            if isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true", "yes", "on"}
        )
        settings.disable_tracing = not enabled
    except (ImportError, ModuleNotFoundError):
        logger.debug("Guardrails tracing settings are unavailable")


def _resolve_guardrails_llm_callable(
    profile_name: str,
    context: dict[str, Any],
    *,
    context_key: str = "workflow_guardrails_model",
) -> str:
    runtime_context = _guardrails_runtime_context(context)
    selected_profile = str(runtime_context.get(context_key) or profile_name).strip()
    profiles = (
        runtime_context.get("llm_profiles")
        if isinstance(runtime_context.get("llm_profiles"), dict)
        else {}
    )
    if selected_profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        env_name = context_key.upper()
        raise GuardrailConfigurationError(
            f"{env_name} references unknown LLM profile '{selected_profile}'. "
            f"Available profiles: {available}."
        )
    profile = _guardrails_profile_from_mapping(
        selected_profile, profiles[selected_profile]
    )
    _configure_litellm_profile_env(selected_profile, profile)
    model_name = profile.llm_model_name
    if profile.llm_provider == "openrouter" and not model_name.startswith(
        "openrouter/"
    ):
        return f"openrouter/{model_name}"
    return model_name


def _guardrails_runtime_context(context: dict[str, Any]) -> dict[str, Any]:
    if isinstance(context.get("llm_profiles"), dict) and context.get("llm_profiles"):
        return context
    try:
        runtime_context = load_runtime_config().as_context()
    except Exception as exc:
        raise GuardrailConfigurationError(
            "Unable to load LLM profiles for workflow guardrails"
        ) from exc
    return {**runtime_context, **context}


def _guardrails_profile_from_mapping(profile_name: str, value: Any) -> LLMProfileConfig:
    try:
        if isinstance(value, LLMProfileConfig):
            return value
        if isinstance(value, dict):
            return LLMProfileConfig.model_validate(value)
    except Exception as exc:
        raise GuardrailConfigurationError(
            f"WORKFLOW_GUARDRAILS_MODEL profile '{profile_name}' has invalid LLM profile configuration"
        ) from exc
    raise GuardrailConfigurationError(
        f"WORKFLOW_GUARDRAILS_MODEL profile '{profile_name}' has invalid LLM profile configuration"
    )


def _configure_litellm_profile_env(
    profile_name: str, profile: LLMProfileConfig
) -> None:
    if not profile.llm_api_key_env:
        return
    api_key = os.getenv(profile.llm_api_key_env)
    if not api_key:
        raise GuardrailConfigurationError(
            f"WORKFLOW_GUARDRAILS_MODEL profile '{profile_name}' references missing or empty environment variable "
            f"'{profile.llm_api_key_env}'."
        )
    if profile.llm_provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif profile.llm_provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = api_key


def _resolve_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _resolve_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config_value(item) for item in value]
    if not isinstance(value, str):
        return value
    match = ENV_VALUE_RE.match(value)
    if not match:
        return value
    env_name = match.group(1)
    default = match.group(2) if match.group(2) is not None else ""
    return os.getenv(env_name, default)


def _validation_failure_reasons(outcome: Any) -> list[str]:
    reasons: list[str] = []
    for summary in getattr(outcome, "validation_summaries", None) or []:
        if isinstance(summary, dict):
            status = summary.get("validator_status")
            reason = summary.get("failure_reason")
        else:
            status = getattr(summary, "validator_status", None)
            reason = getattr(summary, "failure_reason", None)
        if str(status or "").lower() == "fail" and reason:
            reasons.append(str(reason))
    if reasons:
        return reasons
    error = getattr(outcome, "error", None)
    if error:
        return [str(error)]
    return ["Guardrails validation failed."]
