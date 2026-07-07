from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from models import WorkflowType

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "guardrails.yaml"
MODEL_CACHE_DIR = BASE_DIR / ".cache"
GUARDRAIL_REVIEW_FLAG = "GUARDRAIL_REVIEW_REQUIRED"
GUARDRAIL_HIGH_RISK_FLAG = "GUARDRAIL_HIGH_RISK"

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bsk-lf-[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|secret|password|token|access[_-]?token|refresh[_-]?token)\s*[:=]\s*['\"]?[^'\"\s,;]{8,}",
        re.IGNORECASE,
    ),
    re.compile(r"\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore (?:all )?(?:previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\breveal (?:the )?(?:system|developer) prompt\b", re.IGNORECASE),
    re.compile(r"\bexfiltrate\b.*\b(?:secret|token|credential|api key)\b", re.IGNORECASE),
    re.compile(r"\bbypass\b.*\b(?:guardrail|policy|safety|permission)\b", re.IGNORECASE),
)
TOXIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bkill yourself\b", re.IGNORECASE),
    re.compile(r"\b(?:idiot|moron|stupid)\b", re.IGNORECASE),
)
FACTUAL_CLAIM_RE = re.compile(
    r"\b(?:according to|latest|current|guaranteed|proven|benchmark|refund|return approved|tracking number)\b",
    re.IGNORECASE,
)


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
        return self.action in {GuardrailAction.ALLOW, GuardrailAction.MONITOR, GuardrailAction.MASK}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _payload_to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, default=str)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def mask_text(text: str) -> str:
    masked = str(text)
    for pattern in SECRET_PATTERNS:
        masked = pattern.sub("[SECRET]", masked)
    masked = EMAIL_RE.sub(lambda match: _mask_email(match.group(0)), masked)
    masked = PHONE_RE.sub(lambda match: _mask_phone(match.group(0)), masked)
    return masked


def sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return mask_text(payload)
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                sanitized[key_text] = _mask_sensitive_value(value)
            else:
                sanitized[key_text] = sanitize_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [sanitize_payload(item) for item in payload]
    return payload


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
    severity = str(decision.get("severity") or "").lower()
    action = str(decision.get("action") or "").lower()
    return severity in {"high", "critical"} or action in {"review_required", "block"}


def guardrail_requires_override(payload: Any) -> bool:
    return has_high_risk_guardrail(payload)


def apply_support_guardrail_result(result: dict[str, Any], decision: GuardrailDecision) -> dict[str, Any]:
    result = dict(result)
    result["guardrail_decision"] = decision_result_payload(decision)
    result["guardrail_action"] = decision.action.value
    result["guardrail_findings"] = [
        finding.model_dump(mode="json")
        for finding in decision.findings
    ]
    if decision.action in {GuardrailAction.REVIEW_REQUIRED, GuardrailAction.BLOCK} or decision.severity in {
        GuardrailSeverity.HIGH,
        GuardrailSeverity.CRITICAL,
    }:
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
    sanitized = decision.sanitized_payload if isinstance(decision.sanitized_payload, dict) else result
    output = dict(sanitized if decision.action in {GuardrailAction.MASK, GuardrailAction.BLOCK} else result)
    output["guardrail_decision"] = decision_result_payload(decision)
    output["guardrail_action"] = decision.action.value
    output["guardrail_findings"] = [
        finding.model_dump(mode="json")
        for finding in decision.findings
    ]
    if workflow_value == WorkflowType.SUPPORT.value:
        return apply_support_guardrail_result(output, decision)
    return output


class WorkflowGuardrailService:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        _ensure_model_cache_env()
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

    def evaluate_input(
        self,
        workflow_type: WorkflowType | str,
        inputs: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        findings = self._evaluate_payload(
            workflow_value,
            GuardrailStage.INPUT,
            inputs,
            context=context,
        )
        action = self._policy_action(workflow_value, GuardrailStage.INPUT, findings)
        sanitized = sanitize_payload(inputs)
        return self._decision(
            workflow_value,
            GuardrailStage.INPUT,
            action,
            findings,
            sanitized_payload=sanitized,
        )

    def evaluate_output(
        self,
        workflow_type: WorkflowType | str,
        result: dict[str, Any],
        grounding_context: list[str] | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        context = {"grounding_context": grounding_context or []}
        findings = self._evaluate_payload(
            workflow_value,
            GuardrailStage.OUTPUT,
            result,
            context=context,
        )
        action = self._policy_action(workflow_value, GuardrailStage.OUTPUT, findings)
        return self._decision(
            workflow_value,
            GuardrailStage.OUTPUT,
            action,
            findings,
            sanitized_payload=sanitize_payload(result),
            metadata={"grounding_context_available": bool(grounding_context)},
        )

    def evaluate_action(
        self,
        workflow_type: WorkflowType | str,
        action_type: str,
        payload: dict[str, Any],
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        findings = self._evaluate_payload(
            workflow_value,
            GuardrailStage.ACTION,
            payload,
            action_type=action_type,
            context={"action_type": action_type},
        )
        action = self._policy_action(
            workflow_value,
            GuardrailStage.ACTION,
            findings,
            action_type=action_type,
        )
        return self._decision(
            workflow_value,
            GuardrailStage.ACTION,
            action,
            findings,
            sanitized_payload=sanitize_payload(payload),
            metadata={"action_type": action_type},
        )

    def _evaluate_payload(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        payload: Any,
        *,
        action_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailFinding]:
        if self._mode() == "off":
            return []
        text = _payload_to_text(payload)
        validators = self._validators_for(workflow_type, stage, action_type)
        findings = self._deterministic_findings(workflow_type, stage, text, validators, context or {})
        findings.extend(self._guardrails_ai_findings(workflow_type, stage, text, validators, context or {}))
        return _dedupe_findings(findings)

    def _deterministic_findings(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        text: str,
        validators: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[GuardrailFinding]:
        findings: list[GuardrailFinding] = []
        enabled = {str(item.get("id") or item.get("builtin") or ""): item for item in validators}
        if "secrets_present" in enabled:
            for match in _find_matches(SECRET_PATTERNS, text):
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        enabled["secrets_present"],
                        "secrets_present",
                        "Potential secret detected.",
                        match,
                    )
                )
        if "detect_pii" in enabled:
            for match in EMAIL_RE.finditer(text):
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        enabled["detect_pii"],
                        "detect_pii",
                        "Email address detected.",
                        match.group(0),
                    )
                )
            for match in PHONE_RE.finditer(text):
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        enabled["detect_pii"],
                        "detect_pii",
                        "Phone number detected.",
                        match.group(0),
                    )
                )
        if "prompt_injection" in enabled:
            for match in _find_matches(PROMPT_INJECTION_PATTERNS, text):
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        enabled["prompt_injection"],
                        "prompt_injection",
                        "Prompt injection pattern detected.",
                        match,
                    )
                )
        if "toxic_language" in enabled:
            for match in _find_matches(TOXIC_PATTERNS, text):
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        enabled["toxic_language"],
                        "toxic_language",
                        "Toxic language pattern detected.",
                        match,
                    )
                )
        if "provenance_llm" in enabled and not context.get("grounding_context"):
            if FACTUAL_CLAIM_RE.search(text):
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        enabled["provenance_llm"],
                        "provenance_llm",
                        "Factual claim has no grounding context available.",
                        "ungrounded factual claim",
                    )
                )
        return findings

    def _guardrails_ai_findings(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        text: str,
        validators: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[GuardrailFinding]:
        guardrails_validators = [
            item
            for item in validators
            if item.get("hub")
            and item.get("enable_ai_runtime") is not False
            and not (item.get("enabled_when") == "grounding_context_available" and not context.get("grounding_context"))
        ]
        if not guardrails_validators:
            return []
        try:
            from guardrails import Guard
        except Exception as exc:
            logger.info("guardrails-ai is not available; deterministic guardrails are still active: %s", exc)
            return []

        findings: list[GuardrailFinding] = []
        for validator_config in guardrails_validators:
            validator = self._build_hub_validator(validator_config)
            if validator is None:
                continue
            try:
                guard = Guard()
                guard.configure(allow_metrics_collection=False)
                outcome = guard.use(validator).validate(
                    text,
                    metadata=self._metadata_for_validator(validator_config, context),
                )
            except Exception as exc:
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        validator_config,
                        str(validator_config.get("id") or "guardrails_ai"),
                        f"Guardrails validator failed: {type(exc).__name__}",
                        str(exc),
                        metadata={"runtime_error": True},
                    )
                )
                continue
            if not bool(getattr(outcome, "validation_passed", False)):
                findings.append(
                    self._finding(
                        workflow_type,
                        stage,
                        validator_config,
                        str(validator_config.get("id") or "guardrails_ai"),
                        str(getattr(outcome, "error", "") or "Guardrails validation failed."),
                        str(getattr(outcome, "error", "") or ""),
                        metadata={"source": "guardrails_ai"},
                    )
                )
        return findings

    def _build_hub_validator(self, validator_config: dict[str, Any]) -> Any | None:
        try:
            import guardrails.hub as hub
        except Exception as exc:
            logger.info("guardrails hub is not available: %s", exc)
            hub = None
        validator_id = str(validator_config.get("id") or "")
        if validator_id == "toxic_language" and os.getenv("WORKFLOW_GUARDRAILS_ENABLE_TOXIC_AI", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            logger.info("guardrails toxic_language AI runtime is disabled; deterministic toxic policy remains active")
            return None
        class_names = {
            "secrets_present": ("SecretsPresent",),
            "detect_pii": ("DetectPII",),
            "toxic_language": ("ToxicLanguage",),
            "provenance_llm": ("ProvenanceLLM",),
        }.get(validator_id, ())
        klass = self._import_hub_validator(hub, validator_id, class_names)
        if klass is None:
            logger.info("guardrails hub validator %s is not installed", validator_id)
            return None
        kwargs = {"on_fail": str(validator_config.get("on_fail") or self.config.get("defaults", {}).get("on_fail") or "noop")}
        if validator_id == "detect_pii":
            kwargs["pii_entities"] = list(validator_config.get("entities") or [])
        if validator_id == "toxic_language":
            kwargs["threshold"] = float(validator_config.get("threshold", 0.5))
            kwargs["validation_method"] = str(validator_config.get("validation_method") or "sentence")
        if validator_id == "provenance_llm":
            kwargs["validation_method"] = str(validator_config.get("validation_method") or "sentence")
            kwargs["top_k"] = int(validator_config.get("top_k") or 3)
        for candidate in (kwargs, {key: value for key, value in kwargs.items() if key != "pii_entities"}, {}):
            try:
                return klass(**candidate)
            except TypeError:
                continue
        logger.info("guardrails hub validator %s could not be constructed", validator_id)
        return None

    def _import_hub_validator(self, hub: Any, validator_id: str, class_names: tuple[str, ...]) -> Any | None:
        for name in class_names:
            if hub is not None:
                try:
                    candidate = getattr(hub, name, None)
                except Exception as exc:
                    logger.warning(
                        "guardrails hub validator %s could not be imported from registry",
                        validator_id,
                        extra={"validator": validator_id, "error": str(exc)},
                    )
                    candidate = None
                if candidate is not None:
                    return candidate
            module_name = {
                "secrets_present": "guardrails_grhub_secrets_present",
                "detect_pii": "guardrails_grhub_detect_pii",
                "provenance_llm": "guardrails_grhub_provenance_llm",
            }.get(validator_id)
            if module_name is None:
                continue
            try:
                module = importlib.import_module(module_name)
                candidate = getattr(module, name, None)
            except Exception as exc:
                logger.warning(
                    "guardrails hub validator %s could not be imported from package",
                    validator_id,
                    extra={"validator": validator_id, "error": str(exc)},
                )
                return None
            if candidate is not None:
                return candidate
        return None

    def _metadata_for_validator(self, validator_config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if validator_config.get("id") == "provenance_llm":
            sources = context.get("grounding_context") or []
            return {"sources": sources}
        return {}

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
        action = GuardrailAction.MONITOR if self._mode() == "monitor" else GuardrailAction.REVIEW_REQUIRED
        validators = self._validators_for(workflow_type, stage)
        policy_by_id = {str(item.get("id") or item.get("builtin") or ""): item for item in validators}
        for finding in findings:
            validator_config = policy_by_id.get(finding.validator)
            configured = _action_from_text(str((validator_config or {}).get("policy") or "monitor"))
            action = _max_action(action, configured)
        return action

    def _action_policy_action(
        self,
        workflow_type: str,
        findings: list[GuardrailFinding],
        action_type: str | None,
    ) -> GuardrailAction:
        stage_config = self._stage_config(workflow_type, GuardrailStage.ACTION)
        action_config = stage_config.get(action_type or "") if isinstance(stage_config, dict) else None
        if not isinstance(action_config, dict):
            return GuardrailAction.MONITOR if self._mode() == "monitor" else GuardrailAction.REVIEW_REQUIRED
        selected = GuardrailAction.MONITOR
        for finding in findings:
            configured = _action_from_text(str(action_config.get(finding.severity.value) or action_config.get("high") or "monitor"))
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
            message=message,
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
    ) -> list[dict[str, Any]]:
        stage_config = self._stage_config(workflow_type, stage)
        if stage == GuardrailStage.ACTION:
            output_config = self._stage_config(workflow_type, GuardrailStage.OUTPUT)
            return list(output_config.get("validators") or []) if isinstance(output_config, dict) else []
        return list(stage_config.get("validators") or []) if isinstance(stage_config, dict) else []

    def _stage_config(self, workflow_type: str, stage: GuardrailStage) -> dict[str, Any]:
        guards = self.config.get("guards") or {}
        workflow_config = guards.get(workflow_type) or {}
        stage_config = workflow_config.get(stage.value) or {}
        return stage_config if isinstance(stage_config, dict) else {}

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
    return workflow_type.value if isinstance(workflow_type, WorkflowType) else str(workflow_type)


def _is_sensitive_key(key: str) -> bool:
    return bool(re.search(r"(api[_-]?key|authorization|bearer|client[_-]?secret|credential|password|refresh[_-]?token|secret|token)", key, re.IGNORECASE))


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


def _find_matches(patterns: tuple[re.Pattern[str], ...], text: str) -> list[str]:
    return [match.group(0) for pattern in patterns for match in pattern.finditer(text)]


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


def _max_severity(findings: list[GuardrailFinding]) -> GuardrailSeverity:
    if not findings:
        return GuardrailSeverity.NONE
    return max((finding.severity for finding in findings), key=lambda item: SEVERITY_RANK[item.value])


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
