from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import yaml

from models import WorkflowType

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "guardrails.yaml"
MODEL_CACHE_DIR = BASE_DIR / ".cache"
GUARDRAIL_REVIEW_FLAG = "GUARDRAIL_REVIEW_REQUIRED"
GUARDRAIL_HIGH_RISK_FLAG = "GUARDRAIL_HIGH_RISK"

SECRET_REDACTION_RE_LIST: tuple[re.Pattern[str], ...] = (
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
EMAIL_ADDRESS_REDACTOR = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_NUMBER_REDACTOR = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
ENV_VALUE_RE = re.compile(r"^\$\{([A-Z0-9_]+)(?::([^}]*))?\}$")
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


@dataclass
class _PayloadEvaluation:
    findings: list[GuardrailFinding]
    skipped_validators: list[dict[str, Any]]


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
    for pattern in SECRET_REDACTION_RE_LIST:
        masked = pattern.sub("[SECRET]", masked)
    masked = EMAIL_ADDRESS_REDACTOR.sub(lambda match: _mask_email(match.group(0)), masked)
    masked = PHONE_NUMBER_REDACTOR.sub(lambda match: _mask_phone(match.group(0)), masked)
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
        self._embed_model: Any | None = None
        self._validator_cache: dict[str, Any] = {}
        self._validate_config()

    def evaluate_input(
        self,
        workflow_type: WorkflowType | str,
        inputs: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        evaluation = self._evaluate_payload(
            workflow_value,
            GuardrailStage.INPUT,
            inputs,
            context=context,
        )
        action = self._policy_action(workflow_value, GuardrailStage.INPUT, evaluation.findings)
        sanitized = sanitize_payload(inputs)
        return self._decision(
            workflow_value,
            GuardrailStage.INPUT,
            action,
            evaluation.findings,
            sanitized_payload=sanitized,
            metadata={"skipped_validators": evaluation.skipped_validators},
        )

    def evaluate_output(
        self,
        workflow_type: WorkflowType | str,
        result: dict[str, Any],
        grounding_context: list[str] | None = None,
        query_function: Any | None = None,
        embed_function: Any | None = None,
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        context = {
            "embed_function": embed_function,
            "grounding_context": grounding_context or [],
            "query_function": query_function,
        }
        evaluation = self._evaluate_payload(
            workflow_value,
            GuardrailStage.OUTPUT,
            result,
            context=context,
        )
        action = self._policy_action(workflow_value, GuardrailStage.OUTPUT, evaluation.findings)
        return self._decision(
            workflow_value,
            GuardrailStage.OUTPUT,
            action,
            evaluation.findings,
            sanitized_payload=sanitize_payload(result),
            metadata={
                "grounding_context_available": bool(grounding_context),
                "skipped_validators": evaluation.skipped_validators,
            },
        )

    def evaluate_action(
        self,
        workflow_type: WorkflowType | str,
        action_type: str,
        payload: dict[str, Any],
    ) -> GuardrailDecision:
        workflow_value = _workflow_value(workflow_type)
        evaluation = self._evaluate_payload(
            workflow_value,
            GuardrailStage.ACTION,
            payload,
            action_type=action_type,
            context={"action_type": action_type},
        )
        action = self._policy_action(
            workflow_value,
            GuardrailStage.ACTION,
            evaluation.findings,
            action_type=action_type,
        )
        return self._decision(
            workflow_value,
            GuardrailStage.ACTION,
            action,
            evaluation.findings,
            sanitized_payload=sanitize_payload(payload),
            metadata={
                "action_type": action_type,
                "skipped_validators": evaluation.skipped_validators,
            },
        )

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
            return _PayloadEvaluation(findings=[], skipped_validators=[])
        text = _payload_to_text(payload)
        validators = self._validators_for(workflow_type, stage, action_type)
        return self._guardrails_ai_findings(workflow_type, stage, text, validators, context or {})

    def _guardrails_ai_findings(
        self,
        workflow_type: str,
        stage: GuardrailStage,
        text: str,
        validators: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> _PayloadEvaluation:
        guardrails_validators = [item for item in validators if item.get("hub")]
        if not guardrails_validators:
            return _PayloadEvaluation(findings=[], skipped_validators=[])
        try:
            from guardrails import Guard
        except Exception as exc:
            raise GuardrailConfigurationError("guardrails-ai is required for workflow guardrails") from exc

        findings: list[GuardrailFinding] = []
        skipped_validators: list[dict[str, Any]] = []
        for validator_config in guardrails_validators:
            validator_id = str(validator_config.get("id") or validator_config.get("hub") or "guardrails_ai")
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
            validator = self._build_hub_validator(validator_config)
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
                        validator_id,
                        f"Guardrails validator failed: {type(exc).__name__}",
                        str(exc),
                        metadata={"runtime_error": True},
                    )
                )
                continue
            if not bool(getattr(outcome, "validation_passed", False)):
                failure_reason = "\n".join(_validation_failure_reasons(outcome))
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
        return _PayloadEvaluation(findings=_dedupe_findings(findings), skipped_validators=skipped_validators)

    def _build_hub_validator(self, validator_config: dict[str, Any]) -> Any:
        try:
            import guardrails.hub as hub
        except Exception as exc:
            raise GuardrailConfigurationError("guardrails.hub is required for workflow guardrails") from exc
        validator_id = str(validator_config.get("id") or validator_config.get("hub") or "")
        hub_uri = str(validator_config.get("hub") or validator_id)
        class_names = _class_names_for_validator(validator_config)
        if not class_names:
            raise GuardrailConfigurationError(f"Guardrails Hub validator class is not configured for '{hub_uri}'")
        kwargs = self._validator_kwargs(validator_config)
        cache_key = _validator_cache_key(validator_config, kwargs)
        cached = self._validator_cache.get(cache_key)
        if cached is not None:
            return cached
        klass = self._import_hub_validator(hub, hub_uri, class_names)
        try:
            validator = klass(**kwargs)
        except TypeError as exc:
            raise GuardrailConfigurationError(
                f"Guardrails Hub validator '{hub_uri}' could not be constructed with configured args"
            ) from exc
        self._validator_cache[cache_key] = validator
        return validator

    def _import_hub_validator(self, hub: Any, validator_id: str, class_names: tuple[str, ...]) -> Any:
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
        install_hint = f"guardrails hub install {validator_id}" if validator_id.startswith("hub://") else "guardrails hub list"
        raise GuardrailConfigurationError(
            f"Guardrails Hub validator '{validator_id}' is not installed or does not expose {class_names}. "
            f"Run: {install_hint}"
        )

    def _validator_kwargs(self, validator_config: dict[str, Any]) -> dict[str, Any]:
        defaults = self.config.get("defaults") if isinstance(self.config.get("defaults"), dict) else {}
        validator_id = str(validator_config.get("id") or "")
        kwargs = dict(validator_config.get("args") or {})
        for key in ("llm_callable", "match_type", "max_tokens", "regex", "threshold", "top_k", "validation_method"):
            if key in validator_config:
                kwargs.setdefault(key, validator_config[key])
        if validator_id == "detect_pii" and "entities" in validator_config:
            kwargs.setdefault("pii_entities", list(validator_config.get("entities") or []))
        if "forbidden_terms" in validator_config:
            kwargs.setdefault("regex", _negative_regex_from_terms(list(validator_config.get("forbidden_terms") or [])))
            kwargs.setdefault("match_type", "fullmatch")
        kwargs.setdefault("on_fail", str(validator_config.get("on_fail") or defaults.get("on_fail") or "noop"))
        return _resolve_config_value(kwargs)

    def _metadata_for_validator(self, validator_config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if validator_config.get("id") == "provenance_llm":
            query_function = context.get("query_function")
            if query_function is not None:
                return {"query_function": query_function}
            sources = context.get("grounding_context") or []
            embed_function = context.get("embed_function") or self._default_embed_function()
            metadata = {"embed_function": embed_function, "sources": sources}
            if "pass_on_invalid" in validator_config:
                metadata["pass_on_invalid"] = bool(validator_config.get("pass_on_invalid"))
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
            model_name = os.getenv("WORKFLOW_GUARDRAILS_EMBED_MODEL", "paraphrase-MiniLM-L6-v2")
            self._embed_model = SentenceTransformer(model_name)

        def embed(value: str | list[str]) -> Any:
            return self._embed_model.encode(value, normalize_embeddings=True)

        return embed

    def _skip_reason(self, validator_config: dict[str, Any], context: dict[str, Any]) -> str | None:
        if validator_config.get("enabled_when") == "grounding_context_available" and not context.get("grounding_context"):
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

    def validate_runtime(self, *, smoke_toxic: bool = False) -> None:
        try:
            from guardrails import Guard
        except Exception as exc:
            raise GuardrailConfigurationError("guardrails-ai is required for workflow guardrails") from exc
        for validator_config in self._configured_validator_configs():
            validator = self._build_hub_validator(validator_config)
            if smoke_toxic and validator_config.get("id") == "toxic_language":
                try:
                    guard = Guard()
                    guard.configure(allow_metrics_collection=False)
                    guard.use(validator).validate("Please keep the reply professional and respectful.")
                except Exception as exc:
                    raise GuardrailConfigurationError(
                        "hub://guardrails/toxic_language installed but failed smoke validation. "
                        "Try reinstalling CPU torch and the validator, then rerun this script."
                    ) from exc

    def _validate_config(self) -> None:
        if self._mode() == "off":
            return
        for validator_config in self._configured_validator_configs():
            validator_id = str(validator_config.get("id") or validator_config.get("hub") or "")
            if validator_config.get("builtin"):
                raise GuardrailConfigurationError(
                    f"Workflow guardrail '{validator_id}' uses builtin detection. Use a hub:// validator instead."
                )
            if not validator_config.get("hub"):
                raise GuardrailConfigurationError(f"Workflow guardrail '{validator_id}' must configure a hub:// validator")
            if validator_config.get("enable_ai_runtime") is False:
                raise GuardrailConfigurationError(
                    f"Workflow guardrail '{validator_id}' disables Hub runtime. Install the validator instead."
                )
            if str(validator_config.get("id") or "") == "forbidden_terms":
                args = validator_config.get("args") if isinstance(validator_config.get("args"), dict) else {}
                if not args.get("regex") and not validator_config.get("forbidden_terms"):
                    raise GuardrailConfigurationError("forbidden_terms must configure regex or forbidden_terms")

    def _configured_validator_configs(self) -> list[dict[str, Any]]:
        validators: list[dict[str, Any]] = []
        guards = self.config.get("guards") if isinstance(self.config.get("guards"), dict) else {}
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


def _validator_cache_key(validator_config: dict[str, Any], kwargs: dict[str, Any]) -> str:
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
        raise GuardrailConfigurationError("forbidden_terms must contain at least one non-empty term")
    return rf"(?is)^(?!.*(?:{'|'.join(escaped_terms)})).*$"


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
