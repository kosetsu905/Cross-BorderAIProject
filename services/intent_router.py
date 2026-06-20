from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import httpx

from services.pim_connector import PIMConnector
from tools.custom.support_rag_tools import DEFAULT_KNOWLEDGE_DIR, search_knowledge_base
from utils.llm_config import (
    llm_api_key,
    llm_chat_completions_url,
    llm_model_name,
    llm_reasoning_compat_params,
)

logger = logging.getLogger(__name__)

INTENTS = ("pre_sales", "order_fulfillment", "post_sales_support")
SUGGESTED_AGENT = {
    "pre_sales": "pre_sales_specialist",
    "order_fulfillment": "order_fulfillment_specialist",
    "post_sales_support": "senior_support_agent",
}
ORDER_ID_RE = re.compile(r"\b(?:ORD[-_A-Z0-9]*|\#?\d{5,})\b", re.IGNORECASE)
QUANTITY_RE = re.compile(r"\b(?:buy|purchase|order|get|need|want)\s+(?:about\s+)?(\d{1,5})\b", re.IGNORECASE)
MONEY_OR_PRICE_RE = re.compile(r"(?:\$|usd|eur|jpy|price|discount|quote|wholesale|moq|carton)", re.IGNORECASE)
STOPWORDS = {
    "hello", "please", "thanks", "thank", "want", "would", "could", "should", "about", "with",
    "this", "that", "from", "have", "need", "order", "buy", "purchase", "discount", "quote",
    "wholesale", "retail", "price", "product", "products", "details", "three", "piece", "set",
    "to", "of", "can", "i", "get", "a", "an", "the", "for", "me", "my",
}


class HybridIntentRouter:
    def __init__(
        self,
        config_context: dict[str, Any] | None = None,
        classifier: Any | None = None,
        llm_client: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.config_context = config_context or {}
        self.classifier = classifier
        self.llm_client = llm_client
        self.threshold = _float_config(self.config_context, "intent_router_confidence_threshold", 0.75)

    def classify(
        self,
        inquiry_text: str,
        has_order_id: bool = False,
        customer_tier: str = "STANDARD",
        language: str = "en",
    ) -> dict[str, Any]:
        text = str(inquiry_text or "")
        normalized = text.lower()
        scores = {intent: 0.05 for intent in INTENTS}
        signals: list[dict[str, Any]] = []

        self._apply_rule_signals(normalized, has_order_id, customer_tier, scores, signals)
        self._apply_catalog_signal(text, scores, signals)
        self._apply_pim_signal(text, language, scores, signals)
        self._apply_classifier_signal(text, language, scores, signals)

        intent, confidence = _best_intent(scores)
        llm_fallback_used = False
        if confidence < self.threshold and self._llm_enabled():
            llm_result = self._classify_with_llm(text, language)
            if llm_result:
                llm_fallback_used = True
                intent = llm_result["detected_intent"]
                confidence = max(confidence, float(llm_result.get("confidence_score") or 0))
                signals.append({"name": "llm_fallback", "intent": intent, "weight": round(confidence, 4)})

        requires_review = confidence < self.threshold
        return {
            "detected_intent": intent,
            "confidence_score": round(min(confidence, 0.98), 4),
            "requires_human_review": requires_review,
            "suggested_agent": SUGGESTED_AGENT[intent],
            "routing_signals": signals,
            "llm_fallback_used": llm_fallback_used,
            "context_enrichment": {
                "product_category_hint": _extract_product_hint(text, signals),
                "urgency_level": "high" if "urgent" in normalized or "asap" in normalized else "normal",
            },
        }

    def _apply_rule_signals(
        self,
        text: str,
        has_order_id: bool,
        customer_tier: str,
        scores: dict[str, float],
        signals: list[dict[str, Any]],
    ) -> None:
        pre_sales_phrases = [
            "before buy", "which model", "compatible with", "does it work", "compare", "recommend",
            "feature", "specification", "specs", "bulk pricing", "deciding between", "better for",
            "outdoor", "rainy", "works better", "want to buy", "can i get a discount", "discount",
            "quote", "wholesale", "moq", "retail price",
        ]
        order_phrases = [
            "where is my order", "order status", "tracking", "shipping date", "delivery time",
            "change address", "modify order", "when will my package", "delivery eta",
            "物流", "快递", "订单状态", "追踪", "包裹在哪",
        ]
        post_sales_phrases = [
            "return", "refund", "defective", "not working", "damaged", "wrong item", "setup help",
            "cracked", "replacement", "warranty", "arrived broken", "broken",
            "坏了", "不能用", "退款", "退货", "售后", "投诉", "主管", "经理", "换货",
        ]

        if any(phrase in text for phrase in post_sales_phrases):
            _add_signal(scores, signals, "post_sales_keyword", "post_sales_support", 0.7)
        if any(phrase in text for phrase in order_phrases):
            _add_signal(scores, signals, "order_keyword", "order_fulfillment", 0.65)
        if has_order_id or ORDER_ID_RE.search(text):
            _add_signal(scores, signals, "order_identifier", "order_fulfillment", 0.55)
        if any(phrase in text for phrase in pre_sales_phrases):
            _add_signal(scores, signals, "pre_sales_phrase", "pre_sales", 0.55)
        if QUANTITY_RE.search(text) and MONEY_OR_PRICE_RE.search(text):
            _add_signal(scores, signals, "quantity_purchase_signal", "pre_sales", 0.55)
        if "buy" in text and ("discount" in text or "quote" in text or "wholesale" in text):
            _add_signal(scores, signals, "purchase_discount_signal", "pre_sales", 0.45)
        if customer_tier in {"VIP", "PREMIUM"} and max(scores.values()) < 0.5:
            _add_signal(scores, signals, "premium_ambiguous_support_bias", "post_sales_support", 0.2)

    def _apply_catalog_signal(self, text: str, scores: dict[str, float], signals: list[dict[str, Any]]) -> None:
        knowledge_dir = self.config_context.get("support_knowledge_dir") or str(DEFAULT_KNOWLEDGE_DIR)
        try:
            results = search_knowledge_base(text, knowledge_dir=knowledge_dir, top_k=3)
        except Exception as exc:
            logger.debug("Catalog routing signal skipped: %s", exc)
            return
        catalog_results = [
            result for result in results
            if str(result.get("source", "")).lower().endswith(".pdf")
            and float(result.get("score") or 0) >= 0.2
        ]
        if catalog_results:
            _add_signal(scores, signals, "catalog_match", "pre_sales", 0.3, source=catalog_results[0].get("source"))

    def _apply_pim_signal(self, text: str, language: str, scores: dict[str, float], signals: list[dict[str, Any]]) -> None:
        backend = str(self.config_context.get("pim_backend") or "akeneo").lower()
        try:
            result = _run_coroutine_sync(
                PIMConnector(
                    backend=backend,
                    base_url=self.config_context.get(f"pim_{backend}_base_url"),
                    api_key=self.config_context.get(f"pim_{backend}_api_key"),
                ).search_product(_extract_product_hint(text, signals) or text[:120], "US", language)
            )
        except Exception as exc:
            logger.debug("PIM routing signal skipped: %s", exc)
            return
        if result.product_found:
            if result.data_source == "mock_fallback" and not _has_product_or_purchase_terms(text):
                return
            weight = 0.18 if result.data_source == "mock_fallback" else 0.28
            _add_signal(scores, signals, "pim_product_match", "pre_sales", weight, source=result.data_source)

    def _apply_classifier_signal(self, text: str, language: str, scores: dict[str, float], signals: list[dict[str, Any]]) -> None:
        classifier = self.classifier
        if classifier is None and _bool_config(self.config_context, "intent_classifier_enabled", False):
            model_path = self.config_context.get("intent_classifier_model_path")
            if model_path:
                try:
                    from scripts.train_intent_classifier import IntentClassifierTool

                    classifier = IntentClassifierTool(model_path)
                except Exception as exc:
                    logger.debug("Intent classifier unavailable: %s", exc)
                    return
        if classifier is None:
            return
        try:
            result = classifier.predict(text, language) if hasattr(classifier, "predict") else classifier(text, language)
        except Exception as exc:
            logger.debug("Intent classifier prediction skipped: %s", exc)
            return
        intent = result.get("detected_intent")
        confidence = float(result.get("confidence_score") or 0)
        if intent in INTENTS and confidence > 0:
            _add_signal(scores, signals, "trained_classifier", intent, min(confidence * 0.75, 0.75))

    def _llm_enabled(self) -> bool:
        return bool(
            _bool_config(self.config_context, "intent_router_llm_fallback_enabled", True)
            and llm_api_key(self.config_context)
        )

    def _classify_with_llm(self, text: str, language: str) -> dict[str, Any] | None:
        if self.llm_client:
            return _normalize_llm_result(self.llm_client(text, self.config_context))
        try:
            payload = {
                "model": llm_model_name(self.config_context),
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify customer service intent as exactly one of: pre_sales, "
                            "order_fulfillment, post_sales_support. Return JSON with detected_intent "
                            "and confidence_score between 0 and 1."
                        ),
                    },
                    {"role": "user", "content": f"Language: {language}\nInquiry: {text[:1200]}"},
                ],
                "temperature": 0,
                **llm_reasoning_compat_params(self.config_context),
            }
            response = httpx.post(
                llm_chat_completions_url(self.config_context),
                headers={
                    "Authorization": f"Bearer {llm_api_key(self.config_context)}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _normalize_llm_result(json.loads(content))
        except Exception as exc:
            logger.debug("LLM intent fallback failed: %s", exc)
            return None


def classify_intent(
    inquiry_text: str,
    has_order_id: bool = False,
    customer_tier: str = "STANDARD",
    language: str = "en",
    config_context: dict[str, Any] | None = None,
    classifier: Any | None = None,
    llm_client: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return HybridIntentRouter(config_context, classifier=classifier, llm_client=llm_client).classify(
        inquiry_text=inquiry_text,
        has_order_id=has_order_id,
        customer_tier=customer_tier,
        language=language,
    )


def _add_signal(
    scores: dict[str, float],
    signals: list[dict[str, Any]],
    name: str,
    intent: str,
    weight: float,
    **extra: Any,
) -> None:
    scores[intent] += weight
    signals.append({"name": name, "intent": intent, "weight": round(weight, 4), **extra})


def _best_intent(scores: dict[str, float]) -> tuple[str, float]:
    intent = max(scores, key=scores.get)
    top_score = scores[intent]
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    confidence = min(0.98, 0.45 + top_score * 0.45 + margin * 0.2)
    if top_score >= 0.85:
        confidence = max(confidence, 0.86)
    return intent, confidence


def _extract_product_hint(text: str, signals: list[dict[str, Any]] | None = None) -> str | None:
    candidates = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9-]{1,}\b", text)
    useful = [candidate for candidate in candidates if candidate.lower() not in STOPWORDS and not candidate.isdigit()]
    if useful:
        return " ".join(useful[:6]).replace("_", " ").title()
    if signals:
        for signal in signals:
            source = str(signal.get("source") or "")
            if source:
                return Path(source).stem.replace("_", " ").replace("-", " ").title()
    return None


def _has_product_or_purchase_terms(text: str) -> bool:
    normalized = text.lower()
    product_terms = {
        "buy", "purchase", "discount", "quote", "wholesale", "moq", "carton", "price",
        "keyboard", "mouse", "headset", "earphone", "bluetooth", "camera", "tablet",
        "watch", "model", "compatible", "waterproof", "spec", "specification",
    }
    return any(term in normalized for term in product_terms) or bool(QUANTITY_RE.search(normalized))


def _normalize_llm_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    intent = result.get("detected_intent")
    confidence = float(result.get("confidence_score") or result.get("confidence") or 0)
    if intent not in INTENTS:
        return None
    return {"detected_intent": intent, "confidence_score": confidence}


def _bool_config(config_context: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config_context.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _float_config(config_context: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config_context.get(key, default))
    except (TypeError, ValueError):
        return default


def _run_coroutine_sync(coroutine):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coroutine)).result()
