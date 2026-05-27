from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

try:
    from langdetect import LangDetectException, detect as langdetect_detect
except ImportError:  # pragma: no cover - depends on optional local install
    LangDetectException = Exception
    langdetect_detect = None


class LanguageDetector:
    SUPPORTED_MAP = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "zh": "Chinese",
        "ar": "Arabic",
        "pt": "Portuguese",
        "ko": "Korean",
        "it": "Italian",
        "ru": "Russian",
    }

    _SCRIPT_PATTERNS = (
        (re.compile(r"[\u3040-\u309f\u30a0-\u30ff]"), "ja"),
        (re.compile(r"[\u4e00-\u9fff]"), "zh"),
        (re.compile(r"[\uac00-\ud7af]"), "ko"),
        (re.compile(r"[\u0600-\u06ff]"), "ar"),
        (re.compile(r"[\u0400-\u04ff]"), "ru"),
    )
    _PHRASE_PATTERNS = (
        (re.compile(r"\b(hola|gracias|pedido|devoluci[oó]n|reembolso)\b", re.IGNORECASE), "es"),
        (re.compile(r"\b(bonjour|merci|commande|remboursement|retour)\b", re.IGNORECASE), "fr"),
        (re.compile(r"\b(hallo|danke|bestellung|r[üu]ckerstattung|r[üu]ckgabe)\b", re.IGNORECASE), "de"),
        (re.compile(r"\b(ol[aá]|obrigad[ao]|pedido|reembolso|devolu[cç][aã]o)\b", re.IGNORECASE), "pt"),
        (re.compile(r"\b(ciao|grazie|ordine|rimborso|reso)\b", re.IGNORECASE), "it"),
    )
    _ACCENT_PATTERNS = (
        (re.compile(r"[¿¡ñáéíóúü]", re.IGNORECASE), "es"),
        (re.compile(r"[àâçéèêëîïôûùüÿœ]", re.IGNORECASE), "fr"),
        (re.compile(r"[äöüß]", re.IGNORECASE), "de"),
        (re.compile(r"[ãõçáâêéíóôú]", re.IGNORECASE), "pt"),
    )

    @staticmethod
    def detect(text: str, fallback: str = "en") -> str:
        normalized_fallback = LanguageDetector._normalize_code(fallback) or "en"
        clean = (text or "").strip()
        if not clean:
            return normalized_fallback

        heuristic = LanguageDetector._heuristic_detect(clean)
        if heuristic:
            return heuristic
        if len(clean) < 4:
            return normalized_fallback

        if langdetect_detect is not None:
            try:
                detected = LanguageDetector._normalize_code(langdetect_detect(clean))
                if detected in LanguageDetector.SUPPORTED_MAP:
                    return detected
            except LangDetectException:
                logger.warning("Language detection failed for text: %s...", clean[:30])

        return normalized_fallback

    @staticmethod
    def get_crewai_language_plan(code: str) -> str:
        normalized = LanguageDetector._normalize_code(code)
        return LanguageDetector.SUPPORTED_MAP.get(normalized or "", "English")

    @staticmethod
    def _heuristic_detect(text: str) -> str | None:
        for pattern, language in LanguageDetector._SCRIPT_PATTERNS:
            if pattern.search(text):
                return language
        for pattern, language in LanguageDetector._PHRASE_PATTERNS:
            if pattern.search(text):
                return language
        for pattern, language in LanguageDetector._ACCENT_PATTERNS:
            if pattern.search(text):
                return language
        return None

    @staticmethod
    def _normalize_code(code: str | None) -> str | None:
        if not code:
            return None
        normalized = str(code).strip().lower().replace("_", "-")
        return normalized.split("-", 1)[0]
