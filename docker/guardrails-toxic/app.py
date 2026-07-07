from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException
from guardrails import Guard
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_guard_lock = RLock()


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    validator: str


class ToxicValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    validation_method: str = Field(default="sentence")


class ToxicValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_passed: bool
    failure_reasons: list[str] = Field(default_factory=list)
    validator: str = "hub://guardrails/toxic_language"
    source: str = "guardrails_ai_docker"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _ensure_nltk_data()
    _guard(0.5, "sentence")
    yield


app = FastAPI(title="Guardrails Toxic Runtime", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_guard.cache_info().currsize > 0,
        validator="hub://guardrails/toxic_language",
    )


@app.post("/validate", response_model=ToxicValidationResponse)
def validate_toxic(request: ToxicValidationRequest) -> ToxicValidationResponse:
    try:
        guard = _guard(request.threshold, request.validation_method)
        with _guard_lock:
            outcome = guard.validate(request.text)
    except Exception as exc:
        logger.exception("toxic_language validation failed")
        raise HTTPException(status_code=503, detail=type(exc).__name__) from exc
    validation_passed = bool(getattr(outcome, "validation_passed", False))
    return ToxicValidationResponse(
        validation_passed=validation_passed,
        failure_reasons=[] if validation_passed else _failure_reasons(outcome),
    )


@lru_cache(maxsize=8)
def _guard(threshold: float, validation_method: str) -> Guard:
    try:
        from guardrails.hub import ToxicLanguage
    except (AttributeError, ImportError):
        from validator import ToxicLanguage

    guard = Guard()
    guard.configure(allow_metrics_collection=False)
    guard.use(
        ToxicLanguage(
            threshold=threshold,
            validation_method=validation_method,
            on_fail="noop",
            use_local=True,
        )
    )
    return guard


def _failure_reasons(outcome: Any) -> list[str]:
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
    return ["Guardrails toxic_language validation failed."]


def _ensure_nltk_data() -> None:
    import nltk

    data_dir = os.getenv("NLTK_DATA", "/models/nltk")
    for package_name in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{package_name}")
        except LookupError:
            nltk.download(package_name, download_dir=data_dir, quiet=True)
