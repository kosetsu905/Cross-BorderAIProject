from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import redis
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from utils.observability import set_span_attributes, tool_span
from utils.tool_execution import run_tool_async_or_threaded

logger = logging.getLogger(__name__)

DEFAULT_TOOL_CACHE_TTL_SECONDS = 86400
DEFAULT_MAX_VALUE_BYTES = 1024 * 1024
TOOL_CACHE_KEY_PREFIX = "tool_cache:"
SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|client[_-]?secret|credential|password|refresh[_-]?token|secret|token)",
    re.IGNORECASE,
)
PII_KEY_RE = re.compile(
    r"(customer[_-]?email|customer[_-]?handle|email|phone|recipient|sender)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")


@dataclass(frozen=True)
class ToolCacheRead:
    value: Any
    backend: str
    cache_key: str
    metadata: dict[str, Any]


class InMemoryToolCacheStore:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def get(self, cache_key: str) -> dict[str, Any] | None:
        entry = self.entries.get(cache_key)
        if not entry:
            return None
        if entry["expires_at"] <= datetime.now(UTC):
            return None
        return entry

    def set(
        self,
        cache_key: str,
        tool_name: str,
        tool_version: str,
        value: Any,
        metadata: dict[str, Any],
        expires_at: datetime,
    ) -> None:
        self.entries[cache_key] = {
            "cache_key": cache_key,
            "tool_name": tool_name,
            "tool_version": tool_version,
            "value": _json_safe(value),
            "metadata": _json_safe(metadata),
            "expires_at": expires_at,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }


class PostgresToolCacheStore:
    def __init__(self, session_factory: Any | None = None) -> None:
        if session_factory is None:
            from database import SessionLocal

            session_factory = SessionLocal
        self.session_factory = session_factory

    def get(self, cache_key: str) -> dict[str, Any] | None:
        from db_models import ToolCacheEntryRecord

        try:
            with self.session_factory() as session:
                record = session.get(ToolCacheEntryRecord, cache_key)
                if record is None or record.expires_at <= datetime.now(UTC):
                    return None
                return {
                    "cache_key": record.cache_key,
                    "tool_name": record.tool_name,
                    "tool_version": record.tool_version,
                    "value": record.value,
                    "metadata": record.metadata_ or {},
                    "expires_at": record.expires_at,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                }
        except SQLAlchemyError as exc:
            logger.warning("Tool cache DB read failed: %s", exc)
            return None

    def set(
        self,
        cache_key: str,
        tool_name: str,
        tool_version: str,
        value: Any,
        metadata: dict[str, Any],
        expires_at: datetime,
    ) -> None:
        from db_models import ToolCacheEntryRecord

        try:
            with self.session_factory() as session:
                record = session.get(ToolCacheEntryRecord, cache_key)
                if record is None:
                    record = ToolCacheEntryRecord(
                        cache_key=cache_key,
                        tool_name=tool_name,
                        tool_version=tool_version,
                    )
                    session.add(record)
                record.tool_name = tool_name
                record.tool_version = tool_version
                record.value = _json_safe(value)
                record.metadata_ = _json_safe(metadata)
                record.expires_at = expires_at
                record.updated_at = datetime.now(UTC)
                session.commit()
        except SQLAlchemyError as exc:
            logger.warning("Tool cache DB write failed: %s", exc)


class ToolCache:
    def __init__(
        self,
        config_context: dict[str, Any] | None,
        *,
        redis_client: Any | None = None,
        db_store: Any | None = None,
    ) -> None:
        self.config_context = dict(config_context or {})
        self.ttl_seconds = _positive_int_config(
            self.config_context,
            "tool_cache_ttl_seconds",
            DEFAULT_TOOL_CACHE_TTL_SECONDS,
        )
        self.max_value_bytes = _positive_int_config(
            self.config_context,
            "tool_cache_max_value_bytes",
            DEFAULT_MAX_VALUE_BYTES,
        )
        self.redis_client = redis_client if redis_client is not None else self._build_redis_client()
        self.db_store = db_store if db_store is not None else self._build_db_store()

    @property
    def enabled(self) -> bool:
        has_runtime_cache_config = any(
            key in self.config_context
            for key in (
                "tool_cache_enabled",
                "tool_cache_backend",
                "tool_cache_redis_url",
                "tool_cache_ttl_seconds",
            )
        )
        return (
            has_runtime_cache_config
            and self.ttl_seconds > 0
            and _bool_config(self.config_context, "tool_cache_enabled", True)
        )

    def get(self, cache_key: str) -> ToolCacheRead | None:
        if not self.enabled:
            return None
        redis_hit = self._redis_get(cache_key)
        if redis_hit is not None:
            metadata = {
                **(redis_hit.get("metadata") or {}),
                "cache_hit": True,
                "cache_backend": "redis",
                "cache_key": cache_key,
            }
            return ToolCacheRead(
                value=redis_hit["value"],
                backend="redis",
                cache_key=cache_key,
                metadata=metadata,
            )
        db_hit = self._db_get(cache_key)
        if db_hit is None:
            return None
        self._redis_set(cache_key, db_hit["value"], db_hit.get("metadata") or {})
        metadata = {
            **(db_hit.get("metadata") or {}),
            "cache_hit": True,
            "cache_backend": "postgres",
            "cache_key": cache_key,
        }
        return ToolCacheRead(
            value=db_hit["value"],
            backend="postgres",
            cache_key=cache_key,
            metadata=metadata,
        )

    def set(
        self,
        cache_key: str,
        tool_name: str,
        tool_version: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        safe_value = _json_safe(value)
        encoded = json.dumps(safe_value, sort_keys=True, default=str).encode("utf-8")
        if len(encoded) > self.max_value_bytes:
            logger.info(
                "Skipping oversized tool cache value",
                extra={"tool_name": tool_name, "cache_key": cache_key, "bytes": len(encoded)},
            )
            return
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        safe_metadata = {
            **(metadata or {}),
            "cache_hit": False,
            "cache_backend": "write_through",
            "cache_key": cache_key,
            "cached_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        self._redis_set(cache_key, safe_value, safe_metadata)
        self._db_set(cache_key, tool_name, tool_version, safe_value, safe_metadata, expires_at)

    def get_or_set(
        self,
        *,
        tool_name: str,
        tool_version: str,
        arguments: dict[str, Any],
        provider_identity: dict[str, Any] | None,
        fetcher: Callable[[], Any],
    ) -> Any:
        cache_key = build_tool_cache_key(
            tool_name=tool_name,
            tool_version=tool_version,
            arguments=arguments,
            provider_identity=provider_identity or {},
        )
        with tool_span(
            tool_name,
            config_context=self.config_context,
            attributes={
                "tool_version": tool_version,
                "cache_key": cache_key,
                "tool_cache_enabled": self.enabled,
            },
        ):
            hit = self.get(cache_key)
            if hit is not None:
                set_span_attributes(
                    {
                        "cache_hit": True,
                        "cache_backend": hit.backend,
                        "cache_key": hit.cache_key,
                    },
                    config_context=self.config_context,
                )
                return hit.value
            value = fetcher()
            self.set(
                cache_key,
                tool_name,
                tool_version,
                value,
                metadata={"tool_name": tool_name, "tool_version": tool_version},
            )
            set_span_attributes(
                {
                    "cache_hit": False,
                    "cache_backend": "miss",
                    "cache_key": cache_key,
                },
                config_context=self.config_context,
            )
            return value

    def _build_redis_client(self) -> Any | None:
        backend = str(self.config_context.get("tool_cache_backend") or "redis_postgres").lower()
        if "redis" not in backend:
            return None
        redis_url = (
            self.config_context.get("tool_cache_redis_url")
            or os.getenv("TOOL_CACHE_REDIS_URL")
            or os.getenv("CELERY_BROKER_URL")
        )
        if not redis_url:
            return None
        try:
            return redis.Redis.from_url(str(redis_url), socket_connect_timeout=1, socket_timeout=1)
        except RedisError as exc:
            logger.warning("Tool cache Redis initialization failed: %s", exc)
            return None

    def _build_db_store(self) -> Any | None:
        backend = str(self.config_context.get("tool_cache_backend") or "redis_postgres").lower()
        if "postgres" not in backend and "database" not in backend:
            return None
        if not _bool_config(self.config_context, "tool_cache_db_enabled", True):
            return None
        return PostgresToolCacheStore()

    def _redis_get(self, cache_key: str) -> dict[str, Any] | None:
        if self.redis_client is None:
            return None
        try:
            raw_value = self.redis_client.get(_redis_key(cache_key))
        except RedisError as exc:
            logger.warning("Tool cache Redis read failed: %s", exc)
            return None
        if not raw_value:
            return None
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")
        try:
            payload = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) and "value" in payload else None

    def _redis_set(self, cache_key: str, value: Any, metadata: dict[str, Any]) -> None:
        if self.redis_client is None:
            return
        payload = json.dumps(
            {"value": _json_safe(value), "metadata": _json_safe(metadata)},
            sort_keys=True,
            default=str,
        )
        try:
            self.redis_client.setex(_redis_key(cache_key), self.ttl_seconds, payload)
        except RedisError as exc:
            logger.warning("Tool cache Redis write failed: %s", exc)

    def _db_get(self, cache_key: str) -> dict[str, Any] | None:
        if self.db_store is None:
            return None
        return self.db_store.get(cache_key)

    def _db_set(
        self,
        cache_key: str,
        tool_name: str,
        tool_version: str,
        value: Any,
        metadata: dict[str, Any],
        expires_at: datetime,
    ) -> None:
        if self.db_store is None:
            return
        self.db_store.set(cache_key, tool_name, tool_version, value, metadata, expires_at)


def cached_tool_call(
    config_context: dict[str, Any] | None,
    *,
    tool_name: str,
    tool_version: str,
    arguments: dict[str, Any],
    provider_identity: dict[str, Any] | None,
    fetcher: Callable[[], Any],
    cache: ToolCache | None = None,
) -> Any:
    if not config_context:
        return fetcher()
    tool_cache = cache or ToolCache(config_context)
    return tool_cache.get_or_set(
        tool_name=tool_name,
        tool_version=tool_version,
        arguments=arguments,
        provider_identity=provider_identity,
        fetcher=fetcher,
    )


async def cached_tool_call_async(
    config_context: dict[str, Any] | None,
    *,
    tool_name: str,
    tool_version: str,
    arguments: dict[str, Any],
    provider_identity: dict[str, Any] | None,
    fetcher: Callable[[], Any],
    cache: ToolCache | None = None,
) -> Any:
    if not config_context:
        return fetcher()
    return await run_tool_async_or_threaded(
        cached_tool_call,
        config_context,
        tool_name=tool_name,
        tool_version=tool_version,
        arguments=arguments,
        provider_identity=provider_identity,
        fetcher=fetcher,
        cache=cache,
        config_context=config_context,
    )


def build_tool_cache_key(
    *,
    tool_name: str,
    tool_version: str,
    arguments: dict[str, Any],
    provider_identity: dict[str, Any] | None = None,
) -> str:
    material = {
        "tool_name": tool_name,
        "tool_version": tool_version,
        "arguments": normalize_cache_material(arguments),
        "provider_identity": normalize_cache_material(provider_identity or {}),
    }
    payload = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_cache_material(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            key_text = str(key)
            if SECRET_KEY_RE.search(key_text):
                continue
            item = value[key]
            if PII_KEY_RE.search(key_text):
                normalized[key_text] = "[REDACTED_PII]"
            else:
                normalized[key_text] = normalize_cache_material(item)
        return normalized
    if isinstance(value, (list, tuple, set)):
        return [normalize_cache_material(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def build_cached_serper_tool(config_context: dict[str, Any], purpose: str = "search") -> Any:
    from crewai_tools import SerperDevTool
    from pydantic import Field

    try:
        from crewai.tools import BaseTool
    except ImportError:
        from crewai_tools import BaseTool

    class _CachedToolWrapper(BaseTool):
        delegate: Any = Field(exclude=True)
        cache: ToolCache = Field(exclude=True)
        tool_version: str = "serper-dev:v1"
        provider_identity: dict[str, Any] = Field(default_factory=dict)

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            return self.cache.get_or_set(
                tool_name=self.name,
                tool_version=self.tool_version,
                arguments={"args": args, "kwargs": kwargs, "purpose": purpose},
                provider_identity=self.provider_identity,
                fetcher=lambda: self.delegate._run(*args, **kwargs),
            )

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:
            return await run_tool_async_or_threaded(
                self._run,
                *args,
                config_context=config_context,
                **kwargs,
            )

    delegate = SerperDevTool()
    return _CachedToolWrapper(
        name=delegate.name,
        description=delegate.description,
        args_schema=delegate.args_schema,
        delegate=delegate,
        cache=ToolCache(config_context),
        provider_identity={
            "provider": "serper",
            "purpose": purpose,
            "base_url": getattr(delegate, "base_url", None),
            "search_type": getattr(delegate, "search_type", None),
            "n_results": getattr(delegate, "n_results", None),
        },
    )


def build_cached_scrape_tool(config_context: dict[str, Any], purpose: str = "scrape") -> Any:
    from crewai_tools import ScrapeWebsiteTool
    from pydantic import Field

    try:
        from crewai.tools import BaseTool
    except ImportError:
        from crewai_tools import BaseTool

    class _CachedToolWrapper(BaseTool):
        delegate: Any = Field(exclude=True)
        cache: ToolCache = Field(exclude=True)
        tool_version: str = "scrape-website:v1"
        provider_identity: dict[str, Any] = Field(default_factory=dict)

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            return self.cache.get_or_set(
                tool_name=self.name,
                tool_version=self.tool_version,
                arguments={"args": args, "kwargs": kwargs, "purpose": purpose},
                provider_identity=self.provider_identity,
                fetcher=lambda: self.delegate._run(*args, **kwargs),
            )

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:
            return await run_tool_async_or_threaded(
                self._run,
                *args,
                config_context=config_context,
                **kwargs,
            )

    delegate = ScrapeWebsiteTool()
    return _CachedToolWrapper(
        name=delegate.name,
        description=delegate.description,
        args_schema=delegate.args_schema,
        delegate=delegate,
        cache=ToolCache(config_context),
        provider_identity={"provider": "scrape_website", "purpose": purpose},
    )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _redis_key(cache_key: str) -> str:
    return f"{TOOL_CACHE_KEY_PREFIX}{cache_key}"


def _redact_text(value: str) -> str:
    return PHONE_RE.sub("[REDACTED_PHONE]", EMAIL_RE.sub("[REDACTED_EMAIL]", value))


def _bool_config(config_context: dict[str, Any], key: str, default: bool) -> bool:
    value = config_context.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _positive_int_config(config_context: dict[str, Any], key: str, default: int) -> int:
    try:
        parsed = int(config_context.get(key) or default)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default
