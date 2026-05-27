from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import redis
import redis.asyncio as aioredis
from redis.exceptions import RedisError

DEFAULT_SESSION_TTL_SECONDS = 24 * 60 * 60
DEFAULT_HISTORY_LIMIT = 20


class SessionManager:
    def __init__(
        self,
        *,
        redis_url: str | None,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        redis_client: Any | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = max(1, int(ttl_seconds or DEFAULT_SESSION_TTL_SECONDS))
        self.history_limit = max(1, int(history_limit or DEFAULT_HISTORY_LIMIT))
        self._redis = redis_client
        self._enabled = bool(redis_client or redis_url)

    @classmethod
    def from_config(cls, config: Any) -> "SessionManager":
        return cls(
            redis_url=getattr(config, "support_session_redis_url", None),
            ttl_seconds=getattr(config, "support_session_ttl_seconds", DEFAULT_SESSION_TTL_SECONDS),
            history_limit=getattr(config, "support_session_history_limit", DEFAULT_HISTORY_LIMIT),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        client = self._client()
        if client is None:
            return None
        try:
            raw = client.get(self._key(session_id))
        except RedisError:
            return None
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            loaded = json.loads(str(raw))
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None

    def record_inbound_message(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        message: dict[str, Any],
        language_preference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._record_message(
            session_id=session_id,
            channel=channel,
            customer_id=customer_id,
            direction="inbound",
            message=message,
            language_preference=language_preference,
            metadata=metadata,
        )

    def record_outbound_message(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        message: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._record_message(
            session_id=session_id,
            channel=channel,
            customer_id=customer_id,
            direction="outbound",
            message=message,
            language_preference=None,
            metadata=metadata,
        )

    def create_or_update(self, session_id: str, metadata: dict[str, Any], user_turn: str) -> dict[str, Any]:
        return self.record_inbound_message(
            session_id=session_id,
            channel=str(metadata.get("channel") or "unknown"),
            customer_id=metadata.get("customer_id"),
            language_preference=metadata.get("language") or metadata.get("language_preference"),
            metadata=metadata,
            message={"role": "user", "content": user_turn, "text": user_turn},
        )

    def is_window_expired(self, session_id: str) -> bool:
        session = self.load_session(session_id)
        if not session or not session.get("window_expiry"):
            return True
        try:
            expiry = datetime.fromisoformat(str(session["window_expiry"]))
        except ValueError:
            return True
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return datetime.now(UTC) > expiry

    def log_ai_response(self, session_id: str, ai_text: str) -> dict[str, Any]:
        session = self.load_session(session_id) or {}
        return self.record_outbound_message(
            session_id=session_id,
            channel=str(session.get("channel") or "unknown"),
            customer_id=session.get("customer_id"),
            metadata=session.get("metadata") if isinstance(session.get("metadata"), dict) else {},
            message={"role": "ai", "content": ai_text, "text": ai_text},
        )

    def update_language_preference(self, session_id: str, language: str | None) -> dict[str, Any]:
        if not language:
            return {"status": "skipped", "error": None}
        session = self.load_session(session_id) or self._new_session(
            session_id=session_id,
            channel="unknown",
            customer_id=None,
            language_preference=language,
            metadata={},
        )
        session["language_preference"] = language
        return self._save_session(session)

    def _record_message(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        direction: str,
        message: dict[str, Any],
        language_preference: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "error": None}
        session = self.load_session(session_id) or self._new_session(
            session_id=session_id,
            channel=channel,
            customer_id=customer_id,
            language_preference=language_preference,
            metadata=metadata or {},
        )
        now = datetime.now(UTC)
        session["channel"] = channel or session.get("channel")
        session["customer_id"] = customer_id or session.get("customer_id")
        if language_preference:
            session["language_preference"] = language_preference
        session["metadata"] = {**(session.get("metadata") or {}), **(metadata or {})}
        session["window_expiry"] = (now + timedelta(seconds=self.ttl_seconds)).isoformat()
        entry = {
            **message,
            "direction": direction,
            "recorded_at": now.isoformat(),
        }
        history = list(session.get("history") or [])
        history.append(entry)
        session["history"] = history[-self.history_limit :]
        return self._save_session(session)

    def _save_session(self, session: dict[str, Any]) -> dict[str, Any]:
        client = self._client()
        if client is None:
            return {"status": "disabled", "error": None}
        try:
            client.setex(self._key(str(session["session_id"])), self.ttl_seconds, json.dumps(session, default=str))
        except RedisError as exc:
            return {"status": "unavailable", "error": str(exc)}
        return {"status": "ok", "error": None, "session": session}

    def _new_session(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        language_preference: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "session_id": session_id,
            "channel": channel,
            "customer_id": customer_id,
            "language_preference": language_preference,
            "metadata": metadata,
            "history": [],
            "window_started_at": now.isoformat(),
            "window_expiry": (now + timedelta(seconds=self.ttl_seconds)).isoformat(),
        }

    def _client(self) -> Any | None:
        if not self.enabled:
            return None
        if self._redis is None and self.redis_url:
            try:
                self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
            except RedisError:
                return None
        return self._redis

    @staticmethod
    def _key(session_id: str) -> str:
        return f"support:session:{session_id}"


class AsyncSessionManager:
    def __init__(
        self,
        *,
        redis_url: str | None,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        redis_client: Any | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = max(1, int(ttl_seconds or DEFAULT_SESSION_TTL_SECONDS))
        self.history_limit = max(1, int(history_limit or DEFAULT_HISTORY_LIMIT))
        self._redis = redis_client
        self._enabled = bool(redis_client or redis_url)

    @classmethod
    def from_config(cls, config: Any) -> "AsyncSessionManager":
        return cls(
            redis_url=getattr(config, "support_session_redis_url", None),
            ttl_seconds=getattr(config, "support_session_ttl_seconds", DEFAULT_SESSION_TTL_SECONDS),
            history_limit=getattr(config, "support_session_history_limit", DEFAULT_HISTORY_LIMIT),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        client = self._client()
        if client is None:
            return None
        try:
            raw = await client.get(self._key(session_id))
        except RedisError:
            return None
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            loaded = json.loads(str(raw))
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None

    async def record_inbound_message(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        message: dict[str, Any],
        language_preference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._record_message(
            session_id=session_id,
            channel=channel,
            customer_id=customer_id,
            direction="inbound",
            message=message,
            language_preference=language_preference,
            metadata=metadata,
        )

    async def record_outbound_message(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        message: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._record_message(
            session_id=session_id,
            channel=channel,
            customer_id=customer_id,
            direction="outbound",
            message=message,
            language_preference=None,
            metadata=metadata,
        )

    async def create_or_update(self, session_id: str, metadata: dict[str, Any], user_turn: str) -> dict[str, Any]:
        return await self.record_inbound_message(
            session_id=session_id,
            channel=str(metadata.get("channel") or "unknown"),
            customer_id=metadata.get("customer_id"),
            language_preference=metadata.get("language") or metadata.get("language_preference"),
            metadata=metadata,
            message={"role": "user", "content": user_turn, "text": user_turn},
        )

    async def is_window_expired(self, session_id: str) -> bool:
        session = await self.load_session(session_id)
        if not session or not session.get("window_expiry"):
            return True
        try:
            expiry = datetime.fromisoformat(str(session["window_expiry"]))
        except ValueError:
            return True
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return datetime.now(UTC) > expiry

    async def log_ai_response(self, session_id: str, ai_text: str) -> dict[str, Any]:
        session = await self.load_session(session_id) or {}
        return await self.record_outbound_message(
            session_id=session_id,
            channel=str(session.get("channel") or "unknown"),
            customer_id=session.get("customer_id"),
            metadata=session.get("metadata") if isinstance(session.get("metadata"), dict) else {},
            message={"role": "ai", "content": ai_text, "text": ai_text},
        )

    async def update_language_preference(self, session_id: str, language: str | None) -> dict[str, Any]:
        if not language:
            return {"status": "skipped", "error": None}
        session = await self.load_session(session_id) or self._new_session(
            session_id=session_id,
            channel="unknown",
            customer_id=None,
            language_preference=language,
            metadata={},
        )
        session["language_preference"] = language
        return await self._save_session(session)

    async def _record_message(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        direction: str,
        message: dict[str, Any],
        language_preference: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "error": None}
        session = await self.load_session(session_id) or self._new_session(
            session_id=session_id,
            channel=channel,
            customer_id=customer_id,
            language_preference=language_preference,
            metadata=metadata or {},
        )
        now = datetime.now(UTC)
        session["channel"] = channel or session.get("channel")
        session["customer_id"] = customer_id or session.get("customer_id")
        if language_preference:
            session["language_preference"] = language_preference
        session["metadata"] = {**(session.get("metadata") or {}), **(metadata or {})}
        session["window_expiry"] = (now + timedelta(seconds=self.ttl_seconds)).isoformat()
        entry = {
            **message,
            "direction": direction,
            "recorded_at": now.isoformat(),
        }
        history = list(session.get("history") or [])
        history.append(entry)
        session["history"] = history[-self.history_limit :]
        return await self._save_session(session)

    async def _save_session(self, session: dict[str, Any]) -> dict[str, Any]:
        client = self._client()
        if client is None:
            return {"status": "disabled", "error": None}
        try:
            await client.setex(self._key(str(session["session_id"])), self.ttl_seconds, json.dumps(session, default=str))
        except RedisError as exc:
            return {"status": "unavailable", "error": str(exc)}
        return {"status": "ok", "error": None, "session": session}

    def _new_session(
        self,
        *,
        session_id: str,
        channel: str,
        customer_id: str | None,
        language_preference: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "session_id": session_id,
            "channel": channel,
            "customer_id": customer_id,
            "language_preference": language_preference,
            "metadata": metadata,
            "history": [],
            "window_started_at": now.isoformat(),
            "window_expiry": (now + timedelta(seconds=self.ttl_seconds)).isoformat(),
        }

    def _client(self) -> Any | None:
        if not self.enabled:
            return None
        if self._redis is None and self.redis_url:
            try:
                self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            except RedisError:
                return None
        return self._redis

    @staticmethod
    def _key(session_id: str) -> str:
        return f"support:session:{session_id}"
