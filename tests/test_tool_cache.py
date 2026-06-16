import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from utils.tool_cache import (
    InMemoryToolCacheStore,
    ToolCache,
    build_tool_cache_key,
    cached_tool_call,
    normalize_cache_material,
)
from utils.tool_execution import run_tool_async_or_threaded


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.ttls[key] = ttl
        self.values[key] = value


def _config(**overrides: Any) -> dict[str, Any]:
    return {
        "tool_cache_enabled": True,
        "tool_cache_backend": "redis_postgres",
        "tool_cache_ttl_seconds": 60,
        "tool_cache_db_enabled": True,
        "tool_cache_max_value_bytes": 1048576,
        "tool_execution_async_enabled": True,
        "tool_execution_max_workers": 2,
        **overrides,
    }


def test_cache_key_normalizes_argument_order_and_excludes_secrets() -> None:
    left = build_tool_cache_key(
        tool_name="Search",
        tool_version="v1",
        arguments={"b": 2, "a": 1, "api_key": "secret-one"},
        provider_identity={"provider": "serper", "access_token": "token-one"},
    )
    right = build_tool_cache_key(
        tool_name="Search",
        tool_version="v1",
        arguments={"api_key": "secret-two", "a": 1, "b": 2},
        provider_identity={"access_token": "token-two", "provider": "serper"},
    )

    assert left == right


def test_cache_key_includes_provider_model_date_and_deep_read_flags() -> None:
    base = {
        "tool_name": "Benchmark",
        "tool_version": "v1",
        "arguments": {"market": "US", "date_range": "Last 30 Days", "deep_read": False},
        "provider_identity": {"provider": "serper", "model": "default"},
    }

    assert build_tool_cache_key(**base) != build_tool_cache_key(
        **{**base, "arguments": {**base["arguments"], "deep_read": True}}
    )
    assert build_tool_cache_key(**base) != build_tool_cache_key(
        **{**base, "arguments": {**base["arguments"], "date_range": "Last 60 Days"}}
    )
    assert build_tool_cache_key(**base) != build_tool_cache_key(
        **{**base, "provider_identity": {"provider": "serper", "model": "new"}}
    )


def test_normalization_redacts_pii_like_values() -> None:
    normalized = normalize_cache_material(
        {
            "customer_email": "buyer@example.com",
            "note": "Call +1 212 555 0101 or email person@example.com",
        }
    )

    assert normalized["customer_email"] == "[REDACTED_PII]"
    assert "[REDACTED_PHONE]" in normalized["note"]
    assert "[REDACTED_EMAIL]" in normalized["note"]


def test_redis_hit_returns_without_calling_fetcher() -> None:
    redis_client = FakeRedis()
    db_store = InMemoryToolCacheStore()
    cache = ToolCache(_config(), redis_client=redis_client, db_store=db_store)
    calls = {"count": 0}

    def fetcher() -> dict[str, str]:
        calls["count"] += 1
        return {"result": "live"}

    first = cache.get_or_set(
        tool_name="Search",
        tool_version="v1",
        arguments={"q": "market"},
        provider_identity={"provider": "serper"},
        fetcher=fetcher,
    )
    second = cache.get_or_set(
        tool_name="Search",
        tool_version="v1",
        arguments={"q": "market"},
        provider_identity={"provider": "serper"},
        fetcher=fetcher,
    )

    assert first == {"result": "live"}
    assert second == {"result": "live"}
    assert calls["count"] == 1


def test_db_fallback_populates_redis_without_calling_fetcher() -> None:
    redis_client = FakeRedis()
    db_store = InMemoryToolCacheStore()
    config = _config()
    cache_key = build_tool_cache_key(
        tool_name="Search",
        tool_version="v1",
        arguments={"q": "market"},
        provider_identity={"provider": "serper"},
    )
    db_store.set(
        cache_key,
        "Search",
        "v1",
        {"result": "db"},
        {"cache_key": cache_key},
        datetime.now(UTC) + timedelta(seconds=60),
    )
    cache = ToolCache(config, redis_client=redis_client, db_store=db_store)

    result = cache.get_or_set(
        tool_name="Search",
        tool_version="v1",
        arguments={"q": "market"},
        provider_identity={"provider": "serper"},
        fetcher=lambda: {"result": "live"},
    )

    assert result == {"result": "db"}
    assert redis_client.values


def test_write_through_stores_to_redis_and_db() -> None:
    redis_client = FakeRedis()
    db_store = InMemoryToolCacheStore()
    cache = ToolCache(_config(), redis_client=redis_client, db_store=db_store)

    result = cached_tool_call(
        _config(),
        tool_name="Commerce Metrics Read",
        tool_version="v1",
        arguments={"region": "US"},
        provider_identity={"provider": "commerce_api"},
        fetcher=lambda: {"orders": 10},
        cache=cache,
    )

    assert result == {"orders": 10}
    assert redis_client.values
    assert db_store.entries


def test_expired_db_row_is_ignored() -> None:
    redis_client = FakeRedis()
    db_store = InMemoryToolCacheStore()
    cache_key = build_tool_cache_key(
        tool_name="Search",
        tool_version="v1",
        arguments={"q": "old"},
        provider_identity={"provider": "serper"},
    )
    db_store.set(
        cache_key,
        "Search",
        "v1",
        {"result": "old"},
        {"cache_key": cache_key},
        datetime.now(UTC) - timedelta(seconds=1),
    )
    cache = ToolCache(_config(), redis_client=redis_client, db_store=db_store)

    result = cache.get_or_set(
        tool_name="Search",
        tool_version="v1",
        arguments={"q": "old"},
        provider_identity={"provider": "serper"},
        fetcher=lambda: {"result": "fresh"},
    )

    assert result == {"result": "fresh"}


def test_oversized_result_is_not_cached() -> None:
    redis_client = FakeRedis()
    db_store = InMemoryToolCacheStore()
    cache = ToolCache(
        _config(tool_cache_max_value_bytes=8),
        redis_client=redis_client,
        db_store=db_store,
    )

    result = cache.get_or_set(
        tool_name="Search",
        tool_version="v1",
        arguments={"q": "large"},
        provider_identity={"provider": "serper"},
        fetcher=lambda: {"payload": "too-large-to-cache"},
    )

    assert result == {"payload": "too-large-to-cache"}
    assert not redis_client.values
    assert not db_store.entries


def test_async_execution_helper_uses_bounded_executor() -> None:
    thread_names: list[str] = []

    def blocking_call() -> str:
        import threading

        thread_names.append(threading.current_thread().name)
        return "done"

    result = asyncio.run(
        run_tool_async_or_threaded(blocking_call, config_context=_config(tool_execution_max_workers=1))
    )

    assert result == "done"
    assert thread_names
    assert thread_names[0].startswith("tool-io")
