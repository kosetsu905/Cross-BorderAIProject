import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")
DEFAULT_TOOL_EXECUTION_MAX_WORKERS = 8


@lru_cache(maxsize=8)
def _executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=max(1, max_workers),
        thread_name_prefix="tool-io",
    )


async def run_tool_async_or_threaded(
    function: Callable[..., T],
    *args: Any,
    config_context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> T:
    """Run a blocking tool function in a bounded executor when async mode is enabled."""
    context = config_context or {}
    if not _bool_config(context, "tool_execution_async_enabled", True):
        return function(*args, **kwargs)

    max_workers = _positive_int_config(
        context,
        "tool_execution_max_workers",
        DEFAULT_TOOL_EXECUTION_MAX_WORKERS,
    )
    loop = asyncio.get_running_loop()
    bound_call = partial(function, *args, **kwargs)
    return await loop.run_in_executor(_executor(max_workers), bound_call)


class AsyncToolExecutionMixin:
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        context = getattr(self, "tool_cache_context", None) or getattr(
            self,
            "tool_execution_context",
            None,
        )
        return await run_tool_async_or_threaded(
            self._run,
            *args,
            config_context=context,
            **kwargs,
        )


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
    return parsed if parsed > 0 else default
