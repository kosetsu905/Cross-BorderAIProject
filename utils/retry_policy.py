import os
from collections.abc import Iterator


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

RETRYABLE_EXCEPTION_NAMES = {
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionError",
    "ConnectTimeout",
    "InternalServerError",
    "NetworkError",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "ServiceUnavailableError",
    "Timeout",
    "TimeoutError",
    "TimeoutException",
    "TransportError",
    "WriteError",
    "WriteTimeout",
}

NON_RETRYABLE_EXCEPTION_NAMES = {
    "AuthenticationError",
    "BadRequestError",
    "ConfigurationError",
    "KeyError",
    "NotFoundError",
    "PermissionDeniedError",
    "PermissionError",
    "TypeError",
    "UnprocessableEntityError",
    "ValidationError",
    "ValueError",
}


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    return None


def is_retryable_exception(exc: BaseException) -> bool:
    for chained_exc in iter_exception_chain(exc):
        status_code = _status_code(chained_exc)
        if status_code is not None:
            return status_code in RETRYABLE_STATUS_CODES

        name = chained_exc.__class__.__name__
        if name in NON_RETRYABLE_EXCEPTION_NAMES:
            return False
        if name in RETRYABLE_EXCEPTION_NAMES:
            return True

    return False


def retry_countdown_seconds(retries: int) -> int:
    base_delay = int(os.getenv("CELERY_RETRY_BASE_DELAY_SECONDS", "30"))
    max_delay = int(os.getenv("CELERY_RETRY_MAX_DELAY_SECONDS", "300"))
    return min(max_delay, base_delay * (2**max(0, retries)))
