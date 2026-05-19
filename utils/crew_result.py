from typing import Any

from utils.usage_tracking import attach_usage_metrics


def serialize_crew_result(result: Any) -> dict[str, Any]:
    pydantic_result = getattr(result, "pydantic", None)
    if pydantic_result is not None:
        if hasattr(pydantic_result, "model_dump"):
            return attach_usage_metrics(pydantic_result.model_dump(), result)
        if hasattr(pydantic_result, "dict"):
            return attach_usage_metrics(pydantic_result.dict(), result)

    json_dict = getattr(result, "json_dict", None)
    if isinstance(json_dict, dict):
        return attach_usage_metrics(json_dict, result)

    raw = getattr(result, "raw", None)
    if raw is not None:
        return attach_usage_metrics({"raw": raw}, result)

    if isinstance(result, dict):
        return attach_usage_metrics(result, result)

    return attach_usage_metrics({"raw": str(result)}, result)
