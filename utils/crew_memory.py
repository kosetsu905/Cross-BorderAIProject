import re
from pathlib import Path
from typing import Any

from crewai.memory import Memory
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.rag.embeddings.providers.openai.types import OpenAIProviderSpec

from utils.llm_config import build_llm


DEFAULT_MEMORY_WORKFLOWS = {
    "marketing",
    "content",
    "analytics",
    "bizdev",
    "scheduler",
    "sales_improvement",
}
MEMORY_TABLE_RE = re.compile(r"[^A-Za-z0-9_]+")


def build_crew_memory(config_context: dict[str, Any], workflow: str) -> Memory | bool:
    """Build a scoped CrewAI memory object when explicitly enabled.

    Support is intentionally not part of the default allowlist because native
    CrewAI memory may persist customer PII outside the project's support session
    controls.
    """
    workflow_name = str(workflow).strip().lower()
    if not _bool_config(config_context, "crewai_memory_enabled", False):
        return False
    if workflow_name not in _memory_workflows(config_context):
        return False

    api_key = _embedding_api_key(config_context)
    if not api_key:
        raise ValueError(
            "CREWAI_MEMORY_ENABLED requires OPENAI_API_KEY, or LLM_API_KEY with "
            "LLM_PROVIDER=openai, so CrewAI can create memory embeddings."
        )

    storage_path = _memory_storage_path(config_context)
    storage_path.mkdir(parents=True, exist_ok=True)
    embedder: OpenAIProviderSpec = {
        "provider": "openai",
        "config": {
            "api_key": api_key,
            "model_name": str(config_context.get("crewai_memory_embedder_model") or "text-embedding-3-small"),
        },
    }
    root_scope = f"cross_border_ai:{workflow_name}"
    return Memory(
        llm=build_llm(config_context),
        storage=LanceDBStorage(
            path=storage_path,
            table_name=_memory_table_name(workflow_name),
        ),
        embedder=embedder,
        root_scope=root_scope,
    )


def memory_debug_context(memory: Memory | bool, workflow: str) -> dict[str, Any]:
    return {
        "memory_enabled": isinstance(memory, Memory),
        "memory_scope": f"cross_border_ai:{workflow}" if isinstance(memory, Memory) else None,
    }


def _bool_config(config_context: dict[str, Any], key: str, default: bool) -> bool:
    value = config_context.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _memory_workflows(config_context: dict[str, Any]) -> set[str]:
    raw_value = config_context.get("crewai_memory_workflows")
    if raw_value in (None, ""):
        return set(DEFAULT_MEMORY_WORKFLOWS)
    if isinstance(raw_value, str):
        values = raw_value.split(",")
    elif isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        values = []
    workflows = {str(value).strip().lower() for value in values if str(value).strip()}
    return workflows or set(DEFAULT_MEMORY_WORKFLOWS)


def _embedding_api_key(config_context: dict[str, Any]) -> str | None:
    openai_api_key = config_context.get("openai_api_key")
    if openai_api_key:
        return str(openai_api_key)

    provider = str(config_context.get("llm_provider") or "openai").lower()
    llm_api_key = config_context.get("llm_api_key")
    if provider == "openai" and llm_api_key:
        return str(llm_api_key)
    return None


def _memory_storage_path(config_context: dict[str, Any]) -> Path:
    raw_path = str(config_context.get("crewai_memory_storage_path") or "artifacts/crewai_memory")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _memory_table_name(workflow: str) -> str:
    normalized = MEMORY_TABLE_RE.sub("_", workflow.strip().lower()).strip("_")
    return f"{normalized or 'workflow'}_memories"
