import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "docs" / "knowledge_base"
SUPPORTED_KNOWLEDGE_PATTERNS = ("*.md", "*.txt")


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    heading: str
    content: str
    embedding: tuple[float, ...]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _embed(text: str, dimensions: int = 256) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def _similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _split_markdown(path: Path) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    heading = path.stem.replace("_", " ").title()
    buffer: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if buffer:
                chunks.append((heading, "\n".join(buffer).strip()))
                buffer = []
            heading = line.lstrip("#").strip() or heading
            continue
        buffer.append(line)

    if buffer:
        chunks.append((heading, "\n".join(buffer).strip()))

    return [(item_heading, content) for item_heading, content in chunks if content]


@lru_cache(maxsize=8)
def load_knowledge_chunks(knowledge_dir: str) -> tuple[KnowledgeChunk, ...]:
    base_dir = Path(knowledge_dir)
    if not base_dir.exists():
        return tuple()

    chunks: list[KnowledgeChunk] = []
    knowledge_paths = [
        path
        for pattern in SUPPORTED_KNOWLEDGE_PATTERNS
        for path in base_dir.glob(pattern)
    ]
    for path in sorted(knowledge_paths):
        for heading, content in _split_markdown(path):
            chunks.append(
                KnowledgeChunk(
                    source=str(path),
                    heading=heading,
                    content=content,
                    embedding=_embed(f"{heading}\n{content}"),
                )
            )
    return tuple(chunks)


def search_knowledge_base(
    query: str,
    knowledge_dir: str | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    chunks = load_knowledge_chunks(str(Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR))
    query_embedding = _embed(query)
    ranked = sorted(
        (
            (_similarity(query_embedding, chunk.embedding), chunk)
            for chunk in chunks
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        {
            "score": round(score, 4),
            "source": chunk.source,
            "heading": chunk.heading,
            "content": chunk.content,
        }
        for score, chunk in ranked[:top_k]
        if score > 0
    ]


class SupportKnowledgeSearchTool(BaseTool):
    name: str = "Support Knowledge Base Search"
    description: str = (
        "Searches the internal support knowledge base for return policy, shipping, "
        "refund, compensation, exchange, and escalation guidance."
    )
    knowledge_dir: str = str(DEFAULT_KNOWLEDGE_DIR)

    def _run(self, query: str, top_k: int = 3) -> dict[str, Any]:
        results = search_knowledge_base(query, self.knowledge_dir, top_k)
        return {
            "query": query,
            "data_source": "local_vector_knowledge_base",
            "confidence_level": "medium" if results else "low",
            "results": results,
            "status": "found" if results else "not_found",
            "assumption_notice": (
                "Results come from the local markdown/text knowledge base. Validate against official policy updates."
            ),
        }
