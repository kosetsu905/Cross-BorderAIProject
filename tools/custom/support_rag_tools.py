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

from utils.tool_cache import cached_tool_call
from utils.tool_execution import AsyncToolExecutionMixin


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
PRICE_RE = re.compile(r"\$\s*\d+(?:\.\d{1,2})?")
CARTON_QTY_RE = re.compile(r"(?:1|one)\s+carton\s+contains\s+([0-9]+\s*(?:pcs|pcs\.|pieces)?)", re.IGNORECASE)
CARTON_SIZE_RE = re.compile(r"(?:1\s+carton\s+size|box\s+gauge)\s*:?\s*([^,\n]+)(?:,\s*([0-9.]+\s*kg))?", re.IGNORECASE)
SINGLE_PRODUCT_SPEC_RE = re.compile(
    r"single\s+product\s+(?:spec(?:ification|ifications|s)?|size)\s*:?\s*(.+)$",
    re.IGNORECASE,
)
WEIGHT_RE = re.compile(r"([0-9.]+\s*(?:kg|g))\s*$", re.IGNORECASE)
LINE_WEIGHT_RE = re.compile(r"^weight\s*:?\s*([0-9.]+\s*kg)\s*$", re.IGNORECASE)
DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "docs" / "knowledge_base"
SUPPORTED_KNOWLEDGE_PATTERNS = ("*.md", "*.txt", "*.pdf")
PDF_CHUNK_SIZE = 1800
PDF_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    heading: str
    content: str
    embedding: tuple[float, ...]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _product_tokens(text: str) -> set[str]:
    stopwords = {
        "hello",
        "hi",
        "hey",
        "want",
        "would",
        "like",
        "buy",
        "purchase",
        "order",
        "discount",
        "quote",
        "price",
        "pricing",
        "can",
        "get",
        "for",
        "the",
        "and",
        "with",
        "please",
        "share",
        "details",
        "product",
        "spec",
        "specs",
        "specification",
        "specifications",
        "feature",
        "features",
        "catalog",
        "stock",
        "status",
        "available",
        "availability",
        "variant",
        "variants",
        "current",
        "unit",
        "size",
        "weight",
        "site",
        "tell",
        "how",
        "bulk",
        "carton",
        "of",
        "is",
        "this",
        "a",
        "an",
        "to",
        "i",
    }
    return {token for token in _tokens(text) if token not in stopwords and not token.isdigit()}


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


def _split_pdf(path: Path) -> list[tuple[str, str]]:
    try:
        import pdfplumber
    except ImportError:
        return []

    chunks: list[tuple[str, str]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                heading = f"{path.stem} - Page {page_index}"
                chunks.extend((heading, chunk) for chunk in _chunk_text(text) if chunk)
    except Exception:
        return []
    return chunks


def _chunk_text(text: str, chunk_size: int = PDF_CHUNK_SIZE, overlap: int = PDF_CHUNK_OVERLAP) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(normalized) <= chunk_size:
        return [normalized] if normalized else []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_knowledge_file(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".pdf":
        return _split_pdf(path)
    return _split_markdown(path)


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
        for heading, content in _split_knowledge_file(path):
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


def extract_catalog_product_offer(
    query: str,
    knowledge_dir: str | None = None,
) -> dict[str, Any]:
    """Extract a deterministic product offer from local catalog chunks.

    This is intentionally stricter than semantic search: prices and carton terms
    must be copied from a source line near a matched product, never inferred.
    """
    query_tokens = _product_tokens(query)
    if not query_tokens:
        return {"status": "not_found"}

    chunks = load_knowledge_chunks(str(Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR))
    best: tuple[int, int, int, KnowledgeChunk, list[str], str] | None = None
    for chunk in chunks:
        if Path(chunk.source).suffix.lower() != ".pdf":
            continue
        lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if not PRICE_RE.search(line):
                continue
            window = lines[max(0, index - 1) : min(len(lines), index + 4)]
            window_text = " ".join(window)
            matched = query_tokens.intersection(_product_tokens(window_text))
            if not matched:
                continue
            if not _catalog_match_is_specific_enough(query_tokens, matched, window_text):
                continue
            score = len(matched)
            product_name = _catalog_product_name(query, window, PRICE_RE.search(window_text).group(0))
            product_tokens = _product_tokens(product_name)
            exact_bonus = 2 if product_tokens == query_tokens else 0
            extra_penalty = -len(product_tokens - query_tokens)
            candidate = (score, exact_bonus, extra_penalty, chunk, window, product_name)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    if best is None:
        return {"status": "not_found"}

    _, _, _, chunk, evidence_lines, product_name = best
    evidence = " ".join(evidence_lines)
    price_match = PRICE_RE.search(evidence)
    qty_match = CARTON_QTY_RE.search(evidence)
    carton_size, carton_weight = _carton_packaging(evidence_lines)
    single_size, single_weight = _single_product_spec(evidence_lines)

    return {
        "status": "found",
        "product_found": True,
        "product_name": product_name,
        "unit_price": price_match.group(0).replace(" ", "") if price_match else None,
        "carton_quantity": qty_match.group(1).upper().replace("PCS.", "PCS") if qty_match else None,
        "carton_size": carton_size,
        "carton_weight": carton_weight,
        "single_product_size": single_size,
        "single_product_weight": single_weight,
        "discount_policy": "Discounts are not pre-approved in the catalog; customer requests must be reviewed before quoting any reduced price.",
        "source": chunk.source,
        "heading": chunk.heading,
        "evidence": evidence_lines,
        "data_source": "local_pdf_catalog",
    }


def _catalog_match_is_specific_enough(query_tokens: set[str], matched: set[str], window_text: str) -> bool:
    if not query_tokens:
        return False
    model_tokens = {token for token in query_tokens if any(character.isdigit() for character in token)}
    window_tokens = _product_tokens(window_text)
    if model_tokens and not model_tokens.intersection(window_tokens):
        return False
    if len(query_tokens) >= 2 and len(matched) < 2:
        return False
    return True


def _catalog_product_name(query: str, evidence_lines: list[str], price: str | None) -> str:
    query_tokens = _product_tokens(query)
    product_parts: list[str] = []
    for line in evidence_lines:
        if line.lower().startswith("single product"):
            continue
        line = re.sub(r"(?:1|one)\s+carton\s+contains\s+[0-9]+\s*(?:pcs|pcs\.|pieces)?", "", line, flags=re.I)
        if price and price in line:
            before_price = line.split(price, 1)[0]
            line = before_price.strip()
        line = re.split(r"\bbox\s+gauge\b", line, maxsplit=1, flags=re.I)[0].strip()
        if line.lower().startswith(("1 carton", "one carton", "weight")):
            continue
        if _product_tokens(line).intersection(query_tokens):
            product_parts.append(line)
    product = " ".join(product_parts)
    product = re.sub(r"\s+", " ", product).strip(" ,")
    return product or query.strip()


def _carton_packaging(evidence_lines: list[str]) -> tuple[str | None, str | None]:
    carton_size = None
    inline_weight = None
    for line in evidence_lines:
        match = CARTON_SIZE_RE.search(line.strip())
        if match:
            carton_size = match.group(1).strip()
            inline_weight = match.group(2).strip() if match.group(2) else None
            break
    if inline_weight:
        return carton_size, inline_weight
    for line in evidence_lines:
        match = LINE_WEIGHT_RE.search(line.strip())
        if match:
            return carton_size, match.group(1).strip()
    return carton_size, None


def _single_product_spec(evidence_lines: list[str]) -> tuple[str | None, str | None]:
    for line in evidence_lines:
        match = SINGLE_PRODUCT_SPEC_RE.search(line.strip())
        if match:
            raw_spec = match.group(1).strip(" .")
            weight_match = WEIGHT_RE.search(raw_spec)
            if not weight_match:
                return raw_spec, None
            weight = weight_match.group(1).strip()
            size = raw_spec[: weight_match.start()].strip(" ,.，")
            return size or None, weight
    return None, None


class SupportKnowledgeSearchTool(AsyncToolExecutionMixin, BaseTool):
    name: str = "Support Knowledge Base Search"
    description: str = (
        "Searches the internal support knowledge base for return policy, shipping, "
        "refund, compensation, exchange, and escalation guidance."
    )
    knowledge_dir: str = str(DEFAULT_KNOWLEDGE_DIR)
    tool_cache_context: dict[str, Any] | None = None

    def _run(self, query: str, top_k: int = 3) -> dict[str, Any]:
        results = cached_tool_call(
            self.tool_cache_context,
            tool_name=self.name,
            tool_version="v1",
            arguments={"query": query, "knowledge_dir": self.knowledge_dir, "top_k": top_k},
            provider_identity={"provider": "local_support_knowledge"},
            fetcher=lambda: search_knowledge_base(query, self.knowledge_dir, top_k),
        )
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
