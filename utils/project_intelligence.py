"""Project intelligence loader for CrewAI agents.

Loads AGENTS.md content and injects relevant sections into agent backstories
to ensure all runtime agents follow project standards.

Usage in crew files:

    from utils.project_intelligence import augment_agents_config

    agents_config = _load_yaml_config("agents.yaml")
    agents_config = augment_agents_config(
        agents_config,
        workflow="marketing",   # optional: include workflow-specific guidelines
        sections=None,           # optional: override default sections
    )
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AGENTS_MD_CACHE: str | None = None
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default sections injected into every agent
_DEFAULT_SECTIONS = [
    "Security Guidelines",
    "Code Standards",
    "Anti-Patterns",
]


def load_project_intelligence() -> str:
    """Load and cache AGENTS.md content.

    Returns:
        Full content of AGENTS.md, or empty string if not found.
    """
    global _AGENTS_MD_CACHE

    if _AGENTS_MD_CACHE is not None:
        return _AGENTS_MD_CACHE

    agents_md_path = _PROJECT_ROOT / "AGENTS.md"

    if not agents_md_path.exists():
        logger.warning("AGENTS.md not found at %s", agents_md_path)
        _AGENTS_MD_CACHE = ""
        return ""

    try:
        _AGENTS_MD_CACHE = agents_md_path.read_text(encoding="utf-8")
        logger.debug("Loaded AGENTS.md (%d bytes)", len(_AGENTS_MD_CACHE))
        return _AGENTS_MD_CACHE
    except Exception as e:
        logger.error("Failed to load AGENTS.md: %s", e)
        _AGENTS_MD_CACHE = ""
        return ""


def extract_section(content: str, section_heading: str) -> str:
    """Extract a specific section from AGENTS.md by heading.

    Args:
        content: Full AGENTS.md content.
        section_heading: Heading to extract (e.g., "Security Guidelines").

    Returns:
        Section content including the heading, or empty string if not found.
    """
    if not content:
        return ""

    lines = content.split("\n")
    start_idx = None
    heading_level = 0

    # Find start of section (exact heading match)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            # Extract text after the # symbols
            text = stripped.lstrip("#").strip()
            if section_heading.lower() in text.lower():
                start_idx = idx
                heading_level = len(stripped) - len(stripped.lstrip("#"))
                break

    if start_idx is None:
        return ""

    # Find end of section (next heading of same or higher level)
    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx]
        if line.strip().startswith("#"):
            current_level = len(line) - len(line.lstrip("#"))
            if current_level <= heading_level:
                end_idx = idx
                break

    return "\n".join(lines[start_idx:end_idx]).strip()


def _build_appendix(sections: list[str], workflow: str | None = None) -> str:
    """Build the intelligence appendix to append to agent backstories.

    Args:
        sections: List of section headings to include.
        workflow: Optional workflow type for workflow-specific guidelines.

    Returns:
        Formatted appendix string, or empty string if no sections found.
    """
    content = load_project_intelligence()
    if not content:
        return ""

    all_sections = list(sections)
    if workflow:
        # Insert workflow-specific section at the start
        all_sections.insert(0, f"{workflow.replace('_', ' ').title().replace(' ', '-')}-Crew")
        # Also try the exact heading format from AGENTS.md
        workflow_heading = workflow.replace('_', ' ').title()
        all_sections.insert(1, f"{workflow_heading} Crew")

    parts = []
    seen = set()
    for section in all_sections:
        section_content = extract_section(content, section)
        if section_content and section not in seen:
            parts.append(section_content)
            seen.add(section)

    if not parts:
        return ""

    header = "\n\n---\n\n## Project Standards & Guidelines\n\nFollow these standards throughout your work:\n\n"
    return header + "\n\n".join(parts)


def augment_agents_config(
    agents_config: dict[str, Any],
    workflow: str | None = None,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Augment an agents config dict with project intelligence.

    Call this after loading agents.yaml to inject AGENTS.md guidelines
    into every agent's backstory.

    Args:
        agents_config: Dict loaded from agents.yaml (keyed by agent name).
        workflow: Optional workflow type (e.g., "marketing", "content").
                  If provided, workflow-specific guidelines are also injected.
        sections: Optional list of section headings to include.
                  Defaults to Security Guidelines, Code Standards, Anti-Patterns.

    Returns:
        Augmented copy of agents_config with enhanced backstories.
    """
    if sections is None:
        sections = list(_DEFAULT_SECTIONS)

    appendix = _build_appendix(sections, workflow=workflow)
    if not appendix:
        return agents_config

    augmented = {}
    for agent_name, agent_def in agents_config.items():
        new_def = dict(agent_def)
        if "backstory" in new_def:
            new_def["backstory"] = new_def["backstory"].rstrip() + appendix
        augmented[agent_name] = new_def

    logger.debug(
        "Augmented %d agent(s) with project intelligence%s",
        len(augmented),
        f" for workflow '{workflow}'" if workflow else "",
    )
    return augmented


def augment_task_description(task_description: str, workflow: str | None = None) -> str:
    """Append a lightweight standards reminder to task descriptions.

    This is optional and useful when you want tasks themselves to reinforce
    quality expectations without loading the full AGENTS.md into every task.

    Args:
        task_description: Original task description string.
        workflow: Optional workflow type for context.

    Returns:
        Task description with quality reminder appended.
    """
    reminder = (
        "\n\n[Project Standards: Follow Pydantic strict validation, "
        "handle errors with tenacity retry, log operations with context, "
        "validate all inputs, avoid hardcoded values.]"
    )
    return task_description + reminder


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    content = load_project_intelligence()
    print(f"Loaded AGENTS.md: {len(content)} bytes")

    print("\n--- Security section excerpt ---")
    security = extract_section(content, "Security Guidelines")
    print(security[:300] if security else "(not found)")

    print("\n--- Marketing Crew section excerpt ---")
    mkt = extract_section(content, "Marketing Crew")
    print(mkt[:300] if mkt else "(not found)")

    print("\n--- Augment test ---")
    sample_config = {
        "test_agent": {
            "role": "Test Agent",
            "goal": "Run tests",
            "backstory": "You are a test agent.",
        }
    }
    result = augment_agents_config(sample_config, workflow="marketing")
    print(f"Original backstory length: {len(sample_config['test_agent']['backstory'])}")
    print(f"Augmented backstory length: {len(result['test_agent']['backstory'])}")