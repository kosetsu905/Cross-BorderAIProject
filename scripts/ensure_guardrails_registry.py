from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from install_guardrails_hub_validators import REGISTRY_ENTRIES


REGISTRY_PATH = Path.cwd() / ".guardrails" / "hub_registry.json"


def main() -> None:
    if _registry_has_validators(REGISTRY_PATH):
        return

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    installed_at = datetime.now(UTC).isoformat()
    registry = {
        "version": 1,
        "validators": {
            validator_id: {
                **entry,
                "installed_at": installed_at,
            }
            for validator_id, entry in REGISTRY_ENTRIES.items()
        },
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def _registry_has_validators(path: Path) -> bool:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    validators = registry.get("validators")
    return isinstance(validators, dict) and bool(validators)


if __name__ == "__main__":
    main()
