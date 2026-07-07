from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final


VALIDATOR_URIS: Final[tuple[str, ...]] = (
    "hub://guardrails/secrets_present",
    "hub://guardrails/detect_pii",
    "hub://guardrails/regex_match",
    "hub://sainatha/prompt_injection_detector",
    "hub://guardrails/provenance_llm",
    "hub://guardrails/toxic_language",
)

TOXIC_URI: Final[str] = "hub://guardrails/toxic_language"
TOXIC_FALLBACK_PACKAGE: Final[str] = "git+https://github.com/guardrails-ai/toxic_language.git@v0.0.2"
WHEELHOUSE_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "vendor" / "guardrails-hub-wheels"
WHEEL_PACKAGE_NAMES: Final[dict[str, str]] = {
    "hub://guardrails/secrets_present": "guardrails_grhub_secrets_present",
    "hub://guardrails/detect_pii": "guardrails_grhub_detect_pii",
    "hub://guardrails/regex_match": "guardrails_grhub_regex_match",
    "hub://sainatha/prompt_injection_detector": "sainatha_grhub_prompt_injection_detector",
    "hub://guardrails/provenance_llm": "guardrails_grhub_provenance_llm",
}
REGISTRY_ENTRIES: Final[dict[str, dict[str, object]]] = {
    "guardrails/secrets_present": {
        "import_path": "guardrails_grhub_secrets_present",
        "exports": ["SecretsPresent"],
        "package_name": "guardrails-grhub-secrets-present",
    },
    "guardrails/detect_pii": {
        "import_path": "guardrails_grhub_detect_pii",
        "exports": ["DetectPII"],
        "package_name": "guardrails-grhub-detect-pii",
    },
    "guardrails/regex_match": {
        "import_path": "guardrails_grhub_regex_match",
        "exports": ["RegexMatch"],
        "package_name": "guardrails-grhub-regex-match",
    },
    "sainatha/prompt_injection_detector": {
        "import_path": "sainatha_grhub_prompt_injection_detector",
        "exports": ["PromptInjectionDetector"],
        "package_name": "sainatha-grhub-prompt-injection-detector",
    },
    "guardrails/provenance_llm": {
        "import_path": "guardrails_grhub_provenance_llm",
        "exports": ["ProvenanceLLM"],
        "package_name": "guardrails-grhub-provenance-llm",
    },
    "guardrails/toxic_language": {
        "import_path": "guardrails_grhub_toxic_language",
        "exports": ["ToxicLanguage"],
        "package_name": "guardrails-grhub-toxic-language",
    },
}


def main() -> None:
    os.environ.setdefault("GUARDRAILS_INSTALLER", "pip")
    token = os.getenv("GUARDRAILS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GUARDRAILS_TOKEN is required to install Guardrails Hub validators in Docker.")

    guardrails_exe = shutil.which("guardrails")
    if not guardrails_exe:
        raise RuntimeError("guardrails CLI was not found. Install guardrails-ai before running this script.")

    _ensure_model_dirs()
    try:
        _run(
            [
                guardrails_exe,
                "configure",
                "--token",
                token,
                "--disable-metrics",
                "--disable-remote-inferencing",
            ]
        )
        for uri in VALIDATOR_URIS:
            _install_validator(guardrails_exe, uri)
        _download_nltk_data()
        _smoke_imports()
        _smoke_toxic_validation()
        _write_registry_entries()
    finally:
        _remove_guardrails_credentials()


def _install_validator(guardrails_exe: str, uri: str) -> None:
    try:
        _run([guardrails_exe, "hub", "install", uri])
    except RuntimeError:
        if uri == TOXIC_URI:
            _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", TOXIC_FALLBACK_PACKAGE])
            return
        wheel = _wheel_for_uri(uri)
        if wheel is None:
            raise
        _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", str(wheel)])


def _wheel_for_uri(uri: str) -> Path | None:
    package_name = WHEEL_PACKAGE_NAMES.get(uri)
    if not package_name:
        return None
    wheels = sorted(WHEELHOUSE_DIR.glob(f"{package_name}-*.whl"))
    return wheels[-1] if wheels else None


def _smoke_imports() -> None:
    _import_validator("guardrails_grhub_secrets_present", "SecretsPresent")
    _import_validator("guardrails_grhub_detect_pii", "DetectPII")
    _import_validator("guardrails_grhub_regex_match", "RegexMatch")
    _import_validator("sainatha_grhub_prompt_injection_detector", "PromptInjectionDetector")
    _import_validator("guardrails_grhub_provenance_llm", "ProvenanceLLM")
    _import_toxic_language()


def _import_validator(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _smoke_toxic_validation() -> None:
    from guardrails import Guard

    toxic_language = _import_toxic_language()
    guard = Guard()
    guard.configure(allow_metrics_collection=False)
    guard.use(
        toxic_language(
            threshold=0.5,
            validation_method="sentence",
            on_fail="noop",
            use_local=True,
        )
    )
    outcome = guard.validate("Please keep this customer support reply professional and respectful.")
    if not bool(getattr(outcome, "validation_passed", False)):
        raise RuntimeError("ToxicLanguage smoke validation failed for a benign sentence.")


def _import_toxic_language() -> type:
    try:
        from guardrails.hub import ToxicLanguage

        return ToxicLanguage
    except (AttributeError, ImportError):
        with contextlib.suppress((AttributeError, ImportError)):
            return _import_validator("guardrails_grhub_toxic_language", "ToxicLanguage")
        module = importlib.import_module("validator")
        REGISTRY_ENTRIES["guardrails/toxic_language"]["import_path"] = "validator"
        REGISTRY_ENTRIES["guardrails/toxic_language"]["package_name"] = "toxic-language"
        return getattr(module, "ToxicLanguage")


def _write_registry_entries() -> None:
    registry_path = Path.cwd() / ".guardrails" / "hub_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registry = {"version": 1, "validators": {}}
    validators = registry.setdefault("validators", {})
    installed_at = datetime.now(UTC).isoformat()
    for validator_id, entry in REGISTRY_ENTRIES.items():
        validators[validator_id] = {
            **entry,
            "installed_at": installed_at,
        }
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def _download_nltk_data() -> None:
    import nltk

    nltk_data = os.getenv("NLTK_DATA", "/models/nltk")
    for package_name in ("punkt", "punkt_tab"):
        nltk.download(package_name, download_dir=nltk_data, quiet=True)


def _ensure_model_dirs() -> None:
    for env_name, default in (
        ("HF_HOME", "/models/huggingface"),
        ("HF_HUB_CACHE", "/models/huggingface/hub"),
        ("SENTENCE_TRANSFORMERS_HOME", "/models/sentence-transformers"),
        ("NLTK_DATA", "/models/nltk"),
    ):
        Path(os.getenv(env_name, default)).mkdir(parents=True, exist_ok=True)


def _remove_guardrails_credentials() -> None:
    for path in (Path.home() / ".guardrailsrc", Path.home() / ".guardrails"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, text=True)
    if result.returncode == 0:
        return
    safe_command = ["<redacted>" if item == os.getenv("GUARDRAILS_TOKEN", "") else item for item in command]
    raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(safe_command)}")


if __name__ == "__main__":
    main()
