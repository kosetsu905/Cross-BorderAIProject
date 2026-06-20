from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "references" / "scenarios.json"
DEFAULT_SUITE = "multilingual_geo_visual"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
ACTIVE_STATUSES = {"pending", "running"}


def utc_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest.get("scenarios"), dict):
        raise ValueError(f"Scenario manifest {path} must define a scenarios object.")
    if not isinstance(manifest.get("suites"), dict):
        raise ValueError(f"Scenario manifest {path} must define a suites object.")
    return manifest


def resolve_scenarios(manifest: dict[str, Any], suite: str, scenarios: str) -> list[str]:
    scenario_map = manifest["scenarios"]
    if scenarios.strip():
        scenario_ids = [item.strip() for item in scenarios.split(",") if item.strip()]
    else:
        suites = manifest["suites"]
        if suite not in suites:
            known = ", ".join(sorted(str(name) for name in suites))
            raise ValueError(f"Unknown suite {suite!r}. Known suites: {known}.")
        scenario_ids = [str(item) for item in suites[suite]]

    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenario_map]
    if missing:
        raise ValueError(f"Unknown scenario id(s): {', '.join(missing)}.")
    return list(dict.fromkeys(scenario_ids))


def scenario_inputs(manifest: dict[str, Any], scenario_id: str, run_id: str) -> dict[str, Any]:
    scenario = manifest["scenarios"][scenario_id]
    inputs = dict(scenario["inputs"])
    marker = marker_for(manifest, scenario_id, run_id)
    features = str(inputs.get("product_features") or "").strip()
    marker_line = f"Live E2E marker: {marker}"
    inputs["product_features"] = f"{features}\n{marker_line}".strip()
    return inputs


def marker_for(manifest: dict[str, Any], scenario_id: str, run_id: str) -> str:
    prefix = str(manifest.get("marker_prefix") or "CB-CONTENT-E2E")
    return f"{prefix}-{scenario_id}-{run_id}"


def print_ui_fields(
    manifest: dict[str, Any],
    suite: str,
    scenarios_arg: str,
    scenario_ids: list[str],
    run_id: str,
) -> None:
    suite_label = "custom" if scenarios_arg.strip() else suite
    print(f"RUN_ID={run_id}")
    print(f"Suite: {suite_label}")
    print(f"Scenarios: {len(scenario_ids)}")
    print()
    for scenario_id in scenario_ids:
        inputs = scenario_inputs(manifest, scenario_id, run_id)
        marker = marker_for(manifest, scenario_id, run_id)
        print("=" * 88)
        print(f"Scenario: {scenario_id}")
        print(f"Marker: {marker}")
        print("Frontend fields:")
        print(f"  Workflow: content")
        print(f"  Subject: {inputs['subject']}")
        print(f"  Product category: {inputs['product_category']}")
        print(f"  Product features: {inputs['product_features']}")
        print(f"  Target markets: {inputs['target_markets']}")
        print(f"  Target languages: {', '.join(inputs['target_languages'])}")
        print(f"  Platforms: {', '.join(inputs['platforms'])}")
        print(f"  Brand voice: {inputs.get('brand_voice', '')}")
        print(f"  Brand name: {inputs.get('brand_name', '')}")
        print(f"  Product URL: {inputs.get('product_url', '')}")
        print(f"  Primary keywords: {', '.join(inputs.get('primary_keywords') or [])}")
        print(f"  Generate Reddit GEO: true")
        print(f"  Generate visual assets: true")
        print(f"  Image count: {inputs['image_generation_count']}")
        print(f"  Image quality: {inputs['image_quality']}")
        print(f"  Image size: {inputs['image_size']}")
        print()
        print("Equivalent request inputs JSON:")
        print(json.dumps(inputs, indent=2, ensure_ascii=False))
        print()
    print("After frontend submission, verify with:")
    print(
        ".\\.venv\\Scripts\\python.exe "
        ".\\.codex\\skills\\content-generation-live-e2e\\scripts\\run_content_generation_live_e2e.py "
        f"--suite {suite} --run-id {run_id} --job-id <JOB_ID> --verify-job"
    )


class ApiClient:
    def __init__(self, base_url: str, bearer_token: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=60) as response:
                data = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        return json.loads(data) if data else None


def submit_workflow(client: ApiClient, inputs: dict[str, Any]) -> dict[str, Any]:
    result = client.request_json("POST", "/api/v1/workflow", {"workflow_type": "content", "inputs": inputs})
    if not isinstance(result, dict) or not result.get("job_id"):
        raise RuntimeError(f"Workflow submission did not return a job_id: {result!r}")
    return result


def fetch_job(client: ApiClient, job_id: str) -> dict[str, Any]:
    result = client.request_json("GET", f"/api/v1/workflow/{job_id}")
    if not isinstance(result, dict):
        raise RuntimeError(f"Job endpoint did not return an object: {result!r}")
    return result


def fetch_events(client: ApiClient, job_id: str) -> list[dict[str, Any]]:
    result = client.request_json("GET", f"/api/v1/workflow/{job_id}/events")
    return result if isinstance(result, list) else []


def wait_for_job(client: ApiClient, job_id: str, timeout_seconds: int, poll_seconds: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    latest_job: dict[str, Any] = {}
    latest_events: list[dict[str, Any]] = []
    while True:
        latest_job = fetch_job(client, job_id)
        latest_events = fetch_events(client, job_id)
        status = str(latest_job.get("status") or "")
        if status not in ACTIVE_STATUSES or time.monotonic() >= deadline:
            return latest_job, latest_events
        time.sleep(poll_seconds)


def list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def collect_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(collect_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(collect_strings(item))
        return strings
    return [str(value)]


def add_issue(report: dict[str, Any], message: str, severity: str = "error") -> None:
    key = "warnings" if severity == "warning" else "errors"
    report[key].append(message)


def language_value(item: dict[str, Any]) -> str:
    return str(item.get("language") or item.get("language_code") or "").strip()


def result_language_sets(result: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "localized_articles": {
            language_value(item)
            for item in list_or_empty(result.get("localized_articles"))
            if isinstance(item, dict) and language_value(item)
        },
        "localized_entities": {
            language_value(item)
            for item in list_or_empty(result.get("localized_entities"))
            if isinstance(item, dict) and language_value(item)
        },
        "multimodal_outputs": {
            language_value(item)
            for item in list_or_empty(result.get("multimodal_outputs"))
            if isinstance(item, dict) and language_value(item)
        },
    }


def language_index(expected_languages: list[str], language: str) -> int | None:
    try:
        return expected_languages.index(language)
    except ValueError:
        return None


def indexed_payload(items: list[Any], index: int | None) -> list[Any]:
    if index is None:
        return []
    return [items[index]] if 0 <= index < len(items) else []


def generated_language_order(result: dict[str, Any]) -> list[str]:
    return [
        language_value(item)
        for item in list_or_empty(result.get("multimodal_outputs"))
        if isinstance(item, dict) and language_value(item)
    ]


def user_facing_language_payload(
    result: dict[str, Any],
    language: str,
    expected_language_order: list[str],
) -> dict[str, Any]:
    language = language.strip()
    actual_generation_order = generated_language_order(result) or expected_language_order
    index = language_index(actual_generation_order, language)
    return {
        "seo_outputs": indexed_payload(list_or_empty(result.get("seo_outputs")), index),
        "multimodal_outputs": [
            item
            for item in list_or_empty(result.get("multimodal_outputs"))
            if isinstance(item, dict) and payload_matches_language(item, language)
        ],
        "visual_assets": indexed_payload(list_or_empty(result.get("visual_assets")), index),
        "reddit_geo_posts": [
            item
            for item in list_or_empty(result.get("reddit_geo_posts"))
            if isinstance(item, dict) and payload_matches_language(item, language)
        ],
    }


def payload_matches_language(item: dict[str, Any], language: str) -> bool:
    item_language = language_value(item)
    if item_language == language:
        return True
    if not item_language and language in json.dumps(item, ensure_ascii=False):
        return True
    return False


def forbidden_terms_for_language(scenario: dict[str, Any], language: str) -> list[str]:
    terms = [str(term) for term in list_or_empty(scenario.get("source_terms")) if str(term)]
    allowed = set(str(term) for term in list_or_empty(scenario.get("ja_allowed_terms"))) if language == "ja" else set()
    return [term for term in terms if term not in allowed]


def verify_localized_entities(report: dict[str, Any], result: dict[str, Any], expected_languages: set[str]) -> None:
    entities = [
        item
        for item in list_or_empty(result.get("localized_entities"))
        if isinstance(item, dict)
    ]
    by_language = {language_value(item): item for item in entities if language_value(item)}
    missing = sorted(expected_languages - set(by_language))
    if missing:
        add_issue(report, f"Missing localized_entities for language(s): {missing}")
    required_fields = ("subject", "product_category", "brand_voice", "primary_keywords")
    for language in sorted(expected_languages & set(by_language)):
        entity = by_language[language]
        for field in required_fields:
            value = entity.get(field)
            if not value:
                add_issue(report, f"localized_entities[{language}].{field} is empty.")
        if "brand_name" not in entity:
            add_issue(report, f"localized_entities[{language}].brand_name is missing.")


def verify_production_assets(report: dict[str, Any], result: dict[str, Any], scenario: dict[str, Any]) -> None:
    assets = [
        item
        for item in list_or_empty(result.get("production_ready_assets"))
        if isinstance(item, dict)
    ]
    asset_types = {str(item.get("asset_type") or "") for item in assets}
    required = {str(item) for item in list_or_empty(scenario.get("required_production_asset_types"))}
    missing = sorted(required - asset_types)
    if missing:
        add_issue(report, f"Missing production_ready_assets type(s): {missing}")


def resolve_artifact_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("\\", "/")
    candidates = [Path(raw)]
    if normalized.startswith("/app/artifacts/"):
        candidates.append(ARTIFACT_ROOT / normalized.removeprefix("/app/artifacts/"))
    elif normalized.startswith("app/artifacts/"):
        candidates.append(ARTIFACT_ROOT / normalized.removeprefix("app/artifacts/"))
    elif normalized.startswith("artifacts/"):
        candidates.append(PROJECT_ROOT / normalized)
    elif "/artifacts/" in normalized:
        candidates.append(ARTIFACT_ROOT / normalized.split("/artifacts/", 1)[1])
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved.is_file():
            try:
                resolved.relative_to(ARTIFACT_ROOT.resolve())
            except ValueError:
                continue
            return resolved
    return None


def verify_visual_assets(report: dict[str, Any], result: dict[str, Any]) -> None:
    assets = [
        item
        for item in list_or_empty(result.get("visual_assets"))
        if isinstance(item, dict)
    ]
    if not assets:
        add_issue(report, "visual_assets is empty; live image generation did not produce an asset.")
        return
    usable = []
    for asset in assets:
        path = resolve_artifact_path(asset.get("asset_path"))
        if path or str(asset.get("asset_url") or "").strip():
            usable.append(asset)
    if not usable:
        add_issue(report, "No visual asset has an existing artifact path under artifacts/content_creation or an asset_url.")


def verify_leaks(
    report: dict[str, Any],
    result: dict[str, Any],
    scenario: dict[str, Any],
    expected_language_order: list[str],
) -> None:
    for language in expected_language_order:
        payload = user_facing_language_payload(result, language, expected_language_order)
        serialized = json.dumps(payload, ensure_ascii=False)
        for term in forbidden_terms_for_language(scenario, language):
            if term in serialized:
                add_issue(report, f"Source term {term!r} leaked in {language} SEO/visual/Reddit payload.")


def verify_result(
    manifest: dict[str, Any],
    scenario_id: str,
    run_id: str,
    job: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    scenario = manifest["scenarios"][scenario_id]
    expected_language_order = [
        str(item)
        for item in list_or_empty(scenario.get("expected_languages"))
        if str(item).strip()
    ]
    expected_languages = set(expected_language_order)
    result = dict_or_empty(job.get("result"))
    report: dict[str, Any] = {
        "scenario": scenario_id,
        "marker": marker_for(manifest, scenario_id, run_id),
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "errors": [],
        "warnings": [],
        "language_sets": {},
        "event_count": len(events),
    }

    if not os.getenv("OPENAI_API_KEY"):
        add_issue(report, "OPENAI_API_KEY is not configured; live image generation cannot be verified.")
    if not os.getenv("SERPER_API_KEY"):
        add_issue(report, "SERPER_API_KEY is not configured; Reddit GEO may use low-confidence fallback.", "warning")
    if job.get("status") != "completed":
        add_issue(report, f"Expected job status 'completed', got {job.get('status')!r}.")
    if job.get("error"):
        add_issue(report, f"Job error is present: {job.get('error')}")
    if not result:
        add_issue(report, "Job result is missing or not an object.")
        return report

    required_keys = [str(item) for item in list_or_empty(scenario.get("required_result_keys"))]
    missing_keys = [key for key in required_keys if key not in result]
    if missing_keys:
        add_issue(report, f"Missing result key(s): {missing_keys}")

    language_sets = result_language_sets(result)
    report["language_sets"] = {key: sorted(value) for key, value in language_sets.items()}
    article_missing = sorted(expected_languages - language_sets["localized_articles"])
    if article_missing:
        add_issue(report, f"Missing localized_articles for language(s): {article_missing}")
    multimodal_missing = sorted(expected_languages - language_sets["multimodal_outputs"])
    if multimodal_missing:
        add_issue(report, f"Missing multimodal_outputs for language(s): {multimodal_missing}")

    verify_localized_entities(report, result, expected_languages)
    verify_production_assets(report, result, scenario)
    verify_visual_assets(report, result)
    if not list_or_empty(result.get("seo_outputs")):
        add_issue(report, "seo_outputs is empty.")
    if not list_or_empty(result.get("reddit_geo_posts")):
        add_issue(report, "reddit_geo_posts is empty.")
    verify_leaks(report, result, scenario, expected_language_order)

    report["result_counts"] = {
        key: len(list_or_empty(result.get(key)))
        for key in (
            "localized_articles",
            "localized_entities",
            "seo_outputs",
            "multimodal_outputs",
            "visual_assets",
            "visual_asset_scores",
            "reddit_geo_posts",
            "production_ready_assets",
        )
    }
    report["status"] = "failed" if report["errors"] else "passed"
    return report


def write_report(run_id: str, report: dict[str, Any]) -> Path:
    path = ARTIFACT_ROOT / "content_generation_e2e" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_verify(args: argparse.Namespace, manifest: dict[str, Any], scenario_ids: list[str], run_id: str) -> int:
    token = args.bearer_token or os.getenv("API_BEARER_TOKEN")
    client = ApiClient(args.api_base_url, token)
    job_id = args.job_id.strip()
    submitted_via_api = False
    if args.submit_api:
        if len(scenario_ids) != 1:
            raise SystemExit("--submit-api supports exactly one scenario.")
        submission = submit_workflow(client, scenario_inputs(manifest, scenario_ids[0], run_id))
        job_id = str(submission["job_id"])
        submitted_via_api = True
        print(f"Submitted API fallback job {job_id}")
    if not job_id:
        raise SystemExit("--job-id is required with --verify-job unless --submit-api is used.")

    job, events = wait_for_job(client, job_id, args.timeout_seconds, args.poll_seconds)
    scenario_reports = [
        verify_result(manifest, scenario_id, run_id, job, events)
        for scenario_id in scenario_ids
    ]
    report = {
        "run_id": run_id,
        "suite": args.suite,
        "scenario_ids": scenario_ids,
        "job_id": job_id,
        "submitted_via_api": submitted_via_api,
        "api_base_url": args.api_base_url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": scenario_reports,
    }
    path = write_report(run_id, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written: {path}")
    failures = [
        item
        for item in scenario_reports
        if item.get("errors") or item.get("status") != "passed"
    ]
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live Content Generation frontend E2E checks.")
    parser.add_argument("--run-id", default="", help="Run id used in marker text, for example 20260620-153000.")
    parser.add_argument("--print-ui-fields", action="store_true", help="Print fields to submit through the Streamlit dashboard.")
    parser.add_argument("--verify-job", action="store_true", help="Poll and verify a submitted content workflow job.")
    parser.add_argument("--submit-api", action="store_true", help="Explicit fallback: submit the content workflow through the API instead of the frontend.")
    parser.add_argument("--job-id", default="", help="Job id copied from the Streamlit dashboard after frontend submission.")
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="Scenario suite from the manifest.")
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario ids. Overrides --suite when provided.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to the scenario manifest JSON.")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--bearer-token", default="", help="Optional API bearer token; defaults to API_BEARER_TOKEN.")
    parser.add_argument("--timeout-seconds", type=int, default=1200, help="How long to poll for the workflow result.")
    parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval.")
    return parser


def main() -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args()
    if not args.print_ui_fields and not args.verify_job and not args.submit_api:
        parser.print_help()
        return 2

    manifest = load_manifest(Path(args.manifest))
    scenario_ids = resolve_scenarios(manifest, args.suite, args.scenarios)
    run_id = args.run_id or utc_run_id()
    if args.print_ui_fields:
        print_ui_fields(manifest, args.suite, args.scenarios, scenario_ids, run_id)
    if args.verify_job or args.submit_api:
        return run_verify(args, manifest, scenario_ids, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
