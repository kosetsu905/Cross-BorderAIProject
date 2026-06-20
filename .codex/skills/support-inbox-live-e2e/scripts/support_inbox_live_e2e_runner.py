from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "references" / "scenarios.json"
DEFAULT_SUITE = "balanced"
GMAIL_SENDER_EMAIL_ENV = "GMAIL_SENDER_EMAIL"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def utc_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest.get("scenarios"), dict):
        raise ValueError(f"Scenario manifest {path} must define a scenarios object.")
    if not isinstance(manifest.get("suites"), dict):
        raise ValueError(f"Scenario manifest {path} must define a suites object.")
    return manifest


def marker_prefix(manifest: dict[str, Any]) -> str:
    return str(manifest.get("marker_prefix") or "CB-SUPPORT-E2E")


def load_project_env() -> None:
    """Load project .env for standalone skill script runs without overriding shell env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_recipient(manifest: dict[str, Any], explicit_recipient: str | None = None) -> str:
    for value in (
        explicit_recipient,
        os.getenv(GMAIL_SENDER_EMAIL_ENV),
        manifest.get("recipient"),
    ):
        recipient = str(value or "").strip()
        if recipient:
            return recipient
    raise ValueError(
        f"Live E2E recipient is required. Set {GMAIL_SENDER_EMAIL_ENV} in .env "
        "or pass --recipient."
    )


def marker_for(manifest: dict[str, Any], scenario: str, run_id: str) -> str:
    return f"{marker_prefix(manifest)}-{scenario}-{run_id}"


def format_context(run_id: str, marker: str) -> SafeFormatDict:
    return SafeFormatDict(
        {
            "marker": marker,
            "run_id": run_id,
            "run_date": run_id.split("-", 1)[0],
            "run_id_digits": re.sub(r"[^0-9]", "", run_id),
        }
    )


def render_template(value: str, run_id: str, marker: str) -> str:
    return value.format_map(format_context(run_id, marker))


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

    deduped: list[str] = []
    seen: set[str] = set()
    for scenario_id in scenario_ids:
        if scenario_id not in seen:
            deduped.append(scenario_id)
            seen.add(scenario_id)
    return deduped


def messages_for(
    manifest: dict[str, Any],
    run_id: str,
    scenario_ids: list[str],
    resolved_recipient: str,
) -> dict[str, dict[str, str]]:
    messages: dict[str, dict[str, str]] = {}
    scenario_map = manifest["scenarios"]
    for scenario_id in scenario_ids:
        scenario = scenario_map[scenario_id]
        marker = marker_for(manifest, scenario_id, run_id)
        messages[scenario_id] = {
            "recipient": resolved_recipient,
            "marker": marker,
            "subject": render_template(str(scenario["subject"]), run_id, marker),
            "body": render_template(str(scenario["body"]), run_id, marker).strip(),
        }
    return messages


def print_messages(
    manifest: dict[str, Any],
    run_id: str,
    suite: str,
    scenarios_arg: str,
    scenario_ids: list[str],
    resolved_recipient: str,
) -> None:
    suite_label = "custom" if scenarios_arg.strip() else suite
    verify_selector = f"--scenarios {scenarios_arg.strip()}" if scenarios_arg.strip() else f"--suite {suite}"
    print(f"RUN_ID={run_id}")
    print(f"Suite: {suite_label}")
    print(f"Scenarios: {len(scenario_ids)}")
    print(f"Recipient: {resolved_recipient}")
    print()
    print(f"After sending, verify with: {verify_selector} --run-id {run_id} --sync-and-verify")
    print()
    for scenario_id, message in messages_for(manifest, run_id, scenario_ids, resolved_recipient).items():
        print("=" * 88)
        print(f"Scenario: {scenario_id}")
        print(f"To: {message['recipient']}")
        print(f"Subject: {message['subject']}")
        print()
        print(message["body"])
        print()


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
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=60) as response:
                data = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        return json.loads(data) if data else None


def sync_latest(client: ApiClient, manifest: dict[str, Any], run_id: str, max_results: int) -> dict[str, Any]:
    return client.request_json(
        "POST",
        "/api/v1/channels/gmail/sync-latest",
        {
            "max_results": max_results,
            "query": f"{marker_prefix(manifest)} {run_id}",
        },
    )


def list_conversation_details(client: ApiClient) -> list[dict[str, Any]]:
    conversations = client.request_json("GET", "/api/v1/support/conversations?limit=200")
    if not isinstance(conversations, list):
        return []
    details: list[dict[str, Any]] = []
    for item in conversations:
        if not isinstance(item, dict) or not item.get("conversation_id"):
            continue
        detail = client.request_json("GET", f"/api/v1/support/conversations/{item['conversation_id']}")
        if isinstance(detail, dict):
            details.append(detail)
    return details


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


def find_by_marker(details: list[dict[str, Any]], marker: str) -> dict[str, Any] | None:
    for detail in details:
        haystack = "\n".join(collect_strings(detail))
        if marker in haystack:
            return detail
    return None


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).casefold()


def contains_term(haystack: str, term: str) -> bool:
    return normalize_text(term) in normalize_text(haystack)


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def add_issue(result: dict[str, Any], message: str, severity: str = "error") -> None:
    key = "warnings" if severity == "warning" else "errors"
    result[key].append(message)


def is_json_like_draft(draft: str) -> bool:
    stripped = draft.strip()
    if stripped.startswith("```") or stripped.startswith("{"):
        return True
    lowered = stripped.lower()
    return "```json" in lowered or '"response_type"' in lowered or '"final_response"' in lowered


def get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def rendered_terms(terms: list[Any], run_id: str, marker: str) -> list[str]:
    return [render_template(str(term), run_id, marker) for term in terms]


def outbound_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in list_or_empty(conversation.get("messages"))
        if isinstance(message, dict) and str(message.get("direction") or "").lower() == "outbound"
    ]


def has_logistics_output(payload: dict[str, Any]) -> bool:
    support = dict_or_empty(payload.get("support_response"))
    logistics = support.get("logistics_output")
    if logistics is None:
        return False
    if isinstance(logistics, str):
        return logistics.strip().lower() not in {"", "none", "null"}
    return bool(logistics)


def has_cjk_text(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def has_escalation_signal(conversation: dict[str, Any], payload: dict[str, Any]) -> bool:
    if conversation.get("escalation_flag") or payload.get("escalation_needed"):
        return True
    if str(payload.get("qa_status") or "").upper() == "REVIEW_REQUIRED":
        return True
    haystack = normalize_text("\n".join(collect_strings(payload)))
    return any(term in haystack for term in ("handoff", "escalat", "supervisor", "manager", "主管", "经理", "人工"))


def verify_terms(
    result: dict[str, Any],
    draft: str,
    conversation: dict[str, Any],
    checks: dict[str, Any],
    run_id: str,
    marker: str,
) -> None:
    required = rendered_terms(list_or_empty(checks.get("required_draft_terms")), run_id, marker)
    missing = [term for term in required if not contains_term(draft, term)]
    if missing:
        result["missing_required_terms"] = missing
        add_issue(result, f"Draft is missing required term(s): {missing}")

    warning_terms = rendered_terms(list_or_empty(checks.get("warning_draft_terms")), run_id, marker)
    warning_missing = [term for term in warning_terms if not contains_term(draft, term)]
    if warning_missing:
        result["missing_warning_terms"] = warning_missing
        add_issue(result, f"Draft is missing preferred term(s): {warning_missing}", "warning")

    required_any = rendered_terms(list_or_empty(checks.get("required_any_draft_terms")), run_id, marker)
    if required_any and not any(contains_term(draft, term) for term in required_any):
        add_issue(result, f"Draft must include at least one of: {required_any}")

    forbidden = rendered_terms(list_or_empty(checks.get("forbidden_draft_terms")), run_id, marker)
    leaked = [term for term in forbidden if contains_term(draft, term)]
    if leaked:
        add_issue(result, f"Draft contains forbidden term(s): {leaked}")

    conversation_haystack = "\n".join(collect_strings(conversation))
    forbidden_conversation = rendered_terms(
        list_or_empty(checks.get("forbidden_conversation_terms")),
        run_id,
        marker,
    )
    conversation_leaks = [term for term in forbidden_conversation if contains_term(conversation_haystack, term)]
    if conversation_leaks:
        add_issue(result, f"Conversation payload leaked forbidden term(s): {conversation_leaks}")


def verify_payload_equals(
    result: dict[str, Any],
    payload: dict[str, Any],
    checks: dict[str, Any],
) -> None:
    for check in list_or_empty(checks.get("payload_equals")):
        if not isinstance(check, dict):
            continue
        path = str(check.get("path") or "")
        actual = get_path(payload, path)
        expected = check.get("value")
        if actual != expected:
            severity = str(check.get("severity") or "error")
            add_issue(result, f"Expected payload {path}={expected!r}, got {actual!r}.", severity)


def verify_tracking(
    result: dict[str, Any],
    draft: str,
    payload: dict[str, Any],
    checks: dict[str, Any],
    run_id: str,
    marker: str,
    prefix: str,
) -> None:
    order = dict_or_empty(payload.get("order_response"))
    if checks.get("expect_tracking_found") and order.get("tracking_record_found") is not True:
        add_issue(result, "Expected order_response.tracking_record_found == true.")
    if checks.get("expect_tracking_not_found"):
        not_found = order.get("tracking_lookup_status") == "not_found" or order.get("tracking_record_found") is False
        if not not_found:
            add_issue(result, "Expected wrong or missing tracking number to produce not_found tracking status.")

    if checks.get("reject_marker_as_identifier"):
        queries = [str(item) for item in list_or_empty(order.get("tracking_lookup_query"))]
        marker_terms = [marker, prefix, run_id, run_id.split("-", 1)[0], re.sub(r"[^0-9]", "", run_id)]
        polluted = [
            query
            for query in queries
            if any(contains_term(query, term) or contains_term(term, query) for term in marker_terms if term)
        ]
        if polluted:
            add_issue(result, f"Tracking lookup query appears polluted by test marker/run id: {polluted}")
        if any(contains_term(draft, term) for term in marker_terms):
            add_issue(result, "Draft appears to echo the test marker or run id as customer-facing tracking data.")


def verify_post_sales_controls(
    result: dict[str, Any],
    draft: str,
    conversation: dict[str, Any],
    payload: dict[str, Any],
    checks: dict[str, Any],
) -> None:
    if checks.get("forbid_return_logistics_without_payload") and not has_logistics_output(payload):
        risky_terms = (
            "print the prepaid",
            "prepaid return label:",
            "labels.example.local",
            "tracking number rtn",
        )
        leaked = [term for term in risky_terms if contains_term(draft, term)]
        if leaked:
            add_issue(result, f"Draft includes return logistics without logistics_output: {leaked}")

    language = str(checks.get("require_language") or "")
    if language == "zh":
        detected = str(payload.get("detected_language") or payload.get("language_detected") or "").casefold()
        if not detected.startswith("zh") and not has_cjk_text(draft):
            add_issue(result, "Expected Chinese language handling, but draft/payload did not show Chinese.")

    if checks.get("require_escalation_signal") and not has_escalation_signal(conversation, payload):
        add_issue(result, "Expected escalation or human-handoff signal for this scenario.")


def verify_conversation(
    manifest: dict[str, Any],
    scenario_id: str,
    run_id: str,
    conversation: dict[str, Any] | None,
) -> dict[str, Any]:
    marker = marker_for(manifest, scenario_id, run_id)
    scenario = manifest["scenarios"][scenario_id]
    checks = dict_or_empty(scenario.get("checks"))
    result: dict[str, Any] = {
        "scenario": scenario_id,
        "marker": marker,
        "conversation_id": None,
        "status": "missing",
        "errors": [],
        "warnings": [],
    }
    if conversation is None:
        add_issue(result, "Conversation with marker was not found.")
        return result

    result["conversation_id"] = conversation.get("conversation_id")
    draft = str(conversation.get("draft_response") or "")
    payload = dict_or_empty(conversation.get("draft_payload"))
    outbound = outbound_messages(conversation)
    result["conversation_status"] = conversation.get("status")
    result["requires_approval"] = conversation.get("requires_approval")
    result["escalation_flag"] = conversation.get("escalation_flag")
    result["detected_intent"] = payload.get("detected_intent")
    result["detected_language"] = payload.get("detected_language")
    result["draft_is_json_like"] = is_json_like_draft(draft)
    result["outbound_message_count"] = len(outbound)
    result["outbound_channel_message_ids"] = [
        str(message.get("channel_message_id"))
        for message in outbound
        if message.get("channel_message_id") is not None
    ]
    result["auto_dispatch_observed"] = bool(outbound) or conversation.get("status") == "sent"
    max_outbound = int(checks.get("max_outbound_message_count", manifest.get("max_outbound_message_count", 1)))
    if len(outbound) > max_outbound:
        add_issue(
            result,
            f"Conversation has {len(outbound)} outbound message(s), expected no more than {max_outbound}.",
        )

    if not draft.strip():
        result["status"] = "pending"
        add_issue(result, "Draft response is not available yet.")
        return result

    if result["draft_is_json_like"]:
        add_issue(result, "Draft response appears to contain raw JSON or a code fence.")
    if contains_term(draft, "labels.example.local"):
        add_issue(result, "Draft leaked labels.example.local return-label placeholder.")
    if contains_term(draft, marker_prefix(manifest)):
        add_issue(result, "Draft echoed the live test marker prefix.")

    expected_intent = scenario.get("expected_intent")
    if expected_intent and payload.get("detected_intent") != expected_intent:
        add_issue(result, f"Expected detected_intent={expected_intent!r}, got {payload.get('detected_intent')!r}.")

    verify_terms(result, draft, conversation, checks, run_id, marker)
    verify_payload_equals(result, payload, checks)
    verify_tracking(result, draft, payload, checks, run_id, marker, marker_prefix(manifest))
    verify_post_sales_controls(result, draft, conversation, payload, checks)

    result["status"] = "failed" if result["errors"] else "passed"
    result["draft_preview"] = draft[:500]
    return result


def write_report(run_id: str, report: dict[str, Any]) -> Path:
    path = Path("artifacts") / "support_inbox_e2e" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def sync_and_verify(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    scenario_ids: list[str],
    run_id: str,
    resolved_recipient: str,
) -> int:
    if not run_id:
        raise SystemExit("--run-id is required with --sync-and-verify. Run --print-messages first.")

    token = args.bearer_token or os.getenv("API_BEARER_TOKEN")
    client = ApiClient(args.api_base_url, token)
    deadline = time.monotonic() + args.timeout_seconds
    last_report: dict[str, Any] | None = None

    while True:
        sync_result: dict[str, Any] | None = None
        try:
            sync_result = sync_latest(client, manifest, run_id, args.max_results)
            details = list_conversation_details(client)
            scenario_results = [
                verify_conversation(
                    manifest,
                    scenario_id,
                    run_id,
                    find_by_marker(details, marker_for(manifest, scenario_id, run_id)),
                )
                for scenario_id in scenario_ids
            ]
        except RuntimeError as exc:
            scenario_results = []
            sync_result = {"error": str(exc)}

        report = {
            "run_id": run_id,
            "suite": args.suite,
            "scenario_ids": scenario_ids,
            "recipient": resolved_recipient,
            "api_base_url": args.api_base_url,
            "sync_result": sync_result,
            "results": scenario_results,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        last_report = report

        statuses = [item.get("status") for item in scenario_results]
        all_ready = len(scenario_results) == len(scenario_ids) and all(
            status in {"passed", "failed"} for status in statuses
        )
        if all_ready or time.monotonic() >= deadline:
            break
        time.sleep(args.poll_seconds)

    assert last_report is not None
    path = write_report(run_id, last_report)
    failed = [
        item
        for item in last_report.get("results", [])
        if item.get("errors") or item.get("status") != "passed"
    ]
    sync_error = isinstance(last_report.get("sync_result"), dict) and bool(last_report["sync_result"].get("error"))
    incomplete = len(last_report.get("results", [])) != len(scenario_ids)
    print(json.dumps(last_report, indent=2, ensure_ascii=False))
    print(f"Report written: {path}")
    return 1 if sync_error or incomplete or failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QQ-to-Gmail support inbox live E2E regression checks.")
    parser.add_argument("--run-id", default="", help="Run id used in message subjects, for example 20260620-153000.")
    parser.add_argument("--print-messages", action="store_true", help="Print the selected emails to send from QQ Mail.")
    parser.add_argument("--sync-and-verify", action="store_true", help="Sync Gmail latest messages and verify support inbox results.")
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="Scenario suite from the manifest. Defaults to balanced.")
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario ids. Overrides --suite when provided.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to the scenario manifest JSON.")
    parser.add_argument("--recipient", default="", help=f"Override recipient; defaults to {GMAIL_SENDER_EMAIL_ENV}.")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--bearer-token", default="", help="Optional API bearer token; defaults to API_BEARER_TOKEN.")
    parser.add_argument("--max-results", type=int, default=50, help="Gmail sync max_results.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="How long to poll for draft results.")
    parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval.")
    return parser


def main() -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args()
    if not args.print_messages and not args.sync_and_verify:
        parser.print_help()
        return 2

    manifest = load_manifest(Path(args.manifest))
    scenario_ids = resolve_scenarios(manifest, args.suite, args.scenarios)
    try:
        resolved_recipient = resolve_recipient(manifest, args.recipient)
    except ValueError as exc:
        parser.error(str(exc))
    run_id = args.run_id or utc_run_id()
    if args.print_messages:
        print_messages(manifest, run_id, args.suite, args.scenarios, scenario_ids, resolved_recipient)
    if args.sync_and_verify:
        return sync_and_verify(args, manifest, scenario_ids, run_id, resolved_recipient)
    return 0
