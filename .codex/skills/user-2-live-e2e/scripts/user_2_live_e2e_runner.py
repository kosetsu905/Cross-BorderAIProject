from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "references" / "scenarios.json"
DEFAULT_SUITE = "full"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
SENSITIVE_REPORT_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "password_hash",
    "reset_token",
    "session_token",
    "token_hash",
}


class ApiError(RuntimeError):
    def __init__(self, method: str, path: str, status_code: int, detail: str) -> None:
        super().__init__(f"{method} {path} failed: HTTP {status_code}: {detail}")
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class ScenarioData:
    scenario_id: str
    marker: str
    suffix: str
    email: str
    username: str
    initial_password: str
    changed_password: str
    reset_password: str
    phone_country_code: str
    phone: str
    normalized_phone: str
    phone_password: str
    profile_update: dict[str, str]
    real_oauth_providers: list[str]
    oauth_login: dict[str, Any] | None
    oauth_link_keep: dict[str, Any] | None
    oauth_link_then_unlink: dict[str, Any] | None
    payment_methods: list[dict[str, Any]]
    expected_final_payment_reference: str
    removed_payment_reference: str
    subscription_plan: str
    expected_final_subscription_status: str
    forbidden_report_terms: list[str]


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


def marker_for(manifest: dict[str, Any], scenario_id: str, run_id: str) -> str:
    prefix = str(manifest.get("marker_prefix") or "CB-USER2-E2E")
    return f"{prefix}-{scenario_id}-{run_id}"


def scenario_data(manifest: dict[str, Any], scenario_id: str, run_id: str) -> ScenarioData:
    raw = manifest["scenarios"][scenario_id]
    suffix = _suffix_for(run_id)
    marker = marker_for(manifest, scenario_id, run_id)
    email = f"cb-user2-e2e+{suffix}@{raw.get('email_domain', 'example.com')}"
    username = f"user2-e2e-{suffix}"
    country_code = str(raw.get("phone_country_code") or "+86")
    phone_digits = _digits_for(run_id, 8)
    phone = f"{raw.get('phone_prefix', '139')}{phone_digits}"
    return ScenarioData(
        scenario_id=scenario_id,
        marker=marker,
        suffix=suffix,
        email=email,
        username=username,
        initial_password=_format_template(raw["initial_password_template"], suffix),
        changed_password=_format_template(raw["changed_password_template"], suffix),
        reset_password=_format_template(raw["reset_password_template"], suffix),
        phone_country_code=country_code,
        phone=phone,
        normalized_phone=_normalize_phone(phone, country_code),
        phone_password=_format_template(raw["phone_password_template"], suffix),
        profile_update=dict(raw["profile_update"]),
        real_oauth_providers=[str(item) for item in raw.get("real_oauth_providers", [])],
        oauth_login=_format_nested(raw.get("oauth_login"), suffix),
        oauth_link_keep=_format_nested(raw.get("oauth_link_keep"), suffix),
        oauth_link_then_unlink=_format_nested(raw.get("oauth_link_then_unlink"), suffix),
        payment_methods=[_format_nested(item, suffix) for item in raw["payment_methods"]],
        expected_final_payment_reference=_format_template(raw["expected_final_payment_reference"], suffix),
        removed_payment_reference=_format_template(raw["removed_payment_reference"], suffix),
        subscription_plan=str(raw["subscription_plan"]),
        expected_final_subscription_status=str(raw["expected_final_subscription_status"]),
        forbidden_report_terms=[str(item) for item in raw.get("forbidden_report_terms", [])],
    )


def _suffix_for(run_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]", "", run_id).lower()
    return (normalized[-12:] or "manualrun001")[:12]


def _digits_for(run_id: str, count: int) -> str:
    digits = re.sub(r"\D", "", run_id)
    return (digits[-count:] if len(digits) >= count else digits.zfill(count)) or "00000001"


def _format_template(value: str, suffix: str) -> str:
    return value.replace("{suffix}", suffix)


def _format_nested(value: Any, suffix: str) -> Any:
    if isinstance(value, str):
        return _format_template(value, suffix)
    if isinstance(value, list):
        return [_format_nested(item, suffix) for item in value]
    if isinstance(value, dict):
        return {str(key): _format_nested(item, suffix) for key, item in value.items()}
    return value


def _normalize_phone(phone: str, country_code: str) -> str:
    raw_phone = phone.strip()
    if raw_phone.startswith("+"):
        combined = raw_phone
    else:
        prefix = country_code.strip()
        if prefix and not prefix.startswith("+"):
            prefix = f"+{prefix}"
        combined = f"{prefix}{raw_phone}" if prefix else raw_phone
    chars: list[str] = []
    for index, char in enumerate(combined):
        if char.isdigit() or (char == "+" and index == 0):
            chars.append(char)
    return "".join(chars)


def print_ui_scenario(
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
        data = scenario_data(manifest, scenario_id, run_id)
        print("=" * 88)
        print(f"Scenario: {scenario_id}")
        print(f"Marker: {data.marker}")
        print("Open local Chrome, visit the Streamlit dashboard, then select View -> Users.")
        print()
        print("1) Logged-out UI checks")
        print("  - Confirm only Login / Create account entry points are visible.")
        print("  - Confirm real OAuth provider buttons are visible.")
        print("  - Confirm Email and Phone are available under auth methods.")
        print()
        print("2) Create account -> Email")
        print(f"  Email: {data.email}")
        print(f"  Password: {data.initial_password}")
        print(f"  Username: {data.username}")
        print("  First name: E2E")
        print("  Last name: Initial")
        print("  Country: Australia")
        print()
        print("3) Profile")
        for key, value in data.profile_update.items():
            print(f"  {key}: {value}")
        print()
        print("4) Security")
        print(f"  Change password from {data.initial_password} to {data.changed_password}")
        print("  Request password reset for the email account.")
        print(f"  Confirm reset with new password: {data.reset_password}")
        print("  The reset token must stay internal to the local demo UI and must not be displayed.")
        print()
        print("5) Connected accounts")
        print("  If real OAuth env is configured, connect these real providers through the UI:")
        print(json.dumps(data.real_oauth_providers, indent=4, ensure_ascii=False))
        if has_oauth_payload(data.oauth_link_keep):
            print("  Link and keep:")
            print(json.dumps(data.oauth_link_keep, indent=4, ensure_ascii=False))
        if has_oauth_payload(data.oauth_link_then_unlink):
            print("  Link and then unlink:")
            print(json.dumps(data.oauth_link_then_unlink, indent=4, ensure_ascii=False))
        print()
        print("6) Billing")
        for index, method in enumerate(data.payment_methods, start=1):
            print(f"  Payment method {index}:")
            print(json.dumps(method, indent=4, ensure_ascii=False))
        print("  Make the second payment method default, then remove the first payment method.")
        print()
        print("7) Subscription")
        print(f"  Subscribe to: {data.subscription_plan}")
        print("  Then cancel the subscription.")
        print()
        print("8) Sign out and login checks")
        print(f"  Email login final password: {data.reset_password}")
        print(f"  Phone country code: {data.phone_country_code}")
        print(f"  Phone: {data.phone}")
        print(f"  Phone password: {data.phone_password}")
        print()
        if has_oauth_payload(data.oauth_login):
            print("9) Developer OAuth login")
            print(json.dumps(data.oauth_login, indent=2, ensure_ascii=False))
        else:
            print("9) Developer OAuth login")
            print("  - Skipped: all configured providers use real OAuth.")
        print()
        print("Manual UI regression checks:")
        print("  - No access_token, token_hash, reset_token, password_hash, raw JSON, or JSON error body is visible.")
        print("  - Logged-in account tools appear only after login.")
        print("  - Payment data is masked/tokenized only.")
        print()
    print("After completing the UI flow, verify with:")
    print(
        ".\\.venv\\Scripts\\python.exe "
        ".\\.codex\\skills\\user-2-live-e2e\\scripts\\run_user_2_live_e2e.py "
        f"--suite {suite} --run-id {run_id} --verify-ui-run"
    )


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        user_token: str = "",
        timeout_seconds: int = 60,
    ) -> Any:
        body = None
        headers = {"Content-Type": "application/json"}
        if user_token:
            headers["Authorization"] = f"Bearer {user_token}"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                data = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ApiError(method, path, exc.code, detail) from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        return json.loads(data) if data else None


def add_issue(report: dict[str, Any], message: str, severity: str = "error") -> None:
    key = "warnings" if severity == "warning" else "errors"
    report[key].append(message)


def auth_user(result: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(result, dict) or not isinstance(result.get("user"), dict):
        raise RuntimeError(f"Auth endpoint did not return a user object: {result!r}")
    token = str(result.get("access_token") or "")
    if not token:
        raise RuntimeError("Auth endpoint did not return an access token.")
    return result["user"], token


def register_email(client: ApiClient, data: ScenarioData) -> tuple[dict[str, Any], str]:
    return auth_user(
        client.request_json(
            "POST",
            "/api/v1/users/register/email",
            {
                "email": data.email,
                "password": data.initial_password,
                "username": data.username,
                "first_name": "E2E",
                "last_name": "Initial",
                "country": "Australia",
            },
        )
    )


def register_phone(client: ApiClient, data: ScenarioData) -> tuple[dict[str, Any], str]:
    return auth_user(
        client.request_json(
            "POST",
            "/api/v1/users/register/phone",
            {
                "country_code": data.phone_country_code,
                "phone": data.phone,
                "password": data.phone_password,
                "username": f"phone-{data.username}",
                "first_name": "Phone",
                "last_name": "E2E",
                "country": "China",
            },
        )
    )


def login_email(client: ApiClient, email: str, password: str) -> tuple[dict[str, Any], str]:
    return auth_user(client.request_json("POST", "/api/v1/users/login/email", {"email": email, "password": password}))


def login_phone(client: ApiClient, data: ScenarioData) -> tuple[dict[str, Any], str]:
    return auth_user(
        client.request_json(
            "POST",
            "/api/v1/users/login/phone",
            {"country_code": data.phone_country_code, "phone": data.phone, "password": data.phone_password},
        )
    )


def oauth_login(client: ApiClient, request_payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    return auth_user(client.request_json("POST", "/api/v1/users/oauth/login", _oauth_payload(request_payload)))


def has_oauth_payload(value: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("provider")
        and value.get("provider_user_id")
    )


def _oauth_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": value["provider"],
        "provider_user_id": value["provider_user_id"],
        "provider_info": value.get("provider_info") or {},
    }


def expect_login_failure(report: dict[str, Any], client: ApiClient, email: str, password: str, label: str) -> None:
    try:
        login_email(client, email, password)
    except ApiError as exc:
        if exc.status_code in {400, 401, 403}:
            report["checks"][label] = "failed_as_expected"
            return
        add_issue(report, f"{label} returned unexpected HTTP {exc.status_code}: {exc.detail}")
        return
    add_issue(report, f"{label} unexpectedly succeeded.")


def verify_user_response_has_no_secret_fields(report: dict[str, Any], user: dict[str, Any], label: str) -> None:
    serialized = json.dumps(user, ensure_ascii=False)
    forbidden = {"access_token", "password_hash", "reset_token", "session_hash", "token_hash"}
    leaked = sorted(term for term in forbidden if term in serialized)
    if leaked:
        add_issue(report, f"{label} user response leaked forbidden field(s): {leaked}")


def provider_names(user: dict[str, Any]) -> set[str]:
    providers = user.get("auth_providers")
    if not isinstance(providers, list):
        return set()
    return {str(item.get("provider")) for item in providers if isinstance(item, dict)}


def real_oauth_configured(provider: str) -> bool:
    prefix = provider.upper()
    if provider == "apple":
        return bool(
            os.getenv("APPLE_OAUTH_CLIENT_ID")
            and os.getenv("APPLE_OAUTH_TEAM_ID")
            and os.getenv("APPLE_OAUTH_KEY_ID")
            and (os.getenv("APPLE_OAUTH_PRIVATE_KEY") or os.getenv("APPLE_OAUTH_PRIVATE_KEY_PATH"))
        )
    if provider == "wechat":
        return bool(
            (os.getenv("WECHAT_OAUTH_APP_ID") or os.getenv("WECHAT_OAUTH_CLIENT_ID"))
            and os.getenv("WECHAT_OAUTH_CLIENT_SECRET")
        )
    if provider == "alipay":
        return bool(
            (os.getenv("ALIPAY_OAUTH_APP_ID") or os.getenv("ALIPAY_OAUTH_CLIENT_ID"))
            and (os.getenv("ALIPAY_OAUTH_PRIVATE_KEY") or os.getenv("ALIPAY_OAUTH_PRIVATE_KEY_PATH"))
        )
    if provider == "douyin":
        return bool(
            (os.getenv("DOUYIN_OAUTH_CLIENT_KEY") or os.getenv("DOUYIN_OAUTH_CLIENT_ID"))
            and os.getenv("DOUYIN_OAUTH_CLIENT_SECRET")
        )
    if provider == "qq":
        return bool(os.getenv("QQ_OAUTH_CLIENT_ID") and os.getenv("QQ_OAUTH_CLIENT_SECRET"))
    return bool(os.getenv(f"{prefix}_OAUTH_CLIENT_ID") and os.getenv(f"{prefix}_OAUTH_CLIENT_SECRET"))


def payment_references(user: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = user.get("payment_methods")
    if not isinstance(methods, list):
        return {}
    references: dict[str, dict[str, Any]] = {}
    for method in methods:
        if not isinstance(method, dict):
            continue
        payment_data = method.get("payment_data") if isinstance(method.get("payment_data"), dict) else {}
        reference = str(payment_data.get("processor_reference") or payment_data.get("account") or "")
        if reference:
            references[reference] = method
    return references


def check_final_user_state(
    report: dict[str, Any],
    user: dict[str, Any],
    data: ScenarioData,
    *,
    require_real_oauth: bool,
) -> None:
    verify_user_response_has_no_secret_fields(report, user, "final /me")
    for key, expected in data.profile_update.items():
        if user.get(key) != expected:
            add_issue(report, f"Profile field {key!r} expected {expected!r}, got {user.get(key)!r}.")

    providers = provider_names(user)
    if has_oauth_payload(data.oauth_link_keep):
        keep_provider = str(data.oauth_link_keep["provider"])
        if keep_provider not in providers:
            add_issue(report, f"Expected kept OAuth provider {keep_provider!r} to be linked.")
    if has_oauth_payload(data.oauth_link_then_unlink):
        removed_provider = str(data.oauth_link_then_unlink["provider"])
        if removed_provider in providers:
            add_issue(report, f"Expected OAuth provider {removed_provider!r} to be unlinked.")
    if require_real_oauth:
        for provider in data.real_oauth_providers:
            if not real_oauth_configured(provider):
                add_issue(report, f"Real OAuth provider {provider!r} is not configured; live UI check skipped.", "warning")
                continue
            if provider not in providers:
                add_issue(report, f"Expected real OAuth provider {provider!r} to be linked by the UI flow.")
    elif data.real_oauth_providers:
        add_issue(report, "API fallback does not exercise real OAuth providers.", "warning")

    references = payment_references(user)
    final_payment = references.get(data.expected_final_payment_reference)
    if final_payment is None:
        add_issue(report, f"Expected active payment reference {data.expected_final_payment_reference!r}.")
    elif not final_payment.get("is_default"):
        add_issue(report, f"Expected payment reference {data.expected_final_payment_reference!r} to be default.")
    if data.removed_payment_reference in references:
        add_issue(report, f"Removed payment reference {data.removed_payment_reference!r} is still active.")

    if user.get("subscription_plan") != data.subscription_plan:
        add_issue(report, f"Expected subscription_plan {data.subscription_plan!r}, got {user.get('subscription_plan')!r}.")
    if user.get("subscription_status") != data.expected_final_subscription_status:
        add_issue(
            report,
            "Expected subscription_status "
            f"{data.expected_final_subscription_status!r}, got {user.get('subscription_status')!r}.",
        )


def current_user(client: ApiClient, token: str) -> dict[str, Any]:
    result = client.request_json("GET", "/api/v1/users/me", user_token=token)
    if not isinstance(result, dict):
        raise RuntimeError(f"/me did not return an object: {result!r}")
    return result


def run_verify_ui(client: ApiClient, data: ScenarioData) -> dict[str, Any]:
    report = base_scenario_report(data, mode="verify-ui-run")
    try:
        expect_login_failure(report, client, data.email, data.initial_password, "old_email_password")
        expect_login_failure(report, client, data.email, data.changed_password, "changed_email_password_after_reset")
        email_user, email_token = login_email(client, data.email, data.reset_password)
        report["checks"]["email_login_final_password"] = "passed"
        verify_user_response_has_no_secret_fields(report, email_user, "email login")
        user = current_user(client, email_token)
        check_final_user_state(report, user, data, require_real_oauth=True)

        phone_user, phone_token = login_phone(client, data)
        report["checks"]["phone_login"] = "passed"
        verify_user_response_has_no_secret_fields(report, phone_user, "phone login")
        if phone_user.get("phone") != data.normalized_phone:
            add_issue(report, f"Expected phone {data.normalized_phone!r}, got {phone_user.get('phone')!r}.")
        logout(client, phone_token)

        if has_oauth_payload(data.oauth_login):
            oauth_user, oauth_token = oauth_login(client, data.oauth_login)
            report["checks"]["developer_oauth_repeat_login"] = "passed"
            verify_user_response_has_no_secret_fields(report, oauth_user, "oauth login")
            if str(data.oauth_login["provider"]) not in provider_names(oauth_user):
                add_issue(report, "Developer OAuth user response is missing the OAuth provider.")
            logout(client, oauth_token)
        else:
            report["checks"]["developer_oauth_repeat_login"] = "skipped"
        logout(client, email_token)
    except Exception as exc:
        add_issue(report, str(exc))
    finalize_report_status(report)
    return report


def run_api_fallback(client: ApiClient, data: ScenarioData) -> dict[str, Any]:
    report = base_scenario_report(data, mode="run-api-fallback")
    token = ""
    phone_token = ""
    oauth_token = ""
    try:
        email_user, token = register_email(client, data)
        report["checks"]["email_register"] = "passed"
        verify_user_response_has_no_secret_fields(report, email_user, "email register")

        try:
            register_email(client, data)
        except ApiError as exc:
            if exc.status_code == 409:
                report["checks"]["duplicate_email_rejected"] = "passed"
            else:
                add_issue(report, f"Duplicate email returned HTTP {exc.status_code}, expected 409.")
        else:
            add_issue(report, "Duplicate email registration unexpectedly succeeded.")

        updated = client.request_json("PATCH", "/api/v1/users/me", data.profile_update, user_token=token)
        if not isinstance(updated, dict):
            raise RuntimeError("Profile update did not return a user object.")
        report["checks"]["profile_update"] = "passed"
        verify_user_response_has_no_secret_fields(report, updated, "profile update")

        client.request_json(
            "POST",
            "/api/v1/users/me/password",
            {"old_password": data.initial_password, "new_password": data.changed_password},
            user_token=token,
        )
        report["checks"]["change_password"] = "passed"
        expect_login_failure(report, client, data.email, data.initial_password, "old_email_password")
        _, token = login_email(client, data.email, data.changed_password)

        reset_response = client.request_json("POST", "/api/v1/users/password-reset/request", {"email": data.email})
        reset_token = _reset_token_from_response(reset_response)
        client.request_json(
            "POST",
            "/api/v1/users/password-reset/confirm",
            {"token": reset_token, "new_password": data.reset_password},
        )
        report["checks"]["password_reset"] = "passed"
        expect_login_failure(report, client, data.email, data.changed_password, "changed_email_password_after_reset")
        _, token = login_email(client, data.email, data.reset_password)

        if has_oauth_payload(data.oauth_link_keep):
            link_keep = client.request_json(
                "POST",
                "/api/v1/users/me/oauth",
                _oauth_payload(data.oauth_link_keep),
                user_token=token,
            )
            report["checks"]["oauth_link_keep"] = "passed"
            verify_user_response_has_no_secret_fields(report, link_keep, "oauth link keep")
        else:
            add_issue(report, "No dev-only OAuth provider is configured for link-and-keep coverage.", "warning")
        if has_oauth_payload(data.oauth_link_then_unlink):
            client.request_json("POST", "/api/v1/users/me/oauth", _oauth_payload(data.oauth_link_then_unlink), user_token=token)
            client.request_json(
                "DELETE",
                f"/api/v1/users/me/oauth/{data.oauth_link_then_unlink['provider']}",
                user_token=token,
            )
            report["checks"]["oauth_link_then_unlink"] = "passed"
        else:
            add_issue(report, "No extra dev-only OAuth provider is configured for link-then-unlink coverage.", "warning")

        payment_ids: list[str] = []
        for payment in data.payment_methods:
            result = client.request_json(
                "POST",
                "/api/v1/users/me/payment-methods",
                {
                    "payment_type": payment["payment_type"],
                    "payment_data": payment["payment_data"],
                    "is_default": bool(payment.get("is_default")),
                },
                user_token=token,
            )
            if not isinstance(result, dict) or not result.get("payment_method_id"):
                raise RuntimeError(f"Payment method endpoint returned unexpected payload: {result!r}")
            payment_ids.append(str(result["payment_method_id"]))
        if len(payment_ids) >= 2:
            client.request_json("POST", f"/api/v1/users/me/payment-methods/{payment_ids[1]}/default", user_token=token)
            client.request_json("DELETE", f"/api/v1/users/me/payment-methods/{payment_ids[0]}", user_token=token)
        report["checks"]["payment_methods"] = "passed"

        subscribed = client.request_json(
            "POST",
            "/api/v1/users/me/subscription",
            {"plan_id": data.subscription_plan},
            user_token=token,
        )
        if not isinstance(subscribed, dict) or subscribed.get("subscription_plan") != data.subscription_plan:
            add_issue(report, "Subscription endpoint did not return the expected plan.")
        cancelled = client.request_json("POST", "/api/v1/users/me/subscription/cancel", user_token=token)
        if not isinstance(cancelled, dict):
            raise RuntimeError("Subscription cancel endpoint did not return a user object.")
        report["checks"]["subscription_cancel"] = "passed"

        final_user = current_user(client, token)
        check_final_user_state(report, final_user, data, require_real_oauth=False)

        _, phone_token = register_phone(client, data)
        report["checks"]["phone_register"] = "passed"
        try:
            register_phone(client, data)
        except ApiError as exc:
            if exc.status_code == 409:
                report["checks"]["duplicate_phone_rejected"] = "passed"
            else:
                add_issue(report, f"Duplicate phone returned HTTP {exc.status_code}, expected 409.")
        else:
            add_issue(report, "Duplicate phone registration unexpectedly succeeded.")
        phone_user, phone_token = login_phone(client, data)
        report["checks"]["phone_login"] = "passed"
        verify_user_response_has_no_secret_fields(report, phone_user, "phone login")
        if phone_user.get("phone") != data.normalized_phone:
            add_issue(report, f"Expected phone {data.normalized_phone!r}, got {phone_user.get('phone')!r}.")

        if has_oauth_payload(data.oauth_login):
            oauth_user, oauth_token = oauth_login(client, data.oauth_login)
            oauth_user_again, oauth_token_again = oauth_login(client, data.oauth_login)
            report["checks"]["developer_oauth_login"] = "passed"
            if oauth_user.get("user_id") != oauth_user_again.get("user_id"):
                add_issue(report, "Repeated OAuth login did not return the same user.")
            verify_user_response_has_no_secret_fields(report, oauth_user_again, "oauth repeat login")
            logout(client, oauth_token_again)
        else:
            report["checks"]["developer_oauth_login"] = "skipped"
    except Exception as exc:
        add_issue(report, str(exc))
    finally:
        for maybe_token in (oauth_token, phone_token, token):
            if maybe_token:
                try:
                    logout(client, maybe_token)
                except Exception:
                    pass
    finalize_report_status(report)
    return report


def _reset_token_from_response(value: Any) -> str:
    if not isinstance(value, dict):
        raise RuntimeError(f"Password reset request returned unexpected payload: {value!r}")
    reset_token = str(value.get("reset_token") or "")
    if not reset_token:
        raise RuntimeError("Password reset request did not return a demo reset token.")
    return reset_token


def logout(client: ApiClient, token: str) -> None:
    client.request_json("POST", "/api/v1/users/logout", user_token=token)


def base_scenario_report(data: ScenarioData, mode: str) -> dict[str, Any]:
    return {
        "scenario": data.scenario_id,
        "marker": data.marker,
        "mode": mode,
        "email": data.email,
        "phone": data.normalized_phone,
        "oauth_provider": data.oauth_login["provider"] if has_oauth_payload(data.oauth_login) else None,
        "subscription_plan": data.subscription_plan,
        "errors": [],
        "warnings": [],
        "checks": {},
    }


def finalize_report_status(report: dict[str, Any]) -> None:
    report["status"] = "failed" if report["errors"] else "passed"


def sanitize_report(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(term in lowered for term in SENSITIVE_REPORT_KEYS):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_report(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_report(item) for item in value]
    if isinstance(value, str) and _looks_sensitive(value):
        return "[REDACTED]"
    return value


def _looks_sensitive(value: str) -> bool:
    return len(value) >= 32 and bool(re.match(r"^[A-Za-z0-9_-]+$", value))


def verify_report_has_no_forbidden_terms(report: dict[str, Any], forbidden_terms: list[str]) -> list[str]:
    serialized = json.dumps(report, ensure_ascii=False)
    return sorted(term for term in forbidden_terms if term in serialized)


def write_report(run_id: str, report: dict[str, Any]) -> Path:
    path = ARTIFACT_ROOT / "user_2_e2e" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_checks(args: argparse.Namespace, manifest: dict[str, Any], scenario_ids: list[str], run_id: str) -> int:
    client = ApiClient(args.api_base_url)
    scenario_reports: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        data = scenario_data(manifest, scenario_id, run_id)
        if args.run_api_fallback:
            scenario_report = run_api_fallback(client, data)
        else:
            scenario_report = run_verify_ui(client, data)
        scenario_reports.append(scenario_report)

    report = {
        "run_id": run_id,
        "suite": args.suite,
        "scenario_ids": scenario_ids,
        "mode": "run-api-fallback" if args.run_api_fallback else "verify-ui-run",
        "api_base_url": args.api_base_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": scenario_reports,
    }
    report = sanitize_report(report)
    sanitized_results = report.get("results") if isinstance(report.get("results"), list) else []
    leak_failure = False
    for index, scenario_id in enumerate(scenario_ids):
        data = scenario_data(manifest, scenario_id, run_id)
        leaked = verify_report_has_no_forbidden_terms(report, data.forbidden_report_terms)
        if leaked:
            leak_failure = True
            target = sanitized_results[index] if index < len(sanitized_results) and isinstance(sanitized_results[index], dict) else {}
            target.setdefault("errors", []).append(f"Sanitized report still contains forbidden term(s): {leaked}")
            target["status"] = "failed"
    path = write_report(run_id, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written: {path}")
    failures = [item for item in scenario_reports if item.get("errors") or item.get("status") != "passed"]
    return 1 if failures or leak_failure else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live User 2.0 UI/API E2E checks.")
    parser.add_argument("--run-id", default="", help="Run id used for generated test data, for example 20260620-153000.")
    parser.add_argument("--print-ui-scenario", action="store_true", help="Print exact values and steps for the Streamlit Users UI.")
    parser.add_argument("--verify-ui-run", action="store_true", help="Verify a completed Streamlit Users UI run through the User API.")
    parser.add_argument("--run-api-fallback", action="store_true", help="Explicit fallback: exercise the User 2.0 flow through the API.")
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="Scenario suite from the manifest.")
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario ids. Overrides --suite when provided.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to the scenario manifest JSON.")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    return parser


def main() -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args()
    if not args.print_ui_scenario and not args.verify_ui_run and not args.run_api_fallback:
        parser.print_help()
        return 2

    manifest = load_manifest(Path(args.manifest))
    scenario_ids = resolve_scenarios(manifest, args.suite, args.scenarios)
    run_id = args.run_id or utc_run_id()
    if args.print_ui_scenario:
        print_ui_scenario(manifest, args.suite, args.scenarios, scenario_ids, run_id)
    if args.verify_ui_run or args.run_api_fallback:
        return run_checks(args, manifest, scenario_ids, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
