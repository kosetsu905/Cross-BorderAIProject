from collections.abc import Generator
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from admin_dashboard import (
    USER_AUTH_METHODS,
    _friendly_user_error,
    _headers,
    _oauth_payload,
    _payment_method_payload,
    _user_auth_state_from_response,
    _user_headers,
    _user_profile_summary,
)
from api.user_routes import create_user_router
from database import get_db_session
from db_models import (
    UserAuthProviderRecord,
    UserPasswordResetTokenRecord,
    UserPaymentMethodRecord,
    UserRecord,
    UserSessionRecord,
)
from services.user_service import hash_token


USER_TABLES = [
    UserRecord.__table__,
    UserAuthProviderRecord.__table__,
    UserSessionRecord.__table__,
    UserPasswordResetTokenRecord.__table__,
    UserPaymentMethodRecord.__table__,
]


def make_client() -> tuple[TestClient, sessionmaker[Session]]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in USER_TABLES:
        table.create(engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    app = FastAPI()
    app.include_router(create_user_router())

    def override_db() -> Generator[Session, None, None]:
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db_session] = override_db
    return TestClient(app), testing_session


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def register_email(client: TestClient, email: str = "john@example.com", password: str = "SecurePass123!") -> dict[str, Any]:
    response = client.post(
        "/api/v1/users/register/email",
        json={
            "email": email,
            "password": password,
            "username": "john_doe",
            "first_name": "John",
            "last_name": "Doe",
            "country": "US",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def register_phone(client: TestClient, phone: str = "13800138000", password: str = "PhonePass123!") -> dict[str, Any]:
    response = client.post(
        "/api/v1/users/register/phone",
        json={
            "country_code": "+86",
            "phone": phone,
            "password": password,
            "username": "phone_user",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_email_register_duplicate_and_login() -> None:
    client, _ = make_client()
    created = register_email(client)

    assert created["token_type"] == "bearer"
    assert created["access_token"]
    assert created["user"]["email"] == "john@example.com"
    assert "password_hash" not in str(created)
    assert "token_hash" not in str(created)

    duplicate = client.post(
        "/api/v1/users/register/email",
        json={"email": "john@example.com", "password": "SecurePass123!"},
    )
    assert duplicate.status_code == 409

    bad_login = client.post(
        "/api/v1/users/login/email",
        json={"email": "john@example.com", "password": "wrong"},
    )
    assert bad_login.status_code == 401

    login = client.post(
        "/api/v1/users/login/email",
        json={"email": "john@example.com", "password": "SecurePass123!"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["email"] == "john@example.com"


def test_phone_register_duplicate_and_login() -> None:
    client, _ = make_client()
    created = register_phone(client)

    assert created["access_token"]
    assert created["user"]["phone"] == "+8613800138000"
    assert any(provider["provider"] == "phone" for provider in created["user"]["auth_providers"])

    duplicate = client.post(
        "/api/v1/users/register/phone",
        json={"country_code": "+86", "phone": "13800138000", "password": "PhonePass123!"},
    )
    assert duplicate.status_code == 409

    bad_login = client.post(
        "/api/v1/users/login/phone",
        json={"country_code": "+86", "phone": "13800138000", "password": "wrong"},
    )
    assert bad_login.status_code == 401

    login = client.post(
        "/api/v1/users/login/phone",
        json={"country_code": "+86", "phone": "13800138000", "password": "PhonePass123!"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["phone"] == "+8613800138000"


def test_session_can_access_me_and_logout_revokes_token() -> None:
    client, _ = make_client()
    token = register_email(client)["access_token"]

    me = client.get("/api/v1/users/me", headers=auth_header(token))
    assert me.status_code == 200
    assert me.json()["username"] == "john_doe"

    logout = client.post("/api/v1/users/logout", headers=auth_header(token))
    assert logout.status_code == 200

    revoked = client.get("/api/v1/users/me", headers=auth_header(token))
    assert revoked.status_code == 401


def test_profile_update_only_allows_declared_fields() -> None:
    client, _ = make_client()
    token = register_email(client)["access_token"]

    extra = client.patch(
        "/api/v1/users/me",
        headers=auth_header(token),
        json={"first_name": "Jane", "password_hash": "leak"},
    )
    assert extra.status_code == 422

    response = client.patch(
        "/api/v1/users/me",
        headers=auth_header(token),
        json={"first_name": "Jane", "timezone": "America/New_York", "language": "en"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["first_name"] == "Jane"
    assert data["timezone"] == "America/New_York"


def test_change_password_invalidates_old_password() -> None:
    client, _ = make_client()
    token = register_email(client)["access_token"]

    changed = client.post(
        "/api/v1/users/me/password",
        headers=auth_header(token),
        json={"old_password": "SecurePass123!", "new_password": "NewSecurePass456!"},
    )
    assert changed.status_code == 200

    old_login = client.post(
        "/api/v1/users/login/email",
        json={"email": "john@example.com", "password": "SecurePass123!"},
    )
    assert old_login.status_code == 401

    new_login = client.post(
        "/api/v1/users/login/email",
        json={"email": "john@example.com", "password": "NewSecurePass456!"},
    )
    assert new_login.status_code == 200


def test_password_reset_token_is_hash_only_and_one_time_use() -> None:
    client, testing_session = make_client()
    register_email(client)

    requested = client.post(
        "/api/v1/users/password-reset/request",
        json={"email": "john@example.com"},
    )
    assert requested.status_code == 200
    reset_token = requested.json()["reset_token"]
    assert reset_token

    with testing_session() as db:
        stored = db.execute(select(UserPasswordResetTokenRecord)).scalars().one()
        assert stored.token_hash == hash_token(reset_token)
        assert reset_token not in stored.token_hash

    confirmed = client.post(
        "/api/v1/users/password-reset/confirm",
        json={"token": reset_token, "new_password": "ResetPassword789!"},
    )
    assert confirmed.status_code == 200

    reused = client.post(
        "/api/v1/users/password-reset/confirm",
        json={"token": reset_token, "new_password": "AnotherPassword789!"},
    )
    assert reused.status_code == 400

    login = client.post(
        "/api/v1/users/login/email",
        json={"email": "john@example.com", "password": "ResetPassword789!"},
    )
    assert login.status_code == 200


def test_oauth_login_reuses_existing_identity_and_sanitizes_provider_info() -> None:
    client, _ = make_client()
    payload = {
        "provider": "wechat",
        "provider_user_id": "wechat_123",
        "provider_info": {
            "email": "wechat@example.com",
            "name": "Wechat User",
            "access_token": "secret-token",
            "email_verified": True,
        },
    }

    first = client.post("/api/v1/users/oauth/login", json=payload)
    assert first.status_code == 200
    second = client.post("/api/v1/users/oauth/login", json=payload)
    assert second.status_code == 200
    assert second.json()["user"]["user_id"] == first.json()["user"]["user_id"]
    provider_info = second.json()["user"]["auth_providers"][0]["provider_info"]
    assert "access_token" not in provider_info


def test_oauth_link_and_unlink() -> None:
    client, _ = make_client()
    token = register_email(client)["access_token"]

    linked = client.post(
        "/api/v1/users/me/oauth",
        headers=auth_header(token),
        json={"provider": "google", "provider_user_id": "google_john", "provider_info": {"email": "john@gmail.com"}},
    )
    assert linked.status_code == 200
    assert {provider["provider"] for provider in linked.json()["auth_providers"]} == {"email", "google"}

    unlinked = client.delete("/api/v1/users/me/oauth/google", headers=auth_header(token))
    assert unlinked.status_code == 200

    profile = client.get("/api/v1/users/me", headers=auth_header(token))
    assert {provider["provider"] for provider in profile.json()["auth_providers"]} == {"email"}


def test_payment_methods_default_remove_and_sensitive_rejection() -> None:
    client, _ = make_client()
    token = register_email(client)["access_token"]

    first = client.post(
        "/api/v1/users/me/payment-methods",
        headers=auth_header(token),
        json={
            "payment_type": "credit_card",
            "payment_data": {"last_four": "4242", "brand": "visa"},
            "is_default": True,
        },
    )
    assert first.status_code == 200
    first_id = first.json()["payment_method_id"]
    assert first.json()["is_default"] is True

    rejected = client.post(
        "/api/v1/users/me/payment-methods",
        headers=auth_header(token),
        json={
            "payment_type": "credit_card",
            "payment_data": {"card_number": "4242424242424242"},
            "is_default": False,
        },
    )
    assert rejected.status_code == 400

    second = client.post(
        "/api/v1/users/me/payment-methods",
        headers=auth_header(token),
        json={"payment_type": "paypal", "payment_data": {"account": "masked@example.com"}, "is_default": False},
    )
    assert second.status_code == 200
    second_id = second.json()["payment_method_id"]

    defaulted = client.post(
        f"/api/v1/users/me/payment-methods/{second_id}/default",
        headers=auth_header(token),
    )
    assert defaulted.status_code == 200
    assert defaulted.json()["is_default"] is True

    removed = client.delete(
        f"/api/v1/users/me/payment-methods/{first_id}",
        headers=auth_header(token),
    )
    assert removed.status_code == 200

    profile = client.get("/api/v1/users/me", headers=auth_header(token)).json()
    assert [method["payment_method_id"] for method in profile["payment_methods"]] == [second_id]
    assert profile["default_payment_method"] == second_id


def test_subscription_lifecycle() -> None:
    client, _ = make_client()
    token = register_email(client)["access_token"]

    subscribed = client.post(
        "/api/v1/users/me/subscription",
        headers=auth_header(token),
        json={"plan_id": "professional"},
    )
    assert subscribed.status_code == 200
    assert subscribed.json()["subscription_plan"] == "professional"
    assert subscribed.json()["subscription_status"] == "active"

    cancelled = client.post("/api/v1/users/me/subscription/cancel", headers=auth_header(token))
    assert cancelled.status_code == 200
    assert cancelled.json()["subscription_status"] == "cancelled"


def test_response_does_not_expose_secret_hashes() -> None:
    client, _ = make_client()
    token = register_email(client)["access_token"]
    profile = client.get("/api/v1/users/me", headers=auth_header(token)).json()
    serialized = str(profile)

    assert "password_hash" not in serialized
    assert "token_hash" not in serialized
    assert "reset_token" not in serialized


def test_dashboard_user_helpers_do_not_change_workflow_headers() -> None:
    assert USER_AUTH_METHODS == ["Email", "Phone", "Simulated OAuth"]
    assert _headers("workflow-token") == {"Authorization": "Bearer workflow-token"}
    assert _user_headers("user-token") == {"Authorization": "Bearer user-token"}
    assert _oauth_payload("google", "google-1", {"email": "a@example.com"}) == {
        "provider": "google",
        "provider_user_id": "google-1",
        "provider_info": {"email": "a@example.com"},
    }
    assert _payment_method_payload("paypal", {"account": "masked@example.com"}, True) == {
        "payment_type": "paypal",
        "payment_data": {"account": "masked@example.com"},
        "is_default": True,
    }


def test_dashboard_user_errors_are_human_readable() -> None:
    validation_message = _friendly_user_error(
        '422: {"detail":[{"loc":["body","password"],"msg":"String should have at least 8 characters","type":"string_too_short"}]}'
    )
    conflict_message = _friendly_user_error('409: {"detail":"Email already registered"}')
    session_message = _friendly_user_error('401: {"detail":"Invalid or expired user session"}')

    for message in [validation_message, conflict_message, session_message]:
        assert "{" not in message
        assert "}" not in message
        assert "access_token" not in message
        assert "token_hash" not in message
    assert validation_message == "Password must be at least 8 characters."
    assert conflict_message == "This email is already registered. Try logging in instead."
    assert session_message == "Your session has expired. Please sign in again."


def test_dashboard_auth_state_keeps_token_out_of_display_payload() -> None:
    result = {
        "access_token": "secret-session-token",
        "token_type": "bearer",
        "expires_at": "2026-06-22T00:00:00Z",
        "user": {
            "username": "jane",
            "email": "jane@example.com",
            "subscription_plan": "professional",
            "subscription_status": "active",
        },
    }

    state = _user_auth_state_from_response(result, None)

    assert state["ok"] is True
    assert state["session_token"] == "secret-session-token"
    assert state["display"] == {
        "account": "jane",
        "email": "jane@example.com",
        "plan": "Professional",
        "status": "Active",
    }
    assert "access_token" not in str(state["display"])
    assert "secret-session-token" not in str(state["display"])
    assert _user_profile_summary(result["user"]) == state["display"]
