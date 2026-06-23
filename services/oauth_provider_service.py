import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from db_models import UserOAuthFlowRecord
from services.user_service import (
    SessionToken,
    UserAuthenticationError,
    UserService,
    UserValidationError,
    hash_token,
)
from user_models import AuthProvider, OAuthAction, RealOAuthProvider


OAUTH_STATE_TTL = timedelta(minutes=10)
OAUTH_RESULT_TTL = timedelta(minutes=5)
GOOGLE_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GITHUB_AUTHORIZATION_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GOOGLE_SCOPES = "openid email profile"
GITHUB_SCOPES = "read:user user:email"


@dataclass(frozen=True)
class OAuthProviderConfig:
    provider: RealOAuthProvider
    client_id: str
    client_secret: str
    redirect_uri: str
    return_url: str


@dataclass(frozen=True)
class OAuthStartResult:
    provider: RealOAuthProvider
    action: OAuthAction
    authorization_url: str
    expires_at: datetime


@dataclass(frozen=True)
class OAuthCallbackResult:
    redirect_url: str


@dataclass(frozen=True)
class OAuthCompleteResult:
    user_id: str
    session_token: SessionToken


@dataclass(frozen=True)
class ProviderIdentity:
    provider: AuthProvider
    provider_user_id: str
    provider_info: dict[str, Any]


class OAuthProviderService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def start_flow(
        self,
        provider: RealOAuthProvider,
        action: OAuthAction,
        *,
        user_id: str | None = None,
    ) -> OAuthStartResult:
        if action == OAuthAction.LINK and not user_id:
            raise UserAuthenticationError("Linking OAuth provider requires a signed-in user")
        config = self._provider_config(provider)
        state = secrets.token_urlsafe(32)
        expires_at = _utc_now() + OAUTH_STATE_TTL
        record = UserOAuthFlowRecord(
            flow_id=_new_flow_id(),
            provider=provider.value,
            action=action.value,
            user_id=user_id,
            state_hash=hash_token(state),
            state_expires_at=expires_at,
        )
        self.db.add(record)
        self.db.flush()
        return OAuthStartResult(
            provider=provider,
            action=action,
            authorization_url=self._authorization_url(config, state),
            expires_at=expires_at,
        )

    def handle_callback(self, provider: RealOAuthProvider, code: str, state: str) -> OAuthCallbackResult:
        if not code or not state:
            raise UserValidationError("OAuth callback is missing code or state")
        config = self._provider_config(provider)
        record = self._flow_by_state(provider, state)
        now = _utc_now()
        if record.state_consumed_at is not None or _expired(record.state_expires_at, now):
            raise UserValidationError("Invalid or expired OAuth state")

        identity = self._provider_identity(config, code)
        result_code = secrets.token_urlsafe(32)
        record.state_consumed_at = now
        record.result_code_hash = hash_token(result_code)
        record.result_expires_at = now + OAUTH_RESULT_TTL
        record.provider_user_id = identity.provider_user_id
        record.provider_info = identity.provider_info
        record.updated_at = now
        self.db.flush()
        return OAuthCallbackResult(
            redirect_url=_append_query_params(
                config.return_url,
                {"oauth_provider": provider.value, "oauth_result": result_code},
            )
        )

    def complete_flow(self, provider: RealOAuthProvider, result_code: str) -> OAuthCompleteResult:
        record = self._flow_by_result_code(provider, result_code)
        now = _utc_now()
        if (
            record.result_consumed_at is not None
            or record.result_expires_at is None
            or _expired(record.result_expires_at, now)
            or not record.provider_user_id
        ):
            raise UserValidationError("Invalid or expired OAuth result")

        user_service = UserService(self.db)
        auth_provider = AuthProvider(provider.value)
        provider_info = record.provider_info or {}
        if record.action == OAuthAction.LINK.value:
            if not record.user_id:
                raise UserValidationError("OAuth link result is missing user context")
            user = user_service.get_user(record.user_id)
            user = user_service.link_verified_oauth_provider(
                user,
                auth_provider,
                record.provider_user_id,
                provider_info,
            )
            user.last_login = now
            user.updated_at = now
            session_token = user_service.create_session(user.user_id)
        elif record.action == OAuthAction.LOGIN.value:
            merge_verified_email = auth_provider == AuthProvider.GOOGLE and bool(provider_info.get("email_verified"))
            user, session_token = user_service.login_with_verified_oauth(
                auth_provider,
                record.provider_user_id,
                provider_info,
                merge_verified_email=merge_verified_email,
            )
        else:
            raise UserValidationError("Unsupported OAuth action")

        record.result_consumed_at = now
        record.updated_at = now
        self.db.flush()
        return OAuthCompleteResult(user_id=user.user_id, session_token=session_token)

    def _provider_config(self, provider: RealOAuthProvider) -> OAuthProviderConfig:
        prefix = provider.value.upper()
        client_id = os.getenv(f"{prefix}_OAUTH_CLIENT_ID", "").strip()
        client_secret = os.getenv(f"{prefix}_OAUTH_CLIENT_SECRET", "").strip()
        redirect_uri = os.getenv(
            f"{prefix}_OAUTH_REDIRECT_URI",
            f"http://localhost:8000/api/v1/users/oauth/{provider.value}/callback",
        ).strip()
        return_url = os.getenv("OAUTH_RETURN_URL", "http://localhost:8501").strip()
        if not client_id or not client_secret or not redirect_uri:
            raise UserValidationError(f"{provider.value.title()} OAuth is not configured")
        return OAuthProviderConfig(
            provider=provider,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            return_url=return_url or "http://localhost:8501",
        )

    def _authorization_url(self, config: OAuthProviderConfig, state: str) -> str:
        if config.provider == RealOAuthProvider.GOOGLE:
            params = {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": GOOGLE_SCOPES,
                "state": state,
                "prompt": "select_account",
            }
            return f"{GOOGLE_AUTHORIZATION_URL}?{urlencode(params)}"
        params = {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": GITHUB_SCOPES,
            "state": state,
        }
        return f"{GITHUB_AUTHORIZATION_URL}?{urlencode(params)}"

    def _provider_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        if config.provider == RealOAuthProvider.GOOGLE:
            return self._google_identity(config, code)
        if config.provider == RealOAuthProvider.GITHUB:
            return self._github_identity(config, code)
        raise UserValidationError("Unsupported OAuth provider")

    def _google_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._exchange_code(
            GOOGLE_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config.redirect_uri,
            },
            "Google OAuth token exchange failed",
        )
        id_token_value = str(token_payload.get("id_token") or "")
        if not id_token_value:
            raise UserValidationError("Google OAuth response did not include an ID token")
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token as google_id_token
        except ImportError as exc:
            raise UserValidationError("Google OAuth support is not installed") from exc
        try:
            claims = google_id_token.verify_oauth2_token(
                id_token_value,
                google_requests.Request(),
                config.client_id,
            )
        except Exception as exc:
            raise UserValidationError("Google ID token verification failed") from exc
        issuer = claims.get("iss")
        provider_user_id = str(claims.get("sub") or "")
        if issuer not in {"accounts.google.com", "https://accounts.google.com"} or not provider_user_id:
            raise UserValidationError("Google ID token claims are invalid")
        return ProviderIdentity(
            provider=AuthProvider.GOOGLE,
            provider_user_id=provider_user_id,
            provider_info={
                "email": claims.get("email"),
                "email_verified": bool(claims.get("email_verified")),
                "name": claims.get("name"),
                "given_name": claims.get("given_name"),
                "family_name": claims.get("family_name"),
                "picture": claims.get("picture"),
                "locale": claims.get("locale"),
            },
        )

    def _github_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._exchange_code(
            GITHUB_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "redirect_uri": config.redirect_uri,
            },
            "GitHub OAuth token exchange failed",
            headers={"Accept": "application/json"},
        )
        access_token = str(token_payload.get("access_token") or "")
        if not access_token:
            raise UserValidationError("GitHub OAuth response did not include an access token")
        try:
            with httpx.Client(timeout=20) as client:
                response = client.get(
                    GITHUB_USER_URL,
                    headers={
                        "Accept": "application/vnd.github+json",
                        "Authorization": f"Bearer {access_token}",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                )
            response.raise_for_status()
            profile = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UserValidationError("GitHub user profile lookup failed") from exc
        provider_user_id = str(profile.get("id") or "")
        if not provider_user_id:
            raise UserValidationError("GitHub user profile is missing an id")
        return ProviderIdentity(
            provider=AuthProvider.GITHUB,
            provider_user_id=provider_user_id,
            provider_info={
                "login": profile.get("login"),
                "name": profile.get("name"),
                "email": profile.get("email"),
                "avatar_url": profile.get("avatar_url"),
                "html_url": profile.get("html_url"),
            },
        )

    def _exchange_code(
        self,
        url: str,
        payload: dict[str, str],
        error_message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=20) as client:
                response = client.post(url, data=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UserValidationError(error_message) from exc
        if not isinstance(data, dict) or data.get("error"):
            raise UserValidationError(error_message)
        return data

    def _flow_by_state(self, provider: RealOAuthProvider, state: str) -> UserOAuthFlowRecord:
        record = self.db.execute(
            select(UserOAuthFlowRecord)
            .where(UserOAuthFlowRecord.provider == provider.value)
            .where(UserOAuthFlowRecord.state_hash == hash_token(state))
        ).scalars().first()
        if record is None:
            raise UserValidationError("Invalid or expired OAuth state")
        return record

    def _flow_by_result_code(self, provider: RealOAuthProvider, result_code: str) -> UserOAuthFlowRecord:
        record = self.db.execute(
            select(UserOAuthFlowRecord)
            .where(UserOAuthFlowRecord.provider == provider.value)
            .where(UserOAuthFlowRecord.result_code_hash == hash_token(result_code))
        ).scalars().first()
        if record is None:
            raise UserValidationError("Invalid or expired OAuth result")
        return record


def _new_flow_id() -> str:
    return f"oauth_{secrets.token_hex(12)}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _expired(value: datetime, now: datetime) -> bool:
    comparable = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return comparable <= now


def _append_query_params(url: str, params: dict[str, str]) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
