import base64
import hashlib
import json
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
FACEBOOK_AUTHORIZATION_BASE_URL = "https://www.facebook.com"
FACEBOOK_GRAPH_BASE_URL = "https://graph.facebook.com"
X_AUTHORIZATION_URL = "https://x.com/i/oauth2/authorize"
X_TOKEN_URL = "https://api.x.com/2/oauth2/token"
X_USER_ME_URL = "https://api.x.com/2/users/me"
APPLE_AUTHORIZATION_URL = "https://appleid.apple.com/auth/authorize"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
WECHAT_AUTHORIZATION_URL = "https://open.weixin.qq.com/connect/qrconnect"
WECHAT_TOKEN_URL = "https://api.weixin.qq.com/sns/oauth2/access_token"
WECHAT_USERINFO_URL = "https://api.weixin.qq.com/sns/userinfo"
ALIPAY_AUTHORIZATION_URL = "https://openauth.alipay.com/oauth2/publicAppAuthorize.htm"
ALIPAY_GATEWAY_URL = "https://openapi.alipay.com/gateway.do"
WEIBO_AUTHORIZATION_URL = "https://api.weibo.com/oauth2/authorize"
WEIBO_TOKEN_URL = "https://api.weibo.com/oauth2/access_token"
WEIBO_USER_URL = "https://api.weibo.com/2/users/show.json"
DOUYIN_AUTHORIZATION_URL = "https://open.douyin.com/platform/oauth/connect/"
DOUYIN_TOKEN_URL = "https://open.douyin.com/oauth/access_token/"
DOUYIN_USERINFO_URL = "https://open.douyin.com/oauth/userinfo/"
QQ_AUTHORIZATION_URL = "https://graph.qq.com/oauth2.0/authorize"
QQ_TOKEN_URL = "https://graph.qq.com/oauth2.0/token"
QQ_OPENID_URL = "https://graph.qq.com/oauth2.0/me"
QQ_USERINFO_URL = "https://graph.qq.com/user/get_user_info"
GITHUB_AUTHORIZATION_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
MICROSOFT_AUTHORIZATION_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
MICROSOFT_USERINFO_URL = "https://graph.microsoft.com/oidc/userinfo"
LINKEDIN_AUTHORIZATION_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
GOOGLE_SCOPES = "openid email profile"
FACEBOOK_SCOPES = "public_profile,email"
X_SCOPES = "users.read"
APPLE_SCOPES = "name email"
WECHAT_SCOPES = "snsapi_login"
ALIPAY_SCOPES = "auth_user"
GITHUB_SCOPES = "read:user user:email"
MICROSOFT_SCOPES = "openid profile email"
LINKEDIN_SCOPES = "openid profile email"
DOUYIN_SCOPES = "user_info"


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
        pkce_code_verifier = _new_pkce_code_verifier() if provider == RealOAuthProvider.TWITTER else None
        expires_at = _utc_now() + OAUTH_STATE_TTL
        record = UserOAuthFlowRecord(
            flow_id=_new_flow_id(),
            provider=provider.value,
            action=action.value,
            user_id=user_id,
            state_hash=hash_token(state),
            state_expires_at=expires_at,
            pkce_code_verifier=pkce_code_verifier,
        )
        self.db.add(record)
        self.db.flush()
        return OAuthStartResult(
            provider=provider,
            action=action,
            authorization_url=self._authorization_url(config, state, pkce_code_verifier),
            expires_at=expires_at,
        )

    def handle_callback(
        self,
        provider: RealOAuthProvider,
        code: str,
        state: str,
        callback_user: dict[str, Any] | None = None,
    ) -> OAuthCallbackResult:
        if not code or not state:
            raise UserValidationError("OAuth callback is missing code or state")
        config = self._provider_config(provider)
        record = self._flow_by_state(provider, state)
        now = _utc_now()
        if record.state_consumed_at is not None or _expired(record.state_expires_at, now):
            raise UserValidationError("Invalid or expired OAuth state")

        identity = self._provider_identity(config, code, record.pkce_code_verifier, callback_user)
        result_code = secrets.token_urlsafe(32)
        record.state_consumed_at = now
        record.result_code_hash = hash_token(result_code)
        record.result_expires_at = now + OAUTH_RESULT_TTL
        record.provider_user_id = identity.provider_user_id
        record.provider_info = identity.provider_info
        record.pkce_code_verifier = None
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
        if provider == RealOAuthProvider.WECHAT and not client_id:
            client_id = os.getenv("WECHAT_OAUTH_APP_ID", "").strip()
        if provider == RealOAuthProvider.ALIPAY and not client_id:
            client_id = os.getenv("ALIPAY_OAUTH_APP_ID", "").strip()
        if provider == RealOAuthProvider.DOUYIN and not client_id:
            client_id = os.getenv("DOUYIN_OAUTH_CLIENT_KEY", "").strip()
        client_secret = os.getenv(f"{prefix}_OAUTH_CLIENT_SECRET", "").strip()
        redirect_uri = os.getenv(
            f"{prefix}_OAUTH_REDIRECT_URI",
            f"http://localhost:8000/api/v1/users/oauth/{provider.value}/callback",
        ).strip()
        return_url = os.getenv("OAUTH_RETURN_URL", "http://localhost:8501").strip()
        if provider == RealOAuthProvider.APPLE:
            if not client_id or not redirect_uri:
                raise UserValidationError("Apple OAuth is not configured")
        elif provider == RealOAuthProvider.ALIPAY:
            if not client_id or not redirect_uri or not _alipay_private_key():
                raise UserValidationError("Alipay OAuth is not configured")
        elif not client_id or not client_secret or not redirect_uri:
            raise UserValidationError(f"{provider.value.title()} OAuth is not configured")
        return OAuthProviderConfig(
            provider=provider,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            return_url=return_url or "http://localhost:8501",
        )

    def _authorization_url(self, config: OAuthProviderConfig, state: str, pkce_code_verifier: str | None = None) -> str:
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
        if config.provider == RealOAuthProvider.MICROSOFT:
            params = {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "response_mode": "query",
                "scope": MICROSOFT_SCOPES,
                "state": state,
                "prompt": "select_account",
            }
            return f"{MICROSOFT_AUTHORIZATION_URL}?{urlencode(params)}"
        if config.provider == RealOAuthProvider.TWITTER:
            if not pkce_code_verifier:
                raise UserValidationError("X OAuth could not start PKCE")
            params = {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": X_SCOPES,
                "state": state,
                "code_challenge": _pkce_code_challenge(pkce_code_verifier),
                "code_challenge_method": "S256",
            }
            return f"{X_AUTHORIZATION_URL}?{urlencode(params)}"
        if config.provider == RealOAuthProvider.APPLE:
            params = {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "response_mode": "form_post",
                "scope": APPLE_SCOPES,
                "state": state,
            }
            return f"{APPLE_AUTHORIZATION_URL}?{urlencode(params)}"
        if config.provider == RealOAuthProvider.WECHAT:
            params = {
                "appid": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": WECHAT_SCOPES,
                "state": state,
            }
            return f"{WECHAT_AUTHORIZATION_URL}?{urlencode(params)}#wechat_redirect"
        if config.provider == RealOAuthProvider.ALIPAY:
            params = {
                "app_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "scope": ALIPAY_SCOPES,
                "state": state,
            }
            return f"{_alipay_authorization_url()}?{urlencode(params)}"
        if config.provider == RealOAuthProvider.WEIBO:
            params = {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "state": state,
            }
            scope = _weibo_scopes()
            if scope:
                params["scope"] = scope
            return f"{WEIBO_AUTHORIZATION_URL}?{urlencode(params)}"
        if config.provider == RealOAuthProvider.DOUYIN:
            params = {
                "client_key": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": _douyin_scopes(),
                "state": state,
            }
            optional_scope = _douyin_optional_scopes()
            if optional_scope:
                params["optionalScope"] = optional_scope
            return f"{DOUYIN_AUTHORIZATION_URL}?{urlencode(params)}"
        if config.provider == RealOAuthProvider.QQ:
            params = {
                "response_type": "code",
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "state": state,
            }
            scope = _qq_scopes()
            if scope:
                params["scope"] = scope
            return f"{QQ_AUTHORIZATION_URL}?{urlencode(params)}"
        if config.provider == RealOAuthProvider.FACEBOOK:
            version = _facebook_graph_version()
            params = {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": FACEBOOK_SCOPES,
                "state": state,
            }
            return f"{FACEBOOK_AUTHORIZATION_BASE_URL}/{version}/dialog/oauth?{urlencode(params)}"
        if config.provider == RealOAuthProvider.LINKEDIN:
            params = {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": LINKEDIN_SCOPES,
                "state": state,
            }
            return f"{LINKEDIN_AUTHORIZATION_URL}?{urlencode(params)}"
        params = {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": GITHUB_SCOPES,
            "state": state,
        }
        return f"{GITHUB_AUTHORIZATION_URL}?{urlencode(params)}"

    def _provider_identity(
        self,
        config: OAuthProviderConfig,
        code: str,
        pkce_code_verifier: str | None = None,
        callback_user: dict[str, Any] | None = None,
    ) -> ProviderIdentity:
        if config.provider == RealOAuthProvider.GOOGLE:
            return self._google_identity(config, code)
        if config.provider == RealOAuthProvider.FACEBOOK:
            return self._facebook_identity(config, code)
        if config.provider == RealOAuthProvider.TWITTER:
            return self._twitter_identity(config, code, pkce_code_verifier)
        if config.provider == RealOAuthProvider.APPLE:
            return self._apple_identity(config, code, callback_user)
        if config.provider == RealOAuthProvider.WECHAT:
            return self._wechat_identity(config, code)
        if config.provider == RealOAuthProvider.ALIPAY:
            return self._alipay_identity(config, code)
        if config.provider == RealOAuthProvider.WEIBO:
            return self._weibo_identity(config, code)
        if config.provider == RealOAuthProvider.DOUYIN:
            return self._douyin_identity(config, code)
        if config.provider == RealOAuthProvider.QQ:
            return self._qq_identity(config, code)
        if config.provider == RealOAuthProvider.GITHUB:
            return self._github_identity(config, code)
        if config.provider == RealOAuthProvider.MICROSOFT:
            return self._microsoft_identity(config, code)
        if config.provider == RealOAuthProvider.LINKEDIN:
            return self._linkedin_identity(config, code)
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

    def _facebook_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        version = _facebook_graph_version()
        token_payload = self._exchange_code(
            f"{FACEBOOK_GRAPH_BASE_URL}/{version}/oauth/access_token",
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "redirect_uri": config.redirect_uri,
            },
            "Facebook OAuth token exchange failed",
        )
        access_token = str(token_payload.get("access_token") or "")
        profile = self._userinfo_with_access_token(
            f"{FACEBOOK_GRAPH_BASE_URL}/{version}/me",
            access_token,
            "Facebook user profile lookup failed",
            params={"fields": "id,name,email,picture"},
        )
        provider_user_id = str(profile.get("id") or "")
        if not provider_user_id:
            raise UserValidationError("Facebook user profile is missing an id")
        picture = profile.get("picture") if isinstance(profile.get("picture"), dict) else {}
        picture_data = picture.get("data") if isinstance(picture.get("data"), dict) else {}
        return ProviderIdentity(
            provider=AuthProvider.FACEBOOK,
            provider_user_id=provider_user_id,
            provider_info={
                "email": profile.get("email"),
                "name": profile.get("name"),
                "picture": picture_data.get("url"),
            },
        )

    def _twitter_identity(
        self,
        config: OAuthProviderConfig,
        code: str,
        pkce_code_verifier: str | None,
    ) -> ProviderIdentity:
        if not pkce_code_verifier:
            raise UserValidationError("X OAuth PKCE verifier is missing")
        token_payload = self._exchange_code(
            X_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config.redirect_uri,
                "code_verifier": pkce_code_verifier,
            },
            "X OAuth token exchange failed",
            headers={"Accept": "application/json"},
        )
        profile_payload = self._userinfo_with_access_token(
            X_USER_ME_URL,
            str(token_payload.get("access_token") or ""),
            "X user profile lookup failed",
            params={"user.fields": "id,name,username,profile_image_url,verified"},
        )
        profile = profile_payload.get("data") if isinstance(profile_payload.get("data"), dict) else profile_payload
        provider_user_id = str(profile.get("id") or "")
        if not provider_user_id:
            raise UserValidationError("X user profile is missing an id")
        return ProviderIdentity(
            provider=AuthProvider.TWITTER,
            provider_user_id=provider_user_id,
            provider_info={
                "username": profile.get("username"),
                "name": profile.get("name"),
                "picture": profile.get("profile_image_url"),
                "verified": bool(profile.get("verified")),
            },
        )

    def _apple_identity(
        self,
        config: OAuthProviderConfig,
        code: str,
        callback_user: dict[str, Any] | None,
    ) -> ProviderIdentity:
        token_payload = self._exchange_code(
            APPLE_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": _apple_client_secret(config.client_id),
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config.redirect_uri,
            },
            "Apple OAuth token exchange failed",
        )
        id_token_value = str(token_payload.get("id_token") or "")
        if not id_token_value:
            raise UserValidationError("Apple OAuth response did not include an ID token")
        claims = _verify_apple_id_token(id_token_value, config.client_id)
        provider_user_id = str(claims.get("sub") or "")
        if not provider_user_id:
            raise UserValidationError("Apple ID token is missing a subject")
        name_payload = callback_user.get("name") if isinstance(callback_user, dict) else {}
        first_name = name_payload.get("firstName") if isinstance(name_payload, dict) else None
        last_name = name_payload.get("lastName") if isinstance(name_payload, dict) else None
        full_name = " ".join(part for part in [str(first_name or "").strip(), str(last_name or "").strip()] if part)
        return ProviderIdentity(
            provider=AuthProvider.APPLE,
            provider_user_id=provider_user_id,
            provider_info={
                "email": claims.get("email"),
                "email_verified": _truthy_claim(claims.get("email_verified")),
                "is_private_email": _truthy_claim(claims.get("is_private_email")),
                "name": full_name or None,
                "given_name": first_name,
                "family_name": last_name,
            },
        )

    def _wechat_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._get_json(
            WECHAT_TOKEN_URL,
            {
                "appid": config.client_id,
                "secret": config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
            },
            "WeChat OAuth token exchange failed",
        )
        access_token = str(token_payload.get("access_token") or "")
        openid = str(token_payload.get("openid") or "")
        if not access_token or not openid:
            raise UserValidationError("WeChat OAuth response is missing access_token or openid")
        profile = self._get_json(
            WECHAT_USERINFO_URL,
            {
                "access_token": access_token,
                "openid": openid,
                "lang": "en",
            },
            "WeChat user profile lookup failed",
        )
        provider_user_id = str(profile.get("unionid") or profile.get("openid") or openid)
        if not provider_user_id:
            raise UserValidationError("WeChat user profile is missing an id")
        return ProviderIdentity(
            provider=AuthProvider.WECHAT,
            provider_user_id=provider_user_id,
            provider_info={
                "openid": profile.get("openid") or openid,
                "unionid": profile.get("unionid"),
                "nickname": profile.get("nickname"),
                "name": profile.get("nickname"),
                "picture": profile.get("headimgurl"),
                "country": profile.get("country"),
                "province": profile.get("province"),
                "city": profile.get("city"),
                "language": profile.get("language"),
                "sex": profile.get("sex"),
            },
        )

    def _alipay_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._alipay_gateway_request(
            config,
            "alipay.system.oauth.token",
            {
                "grant_type": "authorization_code",
                "code": code,
            },
            "Alipay OAuth token exchange failed",
        )
        access_token = str(token_payload.get("access_token") or "")
        provider_user_id = str(token_payload.get("user_id") or token_payload.get("alipay_user_id") or "")
        if not access_token or not provider_user_id:
            raise UserValidationError("Alipay OAuth response is missing access_token or user_id")

        profile = self._alipay_gateway_request(
            config,
            "alipay.user.info.share",
            {},
            "Alipay user profile lookup failed",
            auth_token=access_token,
        )
        provider_user_id = str(profile.get("user_id") or provider_user_id)
        if not provider_user_id:
            raise UserValidationError("Alipay user profile is missing a user_id")
        return ProviderIdentity(
            provider=AuthProvider.ALIPAY,
            provider_user_id=provider_user_id,
            provider_info={
                "user_id": provider_user_id,
                "avatar": profile.get("avatar"),
                "picture": profile.get("avatar"),
                "nick_name": profile.get("nick_name"),
                "name": profile.get("nick_name") or profile.get("user_name"),
                "user_name": profile.get("user_name"),
                "province": profile.get("province"),
                "city": profile.get("city"),
                "gender": profile.get("gender"),
                "user_type": profile.get("user_type"),
                "is_certified": profile.get("is_certified"),
            },
        )

    def _weibo_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._exchange_code(
            WEIBO_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": config.redirect_uri,
            },
            "Weibo OAuth token exchange failed",
        )
        access_token = str(token_payload.get("access_token") or "")
        uid = str(token_payload.get("uid") or "")
        if not access_token or not uid:
            raise UserValidationError("Weibo OAuth response is missing access_token or uid")
        profile = self._get_json(
            WEIBO_USER_URL,
            {
                "access_token": access_token,
                "uid": uid,
            },
            "Weibo user profile lookup failed",
        )
        provider_user_id = str(profile.get("idstr") or profile.get("id") or uid)
        if not provider_user_id:
            raise UserValidationError("Weibo user profile is missing an id")
        return ProviderIdentity(
            provider=AuthProvider.WEIBO,
            provider_user_id=provider_user_id,
            provider_info={
                "uid": uid,
                "screen_name": profile.get("screen_name"),
                "name": profile.get("name") or profile.get("screen_name"),
                "location": profile.get("location"),
                "description": profile.get("description"),
                "profile_image_url": profile.get("profile_image_url"),
                "avatar_large": profile.get("avatar_large"),
                "picture": profile.get("avatar_large") or profile.get("profile_image_url"),
                "profile_url": profile.get("profile_url"),
                "domain": profile.get("domain"),
                "gender": profile.get("gender"),
                "verified": bool(profile.get("verified")),
                "verified_type": profile.get("verified_type"),
            },
        )

    def _douyin_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._douyin_post_form(
            DOUYIN_TOKEN_URL,
            {
                "client_key": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
            },
            "Douyin OAuth token exchange failed",
        )
        access_token = str(token_payload.get("access_token") or "")
        open_id = str(token_payload.get("open_id") or "")
        if not access_token or not open_id:
            raise UserValidationError("Douyin OAuth response is missing access_token or open_id")
        profile = self._douyin_post_form(
            DOUYIN_USERINFO_URL,
            {
                "access_token": access_token,
                "open_id": open_id,
            },
            "Douyin user profile lookup failed",
        )
        provider_user_id = str(profile.get("open_id") or open_id)
        if not provider_user_id:
            raise UserValidationError("Douyin user profile is missing an open_id")
        return ProviderIdentity(
            provider=AuthProvider.DOUYIN,
            provider_user_id=provider_user_id,
            provider_info={
                "open_id": provider_user_id,
                "union_id": profile.get("union_id") or token_payload.get("union_id"),
                "nickname": profile.get("nickname"),
                "name": profile.get("nickname"),
                "avatar": profile.get("avatar"),
                "picture": profile.get("avatar"),
                "country": profile.get("country"),
                "province": profile.get("province"),
                "city": profile.get("city"),
                "e_account_role": profile.get("e_account_role"),
            },
        )

    def _qq_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._qq_token_response(
            {
                "grant_type": "authorization_code",
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "redirect_uri": config.redirect_uri,
            },
            "QQ OAuth token exchange failed",
        )
        access_token = str(token_payload.get("access_token") or "")
        if not access_token:
            raise UserValidationError("QQ OAuth response is missing access_token")
        openid_payload = self._qq_jsonp_response(
            QQ_OPENID_URL,
            {"access_token": access_token},
            "QQ OpenID lookup failed",
        )
        openid = str(openid_payload.get("openid") or "")
        if not openid:
            raise UserValidationError("QQ OpenID response is missing openid")
        profile = self._get_json(
            QQ_USERINFO_URL,
            {
                "access_token": access_token,
                "oauth_consumer_key": config.client_id,
                "openid": openid,
            },
            "QQ user profile lookup failed",
        )
        if profile.get("ret") not in (None, 0, "0"):
            raise UserValidationError(str(profile.get("msg") or "QQ user profile lookup failed"))
        return ProviderIdentity(
            provider=AuthProvider.QQ,
            provider_user_id=openid,
            provider_info={
                "openid": openid,
                "nickname": profile.get("nickname"),
                "name": profile.get("nickname"),
                "figureurl": profile.get("figureurl"),
                "figureurl_qq_1": profile.get("figureurl_qq_1"),
                "figureurl_qq_2": profile.get("figureurl_qq_2"),
                "picture": profile.get("figureurl_qq_2") or profile.get("figureurl_qq_1") or profile.get("figureurl"),
                "gender": profile.get("gender"),
                "province": profile.get("province"),
                "city": profile.get("city"),
                "year": profile.get("year"),
                "vip": profile.get("vip"),
                "level": profile.get("level"),
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

    def _microsoft_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._exchange_code(
            MICROSOFT_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config.redirect_uri,
                "scope": MICROSOFT_SCOPES,
            },
            "Microsoft OAuth token exchange failed",
        )
        profile = self._userinfo_with_access_token(
            MICROSOFT_USERINFO_URL,
            str(token_payload.get("access_token") or ""),
            "Microsoft user profile lookup failed",
        )
        provider_user_id = str(profile.get("sub") or "")
        if not provider_user_id:
            raise UserValidationError("Microsoft user profile is missing a subject")
        return ProviderIdentity(
            provider=AuthProvider.MICROSOFT,
            provider_user_id=provider_user_id,
            provider_info={
                "email": profile.get("email"),
                "name": profile.get("name"),
                "given_name": profile.get("given_name"),
                "family_name": profile.get("family_name"),
                "locale": profile.get("locale"),
            },
        )

    def _linkedin_identity(self, config: OAuthProviderConfig, code: str) -> ProviderIdentity:
        token_payload = self._exchange_code(
            LINKEDIN_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config.redirect_uri,
            },
            "LinkedIn OAuth token exchange failed",
            headers={"Accept": "application/json"},
        )
        profile = self._userinfo_with_access_token(
            LINKEDIN_USERINFO_URL,
            str(token_payload.get("access_token") or ""),
            "LinkedIn user profile lookup failed",
        )
        provider_user_id = str(profile.get("sub") or "")
        if not provider_user_id:
            raise UserValidationError("LinkedIn user profile is missing a subject")
        return ProviderIdentity(
            provider=AuthProvider.LINKEDIN,
            provider_user_id=provider_user_id,
            provider_info={
                "email": profile.get("email"),
                "email_verified": bool(profile.get("email_verified")),
                "name": profile.get("name"),
                "given_name": profile.get("given_name"),
                "family_name": profile.get("family_name"),
                "picture": profile.get("picture"),
                "locale": profile.get("locale"),
            },
        )

    def _alipay_gateway_request(
        self,
        config: OAuthProviderConfig,
        method: str,
        payload: dict[str, str],
        error_message: str,
        *,
        auth_token: str | None = None,
    ) -> dict[str, Any]:
        params = {
            "app_id": config.client_id,
            "method": method,
            "format": "JSON",
            "charset": "utf-8",
            "sign_type": "RSA2",
            "timestamp": _utc_now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
        }
        if auth_token:
            params["auth_token"] = auth_token
        params.update(payload)
        params["sign"] = _alipay_sign(params)
        try:
            with httpx.Client(timeout=20) as client:
                response = client.post(_alipay_gateway_url(), data=params)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UserValidationError(error_message) from exc
        if not isinstance(data, dict):
            raise UserValidationError(error_message)

        error_payload = data.get("error_response")
        if isinstance(error_payload, dict):
            raise UserValidationError(str(error_payload.get("sub_msg") or error_payload.get("msg") or error_message))
        response_key = f"{method.replace('.', '_')}_response"
        response_payload = data.get(response_key)
        if not isinstance(response_payload, dict):
            raise UserValidationError(error_message)
        if response_payload.get("code") and str(response_payload.get("code")) != "10000":
            raise UserValidationError(str(response_payload.get("sub_msg") or response_payload.get("msg") or error_message))
        return response_payload

    def _douyin_post_form(self, url: str, payload: dict[str, str], error_message: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=20) as client:
                response = client.post(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UserValidationError(error_message) from exc
        if not isinstance(data, dict):
            raise UserValidationError(error_message)
        result = data.get("data") if isinstance(data.get("data"), dict) else data
        if not isinstance(result, dict):
            raise UserValidationError(error_message)
        error_code = result.get("error_code", data.get("err_no"))
        if error_code not in (None, 0, "0"):
            raise UserValidationError(str(result.get("description") or data.get("err_msg") or error_message))
        return result

    def _qq_token_response(self, payload: dict[str, str], error_message: str) -> dict[str, str]:
        try:
            with httpx.Client(timeout=20) as client:
                response = client.get(QQ_TOKEN_URL, params=payload)
            response.raise_for_status()
            data = dict(parse_qsl(response.text, keep_blank_values=True))
        except httpx.HTTPError as exc:
            raise UserValidationError(error_message) from exc
        if not data or data.get("error"):
            raise UserValidationError(str(data.get("error_description") or error_message))
        return data

    def _qq_jsonp_response(self, url: str, params: dict[str, str], error_message: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=20) as client:
                response = client.get(url, params=params)
            response.raise_for_status()
            data = _parse_jsonp_object(response.text)
        except (httpx.HTTPError, ValueError) as exc:
            raise UserValidationError(error_message) from exc
        if not isinstance(data, dict) or data.get("error"):
            raise UserValidationError(str(data.get("error_description") or error_message))
        return data

    def _userinfo_with_access_token(
        self,
        url: str,
        access_token: str,
        error_message: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not access_token:
            raise UserValidationError(error_message)
        try:
            with httpx.Client(timeout=20) as client:
                response = client.get(
                    url,
                    params=params,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {access_token}",
                    },
                )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UserValidationError(error_message) from exc
        if not isinstance(data, dict):
            raise UserValidationError(error_message)
        return data

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

    def _get_json(self, url: str, params: dict[str, str], error_message: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=20) as client:
                response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UserValidationError(error_message) from exc
        if not isinstance(data, dict) or data.get("errcode") or data.get("error_code") or data.get("error"):
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


def _new_pkce_code_verifier() -> str:
    return secrets.token_urlsafe(64)[:128]


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _alipay_sign(params: dict[str, Any]) -> str:
    private_key = _alipay_private_key()
    signing_content = "&".join(
        f"{key}={value}"
        for key, value in sorted(params.items())
        if key != "sign" and value is not None and str(value) != ""
    )
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        key = serialization.load_pem_private_key(private_key.encode("utf-8"), password=None)
        signature = key.sign(signing_content.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
    except ImportError as exc:
        raise UserValidationError("Alipay OAuth RSA2 support is not installed") from exc
    except (TypeError, ValueError) as exc:
        raise UserValidationError("Alipay OAuth private key is invalid") from exc
    return base64.b64encode(signature).decode("ascii")


def _alipay_private_key() -> str:
    raw_key = os.getenv("ALIPAY_OAUTH_PRIVATE_KEY", "").strip()
    if raw_key:
        return raw_key.replace("\\n", "\n")
    key_path = os.getenv("ALIPAY_OAUTH_PRIVATE_KEY_PATH", "").strip()
    if not key_path:
        return ""
    try:
        with open(key_path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError as exc:
        raise UserValidationError("Alipay OAuth private key could not be read") from exc


def _apple_client_secret(client_id: str) -> str:
    team_id = os.getenv("APPLE_OAUTH_TEAM_ID", "").strip()
    key_id = os.getenv("APPLE_OAUTH_KEY_ID", "").strip()
    private_key = _apple_private_key()
    if not team_id or not key_id or not private_key:
        raise UserValidationError("Apple OAuth is not configured")
    try:
        import jwt
    except ImportError as exc:
        raise UserValidationError("Apple OAuth support is not installed") from exc
    now = int(_utc_now().timestamp())
    return str(
        jwt.encode(
            {
                "iss": team_id,
                "iat": now,
                "exp": now + 60 * 60 * 24 * 30,
                "aud": "https://appleid.apple.com",
                "sub": client_id,
            },
            private_key,
            algorithm="ES256",
            headers={"kid": key_id},
        )
    )


def _apple_private_key() -> str:
    raw_key = os.getenv("APPLE_OAUTH_PRIVATE_KEY", "").strip()
    if raw_key:
        return raw_key.replace("\\n", "\n")
    key_path = os.getenv("APPLE_OAUTH_PRIVATE_KEY_PATH", "").strip()
    if not key_path:
        return ""
    try:
        with open(key_path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError as exc:
        raise UserValidationError("Apple OAuth private key could not be read") from exc


def _verify_apple_id_token(id_token_value: str, client_id: str) -> dict[str, Any]:
    try:
        import jwt
    except ImportError as exc:
        raise UserValidationError("Apple OAuth support is not installed") from exc
    try:
        signing_key = jwt.PyJWKClient(APPLE_JWKS_URL).get_signing_key_from_jwt(id_token_value)
        claims = jwt.decode(
            id_token_value,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id,
            issuer="https://appleid.apple.com",
        )
    except Exception as exc:
        raise UserValidationError("Apple ID token verification failed") from exc
    if not isinstance(claims, dict):
        raise UserValidationError("Apple ID token claims are invalid")
    return claims


def _truthy_claim(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


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


def _facebook_graph_version() -> str:
    return os.getenv("FACEBOOK_GRAPH_API_VERSION", "v23.0").strip() or "v23.0"


def _alipay_authorization_url() -> str:
    return os.getenv("ALIPAY_OAUTH_AUTHORIZE_URL", ALIPAY_AUTHORIZATION_URL).strip() or ALIPAY_AUTHORIZATION_URL


def _alipay_gateway_url() -> str:
    return os.getenv("ALIPAY_OAUTH_GATEWAY_URL", ALIPAY_GATEWAY_URL).strip() or ALIPAY_GATEWAY_URL


def _weibo_scopes() -> str:
    return os.getenv("WEIBO_OAUTH_SCOPE", "").strip()


def _douyin_scopes() -> str:
    return os.getenv("DOUYIN_OAUTH_SCOPE", DOUYIN_SCOPES).strip() or DOUYIN_SCOPES


def _douyin_optional_scopes() -> str:
    return os.getenv("DOUYIN_OAUTH_OPTIONAL_SCOPE", "").strip()


def _qq_scopes() -> str:
    return os.getenv("QQ_OAUTH_SCOPE", "").strip()


def _parse_jsonp_object(raw_value: str) -> dict[str, Any]:
    stripped = raw_value.strip()
    if stripped.startswith("callback"):
        start = stripped.find("(")
        end = stripped.rfind(")")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Invalid JSONP payload")
        stripped = stripped[start + 1 : end]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("JSONP payload is not an object")
    return parsed
