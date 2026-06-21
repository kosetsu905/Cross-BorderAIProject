import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db_models import (
    UserAuthProviderRecord,
    UserPasswordResetTokenRecord,
    UserPaymentMethodRecord,
    UserRecord,
    UserSessionRecord,
)
from user_models import (
    AddPaymentMethodRequest,
    AuthProvider,
    AuthProviderResponse,
    EmailRegisterRequest,
    OAuthLinkRequest,
    OAuthLoginRequest,
    PhoneLoginRequest,
    PaymentMethodResponse,
    PhoneRegisterRequest,
    SubscriptionRequest,
    SubscriptionStatus,
    UserResponse,
    UserUpdateRequest,
)


PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 210_000
SESSION_TTL = timedelta(days=30)
PASSWORD_RESET_TTL = timedelta(hours=24)
SENSITIVE_METADATA_KEYS = {
    "access_token",
    "api_key",
    "card_number",
    "client_secret",
    "credential",
    "credentials",
    "cvc",
    "cvv",
    "password",
    "pan",
    "refresh_token",
    "secret",
    "security_code",
    "token",
}
OAUTH_ONLY_PROVIDERS = {
    AuthProvider.GOOGLE,
    AuthProvider.FACEBOOK,
    AuthProvider.TWITTER,
    AuthProvider.LINKEDIN,
    AuthProvider.APPLE,
    AuthProvider.GITHUB,
    AuthProvider.MICROSOFT,
    AuthProvider.WECHAT,
    AuthProvider.ALIPAY,
    AuthProvider.WEIBO,
    AuthProvider.DOUYIN,
    AuthProvider.QQ,
}


class UserServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class UserConflictError(UserServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)


class UserAuthenticationError(UserServiceError):
    def __init__(self, message: str = "Invalid user credentials") -> None:
        super().__init__(message, status_code=401)


class UserNotFoundError(UserServiceError):
    def __init__(self, message: str = "User not found") -> None:
        super().__init__(message, status_code=404)


class UserValidationError(UserServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


@dataclass(frozen=True)
class SessionToken:
    token: str
    expires_at: datetime


@dataclass(frozen=True)
class PasswordResetToken:
    token: str
    expires_at: datetime


class UserService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def register_with_email(self, request: EmailRegisterRequest) -> UserRecord:
        email = _normalize_email(request.email)
        if self._user_by_email(email):
            raise UserConflictError("Email already registered")
        user = UserRecord(
            user_id=_new_id("user"),
            email=email,
            username=request.username or email.split("@", 1)[0],
            password_hash=hash_password(request.password),
            first_name=request.first_name,
            last_name=request.last_name,
            country=request.country,
        )
        self.db.add(user)
        self.db.flush()
        self._add_auth_provider(user.user_id, AuthProvider.EMAIL, email, None)
        return user

    def register_with_phone(self, request: PhoneRegisterRequest) -> UserRecord:
        phone = _normalize_phone(request.phone, request.country_code)
        if self._user_by_phone(phone):
            raise UserConflictError("Phone number already registered")
        user = UserRecord(
            user_id=_new_id("user"),
            phone=phone,
            username=request.username,
            password_hash=hash_password(request.password),
            first_name=request.first_name,
            last_name=request.last_name,
            country=request.country,
        )
        self.db.add(user)
        self.db.flush()
        self._add_auth_provider(user.user_id, AuthProvider.PHONE, phone, None)
        return user

    def login_with_email(self, email: str, password: str) -> tuple[UserRecord, SessionToken]:
        user = self._user_by_email(_normalize_email(email))
        if user is None or not user.password_hash or not user.is_active:
            raise UserAuthenticationError()
        if not verify_password(password, user.password_hash):
            raise UserAuthenticationError()
        user.last_login = _utc_now()
        user.updated_at = _utc_now()
        return user, self.create_session(user.user_id)

    def login_with_phone(self, request: PhoneLoginRequest) -> tuple[UserRecord, SessionToken]:
        user = self._user_by_phone(_normalize_phone(request.phone, request.country_code))
        if user is None or not user.password_hash or not user.is_active:
            raise UserAuthenticationError()
        if not verify_password(request.password, user.password_hash):
            raise UserAuthenticationError()
        user.last_login = _utc_now()
        user.updated_at = _utc_now()
        return user, self.create_session(user.user_id)

    def login_with_oauth(self, request: OAuthLoginRequest) -> tuple[UserRecord, SessionToken]:
        self._ensure_oauth_provider(request.provider)
        provider_info = _sanitize_metadata(request.provider_info)
        existing_provider = self._provider_identity(request.provider, request.provider_user_id)
        if existing_provider is not None:
            user = self.get_user(existing_provider.user_id)
            user.last_login = _utc_now()
            user.updated_at = _utc_now()
            return user, self.create_session(user.user_id)

        email = _normalize_email_or_none(provider_info.get("email"))
        user = self._user_by_email(email) if email else None
        if user is None:
            user = UserRecord(
                user_id=_new_id("user"),
                email=email,
                username=_string_or_none(provider_info.get("username")) or _string_or_none(provider_info.get("name")),
                first_name=_string_or_none(provider_info.get("first_name")),
                last_name=_string_or_none(provider_info.get("last_name")),
                country=_string_or_none(provider_info.get("country")),
                language=_string_or_none(provider_info.get("language")) or "en",
                is_email_verified=bool(provider_info.get("email_verified")) if email else False,
            )
            self.db.add(user)
            self.db.flush()

        self._add_auth_provider(user.user_id, request.provider, request.provider_user_id, provider_info)
        user.last_login = _utc_now()
        user.updated_at = _utc_now()
        return user, self.create_session(user.user_id)

    def create_session(self, user_id: str) -> SessionToken:
        token = secrets.token_urlsafe(32)
        expires_at = _utc_now() + SESSION_TTL
        self.db.add(
            UserSessionRecord(
                session_id=_new_id("sess"),
                user_id=user_id,
                token_hash=hash_token(token),
                expires_at=expires_at,
            )
        )
        self.db.flush()
        return SessionToken(token=token, expires_at=expires_at)

    def get_user_by_session_token(self, token: str) -> UserRecord:
        session = self.db.execute(
            select(UserSessionRecord).where(UserSessionRecord.token_hash == hash_token(token))
        ).scalars().first()
        now = _utc_now()
        if session is None or session.revoked_at is not None or _datetime_expired(session.expires_at, now):
            raise UserAuthenticationError("Invalid or expired user session")
        user = self.get_user(session.user_id)
        if not user.is_active:
            raise UserAuthenticationError("User account is inactive")
        return user

    def revoke_session(self, token: str) -> None:
        session = self.db.execute(
            select(UserSessionRecord).where(UserSessionRecord.token_hash == hash_token(token))
        ).scalars().first()
        if session is not None and session.revoked_at is None:
            session.revoked_at = _utc_now()
            self.db.flush()

    def get_user(self, user_id: str) -> UserRecord:
        user = self.db.get(UserRecord, user_id)
        if user is None:
            raise UserNotFoundError()
        return user

    def update_user(self, user: UserRecord, request: UserUpdateRequest) -> UserRecord:
        fields = request.model_dump(exclude_unset=True)
        if "email" in fields and fields["email"] is not None:
            existing = self._user_by_email(_normalize_email(fields["email"]))
            if existing is not None and existing.user_id != user.user_id:
                raise UserConflictError("Email already registered")
            fields["email"] = _normalize_email(fields["email"])
        if "phone" in fields and fields["phone"] is not None:
            normalized_phone = _normalize_phone(fields["phone"], "")
            existing = self._user_by_phone(normalized_phone)
            if existing is not None and existing.user_id != user.user_id:
                raise UserConflictError("Phone number already registered")
            fields["phone"] = normalized_phone

        for field, value in fields.items():
            if value is not None:
                setattr(user, field, value)
        user.updated_at = _utc_now()
        self.db.flush()
        return user

    def change_password(self, user: UserRecord, old_password: str, new_password: str) -> None:
        if not user.password_hash or not verify_password(old_password, user.password_hash):
            raise UserAuthenticationError("Old password is incorrect")
        user.password_hash = hash_password(new_password)
        user.updated_at = _utc_now()
        self.db.flush()

    def request_password_reset(self, email: str) -> PasswordResetToken | None:
        user = self._user_by_email(_normalize_email(email))
        if user is None:
            return None
        token = secrets.token_urlsafe(32)
        expires_at = _utc_now() + PASSWORD_RESET_TTL
        self.db.add(
            UserPasswordResetTokenRecord(
                reset_token_id=_new_id("reset"),
                user_id=user.user_id,
                token_hash=hash_token(token),
                expires_at=expires_at,
            )
        )
        self.db.flush()
        return PasswordResetToken(token=token, expires_at=expires_at)

    def reset_password(self, token: str, new_password: str) -> None:
        reset_record = self.db.execute(
            select(UserPasswordResetTokenRecord)
            .where(UserPasswordResetTokenRecord.token_hash == hash_token(token))
            .where(UserPasswordResetTokenRecord.used_at.is_(None))
        ).scalars().first()
        now = _utc_now()
        if reset_record is None or _datetime_expired(reset_record.expires_at, now):
            raise UserValidationError("Invalid or expired password reset token")
        user = self.get_user(reset_record.user_id)
        user.password_hash = hash_password(new_password)
        user.updated_at = now
        reset_record.used_at = now
        self.db.flush()

    def link_oauth_provider(self, user: UserRecord, request: OAuthLinkRequest | OAuthLoginRequest) -> UserRecord:
        self._ensure_oauth_provider(request.provider)
        existing_identity = self._provider_identity(request.provider, request.provider_user_id)
        if existing_identity is not None and existing_identity.user_id != user.user_id:
            raise UserConflictError("OAuth provider identity is already linked to another user")
        existing_for_user = self._provider_for_user(user.user_id, request.provider)
        if existing_for_user is not None:
            if existing_for_user.provider_user_id == request.provider_user_id:
                return user
            raise UserConflictError("OAuth provider is already linked for this user")
        self._add_auth_provider(
            user.user_id,
            request.provider,
            request.provider_user_id,
            _sanitize_metadata(request.provider_info),
        )
        user.updated_at = _utc_now()
        self.db.flush()
        return user

    def unlink_oauth_provider(self, user: UserRecord, provider: AuthProvider) -> None:
        self._ensure_oauth_provider(provider)
        record = self._provider_for_user(user.user_id, provider)
        if record is None:
            raise UserNotFoundError("OAuth provider is not linked")
        self.db.delete(record)
        user.updated_at = _utc_now()
        self.db.flush()

    def add_payment_method(self, user: UserRecord, request: AddPaymentMethodRequest) -> UserPaymentMethodRecord:
        payment_data = _sanitize_payment_data(request.payment_data)
        is_default = request.is_default or not self._active_payment_methods(user.user_id)
        if is_default:
            self._clear_default_payment_methods(user.user_id)
        record = UserPaymentMethodRecord(
            payment_method_id=_new_id("pm"),
            user_id=user.user_id,
            payment_type=request.payment_type.value,
            payment_data=payment_data,
            is_default=is_default,
        )
        self.db.add(record)
        user.updated_at = _utc_now()
        self.db.flush()
        return record

    def remove_payment_method(self, user: UserRecord, payment_method_id: str) -> None:
        record = self._payment_method(user.user_id, payment_method_id)
        record.is_active = False
        record.is_default = False
        record.updated_at = _utc_now()
        user.updated_at = _utc_now()
        self.db.flush()

    def set_default_payment_method(self, user: UserRecord, payment_method_id: str) -> UserPaymentMethodRecord:
        record = self._payment_method(user.user_id, payment_method_id)
        self._clear_default_payment_methods(user.user_id)
        record.is_default = True
        record.updated_at = _utc_now()
        user.updated_at = _utc_now()
        self.db.flush()
        return record

    def subscribe_to_plan(self, user: UserRecord, request: SubscriptionRequest) -> UserRecord:
        start = _utc_now()
        user.subscription_plan = request.plan_id
        user.subscription_status = SubscriptionStatus.ACTIVE.value
        user.subscription_start = start
        user.subscription_end = request.end_date or start + timedelta(days=30)
        user.updated_at = start
        self.db.flush()
        return user

    def cancel_subscription(self, user: UserRecord) -> UserRecord:
        user.subscription_status = SubscriptionStatus.CANCELLED.value
        user.updated_at = _utc_now()
        self.db.flush()
        return user

    def user_response(self, user: UserRecord) -> UserResponse:
        providers = self.db.execute(
            select(UserAuthProviderRecord)
            .where(UserAuthProviderRecord.user_id == user.user_id)
            .order_by(UserAuthProviderRecord.connected_at.asc())
        ).scalars().all()
        payment_methods = self._active_payment_methods(user.user_id)
        default_payment = next((method.payment_method_id for method in payment_methods if method.is_default), None)
        return UserResponse(
            user_id=user.user_id,
            email=user.email,
            phone=user.phone,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            country=user.country,
            timezone=user.timezone,
            language=user.language,
            auth_providers=[
                AuthProviderResponse(
                    provider=AuthProvider(provider.provider),
                    provider_user_id=provider.provider_user_id,
                    provider_info=provider.provider_info,
                    connected_at=provider.connected_at,
                )
                for provider in providers
            ],
            is_email_verified=user.is_email_verified,
            is_phone_verified=user.is_phone_verified,
            subscription_plan=user.subscription_plan,
            subscription_status=SubscriptionStatus(user.subscription_status),
            subscription_start=user.subscription_start,
            subscription_end=user.subscription_end,
            payment_methods=[
                PaymentMethodResponse(
                    payment_method_id=method.payment_method_id,
                    payment_type=method.payment_type,
                    payment_data=method.payment_data or {},
                    is_default=method.is_default,
                    is_active=method.is_active,
                    added_at=method.added_at,
                )
                for method in payment_methods
            ],
            default_payment_method=default_payment,
            total_workflows_run=user.total_workflows_run,
            total_tokens_used=user.total_tokens_used,
            total_api_calls=user.total_api_calls,
            is_active=user.is_active,
            last_login=user.last_login,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    def _user_by_email(self, email: str | None) -> UserRecord | None:
        if not email:
            return None
        return self.db.execute(select(UserRecord).where(UserRecord.email == email)).scalars().first()

    def _user_by_phone(self, phone: str | None) -> UserRecord | None:
        if not phone:
            return None
        return self.db.execute(select(UserRecord).where(UserRecord.phone == phone)).scalars().first()

    def _provider_identity(
        self,
        provider: AuthProvider,
        provider_user_id: str,
    ) -> UserAuthProviderRecord | None:
        return self.db.execute(
            select(UserAuthProviderRecord)
            .where(UserAuthProviderRecord.provider == provider.value)
            .where(UserAuthProviderRecord.provider_user_id == provider_user_id)
        ).scalars().first()

    def _provider_for_user(self, user_id: str, provider: AuthProvider) -> UserAuthProviderRecord | None:
        return self.db.execute(
            select(UserAuthProviderRecord)
            .where(UserAuthProviderRecord.user_id == user_id)
            .where(UserAuthProviderRecord.provider == provider.value)
        ).scalars().first()

    def _add_auth_provider(
        self,
        user_id: str,
        provider: AuthProvider,
        provider_user_id: str,
        provider_info: dict[str, Any] | None,
    ) -> UserAuthProviderRecord:
        record = UserAuthProviderRecord(
            auth_provider_id=_new_id("auth"),
            user_id=user_id,
            provider=provider.value,
            provider_user_id=provider_user_id,
            provider_info=provider_info,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def _active_payment_methods(self, user_id: str) -> list[UserPaymentMethodRecord]:
        return list(
            self.db.execute(
                select(UserPaymentMethodRecord)
                .where(UserPaymentMethodRecord.user_id == user_id)
                .where(UserPaymentMethodRecord.is_active.is_(True))
                .order_by(UserPaymentMethodRecord.added_at.asc())
            ).scalars()
        )

    def _payment_method(self, user_id: str, payment_method_id: str) -> UserPaymentMethodRecord:
        record = self.db.execute(
            select(UserPaymentMethodRecord)
            .where(UserPaymentMethodRecord.user_id == user_id)
            .where(UserPaymentMethodRecord.payment_method_id == payment_method_id)
            .where(UserPaymentMethodRecord.is_active.is_(True))
        ).scalars().first()
        if record is None:
            raise UserNotFoundError("Payment method not found")
        return record

    def _clear_default_payment_methods(self, user_id: str) -> None:
        for method in self._active_payment_methods(user_id):
            method.is_default = False
            method.updated_at = _utc_now()

    @staticmethod
    def _ensure_oauth_provider(provider: AuthProvider) -> None:
        if provider not in OAUTH_ONLY_PROVIDERS:
            raise UserValidationError("Only OAuth-style providers are allowed for this endpoint")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_ITERATIONS,
    ).hex()
    return f"{PASSWORD_ALGORITHM}${PASSWORD_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt, expected = stored_hash.split("$", 3)
        iterations = int(iterations_text)
    except ValueError:
        return False
    if algorithm != PASSWORD_ALGORITHM:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(digest, expected)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_expired(value: datetime, now: datetime) -> bool:
    if value.tzinfo is None:
        return value <= now.replace(tzinfo=None)
    return value <= now


def _normalize_email(value: str) -> str:
    return value.strip().lower()


def _normalize_email_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    return stripped or None


def _normalize_phone(phone: str, country_code: str) -> str:
    raw_phone = phone.strip()
    if raw_phone.startswith("+"):
        combined = raw_phone
    else:
        prefix = country_code.strip()
        if prefix and not prefix.startswith("+"):
            prefix = f"+{prefix}"
        combined = f"{prefix}{raw_phone}" if prefix else raw_phone
    normalized = []
    for index, char in enumerate(combined):
        if char.isdigit() or (char == "+" and index == 0):
            normalized.append(char)
    return "".join(normalized)


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _sanitize_metadata(value: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        lowered = normalized_key.lower()
        if lowered in SENSITIVE_METADATA_KEYS or any(marker in lowered for marker in ("secret", "token", "password")):
            continue
        sanitized[normalized_key] = item
    return sanitized


def _sanitize_payment_data(value: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_metadata(value)
    if len(sanitized) != len(value):
        raise UserValidationError("payment_data must contain tokenized or masked metadata only")
    for key, item in sanitized.items():
        lowered = key.lower()
        if lowered in SENSITIVE_METADATA_KEYS or "card_number" in lowered:
            raise UserValidationError("payment_data must not contain raw card or secret fields")
        if isinstance(item, str) and item.isdigit() and len(item) > 6 and lowered not in {"last_four", "last4"}:
            raise UserValidationError("payment_data must not contain raw payment account numbers")
    return sanitized
