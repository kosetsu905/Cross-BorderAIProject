from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


EMAIL_PATTERN = r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$"
PHONE_PATTERN = r"^\+?[0-9][0-9\s().-]{5,31}$"
SUBSCRIPTION_PLANS = {"starter", "professional", "enterprise"}


class StrictUserModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthProvider(str, Enum):
    GOOGLE = "google"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    APPLE = "apple"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    WECHAT = "wechat"
    ALIPAY = "alipay"
    WEIBO = "weibo"
    DOUYIN = "douyin"
    QQ = "qq"
    EMAIL = "email"
    PHONE = "phone"


class RealOAuthProvider(str, Enum):
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    APPLE = "apple"
    WECHAT = "wechat"
    ALIPAY = "alipay"
    WEIBO = "weibo"
    DOUYIN = "douyin"
    QQ = "qq"


class OAuthAction(str, Enum):
    LOGIN = "login"
    LINK = "link"


class PaymentMethodType(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    STRIPE = "stripe"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    ALIPAY_CN = "alipay_cn"
    WECHAT_PAY = "wechat_pay"
    UNION_PAY = "union_pay"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PENDING = "pending"
    TRIAL = "trial"


class EmailRegisterRequest(StrictUserModel):
    email: str = Field(..., min_length=3, max_length=255, pattern=EMAIL_PATTERN)
    password: str = Field(..., min_length=8, max_length=256)
    username: str | None = Field(None, min_length=1, max_length=120)
    first_name: str | None = Field(None, min_length=1, max_length=120)
    last_name: str | None = Field(None, min_length=1, max_length=120)
    country: str | None = Field(None, min_length=1, max_length=80)

    @field_validator("email", mode="after")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class PhoneRegisterRequest(StrictUserModel):
    phone: str = Field(..., min_length=6, max_length=32, pattern=PHONE_PATTERN)
    password: str = Field(..., min_length=8, max_length=256)
    country_code: str = Field("+86", min_length=1, max_length=8)
    username: str | None = Field(None, min_length=1, max_length=120)
    first_name: str | None = Field(None, min_length=1, max_length=120)
    last_name: str | None = Field(None, min_length=1, max_length=120)
    country: str | None = Field(None, min_length=1, max_length=80)


class PhoneLoginRequest(StrictUserModel):
    phone: str = Field(..., min_length=6, max_length=32, pattern=PHONE_PATTERN)
    password: str = Field(..., min_length=1, max_length=256)
    country_code: str = Field("+86", min_length=1, max_length=8)


class EmailLoginRequest(StrictUserModel):
    email: str = Field(..., min_length=3, max_length=255, pattern=EMAIL_PATTERN)
    password: str = Field(..., min_length=1, max_length=256)

    @field_validator("email", mode="after")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class OAuthLoginRequest(StrictUserModel):
    provider: AuthProvider
    provider_user_id: str = Field(..., min_length=1, max_length=255)
    provider_info: dict[str, Any] = Field(default_factory=dict)


class OAuthStartRequest(StrictUserModel):
    action: OAuthAction


class OAuthStartResponse(StrictUserModel):
    provider: RealOAuthProvider
    action: OAuthAction
    authorization_url: str
    expires_at: datetime


class OAuthCompleteRequest(StrictUserModel):
    result_code: str = Field(..., min_length=20, max_length=256)


class UserUpdateRequest(StrictUserModel):
    email: str | None = Field(None, min_length=3, max_length=255, pattern=EMAIL_PATTERN)
    phone: str | None = Field(None, min_length=6, max_length=64)
    username: str | None = Field(None, min_length=1, max_length=120)
    first_name: str | None = Field(None, min_length=1, max_length=120)
    last_name: str | None = Field(None, min_length=1, max_length=120)
    country: str | None = Field(None, min_length=1, max_length=80)
    timezone: str | None = Field(None, min_length=1, max_length=64)
    language: str | None = Field(None, min_length=1, max_length=32)

    @field_validator("email", mode="after")
    @classmethod
    def normalize_optional_email(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else value


class ChangePasswordRequest(StrictUserModel):
    old_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=8, max_length=256)


class PasswordResetRequest(StrictUserModel):
    email: str = Field(..., min_length=3, max_length=255, pattern=EMAIL_PATTERN)

    @field_validator("email", mode="after")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class PasswordResetConfirmRequest(StrictUserModel):
    token: str = Field(..., min_length=20, max_length=256)
    new_password: str = Field(..., min_length=8, max_length=256)


class OAuthLinkRequest(StrictUserModel):
    provider: AuthProvider
    provider_user_id: str = Field(..., min_length=1, max_length=255)
    provider_info: dict[str, Any] = Field(default_factory=dict)


class AddPaymentMethodRequest(StrictUserModel):
    payment_type: PaymentMethodType
    payment_data: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class SubscriptionRequest(StrictUserModel):
    plan_id: str = Field(..., min_length=1, max_length=64)
    end_date: datetime | None = None

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUBSCRIPTION_PLANS:
            raise ValueError("plan_id must be one of starter, professional, enterprise")
        return normalized


class AuthProviderResponse(StrictUserModel):
    provider: AuthProvider
    provider_user_id: str
    provider_info: dict[str, Any] | None = None
    connected_at: datetime


class PaymentMethodResponse(StrictUserModel):
    payment_method_id: str
    payment_type: PaymentMethodType
    payment_data: dict[str, Any]
    is_default: bool
    is_active: bool
    added_at: datetime


class UserResponse(StrictUserModel):
    user_id: str
    email: str | None = None
    phone: str | None = None
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    country: str | None = None
    timezone: str
    language: str
    auth_providers: list[AuthProviderResponse] = Field(default_factory=list)
    is_email_verified: bool
    is_phone_verified: bool
    subscription_plan: str
    subscription_status: SubscriptionStatus
    subscription_start: datetime | None = None
    subscription_end: datetime | None = None
    payment_methods: list[PaymentMethodResponse] = Field(default_factory=list)
    default_payment_method: str | None = None
    total_workflows_run: int
    total_tokens_used: int
    total_api_calls: int
    is_active: bool
    last_login: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AuthResponse(StrictUserModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class PasswordResetRequestResponse(StrictUserModel):
    status: str
    reset_token: str | None = None
    expires_at: datetime | None = None


class StatusResponse(StrictUserModel):
    status: str
