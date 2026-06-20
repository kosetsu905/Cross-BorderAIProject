--- ecommerce_ai_agents/modules/users/user_manager.py (原始)


+++ ecommerce_ai_agents/modules/users/user_manager.py (修改后)
"""
用户模块 - 跨境电商AI工具集合
支持中国和国际主流平台一键登录、密码管理、支付方式管理、套餐订阅
"""

import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass, field


class AuthProvider(Enum):
    """认证提供商"""
    # 国际主流平台
    GOOGLE = "google"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    APPLE = "apple"
    GITHUB = "github"
    MICROSOFT = "microsoft"

    # 中国主流平台
    WECHAT = "wechat"
    ALIPAY = "alipay"
    WEIBO = "weibo"
    DOUYIN = "douyin"
    QQ = "qq"

    # 传统方式
    EMAIL = "email"
    PHONE = "phone"


class PaymentMethodType(Enum):
    """支付方式类型"""
    # 国际支付方式
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    STRIPE = "stripe"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"

    # 中国支付方式
    ALIPAY_CN = "alipay_cn"
    WECHAT_PAY = "wechat_pay"
    UNION_PAY = "union_pay"

    # 其他方式
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"


class SubscriptionStatus(Enum):
    """订阅状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PENDING = "pending"
    TRIAL = "trial"


@dataclass
class User:
    """用户数据模型"""
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    username: Optional[str] = None
    password_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 个人信息
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country: Optional[str] = None
    timezone: str = "UTC"
    language: str = "en"

    # 认证信息
    auth_providers: List[Dict[str, Any]] = field(default_factory=list)
    is_email_verified: bool = False
    is_phone_verified: bool = False

    # 订阅信息
    subscription_plan: str = "starter"
    subscription_status: SubscriptionStatus = SubscriptionStatus.INACTIVE
    subscription_start: Optional[datetime] = None
    subscription_end: Optional[datetime] = None

    # 支付方式
    payment_methods: List[Dict[str, Any]] = field(default_factory=list)
    default_payment_method: Optional[str] = None

    # 使用统计
    total_workflows_run: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0

    # 账户状态
    is_active: bool = True
    last_login: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "phone": self.phone,
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "first_name": self.first_name,
            "last_name": self.last_name,
            "country": self.country,
            "timezone": self.timezone,
            "language": self.language,
            "auth_providers": self.auth_providers,
            "is_email_verified": self.is_email_verified,
            "is_phone_verified": self.is_phone_verified,
            "subscription_plan": self.subscription_plan,
            "subscription_status": self.subscription_status.value,
            "subscription_start": self.subscription_start.isoformat() if self.subscription_start else None,
            "subscription_end": self.subscription_end.isoformat() if self.subscription_end else None,
            "payment_methods": self.payment_methods,
            "default_payment_method": self.default_payment_method,
            "total_workflows_run": self.total_workflows_run,
            "total_tokens_used": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "is_active": self.is_active,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


@dataclass
class PasswordResetToken:
    """密码重置令牌"""
    user_id: str
    token: str
    expires_at: datetime
    is_used: bool = False


class PasswordManager:
    """密码管理器"""

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        哈希密码
        返回: (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)

        # 使用SHA-256进行哈希（生产环境建议使用bcrypt或argon2）
        salted_password = f"{salt}{password}"
        hashed = hashlib.sha256(salted_password.encode()).hexdigest()

        return hashed, salt

    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """验证密码"""
        computed_hash, _ = PasswordManager.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, hashed_password)

    @staticmethod
    def generate_reset_token(user_id: str) -> PasswordResetToken:
        """生成密码重置令牌"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)

        return PasswordResetToken(
            user_id=user_id,
            token=token,
            expires_at=expires_at
        )

    @staticmethod
    def validate_reset_token(reset_token: PasswordResetToken) -> bool:
        """验证重置令牌"""
        if reset_token.is_used:
            return False
        if datetime.now() > reset_token.expires_at:
            return False
        return True


class UserManager:
    """用户管理器 - 内存存储（生产环境应使用数据库）"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.email_to_user_id: Dict[str, str] = {}
        self.phone_to_user_id: Dict[str, str] = {}
        self.provider_to_user_id: Dict[str, Dict[str, str]] = {}  # provider -> {provider_user_id: local_user_id}
        self.reset_tokens: Dict[str, PasswordResetToken] = {}

    def generate_user_id(self) -> str:
        """生成用户ID"""
        return f"user_{secrets.token_hex(8)}"

    def register_with_email(
        self,
        email: str,
        password: str,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        country: Optional[str] = None
    ) -> User:
        """通过邮箱注册"""
        if email in self.email_to_user_id:
            raise ValueError("Email already registered")

        hashed_password, salt = PasswordManager.hash_password(password)

        user = User(
            user_id=self.generate_user_id(),
            email=email,
            username=username or email.split('@')[0],
            password_hash=f"{salt}${hashed_password}",
            first_name=first_name,
            last_name=last_name,
            country=country
        )

        self.users[user.user_id] = user
        self.email_to_user_id[email] = user.user_id

        # 添加邮箱认证提供者
        user.auth_providers.append({
            "provider": AuthProvider.EMAIL.value,
            "provider_user_id": email,
            "connected_at": datetime.now().isoformat()
        })

        return user

    def register_with_phone(
        self,
        phone: str,
        password: str,
        country_code: str = "+86",
        **kwargs
    ) -> User:
        """通过手机号注册"""
        full_phone = f"{country_code}{phone}"

        if full_phone in self.phone_to_user_id:
            raise ValueError("Phone number already registered")

        hashed_password, salt = PasswordManager.hash_password(password)

        user = User(
            user_id=self.generate_user_id(),
            phone=full_phone,
            password_hash=f"{salt}${hashed_password}",
            **kwargs
        )

        self.users[user.user_id] = user
        self.phone_to_user_id[full_phone] = user.user_id

        # 添加手机认证提供者
        user.auth_providers.append({
            "provider": AuthProvider.PHONE.value,
            "provider_user_id": full_phone,
            "connected_at": datetime.now().isoformat()
        })

        return user

    def login_with_oauth(
        self,
        provider: AuthProvider,
        provider_user_id: str,
        provider_info: Dict[str, Any]
    ) -> User:
        """
        OAuth登录（支持一键登录）
        如果用户不存在则自动创建
        """
        provider_key = provider.value

        # 检查是否已存在该OAuth用户
        if provider_key in self.provider_to_user_id:
            if provider_user_id in self.provider_to_user_id[provider_key]:
                user_id = self.provider_to_user_id[provider_key][provider_user_id]
                user = self.users[user_id]
                user.last_login = datetime.now()
                return user

        # 创建新用户
        user = User(
            user_id=self.generate_user_id(),
            email=provider_info.get("email"),
            username=provider_info.get("username") or provider_info.get("name"),
            first_name=provider_info.get("first_name"),
            last_name=provider_info.get("last_name"),
            country=provider_info.get("country"),
            language=provider_info.get("language", "en"),
            is_email_verified=provider_info.get("email_verified", False)
        )

        # 添加OAuth提供者
        user.auth_providers.append({
            "provider": provider_key,
            "provider_user_id": provider_user_id,
            "connected_at": datetime.now().isoformat(),
            "provider_info": provider_info
        })

        self.users[user.user_id] = user

        # 建立映射
        if provider_key not in self.provider_to_user_id:
            self.provider_to_user_id[provider_key] = {}
        self.provider_to_user_id[provider_key][provider_user_id] = user.user_id

        # 如果提供了邮箱，也建立邮箱映射
        if user.email and user.email not in self.email_to_user_id:
            self.email_to_user_id[user.email] = user.user_id

        user.last_login = datetime.now()

        return user

    def login_with_email(self, email: str, password: str) -> Optional[User]:
        """邮箱密码登录"""
        if email not in self.email_to_user_id:
            return None

        user_id = self.email_to_user_id[email]
        user = self.users[user_id]

        if not user.password_hash:
            return None  # 用户是通过OAuth注册的

        salt, stored_hash = user.password_hash.split('$', 1)

        if PasswordManager.verify_password(password, stored_hash, salt):
            user.last_login = datetime.now()
            return user

        return None

    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self.users.get(user_id)

    def update_user_info(
        self,
        user_id: str,
        **kwargs
    ) -> User:
        """更新用户信息"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")

        allowed_fields = [
            'email', 'phone', 'username', 'first_name', 'last_name',
            'country', 'timezone', 'language'
        ]

        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                setattr(user, field, value)

        user.updated_at = datetime.now()
        return user

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """修改密码"""
        user = self.users.get(user_id)
        if not user or not user.password_hash:
            return False

        salt, stored_hash = user.password_hash.split('$', 1)

        if not PasswordManager.verify_password(old_password, stored_hash, salt):
            return False

        new_hashed, new_salt = PasswordManager.hash_password(new_password)
        user.password_hash = f"{new_salt}${new_hashed}"
        user.updated_at = datetime.now()

        return True

    def request_password_reset(self, email: str) -> Optional[PasswordResetToken]:
        """请求密码重置"""
        if email not in self.email_to_user_id:
            return None  # 不暴露用户是否存在

        user_id = self.email_to_user_id[email]
        reset_token = PasswordManager.generate_reset_token(user_id)

        self.reset_tokens[reset_token.token] = reset_token

        return reset_token

    def reset_password(self, token: str, new_password: str) -> bool:
        """重置密码"""
        if token not in self.reset_tokens:
            return False

        reset_token = self.reset_tokens[token]

        if not PasswordManager.validate_reset_token(reset_token):
            return False

        user = self.users.get(reset_token.user_id)
        if not user:
            return False

        new_hashed, new_salt = PasswordManager.hash_password(new_password)
        user.password_hash = f"{new_salt}${new_hashed}"
        user.updated_at = datetime.now()

        reset_token.is_used = True

        return True

    def add_payment_method(
        self,
        user_id: str,
        payment_type: PaymentMethodType,
        payment_data: Dict[str, Any],
        is_default: bool = False
    ) -> Dict[str, Any]:
        """添加支付方式"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")

        payment_method = {
            "id": f"pm_{secrets.token_hex(8)}",
            "type": payment_type.value,
            "data": payment_data,
            "is_default": is_default,
            "added_at": datetime.now().isoformat(),
            "is_active": True
        }

        # 如果是默认支付方式，取消其他默认
        if is_default:
            for pm in user.payment_methods:
                pm["is_default"] = False
            user.default_payment_method = payment_method["id"]

        user.payment_methods.append(payment_method)

        return payment_method

    def remove_payment_method(self, user_id: str, payment_method_id: str) -> bool:
        """移除支付方式"""
        user = self.users.get(user_id)
        if not user:
            return False

        for i, pm in enumerate(user.payment_methods):
            if pm["id"] == payment_method_id:
                user.payment_methods.pop(i)
                if user.default_payment_method == payment_method_id:
                    user.default_payment_method = None
                return True

        return False

    def set_default_payment_method(self, user_id: str, payment_method_id: str) -> bool:
        """设置默认支付方式"""
        user = self.users.get(user_id)
        if not user:
            return False

        found = False
        for pm in user.payment_methods:
            if pm["id"] == payment_method_id:
                pm["is_default"] = True
                found = True
            else:
                pm["is_default"] = False

        if found:
            user.default_payment_method = payment_method_id
            return True

        return False

    def subscribe_to_plan(
        self,
        user_id: str,
        plan_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> User:
        """订阅套餐计划"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")

        user.subscription_plan = plan_id
        user.subscription_status = SubscriptionStatus.ACTIVE
        user.subscription_start = start_date or datetime.now()

        # 如果没有指定结束日期，默认为1个月后
        if end_date is None:
            user.subscription_end = user.subscription_start + timedelta(days=30)
        else:
            user.subscription_end = end_date

        user.updated_at = datetime.now()

        return user

    def cancel_subscription(self, user_id: str) -> User:
        """取消订阅"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")

        user.subscription_status = SubscriptionStatus.CANCELLED
        user.updated_at = datetime.now()

        return user

    def link_oauth_provider(
        self,
        user_id: str,
        provider: AuthProvider,
        provider_user_id: str,
        provider_info: Dict[str, Any]
    ) -> User:
        """为用户绑定新的OAuth提供者"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")

        # 检查是否已绑定
        for auth in user.auth_providers:
            if auth["provider"] == provider.value and auth["provider_user_id"] == provider_user_id:
                return user

        # 添加新的认证提供者
        user.auth_providers.append({
            "provider": provider.value,
            "provider_user_id": provider_user_id,
            "connected_at": datetime.now().isoformat(),
            "provider_info": provider_info
        })

        # 建立映射
        provider_key = provider.value
        if provider_key not in self.provider_to_user_id:
            self.provider_to_user_id[provider_key] = {}
        self.provider_to_user_id[provider_key][provider_user_id] = user_id

        user.updated_at = datetime.now()

        return user

    def unlink_oauth_provider(self, user_id: str, provider: AuthProvider) -> bool:
        """解绑OAuth提供者"""
        user = self.users.get(user_id)
        if not user:
            return False

        provider_key = provider.value

        # 从用户认证列表中移除
        initial_count = len(user.auth_providers)
        user.auth_providers = [
            auth for auth in user.auth_providers
            if auth["provider"] != provider_key
        ]

        if len(user.auth_providers) == initial_count:
            return False

        user.updated_at = datetime.now()
        return True


# 示例用法
if __name__ == "__main__":
    # 初始化用户管理器
    user_manager = UserManager()

    print("=" * 60)
    print("用户模块演示")
    print("=" * 60)

    # 1. 邮箱注册
    print("\n1. 邮箱注册:")
    user1 = user_manager.register_with_email(
        email="john@example.com",
        password="SecurePass123!",
        username="john_doe",
        first_name="John",
        last_name="Doe",
        country="US"
    )
    print(f"   用户ID: {user1.user_id}")
    print(f"   邮箱: {user1.email}")
    print(f"   用户名: {user1.username}")

    # 2. 微信一键登录（自动注册）
    print("\n2. 微信一键登录:")
    wechat_info = {
        "name": "张三",
        "email": "zhangsan@wechat.com",
        "country": "CN",
        "language": "zh",
        "email_verified": True
    }
    user2 = user_manager.login_with_oauth(
        provider=AuthProvider.WECHAT,
        provider_user_id="wechat_123456",
        provider_info=wechat_info
    )
    print(f"   用户ID: {user2.user_id}")
    print(f"   名称: {user2.first_name} {user2.last_name}")
    print(f"   认证方式: {[p['provider'] for p in user2.auth_providers]}")

    # 3. Google一键登录
    print("\n3. Google一键登录:")
    google_info = {
        "name": "Alice Smith",
        "email": "alice@gmail.com",
        "first_name": "Alice",
        "last_name": "Smith",
        "country": "UK",
        "email_verified": True
    }
    user3 = user_manager.login_with_oauth(
        provider=AuthProvider.GOOGLE,
        provider_user_id="google_789012",
        provider_info=google_info
    )
    print(f"   用户ID: {user3.user_id}")
    print(f"   邮箱: {user3.email}")

    # 4. 邮箱登录
    print("\n4. 邮箱登录:")
    logged_user = user_manager.login_with_email("john@example.com", "SecurePass123!")
    if logged_user:
        print(f"   登录成功: {logged_user.username}")
        print(f"   上次登录: {logged_user.last_login}")
    else:
        print("   登录失败")

    # 5. 更新用户信息
    print("\n5. 更新用户信息:")
    updated_user = user_manager.update_user_info(
        user1.user_id,
        timezone="America/New_York",
        language="en"
    )
    print(f"   时区: {updated_user.timezone}")
    print(f"   语言: {updated_user.language}")

    # 6. 修改密码
    print("\n6. 修改密码:")
    success = user_manager.change_password(
        user1.user_id,
        "SecurePass123!",
        "NewSecurePass456!"
    )
    print(f"   修改结果: {'成功' if success else '失败'}")

    # 7. 添加支付方式（支付宝）
    print("\n7. 添加支付方式（支付宝）:")
    alipay_method = user_manager.add_payment_method(
        user1.user_id,
        PaymentMethodType.ALIPAY_CN,
        {
            "account": "138****1234",
            "real_name": "张*"
        },
        is_default=True
    )
    print(f"   支付方式ID: {alipay_method['id']}")
    print(f"   类型: {alipay_method['type']}")

    # 8. 添加支付方式（信用卡）
    print("\n8. 添加支付方式（信用卡）:")
    card_method = user_manager.add_payment_method(
        user1.user_id,
        PaymentMethodType.CREDIT_CARD,
        {
            "last_four": "4242",
            "brand": "visa",
            "exp_month": 12,
            "exp_year": 2025
        }
    )
    print(f"   支付方式ID: {card_method['id']}")
    print(f"   卡号后四位: {card_method['data']['last_four']}")

    # 9. 订阅套餐
    print("\n9. 订阅Professional套餐:")
    subscribed_user = user_manager.subscribe_to_plan(
        user1.user_id,
        plan_id="professional"
    )
    print(f"   套餐: {subscribed_user.subscription_plan}")
    print(f"   状态: {subscribed_user.subscription_status.value}")
    print(f"   开始日期: {subscribed_user.subscription_start}")
    print(f"   结束日期: {subscribed_user.subscription_end}")

    # 10. 绑定更多OAuth提供者
    print("\n10. 为用户绑定Google账号:")
    linked_user = user_manager.link_oauth_provider(
        user1.user_id,
        AuthProvider.GOOGLE,
        "google_john_123",
        {"email": "john.work@gmail.com"}
    )
    print(f"   已绑定的认证方式: {[p['provider'] for p in linked_user.auth_providers]}")

    # 11. 密码重置流程
    print("\n11. 密码重置流程:")
    reset_token = user_manager.request_password_reset("john@example.com")
    if reset_token:
        print(f"   重置令牌已生成: {reset_token.token[:20]}...")

        # 执行重置
        reset_success = user_manager.reset_password(
            reset_token.token,
            "ResetPassword789!"
        )
        print(f"   重置结果: {'成功' if reset_success else '失败'}")

        # 验证新密码
        test_login = user_manager.login_with_email("john@example.com", "ResetPassword789!")
        print(f"   新密码登录: {'成功' if test_login else '失败'}")

    # 12. 查看用户完整信息
    print("\n12. 用户完整信息:")
    final_user = user_manager.get_user(user1.user_id)
    print(json.dumps(final_user.to_dict(), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("用户模块演示完成")
    print("=" * 60)