from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db_session
from db_models import UserRecord
from services.user_service import UserService, UserServiceError
from user_models import (
    AddPaymentMethodRequest,
    AuthProvider,
    AuthResponse,
    ChangePasswordRequest,
    EmailLoginRequest,
    EmailRegisterRequest,
    OAuthLinkRequest,
    OAuthLoginRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    PasswordResetRequestResponse,
    PaymentMethodResponse,
    PhoneRegisterRequest,
    StatusResponse,
    SubscriptionRequest,
    UserResponse,
    UserUpdateRequest,
)


DbDependency = Annotated[Session, Depends(get_db_session)]
user_bearer_scheme = HTTPBearer(auto_error=False)


def create_user_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/users", tags=["users"])

    @router.post("/register/email", response_model=AuthResponse)
    async def register_with_email(request: EmailRegisterRequest, db: DbDependency) -> AuthResponse:
        service = UserService(db)
        try:
            user = service.register_with_email(request)
            session_token = service.create_session(user.user_id)
            _commit(db)
            return AuthResponse(
                user=service.user_response(user),
                access_token=session_token.token,
                expires_at=session_token.expires_at,
            )
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/register/phone", response_model=AuthResponse)
    async def register_with_phone(request: PhoneRegisterRequest, db: DbDependency) -> AuthResponse:
        service = UserService(db)
        try:
            user = service.register_with_phone(request)
            session_token = service.create_session(user.user_id)
            _commit(db)
            return AuthResponse(
                user=service.user_response(user),
                access_token=session_token.token,
                expires_at=session_token.expires_at,
            )
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/login/email", response_model=AuthResponse)
    async def login_with_email(request: EmailLoginRequest, db: DbDependency) -> AuthResponse:
        service = UserService(db)
        try:
            user, session_token = service.login_with_email(request.email, request.password)
            _commit(db)
            return AuthResponse(
                user=service.user_response(user),
                access_token=session_token.token,
                expires_at=session_token.expires_at,
            )
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/oauth/login", response_model=AuthResponse)
    async def login_with_oauth(request: OAuthLoginRequest, db: DbDependency) -> AuthResponse:
        service = UserService(db)
        try:
            user, session_token = service.login_with_oauth(request)
            _commit(db)
            return AuthResponse(
                user=service.user_response(user),
                access_token=session_token.token,
                expires_at=session_token.expires_at,
            )
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/logout", response_model=StatusResponse)
    async def logout(
        db: DbDependency,
        credentials: HTTPAuthorizationCredentials | None = Security(user_bearer_scheme),
    ) -> StatusResponse:
        token = _bearer_token(credentials)
        service = UserService(db)
        try:
            service.get_user_by_session_token(token)
            service.revoke_session(token)
            _commit(db)
            return StatusResponse(status="logged_out")
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.get("/me", response_model=UserResponse)
    async def get_me(current_user: Annotated[UserRecord, Depends(current_user_dependency)], db: DbDependency) -> UserResponse:
        return UserService(db).user_response(current_user)

    @router.patch("/me", response_model=UserResponse)
    async def update_me(
        request: UserUpdateRequest,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> UserResponse:
        service = UserService(db)
        try:
            user = service.update_user(current_user, request)
            _commit(db)
            return service.user_response(user)
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/me/password", response_model=StatusResponse)
    async def change_password(
        request: ChangePasswordRequest,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> StatusResponse:
        service = UserService(db)
        try:
            service.change_password(current_user, request.old_password, request.new_password)
            _commit(db)
            return StatusResponse(status="password_changed")
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/password-reset/request", response_model=PasswordResetRequestResponse)
    async def request_password_reset(
        request: PasswordResetRequest,
        db: DbDependency,
    ) -> PasswordResetRequestResponse:
        service = UserService(db)
        try:
            reset_token = service.request_password_reset(request.email)
            _commit(db)
            if reset_token is None:
                return PasswordResetRequestResponse(status="accepted")
            return PasswordResetRequestResponse(
                status="accepted",
                reset_token=reset_token.token,
                expires_at=reset_token.expires_at,
            )
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/password-reset/confirm", response_model=StatusResponse)
    async def confirm_password_reset(
        request: PasswordResetConfirmRequest,
        db: DbDependency,
    ) -> StatusResponse:
        service = UserService(db)
        try:
            service.reset_password(request.token, request.new_password)
            _commit(db)
            return StatusResponse(status="password_reset")
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/me/oauth", response_model=UserResponse)
    async def link_oauth_provider(
        request: OAuthLinkRequest,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> UserResponse:
        service = UserService(db)
        try:
            user = service.link_oauth_provider(current_user, request)
            _commit(db)
            return service.user_response(user)
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.delete("/me/oauth/{provider}", response_model=StatusResponse)
    async def unlink_oauth_provider(
        provider: AuthProvider,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> StatusResponse:
        service = UserService(db)
        try:
            service.unlink_oauth_provider(current_user, provider)
            _commit(db)
            return StatusResponse(status="oauth_unlinked")
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/me/payment-methods", response_model=PaymentMethodResponse)
    async def add_payment_method(
        request: AddPaymentMethodRequest,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> PaymentMethodResponse:
        service = UserService(db)
        try:
            payment_method = service.add_payment_method(current_user, request)
            _commit(db)
            return PaymentMethodResponse(
                payment_method_id=payment_method.payment_method_id,
                payment_type=payment_method.payment_type,
                payment_data=payment_method.payment_data or {},
                is_default=payment_method.is_default,
                is_active=payment_method.is_active,
                added_at=payment_method.added_at,
            )
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.delete("/me/payment-methods/{payment_method_id}", response_model=StatusResponse)
    async def remove_payment_method(
        payment_method_id: str,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> StatusResponse:
        service = UserService(db)
        try:
            service.remove_payment_method(current_user, payment_method_id)
            _commit(db)
            return StatusResponse(status="payment_method_removed")
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/me/payment-methods/{payment_method_id}/default", response_model=PaymentMethodResponse)
    async def set_default_payment_method(
        payment_method_id: str,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> PaymentMethodResponse:
        service = UserService(db)
        try:
            payment_method = service.set_default_payment_method(current_user, payment_method_id)
            _commit(db)
            return PaymentMethodResponse(
                payment_method_id=payment_method.payment_method_id,
                payment_type=payment_method.payment_type,
                payment_data=payment_method.payment_data or {},
                is_default=payment_method.is_default,
                is_active=payment_method.is_active,
                added_at=payment_method.added_at,
            )
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/me/subscription", response_model=UserResponse)
    async def subscribe_to_plan(
        request: SubscriptionRequest,
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> UserResponse:
        service = UserService(db)
        try:
            user = service.subscribe_to_plan(current_user, request)
            _commit(db)
            return service.user_response(user)
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    @router.post("/me/subscription/cancel", response_model=UserResponse)
    async def cancel_subscription(
        current_user: Annotated[UserRecord, Depends(current_user_dependency)],
        db: DbDependency,
    ) -> UserResponse:
        service = UserService(db)
        try:
            user = service.cancel_subscription(current_user)
            _commit(db)
            return service.user_response(user)
        except UserServiceError as exc:
            _rollback_and_raise(db, exc)

    return router


def current_user_dependency(
    db: DbDependency,
    credentials: HTTPAuthorizationCredentials | None = Security(user_bearer_scheme),
) -> UserRecord:
    token = _bearer_token(credentials)
    try:
        return UserService(db).get_user_by_session_token(token)
    except UserServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _bearer_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _commit(db: Session) -> None:
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="User record conflicts with existing data") from exc


def _rollback_and_raise(db: Session, exc: UserServiceError) -> None:
    db.rollback()
    raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
