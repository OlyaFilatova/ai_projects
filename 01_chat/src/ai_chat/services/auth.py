"""Authentication services built around FastAPI Users and local refresh tokens."""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

from fastapi import status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users import BaseUserManager, exceptions, schemas
from fastapi_users.authentication import JWTStrategy
from fastapi_users.db import BaseUserDatabase
from fastapi_users.models import UserProtocol
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from ai_chat.config import JwtSettings
from ai_chat.errors import AppError
from ai_chat.persistence.models import RefreshTokenModel, UserModel
from ai_chat.repositories.refresh_tokens import RefreshTokenRepository
from ai_chat.repositories.users import UserRepository


@dataclass(slots=True)
class AuthError(AppError):
    """Domain-level authentication error with an HTTP-friendly status code."""


class TokenPayload(BaseModel):
    """Authenticated user identity extracted from a validated access token."""

    sub: str
    email: EmailStr


class UserCreate(schemas.BaseUserCreate):
    """Internal FastAPI Users create schema for local users."""


class SyncUserDatabase(BaseUserDatabase[UserProtocol[str], str]):
    """Minimal FastAPI Users database adapter on top of the sync SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._users = UserRepository(session)
        self._session = session

    async def get(self, id: str) -> UserProtocol[str] | None:
        return cast(UserProtocol[str] | None, self._users.get_by_id(user_id=id))

    async def get_by_email(self, email: str) -> UserProtocol[str] | None:
        return cast(UserProtocol[str] | None, self._users.get_by_email(email=email))

    async def get_by_oauth_account(
        self,
        oauth: str,
        account_id: str,
    ) -> UserProtocol[str] | None:
        return None

    async def create(self, create_dict: dict[str, object]) -> UserProtocol[str]:
        user = UserModel(
            email=str(create_dict["email"]),
            hashed_password=str(create_dict["hashed_password"]),
            is_active=bool(create_dict.get("is_active", True)),
            is_superuser=bool(create_dict.get("is_superuser", False)),
            is_verified=bool(create_dict.get("is_verified", False)),
        )
        self._session.add(user)
        self._session.flush()
        return cast(UserProtocol[str], user)

    async def update(
        self,
        user: UserProtocol[str],
        update_dict: dict[str, object],
    ) -> UserProtocol[str]:
        for key, value in update_dict.items():
            setattr(user, key, value)
        self._session.add(user)
        self._session.flush()
        return user

    async def delete(self, user: UserProtocol[str]) -> None:
        self._session.delete(user)

    async def add_oauth_account(
        self,
        user: UserProtocol[str],
        create_dict: dict[str, object],
    ) -> UserProtocol[str]:
        raise NotImplementedError("OAuth accounts are not supported in this project.")

    async def update_oauth_account(
        self,
        user: UserProtocol[str],
        oauth_account: object,
        update_dict: dict[str, object],
    ) -> UserProtocol[str]:
        raise NotImplementedError("OAuth accounts are not supported in this project.")


class AiChatUserManager(BaseUserManager[UserProtocol[str], str]):
    """FastAPI Users manager customized for the local-user showcase scope."""

    reset_password_token_secret = "unused-reset-secret"
    verification_token_secret = "unused-verification-secret"

    def parse_id(self, value: object) -> str:
        """Validate UUID-shaped identifiers while keeping string IDs in the app."""

        if isinstance(value, str):
            UUID(value)
            return value
        if isinstance(value, UUID):
            return str(value)
        raise exceptions.InvalidID()

    async def validate_password(self, password: str, user: object) -> None:
        """Reject blank passwords beyond the request-model validation layer."""

        if not password.strip():
            raise exceptions.InvalidPasswordException(
                reason="Password must not be blank."
            )


@dataclass(slots=True)
class TokenPair:
    """Issued access and refresh token pair."""

    access_token: str
    refresh_token: str


class UserService:
    """Authentication-focused user workflows backed by FastAPI Users."""

    def __init__(self, *, session: Session, jwt_settings: JwtSettings) -> None:
        self._user_db = SyncUserDatabase(session)
        self._user_manager = AiChatUserManager(self._user_db)
        self._refresh_tokens = RefreshTokenRepository(session)
        self._jwt_settings = jwt_settings

    async def register_user(self, *, email: str, password: str) -> UserProtocol[str]:
        """Register a new local user."""

        try:
            return await self._user_manager.create(
                UserCreate(email=email, password=password),
                safe=True,
            )
        except exceptions.UserAlreadyExists as exc:
            raise AuthError(
                message="A user with that email already exists.",
                status_code=status.HTTP_409_CONFLICT,
                code="auth_user_exists",
            ) from exc

    async def authenticate(self, *, email: str, password: str) -> TokenPair:
        """Authenticate a user and issue access and refresh tokens."""

        credentials = OAuth2PasswordRequestForm(
            username=email,
            password=password,
            scope="",
        )
        user = await self._user_manager.authenticate(credentials)
        if user is None:
            raise AuthError(
                message="Invalid email or password.",
                status_code=status.HTTP_401_UNAUTHORIZED,
                code="auth_credentials_invalid",
            )
        return await self._issue_token_pair(user_id=user.id)

    async def read_access_token(self, *, token: str) -> TokenPayload:
        """Validate an access token and return the current user identity."""

        user = await self._get_access_token_strategy().read_token(token, self._user_manager)
        if user is None:
            raise AuthError(
                message="Invalid or expired access token.",
                status_code=status.HTTP_401_UNAUTHORIZED,
                code="auth_token_invalid",
            )
        return TokenPayload(sub=user.id, email=user.email)

    async def refresh_access_token(self, *, refresh_token: str) -> TokenPair:
        """Exchange a valid refresh token for a new token pair."""

        stored_token = self._get_refresh_token(refresh_token)
        now = datetime.now(UTC)
        expires_at = self._normalize_timestamp(stored_token.expires_at)
        if expires_at <= now:
            self._raise_invalid_refresh_token()

        user = await self._user_manager.get(stored_token.user_id)
        if user is None:  # pragma: no cover
            self._raise_invalid_refresh_token()

        self._refresh_tokens.revoke(refresh_token=stored_token, revoked_at=now)
        return await self._issue_token_pair(user_id=user.id)

    async def revoke_refresh_token(self, *, refresh_token: str) -> None:
        """Revoke a refresh token if it exists and is still active."""

        stored_token = self._get_refresh_token(refresh_token)
        self._refresh_tokens.revoke(refresh_token=stored_token, revoked_at=datetime.now(UTC))

    async def _issue_token_pair(self, *, user_id: str) -> TokenPair:
        """Issue a new access token and a persisted opaque refresh token."""

        user = await self._user_manager.get(user_id)
        access_token = await self._get_access_token_strategy().write_token(user)
        refresh_token = secrets.token_urlsafe(48)
        expires_at = datetime.now(UTC) + timedelta(
            days=self._jwt_settings.refresh_token_expire_days
        )
        self._refresh_tokens.create(
            user_id=user_id,
            token_hash=self._hash_refresh_token(refresh_token),
            expires_at=expires_at,
        )
        return TokenPair(access_token=access_token, refresh_token=refresh_token)

    @staticmethod
    def _hash_refresh_token(refresh_token: str) -> str:
        """Hash a refresh token before storage or lookup."""

        return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()

    def _get_refresh_token(self, refresh_token: str) -> RefreshTokenModel:
        """Load an active refresh token record or raise the shared auth error."""

        token_hash = self._hash_refresh_token(refresh_token)
        stored_token = self._refresh_tokens.get_by_hash(token_hash=token_hash)
        if stored_token is None or stored_token.revoked_at is not None:
            self._raise_invalid_refresh_token()
        assert stored_token is not None
        return stored_token

    @staticmethod
    def _raise_invalid_refresh_token() -> None:
        """Raise the shared invalid-refresh-token error."""

        raise AuthError(
            message="Invalid or expired refresh token.",
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="auth_refresh_token_invalid",
        )

    @staticmethod
    def _normalize_timestamp(timestamp: datetime) -> datetime:
        """Treat naive database timestamps as UTC for consistent comparisons."""

        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)

    def _get_access_token_strategy(self) -> JWTStrategy[UserProtocol[str], str]:
        """Build the FastAPI Users JWT strategy for access tokens."""

        if self._jwt_settings.secret is None:
            raise AuthError(
                message="JWT secret is not configured.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="auth_secret_missing",
            )

        return JWTStrategy(
            secret=self._jwt_settings.secret.get_secret_value(),
            lifetime_seconds=self._jwt_settings.access_token_expire_minutes * 60,
            algorithm=self._jwt_settings.algorithm,
        )
