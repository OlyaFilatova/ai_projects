"""Authentication routes for user registration and token issuance."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from ai_chat.services.auth import AuthError, TokenPayload, UserService
from ai_chat.transport.http.dependencies import (
    get_current_user,
    get_user_service,
    limit_register_requests,
    limit_token_requests,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class EmailPasswordRequest(BaseModel):
    """Incoming payload that carries an email and password."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        """Reject passwords that are only whitespace."""

        if not value.strip():
            raise ValueError("Password must not be blank.")
        return value


class RegistrationRequest(EmailPasswordRequest):
    """Incoming payload for user registration."""


class TokenRequest(EmailPasswordRequest):
    """Incoming payload for token creation."""


class TokenResponse(BaseModel):
    """Outgoing token payload."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RegistrationResponse(BaseModel):
    """Public registration response that avoids account-existence disclosure."""

    detail: str


class RefreshTokenRequest(BaseModel):
    """Incoming payload for refresh and revoke flows."""

    refresh_token: str = Field(min_length=32, max_length=512)


class UserResponse(BaseModel):
    """Outgoing authenticated user payload."""

    id: str
    email: EmailStr


@router.post(
    "/register",
    response_model=RegistrationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def register_user(
    payload: RegistrationRequest,
    _: Annotated[None, Depends(limit_register_requests)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> RegistrationResponse:
    """Register a new local user."""

    try:
        await service.register_user(email=payload.email, password=payload.password)
    except AuthError as exc:
        if exc.code != "auth_user_exists":
            raise
    return RegistrationResponse(
        detail="If registration is available for this email, you can now sign in."
    )


@router.post("/token", response_model=TokenResponse)
async def create_token(
    payload: TokenRequest,
    _: Annotated[None, Depends(limit_token_requests)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> TokenResponse:
    """Authenticate a user and issue a JWT access token."""

    token_pair = await service.authenticate(email=payload.email, password=payload.password)
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    payload: RefreshTokenRequest,
    service: Annotated[UserService, Depends(get_user_service)],
) -> TokenResponse:
    """Exchange a valid refresh token for a new token pair."""

    token_pair = await service.refresh_access_token(refresh_token=payload.refresh_token)
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
    )


@router.post("/revoke", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_refresh_token(
    payload: RefreshTokenRequest,
    service: Annotated[UserService, Depends(get_user_service)],
) -> None:
    """Revoke a refresh token."""

    await service.revoke_refresh_token(refresh_token=payload.refresh_token)


@router.get("/me", response_model=UserResponse)
def get_me(
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
) -> UserResponse:
    """Return the authenticated user identity extracted from the access token."""

    return UserResponse(id=current_user.sub, email=current_user.email)
