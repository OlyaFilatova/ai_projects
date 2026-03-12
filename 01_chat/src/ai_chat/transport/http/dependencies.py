"""HTTP-specific dependency builders."""

from collections.abc import Generator
from typing import Annotated, cast

from fastapi import Depends, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ai_chat.adapters.llm.factory import build_provider
from ai_chat.dependencies import ServiceContainer
from ai_chat.services.auth import AuthError, TokenPayload, UserService
from ai_chat.services.chat import ChatService
from ai_chat.transport.http.rate_limit import enforce_request_rate_limit

bearer_scheme = HTTPBearer(auto_error=False)


def get_container(request: Request) -> ServiceContainer:
    """Return the application service container."""

    return cast(ServiceContainer, request.app.state.container)


def get_db_session(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> Generator[Session, None, None]:
    """Provide a database session for request-scoped work."""

    session = container.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_user_service(
    session: Annotated[Session, Depends(get_db_session)],
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> UserService:
    """Build the user service from request dependencies."""

    return UserService(session=session, jwt_settings=container.settings.jwt)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> TokenPayload:
    """Decode and validate the current bearer token."""

    if credentials is None:
        raise AuthError(
            message="Authentication credentials were not provided.",
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="auth_credentials_missing",
        )

    return await user_service.read_access_token(token=credentials.credentials)


def get_chat_service(
    session: Annotated[Session, Depends(get_db_session)],
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> ChatService:
    """Build the chat service from request-scoped dependencies."""

    return ChatService(
        session=session,
        provider=build_provider(container.settings),
        quota_tracker=container.quota_limiter,
        quota_window_seconds=container.settings.usage_quota.window_seconds,
        chat_request_quota=container.settings.usage_quota.chat_requests,
        chat_stream_request_quota=container.settings.usage_quota.chat_stream_requests,
    )


def limit_register_requests(
    request: Request,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> None:
    """Limit registration attempts by client IP."""

    enforce_request_rate_limit(
        request=request,
        limiter=container.rate_limiter,
        route_key="auth_register",
        limit=container.settings.rate_limit.register_requests,
        window_seconds=container.settings.rate_limit.window_seconds,
    )


def limit_token_requests(
    request: Request,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> None:
    """Limit token creation attempts by client IP."""

    enforce_request_rate_limit(
        request=request,
        limiter=container.rate_limiter,
        route_key="auth_token",
        limit=container.settings.rate_limit.token_requests,
        window_seconds=container.settings.rate_limit.window_seconds,
    )


def limit_chat_requests(
    request: Request,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> None:
    """Limit synchronous chat requests by client IP."""

    enforce_request_rate_limit(
        request=request,
        limiter=container.rate_limiter,
        route_key="chat_messages",
        limit=container.settings.rate_limit.chat_requests,
        window_seconds=container.settings.rate_limit.window_seconds,
    )


def limit_chat_stream_requests(
    request: Request,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> None:
    """Limit streaming chat requests by client IP."""

    enforce_request_rate_limit(
        request=request,
        limiter=container.rate_limiter,
        route_key="chat_stream",
        limit=container.settings.rate_limit.chat_stream_requests,
        window_seconds=container.settings.rate_limit.window_seconds,
    )
