"""Helpers for request rate limiting."""

from dataclasses import dataclass

from fastapi import Request, status
from slowapi.util import get_remote_address

from ai_chat.errors import AppError
from ai_chat.limits import FixedWindowLimitEngine


@dataclass(slots=True)
class RateLimitError(AppError):
    """Raised when a client exceeds a configured request limit."""


def enforce_request_rate_limit(
    *,
    request: Request,
    limiter: FixedWindowLimitEngine,
    route_key: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Apply a fixed-window request limit to the current client address."""

    bucket_key = f"{route_key}:{get_remote_address(request)}"
    if limiter.hit(
        bucket_key=bucket_key,
        limit=limit,
        window_seconds=window_seconds,
    ):
        return

    raise RateLimitError(
        message="Rate limit exceeded. Please retry later.",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        code="rate_limit_exceeded",
    )
