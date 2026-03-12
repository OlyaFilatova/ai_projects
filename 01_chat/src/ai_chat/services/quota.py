"""Helpers for per-user usage quotas."""

from dataclasses import dataclass

from fastapi import status

from ai_chat.errors import AppError
from ai_chat.limits import FixedWindowLimitEngine


@dataclass(slots=True)
class QuotaExceededError(AppError):
    """Raised when a user exceeds a configured quota."""


def enforce_usage_quota(
    *,
    limiter: FixedWindowLimitEngine,
    user_id: str,
    action: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Apply a fixed-window quota to one user and action type."""

    if limiter.hit(
        bucket_key=f"{action}:{user_id}",
        limit=limit,
        window_seconds=window_seconds,
    ):
        return

    raise QuotaExceededError(
        message="Usage quota exceeded. Please retry later.",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        code="quota_exceeded",
    )
