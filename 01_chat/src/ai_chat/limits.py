"""Shared fixed-window limit helpers built on `limits`."""

from limits import RateLimitItem, parse
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter


class FixedWindowLimitEngine:
    """Small shared wrapper over the `limits` fixed-window strategy."""

    def __init__(self, *, storage_uri: str) -> None:
        self._limiter = FixedWindowRateLimiter(storage_from_string(storage_uri))
        self._rate_items: dict[str, RateLimitItem] = {}

    def hit(self, *, bucket_key: str, limit: int, window_seconds: int) -> bool:
        """Consume one hit from the configured fixed-window bucket."""

        rate_definition = f"{limit}/{window_seconds} second"
        rate_item = self._rate_items.setdefault(rate_definition, parse(rate_definition))
        return self._limiter.hit(rate_item, bucket_key)
