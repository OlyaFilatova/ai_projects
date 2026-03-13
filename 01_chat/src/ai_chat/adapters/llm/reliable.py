"""Reliability wrapper for LLM providers."""

from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import status
from openai import APITimeoutError, OpenAIError

from ai_chat.adapters.llm.base import ChatTurn, LlmProvider
from ai_chat.adapters.llm.mock import ProviderError


@dataclass(slots=True)
class ReliabilityPolicy:
    """Retry and timeout-related settings for provider calls."""

    timeout_seconds: float
    max_retries: int


class ReliableLlmProvider:
    """Wrap a provider with conservative retry and timeout error mapping."""

    def __init__(self, provider: LlmProvider, policy: ReliabilityPolicy) -> None:
        self._provider = provider
        self._policy = policy

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """Generate a reply with conservative retries."""

        last_error: ProviderError | None = None
        for _ in range(self._policy.max_retries + 1):
            try:
                return self._provider.generate_reply(messages=messages)
            except (TimeoutError, APITimeoutError, OpenAIError, ProviderError) as exc:
                last_error = self._map_provider_exception(exc)
            if last_error.code == "provider_unavailable":
                continue
            break

        assert last_error is not None
        raise last_error

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Stream a reply with retries only before the first chunk is emitted."""

        last_error: ProviderError | None = None
        for _ in range(self._policy.max_retries + 1):
            emitted_chunk = False
            try:
                for chunk in self._provider.stream_reply(messages=messages):
                    emitted_chunk = True
                    yield chunk
                return
            except (TimeoutError, APITimeoutError, OpenAIError, ProviderError) as exc:
                last_error = self._map_provider_exception(exc)

            if emitted_chunk:
                assert last_error is not None
                raise last_error

            if last_error.code == "provider_unavailable":
                continue
            break

        assert last_error is not None
        raise last_error

    def _map_provider_exception(self, exc: Exception) -> ProviderError:
        """Convert provider-specific failures into application-facing errors."""

        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, (TimeoutError, APITimeoutError)):
            return ProviderError(
                message=(
                    "LLM provider timed out after "
                    f"{self._policy.timeout_seconds:.1f} seconds."
                ),
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                code="provider_timeout",
            )
        if isinstance(exc, OpenAIError):
            provider_status = getattr(exc, "status_code", "unknown")
            return ProviderError(
                message=(
                    "OpenAI-compatible provider request failed "
                    f"with status {provider_status}."
                ),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="provider_request_failed",
            )
        raise TypeError(f"Unsupported provider exception type: {type(exc)!r}")
