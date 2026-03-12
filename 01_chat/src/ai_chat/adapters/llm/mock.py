"""Mock LLM adapter for deterministic local development."""

from collections.abc import Iterator

from fastapi import status

from ai_chat.adapters.llm.base import ChatTurn
from ai_chat.errors import AppError


class ProviderError(AppError):
    """Provider-level error exposed to the service layer."""


class MockLlmProvider:
    """Deterministic mock provider for local development and tests."""

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """Return a stable mock response from the latest user turn."""

        latest_user_message = next(
            turn.content for turn in reversed(messages) if turn.role == "user"
        )
        return f"Mock assistant reply: {latest_user_message}"

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Return deterministic streaming chunks for the latest user turn."""

        reply = self.generate_reply(messages=messages)
        words = reply.split(" ")
        yield words[0] + " "
        yield words[1] + " "
        yield " ".join(words[2:])


class FailingMockLlmProvider:
    """Provider used to exercise failure handling in tests."""

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """Raise a deterministic provider failure."""

        raise ProviderError(
            message="LLM provider is currently unavailable.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="provider_unavailable",
        )

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Raise a deterministic provider failure during streaming."""

        self.generate_reply(messages=messages)
        yield ""


class PartialFailingMockLlmProvider:
    """Provider used to exercise partial-stream failure handling."""

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """This provider is streaming-only in tests."""

        raise ProviderError(
            message="LLM provider is currently unavailable.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="provider_unavailable",
        )

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Yield one chunk and then fail before completion."""

        yield "Mock "
        raise ProviderError(
            message="LLM provider stream interrupted.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="provider_stream_interrupted",
        )


class FlakyMockLlmProvider:
    """Provider that fails once before succeeding to test retries."""

    def __init__(self) -> None:
        self._attempts = 0

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """Fail on the first attempt and succeed on the next one."""

        self._attempts += 1
        if self._attempts == 1:
            raise ProviderError(
                message="LLM provider is currently unavailable.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="provider_unavailable",
            )
        latest_user_message = next(
            turn.content for turn in reversed(messages) if turn.role == "user"
        )
        return f"Mock assistant reply: {latest_user_message}"

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Fail on the first attempt and succeed on the next one."""

        self._attempts += 1
        if self._attempts == 1:
            raise ProviderError(
                message="LLM provider is currently unavailable.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="provider_unavailable",
            )
        yield from MockLlmProvider().stream_reply(messages=messages)


class TimeoutMockLlmProvider:
    """Provider that always times out to test timeout mapping."""

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """Raise a timeout error."""

        raise TimeoutError("simulated timeout")

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Raise a timeout error before any chunk is emitted."""

        raise TimeoutError("simulated timeout")
        yield ""
