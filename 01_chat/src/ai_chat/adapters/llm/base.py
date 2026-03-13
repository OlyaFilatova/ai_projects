"""Base protocol for LLM provider adapters."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(slots=True)
class ChatTurn:
    """Represents a message sent to an LLM provider."""

    role: Literal["developer", "system", "user", "assistant"]
    content: str


class LlmProvider(Protocol):
    """Protocol for synchronous assistant response generation."""

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """Generate a full assistant reply for the provided message history."""

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Generate assistant reply chunks for streaming delivery."""
