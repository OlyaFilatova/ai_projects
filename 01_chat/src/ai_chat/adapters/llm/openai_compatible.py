"""OpenAI-compatible provider adapter."""

from collections.abc import Iterator

import httpx
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ai_chat.adapters.llm.base import ChatTurn


class OpenAiCompatibleProvider:
    """Provider adapter for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=timeout_seconds,
            max_retries=0,
            http_client=http_client or httpx.Client(timeout=timeout_seconds),
        )

    def generate_reply(self, *, messages: list[ChatTurn]) -> str:
        """Call the chat-completions API and return the full assistant content."""

        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages),
        )
        return response.choices[0].message.content or ""

    def stream_reply(self, *, messages: list[ChatTurn]) -> Iterator[str]:
        """Stream chat-completion deltas from an OpenAI-compatible API."""

        stream = self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=self._build_messages(messages),
        )
        with stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    @staticmethod
    def _build_messages(messages: list[ChatTurn]) -> list[ChatCompletionMessageParam]:
        """Convert internal chat turns into OpenAI SDK message payloads."""

        payload: list[ChatCompletionMessageParam] = []
        for turn in messages:
            message: ChatCompletionMessageParam
            if turn.role == "user":
                user_message: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": turn.content,
                }
                message = user_message
            elif turn.role == "assistant":
                assistant_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": turn.content,
                }
                message = assistant_message
            elif turn.role == "system":
                system_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": turn.content,
                }
                message = system_message
            else:
                developer_message: ChatCompletionDeveloperMessageParam = {
                    "role": "developer",
                    "content": turn.content,
                }
                message = developer_message
            payload.append(message)
        return payload
