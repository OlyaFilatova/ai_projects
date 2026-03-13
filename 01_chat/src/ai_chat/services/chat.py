"""Services for synchronous chat orchestration."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, cast

from fastapi import status
from sqlalchemy.orm import Session

from ai_chat.adapters.llm.base import ChatTurn, LlmProvider
from ai_chat.errors import AppError
from ai_chat.limits import FixedWindowLimitEngine
from ai_chat.persistence.models import ConversationModel
from ai_chat.repositories.conversations import ConversationRepository, MessageRepository
from ai_chat.services.quota import enforce_usage_quota

ChatRole = Literal["developer", "system", "user", "assistant"]


@dataclass(slots=True)
class ChatError(AppError):
    """Domain-level chat error with an HTTP-friendly status code."""


@dataclass(slots=True)
class ChatResult:
    """Returned synchronous chat result."""

    conversation_id: str
    user_message: str
    assistant_message: str


@dataclass(slots=True)
class StreamingChatResult:
    """Returned streaming chat result."""

    conversation_id: str
    user_message: str
    chunk_iterator: Iterator[str]


@dataclass(slots=True)
class ConversationSummary:
    """Summary information for a conversation list item."""

    id: str
    title: str
    message_count: int
    created_at: str


@dataclass(slots=True)
class ConversationMessage:
    """History item for a stored message."""

    id: str
    role: str
    content: str
    created_at: str


@dataclass(slots=True)
class ConversationHistory:
    """Conversation details including full message history."""

    id: str
    title: str
    messages: list[ConversationMessage]


class ChatService:
    """Coordinate user messages, provider replies, and persistence."""

    def __init__(
        self,
        *,
        session: Session,
        provider: LlmProvider,
        quota_tracker: FixedWindowLimitEngine,
        quota_window_seconds: int,
        chat_request_quota: int,
        chat_stream_request_quota: int,
    ) -> None:
        self._session = session
        self._provider = provider
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)
        self._quota_tracker = quota_tracker
        self._quota_window_seconds = quota_window_seconds
        self._chat_request_quota = chat_request_quota
        self._chat_stream_request_quota = chat_stream_request_quota

    def reply(
        self,
        *,
        user_id: str,
        message: str,
        conversation_id: str | None,
    ) -> ChatResult:
        """Persist a user message, generate a reply, and persist the assistant output."""

        clean_message = self._normalize_message(message)
        self._enforce_quota(user_id=user_id, action="chat", limit=self._chat_request_quota)
        conversation = self._load_or_create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            first_message=clean_message,
        )
        history = self._store_user_message_and_build_history(
            conversation_id=conversation.id,
            message=clean_message,
        )
        assistant_reply = self._provider.generate_reply(messages=history)
        self._messages.create(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_reply,
        )

        return ChatResult(
            conversation_id=conversation.id,
            user_message=clean_message,
            assistant_message=assistant_reply,
        )

    def stream_reply(
        self,
        *,
        user_id: str,
        message: str,
        conversation_id: str | None,
    ) -> StreamingChatResult:
        """Persist a user message and stream the assistant output."""

        clean_message = self._normalize_message(message)
        self._enforce_quota(
            user_id=user_id,
            action="chat_stream",
            limit=self._chat_stream_request_quota,
        )
        conversation = self._load_or_create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            first_message=clean_message,
        )
        history = self._store_user_message_and_build_history(
            conversation_id=conversation.id,
            message=clean_message,
        )

        def chunk_iterator() -> Iterator[str]:
            collected_chunks: list[str] = []
            for chunk in self._provider.stream_reply(messages=history):
                collected_chunks.append(chunk)
                yield chunk

            self._messages.create(
                conversation_id=conversation.id,
                role="assistant",
                content="".join(collected_chunks),
            )

        return StreamingChatResult(
            conversation_id=conversation.id,
            user_message=clean_message,
            chunk_iterator=chunk_iterator(),
        )

    def _load_or_create_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        first_message: str,
    ) -> ConversationModel:
        """Find an existing conversation or create a new one for the user."""

        if conversation_id is None:
            return self._conversations.create(user_id=user_id, title=first_message[:80])

        conversation = self._conversations.get_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if conversation is None:
            self._raise_missing_conversation()
        assert conversation is not None
        return conversation

    def list_conversations(self, *, user_id: str) -> list[ConversationSummary]:
        """List conversations visible to the current user."""

        conversations = self._conversations.list_for_user(user_id=user_id)
        return [
            ConversationSummary(
                id=conversation.id,
                title=conversation.title,
                message_count=len(conversation.messages),
                created_at=conversation.created_at.isoformat(),
            )
            for conversation in conversations
        ]

    def get_conversation_history(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> ConversationHistory:
        """Return the full message history for one user-owned conversation."""

        conversation = self._conversations.get_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if conversation is None:
            self._raise_missing_conversation()
        assert conversation is not None
        return ConversationHistory(
            id=conversation.id,
            title=conversation.title,
            messages=[
                ConversationMessage(
                    id=message.id,
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at.isoformat(),
                )
                for message in conversation.messages
            ],
        )

    def get_usage_summary(self, *, user_id: str) -> dict[str, int]:
        """Return per-user usage counts."""

        return self._messages.usage_summary_for_user(user_id=user_id)

    def _enforce_quota(self, *, user_id: str, action: str, limit: int) -> None:
        """Apply the configured fixed-window quota for the current user."""

        enforce_usage_quota(
            limiter=self._quota_tracker,
            user_id=user_id,
            action=action,
            limit=limit,
            window_seconds=self._quota_window_seconds,
        )

    def _normalize_message(self, message: str) -> str:
        """Return a validated user message."""

        clean_message = message.strip()
        if not clean_message:
            raise ChatError(
                message="Message content must not be empty.",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code="chat_message_empty",
            )
        return clean_message

    def _store_user_message_and_build_history(
        self,
        *,
        conversation_id: str,
        message: str,
    ) -> list[ChatTurn]:
        """Persist the incoming user turn and rebuild provider history."""

        self._messages.create(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )
        return self._build_history(conversation_id=conversation_id)

    def _build_history(self, *, conversation_id: str) -> list[ChatTurn]:
        """Convert stored messages into provider chat turns."""

        return [
            ChatTurn(
                role=cast(ChatRole, stored_message.role),
                content=stored_message.content,
            )
            for stored_message in self._messages.list_for_conversation(
                conversation_id=conversation_id
            )
        ]

    @staticmethod
    def _raise_missing_conversation() -> None:
        """Raise the shared missing-conversation error."""

        raise ChatError(
            message="Conversation not found for the current user.",
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_conversation_missing",
        )
