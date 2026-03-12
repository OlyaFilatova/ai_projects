"""Chat routes."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from sse_starlette import EventSourceResponse

from ai_chat.services.auth import TokenPayload
from ai_chat.services.chat import ChatService
from ai_chat.transport.http.dependencies import (
    get_chat_service,
    get_current_user,
    limit_chat_requests,
    limit_chat_stream_requests,
)
from ai_chat.transport.http.streaming import build_chat_sse_events, create_sse_response

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Reject whitespace-only message content."""

        if not value.strip():
            raise ValueError("Message content must not be blank.")
        return value


class ChatResponse(BaseModel):
    """Outgoing synchronous chat response payload."""

    conversation_id: str
    user_message: str
    assistant_message: str


class ConversationSummaryResponse(BaseModel):
    """Conversation list item."""

    id: str
    title: str
    message_count: int
    created_at: str


class ConversationMessageResponse(BaseModel):
    """Message in a conversation history response."""

    id: str
    role: str
    content: str
    created_at: str


class ConversationHistoryResponse(BaseModel):
    """Full conversation history response."""

    id: str
    title: str
    messages: list[ConversationMessageResponse]


class UsageSummaryResponse(BaseModel):
    """Basic usage counters for the current user."""

    conversation_count: int
    message_count: int
    user_message_count: int
    assistant_message_count: int


@router.post("/messages", response_model=ChatResponse)
def create_chat_message(
    payload: ChatRequest,
    _: Annotated[None, Depends(limit_chat_requests)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Create a user message and return the assistant response."""

    result = service.reply(
        user_id=current_user.sub,
        message=payload.message,
        conversation_id=payload.conversation_id,
    )

    return ChatResponse(
        conversation_id=result.conversation_id,
        user_message=result.user_message,
        assistant_message=result.assistant_message,
    )


@router.get("/conversations", response_model=list[ConversationSummaryResponse])
def list_conversations(
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> list[ConversationSummaryResponse]:
    """List conversations for the current user."""

    conversations = service.list_conversations(user_id=current_user.sub)
    return [
        ConversationSummaryResponse(
            id=conversation.id,
            title=conversation.title,
            message_count=conversation.message_count,
            created_at=conversation.created_at,
        )
        for conversation in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
def get_conversation(
    conversation_id: str,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ConversationHistoryResponse:
    """Return the full history for a user-owned conversation."""

    conversation = service.get_conversation_history(
        user_id=current_user.sub,
        conversation_id=conversation_id,
    )
    return ConversationHistoryResponse(
        id=conversation.id,
        title=conversation.title,
        messages=[
            ConversationMessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in conversation.messages
        ],
    )


@router.get("/usage", response_model=UsageSummaryResponse)
def get_usage_summary(
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> UsageSummaryResponse:
    """Return basic usage counters for the current user."""

    return UsageSummaryResponse(
        **service.get_usage_summary(user_id=current_user.sub)
    )


@router.post("/messages/stream")
def stream_chat_message(
    payload: ChatRequest,
    _: Annotated[None, Depends(limit_chat_stream_requests)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> EventSourceResponse:
    """Stream assistant response chunks via Server-Sent Events."""

    result = service.stream_reply(
        user_id=current_user.sub,
        message=payload.message,
        conversation_id=payload.conversation_id,
    )
    return create_sse_response(
        build_chat_sse_events(
            conversation_id=result.conversation_id,
            user_message=result.user_message,
            chunk_iterator=result.chunk_iterator,
        )
    )
