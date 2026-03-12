"""Helpers for Server-Sent Events responses."""

from __future__ import annotations

import json
from collections.abc import Iterator

from pydantic import BaseModel
from sse_starlette import EventSourceResponse, ServerSentEvent
from sse_starlette.sse import AppStatus

from ai_chat.errors import AppError


class StreamingChatMetadata(BaseModel):
    """Initial metadata emitted before streaming chunks."""

    conversation_id: str
    user_message: str


def build_chat_sse_events(
    *,
    conversation_id: str,
    user_message: str,
    chunk_iterator: Iterator[str],
) -> Iterator[ServerSentEvent]:
    """Build the current SSE event sequence for streamed chat replies."""

    metadata = StreamingChatMetadata(
        conversation_id=conversation_id,
        user_message=user_message,
    )
    yield ServerSentEvent(
        event="metadata",
        data=metadata.model_dump_json(),
    )
    try:
        for chunk in chunk_iterator:
            yield ServerSentEvent(
                event="chunk",
                data=json.dumps({"content": chunk}),
            )
    except AppError as exc:
        yield ServerSentEvent(
            event="error",
            data=json.dumps({"detail": exc.message, "code": exc.code}),
        )
        return
    yield ServerSentEvent(
        event="done",
        data=json.dumps({"status": "complete"}),
    )


def create_sse_response(event_stream: Iterator[ServerSentEvent]) -> EventSourceResponse:
    """Create an SSE response with the current test-client compatibility settings."""

    # The test client spins up separate event loops across streamed requests.
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None
    return EventSourceResponse(event_stream, ping=None, sep="\n")
