"""Synchronous chat API tests."""

from fastapi.testclient import TestClient
from sqlalchemy import select

from ai_chat.app import create_app
from ai_chat.persistence.models import MessageModel
from tests.support import (
    create_test_client as shared_create_test_client,
)
from tests.support import (
    create_test_settings,
    get_test_container,
    register_and_login,
)


def create_authenticated_client() -> tuple[TestClient, str]:
    """Create a client and return it with a bearer token."""

    client = shared_create_test_client()
    access_token = register_and_login(client)
    return client, access_token


def test_authenticated_chat_flow_persists_and_returns_mock_reply() -> None:
    """The chat endpoint should return a synchronous assistant message."""

    client, access_token = create_authenticated_client()
    response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"]
    assert body["user_message"] == "Hello there"
    assert body["assistant_message"] == "Mock assistant reply: Hello there"


def test_existing_conversation_can_be_reused() -> None:
    """A later request should be able to append to the same conversation."""

    client, access_token = create_authenticated_client()
    first_response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    conversation_id = first_response.json()["conversation_id"]

    second_response = client.post(
        "/chat/messages",
        json={"message": "Continue", "conversation_id": conversation_id},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert second_response.status_code == 200
    assert second_response.json()["conversation_id"] == conversation_id
    assert second_response.json()["assistant_message"] == "Mock assistant reply: Continue"


def test_chat_requires_authentication() -> None:
    """The chat endpoint should reject unauthenticated access."""

    client, _ = create_authenticated_client()

    response = client.post("/chat/messages", json={"message": "Hello there"})

    assert response.status_code == 401
    assert response.json()["code"] == "auth_credentials_missing"


def test_unknown_conversation_is_rejected() -> None:
    """The chat endpoint should reject conversation IDs not owned by the user."""

    client, access_token = create_authenticated_client()
    response = client.post(
        "/chat/messages",
        json={"message": "Hello there", "conversation_id": "missing-id"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found for the current user."
    assert response.json()["code"] == "chat_conversation_missing"


def test_blank_chat_message_is_rejected() -> None:
    """Whitespace-only messages should fail request validation."""

    client, access_token = create_authenticated_client()
    response = client.post(
        "/chat/messages",
        json={"message": "   "},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 422


def test_provider_failure_is_mapped_to_service_unavailable() -> None:
    """Provider failures should return a stable API error payload."""

    settings = create_test_settings(llm={"provider": "failing-mock"})
    client = TestClient(create_app(settings))
    register_payload = {
        "email": "alice@example.com",
        "password": "correct-horse-battery",
    }
    client.post("/auth/register", json=register_payload)
    token_response = client.post("/auth/token", json=register_payload)
    access_token = token_response.json()["access_token"]

    response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "LLM provider is currently unavailable.",
        "code": "provider_unavailable",
    }


def test_streaming_chat_endpoint_returns_sse_events() -> None:
    """The streaming endpoint should return metadata, chunks, and completion events."""

    client, access_token = create_authenticated_client()
    with client.stream(
        "POST",
        "/chat/messages/stream",
        json={"message": "Hello there"},
        headers={"Authorization": f"Bearer {access_token}"},
    ) as response:
        body = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: metadata" in body
    assert "event: chunk" in body
    assert "event: done" in body
    assert '{"content": "Mock "}' in body
    assert '{"content": "assistant "}' in body
    assert '{"content": "reply: Hello there"}' in body


def test_partial_stream_failure_does_not_persist_incomplete_assistant_message() -> None:
    """Interrupted streams should not leave a partial assistant message in storage."""

    settings = create_test_settings(llm={"provider": "partial-failing-mock"})
    client = TestClient(create_app(settings))
    access_token = register_and_login(client)

    with client.stream(
        "POST",
        "/chat/messages/stream",
        json={"message": "Hello there"},
        headers={"Authorization": f"Bearer {access_token}"},
    ) as response:
        body = "".join(response.iter_text())

    session = get_test_container(client).session_factory()
    try:
        messages = list(session.scalars(select(MessageModel).order_by(MessageModel.created_at)))
    finally:
        session.close()

    assert response.status_code == 200
    assert "event: chunk" in body
    assert "event: error" in body
    assert "provider_stream_interrupted" in body
    assert [message.role for message in messages] == ["user"]
    assert [message.content for message in messages] == ["Hello there"]


def test_transient_provider_failure_is_retried_successfully() -> None:
    """A transient provider failure should succeed within the retry budget."""

    settings = create_test_settings(llm={"provider": "flaky-mock", "max_retries": 1})
    client = TestClient(create_app(settings))
    access_token = register_and_login(client)
    response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    assert response.json()["assistant_message"] == "Mock assistant reply: Hello there"


def test_provider_timeout_is_mapped_cleanly() -> None:
    """Timeouts should surface as an understandable provider error."""

    settings = create_test_settings(
        llm={"provider": "timeout-mock", "timeout_seconds": 2.5, "max_retries": 1}
    )
    client = TestClient(create_app(settings))
    access_token = register_and_login(client)
    response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 504
    assert response.json() == {
        "detail": "LLM provider timed out after 2.5 seconds.",
        "code": "provider_timeout",
    }


def test_conversation_listing_history_and_usage_summary() -> None:
    """Users should be able to inspect conversations and usage summary."""

    client, access_token = create_authenticated_client()
    headers = {"Authorization": f"Bearer {access_token}"}
    create_response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers=headers,
    )
    conversation_id = create_response.json()["conversation_id"]

    list_response = client.get("/chat/conversations", headers=headers)
    history_response = client.get(f"/chat/conversations/{conversation_id}", headers=headers)
    usage_response = client.get("/chat/usage", headers=headers)

    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == conversation_id
    assert list_response.json()[0]["message_count"] == 2
    assert history_response.status_code == 200
    assert [message["role"] for message in history_response.json()["messages"]] == [
        "user",
        "assistant",
    ]
    assert usage_response.status_code == 200
    assert usage_response.json() == {
        "conversation_count": 1,
        "message_count": 2,
        "user_message_count": 1,
        "assistant_message_count": 1,
    }


def test_chat_rate_limit_is_enforced() -> None:
    """Chat endpoints should return 429 once the configured limit is exceeded."""

    client = TestClient(
        create_app(create_test_settings(rate_limit={"window_seconds": 60, "chat_requests": 2}))
    )
    access_token = register_and_login(client)
    headers = {"Authorization": f"Bearer {access_token}"}

    response_one = client.post("/chat/messages", json={"message": "One"}, headers=headers)
    response_two = client.post("/chat/messages", json={"message": "Two"}, headers=headers)
    response_three = client.post("/chat/messages", json={"message": "Three"}, headers=headers)

    assert response_one.status_code == 200
    assert response_two.status_code == 200
    assert response_three.status_code == 429
    assert response_three.json()["code"] == "rate_limit_exceeded"


def test_chat_stream_rate_limit_is_enforced() -> None:
    """Streaming chat should return 429 once the configured limit is exceeded."""

    client = TestClient(
        create_app(
            create_test_settings(
                rate_limit={"window_seconds": 60, "chat_stream_requests": 1}
            )
        )
    )
    access_token = register_and_login(client)
    headers = {"Authorization": f"Bearer {access_token}"}

    with client.stream(
        "POST",
        "/chat/messages/stream",
        json={"message": "One"},
        headers=headers,
    ) as first_response:
        assert first_response.status_code == 200
        _ = "".join(first_response.iter_text())

    second_response = client.post(
        "/chat/messages/stream",
        json={"message": "Two"},
        headers=headers,
    )

    assert second_response.status_code == 429
    assert second_response.json()["code"] == "rate_limit_exceeded"


def test_chat_usage_quota_is_enforced_per_user() -> None:
    """Authenticated users should receive 429 when their chat quota is exhausted."""

    client = TestClient(
        create_app(
            create_test_settings(
                usage_quota={
                    "window_seconds": 3600,
                    "chat_requests": 2,
                    "chat_stream_requests": 5,
                }
            )
        )
    )
    access_token = register_and_login(client)
    headers = {"Authorization": f"Bearer {access_token}"}

    response_one = client.post("/chat/messages", json={"message": "One"}, headers=headers)
    response_two = client.post("/chat/messages", json={"message": "Two"}, headers=headers)
    response_three = client.post("/chat/messages", json={"message": "Three"}, headers=headers)

    assert response_one.status_code == 200
    assert response_two.status_code == 200
    assert response_three.status_code == 429
    assert response_three.json()["code"] == "quota_exceeded"


def test_chat_stream_usage_quota_is_enforced_per_user() -> None:
    """Streaming requests should respect the configured per-user quota."""

    client = TestClient(
        create_app(
            create_test_settings(
                usage_quota={
                    "window_seconds": 3600,
                    "chat_requests": 5,
                    "chat_stream_requests": 1,
                }
            )
        )
    )
    access_token = register_and_login(client)
    headers = {"Authorization": f"Bearer {access_token}"}

    with client.stream(
        "POST",
        "/chat/messages/stream",
        json={"message": "One"},
        headers=headers,
    ) as first_response:
        assert first_response.status_code == 200
        _ = "".join(first_response.iter_text())

    second_response = client.post(
        "/chat/messages/stream",
        json={"message": "Two"},
        headers=headers,
    )

    assert second_response.status_code == 429
    assert second_response.json()["code"] == "quota_exceeded"
