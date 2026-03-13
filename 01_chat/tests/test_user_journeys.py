"""Integration tests that exercise current user journeys end to end."""

from sqlalchemy import select

from ai_chat.persistence.models import ConversationModel, MessageModel
from tests.support import create_test_client, get_test_container, register_and_login


def test_user_can_register_authenticate_and_send_multiple_chat_messages() -> None:
    """A supported user should complete the current happy path end to end."""

    client = create_test_client()
    access_token = register_and_login(client)
    headers = {"Authorization": f"Bearer {access_token}"}

    me_response = client.get("/auth/me", headers=headers)
    first_chat_response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers=headers,
    )
    conversation_id = first_chat_response.json()["conversation_id"]
    second_chat_response = client.post(
        "/chat/messages",
        json={"message": "Tell me more", "conversation_id": conversation_id},
        headers=headers,
    )

    assert me_response.status_code == 200
    assert me_response.json()["email"] == "alice@example.com"
    assert first_chat_response.status_code == 200
    assert second_chat_response.status_code == 200
    assert second_chat_response.json()["conversation_id"] == conversation_id


def test_chat_requests_persist_expected_conversation_history() -> None:
    """Chat requests should leave a complete stored history behind them."""

    client = create_test_client()
    access_token = register_and_login(client)
    headers = {"Authorization": f"Bearer {access_token}"}

    first_chat_response = client.post(
        "/chat/messages",
        json={"message": "Hello there"},
        headers=headers,
    )
    conversation_id = first_chat_response.json()["conversation_id"]
    client.post(
        "/chat/messages",
        json={"message": "Continue", "conversation_id": conversation_id},
        headers=headers,
    )

    session = get_test_container(client).session_factory()
    try:
        conversation = session.scalar(
            select(ConversationModel).where(ConversationModel.id == conversation_id)
        )
        messages = list(
            session.scalars(
                select(MessageModel)
                .where(MessageModel.conversation_id == conversation_id)
                .order_by(MessageModel.created_at)
            )
        )
    finally:
        session.close()

    assert conversation is not None
    assert conversation.title == "Hello there"
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message.content for message in messages] == [
        "Hello there",
        "Mock assistant reply: Hello there",
        "Continue",
        "Mock assistant reply: Continue",
    ]


def test_invalid_token_and_missing_conversation_fail_cleanly() -> None:
    """Important failure cases should remain predictable end to end."""

    client = create_test_client()
    access_token = register_and_login(client)

    invalid_token_response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )
    missing_conversation_response = client.post(
        "/chat/messages",
        json={"message": "Hello there", "conversation_id": "missing-id"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert invalid_token_response.status_code == 401
    assert invalid_token_response.json()["code"] == "auth_token_invalid"
    assert missing_conversation_response.status_code == 404
    assert missing_conversation_response.json()["code"] == "chat_conversation_missing"
