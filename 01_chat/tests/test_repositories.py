"""Repository tests for the initial persistence slice."""

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ai_chat.repositories.conversations import ConversationRepository, MessageRepository
from ai_chat.repositories.users import UserRepository


def test_schema_contains_expected_tables(engine: Engine) -> None:
    """The current schema should include chat and refresh-token tables."""

    inspector = inspect(engine)

    assert set(inspector.get_table_names()) == {
        "conversations",
        "messages",
        "refresh_tokens",
        "users",
    }


def test_user_repository_can_create_and_fetch_by_email(
    session_factory: sessionmaker[Session],
) -> None:
    """Users should round-trip through the repository cleanly."""

    with session_factory() as session:
        repository = UserRepository(session)
        created_user = repository.create(
            email="alice@example.com",
            hashed_password="hashed-password",
        )
        created_user_id = created_user.id
        session.commit()

    with session_factory() as session:
        repository = UserRepository(session)
        fetched_user = repository.get_by_email(email="alice@example.com")

    assert fetched_user is not None
    assert fetched_user.id == created_user_id
    assert fetched_user.hashed_password == "hashed-password"


def test_conversation_and_message_repositories_persist_relationships(
    session_factory: sessionmaker[Session],
) -> None:
    """Conversations should load with their persisted messages."""

    with session_factory() as session:
        user = UserRepository(session).create(
            email="bob@example.com",
            hashed_password="hashed-password",
        )
        user_id = user.id
        conversation = ConversationRepository(session).create(
            user_id=user.id,
            title="Support chat",
        )
        conversation_id = conversation.id
        MessageRepository(session).create(
            conversation_id=conversation.id,
            role="user",
            content="Hello",
        )
        MessageRepository(session).create(
            conversation_id=conversation.id,
            role="assistant",
            content="Hi there",
        )
        session.commit()

    with session_factory() as session:
        loaded_conversation = ConversationRepository(session).get_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        messages = MessageRepository(session).list_for_conversation(
            conversation_id=conversation_id
        )

    assert loaded_conversation is not None
    assert [message.content for message in loaded_conversation.messages] == ["Hello", "Hi there"]
    assert [message.role for message in messages] == ["user", "assistant"]
