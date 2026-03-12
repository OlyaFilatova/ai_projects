"""Repositories for working with conversations and messages."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ai_chat.persistence.models import ConversationModel, MessageModel


class ConversationRepository:
    """Persist and load conversations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, user_id: str, title: str = "Untitled conversation") -> ConversationModel:
        """Create and persist a conversation for the given user."""

        conversation = ConversationModel(user_id=user_id, title=title)
        self._session.add(conversation)
        self._session.flush()
        return conversation

    def get_for_user(self, *, conversation_id: str, user_id: str) -> ConversationModel | None:
        """Load a single conversation owned by a specific user."""

        statement = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id,
            )
        )
        return self._session.scalar(statement)

    def list_for_user(self, *, user_id: str) -> list[ConversationModel]:
        """List conversations for a user ordered by newest first."""

        statement = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.created_at.desc())
        )
        return list(self._session.scalars(statement))


class MessageRepository:
    """Persist and load conversation messages."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, conversation_id: str, role: str, content: str) -> MessageModel:
        """Create and persist a message in a conversation."""

        message = MessageModel(conversation_id=conversation_id, role=role, content=content)
        self._session.add(message)
        self._session.flush()
        return message

    def list_for_conversation(self, *, conversation_id: str) -> list[MessageModel]:
        """Return all messages for a conversation ordered by creation time."""

        statement = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at)
        )
        return list(self._session.scalars(statement))

    def usage_summary_for_user(self, *, user_id: str) -> dict[str, int]:
        """Return basic per-user usage counts."""

        conversation_count = self._session.scalar(
            select(func.count()).select_from(ConversationModel).where(
                ConversationModel.user_id == user_id
            )
        )
        message_count = self._session.scalar(
            select(func.count())
            .select_from(MessageModel)
            .join(ConversationModel, MessageModel.conversation_id == ConversationModel.id)
            .where(ConversationModel.user_id == user_id)
        )
        user_message_count = self._session.scalar(
            select(func.count())
            .select_from(MessageModel)
            .join(ConversationModel, MessageModel.conversation_id == ConversationModel.id)
            .where(
                ConversationModel.user_id == user_id,
                MessageModel.role == "user",
            )
        )
        assistant_message_count = self._session.scalar(
            select(func.count())
            .select_from(MessageModel)
            .join(ConversationModel, MessageModel.conversation_id == ConversationModel.id)
            .where(
                ConversationModel.user_id == user_id,
                MessageModel.role == "assistant",
            )
        )
        return {
            "conversation_count": conversation_count or 0,
            "message_count": message_count or 0,
            "user_message_count": user_message_count or 0,
            "assistant_message_count": assistant_message_count or 0,
        }
