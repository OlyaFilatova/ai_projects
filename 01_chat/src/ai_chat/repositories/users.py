"""Repositories for working with persisted users."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_chat.persistence.models import UserModel


class UserRepository:
    """Persist and load application users."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, email: str, hashed_password: str) -> UserModel:
        """Create and persist a new user."""

        user = UserModel(email=email, hashed_password=hashed_password)
        self._session.add(user)
        self._session.flush()
        return user

    def get_by_email(self, *, email: str) -> UserModel | None:
        """Find a user by email address."""

        statement = select(UserModel).where(UserModel.email == email)
        return self._session.scalar(statement)

    def get_by_id(self, *, user_id: str) -> UserModel | None:
        """Find a user by identifier."""

        statement = select(UserModel).where(UserModel.id == user_id)
        return self._session.scalar(statement)
