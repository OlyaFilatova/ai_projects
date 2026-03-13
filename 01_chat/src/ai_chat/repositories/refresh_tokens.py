"""Repositories for persisted refresh tokens."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_chat.persistence.models import RefreshTokenModel


class RefreshTokenRepository:
    """Persist and manage refresh token records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, user_id: str, token_hash: str, expires_at: datetime) -> RefreshTokenModel:
        """Create and persist a refresh token."""

        refresh_token = RefreshTokenModel(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self._session.add(refresh_token)
        self._session.flush()
        return refresh_token

    def get_by_hash(self, *, token_hash: str) -> RefreshTokenModel | None:
        """Return a refresh token by its stored hash."""

        statement = select(RefreshTokenModel).where(RefreshTokenModel.token_hash == token_hash)
        return self._session.scalar(statement)

    def revoke(self, *, refresh_token: RefreshTokenModel, revoked_at: datetime) -> None:
        """Mark a refresh token as revoked."""

        refresh_token.revoked_at = revoked_at
        self._session.add(refresh_token)
