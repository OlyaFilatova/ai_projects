"""Persistence health helpers."""

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


def database_is_ready(session_factory: sessionmaker[Session]) -> bool:
    """Return whether the configured database can be queried."""

    session = session_factory()
    try:
        session.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False
    finally:
        session.close()
