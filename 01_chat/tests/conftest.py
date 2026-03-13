"""Shared test fixtures for persistence and API tests."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ai_chat.persistence.base import Base


@pytest.fixture()
def engine() -> Engine:
    """Create an isolated in-memory database engine for a test."""

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to the isolated test engine."""

    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
