"""Database engine and session helpers."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ai_chat.config import Settings


def create_engine_from_settings(settings: Settings) -> Engine:
    """Create a database engine from application settings."""

    connect_args: dict[str, object] = {}
    engine_options: dict[str, object] = {"future": True}
    if settings.database.url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        engine_options["connect_args"] = connect_args
        if ":memory:" in settings.database.url:
            engine_options["poolclass"] = StaticPool
    return create_engine(settings.database.url, **engine_options)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to the given engine."""

    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
