"""Dependency wiring placeholders for future infrastructure integrations."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from ai_chat.config import Settings
from ai_chat.limits import FixedWindowLimitEngine
from ai_chat.persistence.migrations import run_migrations_to_head
from ai_chat.persistence.session import create_engine_from_settings, create_session_factory


@dataclass(slots=True)
class ServiceContainer:
    """Holds application-wide dependencies and future service adapters."""

    settings: Settings
    session_factory: sessionmaker[Session]
    rate_limiter: FixedWindowLimitEngine
    quota_limiter: FixedWindowLimitEngine


def build_container(settings: Settings) -> ServiceContainer:
    """Create the application dependency container."""

    run_migrations_to_head(
        database_url=settings.database.url,
        project_root=Path(__file__).resolve().parents[2],
    )
    engine = create_engine_from_settings(settings)
    return ServiceContainer(
        settings=settings,
        session_factory=create_session_factory(engine),
        rate_limiter=FixedWindowLimitEngine(
            storage_uri=settings.rate_limit.storage_uri,
        ),
        quota_limiter=FixedWindowLimitEngine(
            storage_uri=settings.usage_quota.storage_uri,
        ),
    )
