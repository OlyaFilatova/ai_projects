"""Migration tests for the Alembic workflow."""

from pathlib import Path

from alembic import command
from sqlalchemy import create_engine, inspect

from ai_chat.persistence.migrations import create_alembic_config


def test_alembic_upgrade_creates_initial_schema(tmp_path: Path) -> None:
    """Applying migrations should create the expected initial tables."""

    database_path = tmp_path / "migration_test.db"
    config = create_alembic_config(
        database_url=f"sqlite+pysqlite:///{database_path}",
        project_root=Path(__file__).resolve().parents[1],
    )

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite+pysqlite:///{database_path}")
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) >= {"users", "conversations", "messages"}
