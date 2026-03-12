"""Helpers for working with Alembic migrations."""

from pathlib import Path

from alembic import command
from alembic.config import Config


def resolve_project_root(project_root: Path | None = None) -> Path:
    """Find the project root that contains Alembic config and migrations."""

    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root)

    candidates.append(Path.cwd())
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        if (candidate / "alembic.ini").exists() and (candidate / "migrations").exists():
            return candidate

    raise RuntimeError(
        "Could not locate project root with alembic.ini and migrations directory."
    )


def create_alembic_config(*, database_url: str, project_root: Path) -> Config:
    """Create an Alembic config pointing at the given database URL."""

    resolved_project_root = resolve_project_root(project_root)
    config = Config(str(resolved_project_root / "alembic.ini"))
    config.set_main_option("script_location", str(resolved_project_root / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def run_migrations_to_head(*, database_url: str, project_root: Path) -> None:
    """Apply all Alembic migrations to the configured database."""

    config = create_alembic_config(
        database_url=database_url,
        project_root=project_root,
    )
    command.upgrade(config, "head")
