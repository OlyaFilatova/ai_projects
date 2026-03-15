"""Shared path-coercion helpers for interface and service flows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def coerce_paths(paths: Sequence[str | Path]) -> list[Path]:
    """Convert a sequence of path-like values into `Path` objects."""
    return [Path(path) for path in paths]


def resolve_data_dir(data_dir: str | Path | None, default: Path) -> Path:
    """Resolve an explicit data directory or fall back to the configured default."""
    if data_dir is None:
        return default
    return Path(data_dir)


def is_path_within_allowed_root(path: Path, allowed_root: Path | None) -> bool:
    """Return whether a path stays within the configured allowed root."""
    if allowed_root is None:
        return True
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_root = allowed_root.expanduser().resolve(strict=False)
    return resolved_path.is_relative_to(resolved_root)
