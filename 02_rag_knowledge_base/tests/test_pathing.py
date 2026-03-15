"""Tests for shared path-coercion helpers."""

from __future__ import annotations

from pathlib import Path

from rag_knb.pathing import coerce_paths, is_path_within_allowed_root, resolve_data_dir


def test_coerce_paths_preserves_order() -> None:
    """Path coercion should keep the incoming order stable."""
    assert coerce_paths(["first.txt", Path("second.txt")]) == [Path("first.txt"), Path("second.txt")]


def test_resolve_data_dir_uses_default_when_missing() -> None:
    """Data-dir resolution should fall back to the provided default."""
    default = Path(".rag-knb")

    assert resolve_data_dir(None, default) == default


def test_resolve_data_dir_prefers_explicit_value() -> None:
    """Data-dir resolution should preserve an explicit override."""
    default = Path(".rag-knb")

    assert resolve_data_dir("custom-dir", default) == Path("custom-dir")


def test_is_path_within_allowed_root_allows_descendants() -> None:
    """Allowed-root checks should accept descendant paths."""
    assert is_path_within_allowed_root(Path("/tmp/root/child.txt"), Path("/tmp/root")) is True


def test_is_path_within_allowed_root_rejects_paths_outside_root() -> None:
    """Allowed-root checks should reject paths outside the configured root."""
    assert is_path_within_allowed_root(Path("/tmp/other/file.txt"), Path("/tmp/root")) is False
