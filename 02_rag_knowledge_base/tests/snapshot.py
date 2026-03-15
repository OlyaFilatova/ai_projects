"""Minimal snapshot helpers for stable test outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def assert_text_snapshot(name: str, content: str) -> None:
    """Assert a text snapshot matches the stored value."""
    snapshot_path = SNAPSHOT_DIR / f"{name}.txt"
    expected = snapshot_path.read_text(encoding="utf-8")
    assert content == expected


def assert_json_snapshot(name: str, payload: Any) -> None:
    """Assert a JSON snapshot matches the stored value."""
    snapshot_path = SNAPSHOT_DIR / f"{name}.json"
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload == expected
