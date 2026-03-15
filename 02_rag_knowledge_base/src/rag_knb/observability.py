"""Lightweight structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger for the module."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log a structured event as deterministic JSON."""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, sort_keys=True))
