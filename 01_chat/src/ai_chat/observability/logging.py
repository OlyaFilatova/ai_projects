"""Structured logging configuration for the application."""

import json
import logging
import logging.config
from datetime import UTC, datetime

from ai_chat.config import Settings


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record with stable top-level fields."""

        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("path", "method", "status_code", "duration_ms", "event"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload)


def configure_logging(settings: Settings) -> None:
    """Configure application logging for the current process."""

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": JsonFormatter}},
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                }
            },
            "root": {"handlers": ["default"], "level": settings.log_level},
        }
    )
