"""Shared application error types."""

from dataclasses import dataclass


@dataclass(slots=True)
class AppError(Exception):
    """Application error with a stable code and HTTP status."""

    message: str
    status_code: int
    code: str

    def __str__(self) -> str:
        """Return the human-readable error message."""

        return self.message
