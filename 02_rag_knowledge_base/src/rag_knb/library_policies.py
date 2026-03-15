"""Shared validation helpers for library-owned guardrails."""

from __future__ import annotations

from pathlib import Path

from rag_knb.errors import DocumentLoadError, ValidationError


def resolve_positive_limit(
    value: int | None,
    default: int,
    error_message: str,
) -> int:
    """Resolve a positive configured limit or fail with a validation error."""
    resolved_value = value if value is not None else default
    if resolved_value <= 0:
        raise ValidationError(error_message)
    return resolved_value


def validate_max_count(count: int, maximum: int, error_message: str) -> None:
    """Validate that an item count stays within the configured maximum."""
    if count > maximum:
        raise ValidationError(error_message)


def validate_text_length(text: str, maximum: int, error_message: str) -> None:
    """Validate that a text value stays within the configured maximum length."""
    if len(text) > maximum:
        raise ValidationError(error_message)


def validate_positive_request_limit(limit: int, maximum: int, error_message: str) -> None:
    """Validate that a request-scoped limit is positive and within the configured maximum."""
    if limit <= 0:
        raise ValidationError("Retrieval limit must be greater than zero.")
    if limit > maximum:
        raise ValidationError(error_message)


def validate_document_file_size(
    resolved_path: Path,
    file_size: int,
    maximum_bytes: int,
) -> None:
    """Validate that a source document fits within the configured byte limit."""
    if file_size > maximum_bytes:
        raise DocumentLoadError(
            f"Document '{resolved_path}' exceeds the configured maximum size of {maximum_bytes} bytes."
        )
