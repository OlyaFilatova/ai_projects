"""Application errors with explicit user-facing messages."""


class RagKnbError(Exception):
    """Base class for expected application errors."""


class UnsupportedFileTypeError(RagKnbError):
    """Raised when a document file type is not supported."""


class DocumentLoadError(RagKnbError):
    """Raised when a document cannot be loaded."""


class ValidationError(RagKnbError):
    """Raised when user-provided input is invalid."""


class PersistedStateError(RagKnbError):
    """Raised when persisted knowledge-base state is missing or invalid."""


class DependencyUnavailableError(RagKnbError):
    """Raised when an optional dependency is selected but unavailable."""
