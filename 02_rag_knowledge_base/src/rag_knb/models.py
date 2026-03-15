"""Domain models used across the application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A source document ingested into the knowledge base."""

    document_id: str
    source_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document into a JSON-compatible shape."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StructuredRecord:
    """One small structured record that can be indexed like a document."""

    record_id: str
    source_path: str
    fields: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_content(self) -> str:
        """Render the record into deterministic field-aware text."""
        return "\n".join(f"{field_name}: {field_value}" for field_name, field_value in self.fields.items())

    def to_document(self, document_id: str) -> Document:
        """Convert the structured record into a standard document payload."""
        return Document(
            document_id=document_id,
            source_path=self.source_path,
            content=self.render_content(),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable chunk derived from a source document."""

    chunk_id: str
    document_id: str
    content: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk into a JSON-compatible shape."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A retrieved chunk and its retrieval metadata."""

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retrieval result into a JSON-compatible shape."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """An answer grounded in retrieval results."""

    answer_text: str
    matches: list[RetrievalResult]
    matched: bool
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the answer into a JSON-compatible shape."""
        return {
            "answer_text": self.answer_text,
            "matches": [match.to_dict() for match in self.matches],
            "matched": self.matched,
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One prior conversation turn supplied by an embedding application."""

    question: str
    answer_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the conversation turn into a JSON-compatible shape."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Summary of one ingestion run."""

    documents: list[Document]
    chunks: list[Chunk]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ingest summary into a JSON-compatible shape."""
        return {
            "documents": [document.to_dict() for document in self.documents],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


@dataclass(frozen=True, slots=True)
class RefreshResult:
    """Summary of one incremental refresh run."""

    documents: list[Document]
    chunks: list[Chunk]
    added_source_paths: list[str]
    updated_source_paths: list[str]
    removed_source_paths: list[str]
    unchanged_source_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the refresh summary into a JSON-compatible shape."""
        return {
            "documents": [document.to_dict() for document in self.documents],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "added_source_paths": self.added_source_paths,
            "updated_source_paths": self.updated_source_paths,
            "removed_source_paths": self.removed_source_paths,
            "unchanged_source_paths": self.unchanged_source_paths,
        }
