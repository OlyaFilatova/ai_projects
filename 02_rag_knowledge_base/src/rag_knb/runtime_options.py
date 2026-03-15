"""Shared runtime-option helpers for interface layers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rag_knb.config import RuntimeConfig

RUNTIME_OPTION_FIELD_NAMES = (
    "data_dir",
    "chunk_size",
    "chunk_overlap",
    "embedding_backend",
    "answer_mode",
    "answer_verbosity",
    "retrieval_strategy",
    "vector_backend",
    "llm_model",
    "llm_base_url",
    "allow_custom_llm_base_url",
    "allowed_root",
    "max_question_length",
    "max_retrieval_limit",
    "max_documents_per_ingest",
    "max_document_bytes",
    "max_chunks_per_ingest",
    "llm_request_timeout_seconds",
)


class RuntimeOptionValues(BaseModel):
    """Normalized runtime-option values shared across entry points."""

    model_config = ConfigDict(frozen=True)

    data_dir: str | Path | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_backend: str | None = None
    answer_mode: str | None = None
    answer_verbosity: str | None = None
    retrieval_strategy: str | None = None
    vector_backend: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    allow_custom_llm_base_url: bool | None = None
    allowed_root: str | Path | None = None
    max_question_length: int | None = None
    max_retrieval_limit: int | None = None
    max_documents_per_ingest: int | None = None
    max_document_bytes: int | None = None
    max_chunks_per_ingest: int | None = None
    llm_request_timeout_seconds: int | None = None

    @classmethod
    def from_object(cls, source: object) -> RuntimeOptionValues:
        """Read runtime-option values from an object with matching attributes."""
        return cls(
            data_dir=getattr(source, "data_dir", None),
            chunk_size=getattr(source, "chunk_size", None),
            chunk_overlap=getattr(source, "chunk_overlap", None),
            embedding_backend=getattr(source, "embedding_backend", None),
            answer_mode=getattr(source, "answer_mode", None),
            answer_verbosity=getattr(source, "answer_verbosity", None),
            retrieval_strategy=getattr(source, "retrieval_strategy", None),
            vector_backend=getattr(source, "vector_backend", None),
            llm_model=getattr(source, "llm_model", None),
            llm_base_url=getattr(source, "llm_base_url", None),
            allow_custom_llm_base_url=getattr(source, "allow_custom_llm_base_url", None),
            allowed_root=getattr(source, "allowed_root", None),
            max_question_length=getattr(source, "max_question_length", None),
            max_retrieval_limit=getattr(source, "max_retrieval_limit", None),
            max_documents_per_ingest=getattr(source, "max_documents_per_ingest", None),
            max_document_bytes=getattr(source, "max_document_bytes", None),
            max_chunks_per_ingest=getattr(source, "max_chunks_per_ingest", None),
            llm_request_timeout_seconds=getattr(source, "llm_request_timeout_seconds", None),
        )

    def has_overrides(self) -> bool:
        """Return whether any runtime option is explicitly set."""
        return any(getattr(self, field_name) is not None for field_name in RUNTIME_OPTION_FIELD_NAMES)

    def to_runtime_config(self) -> RuntimeConfig:
        """Build a validated runtime config from the normalized values."""
        return RuntimeConfig.build(**self.model_dump(exclude_none=True))
