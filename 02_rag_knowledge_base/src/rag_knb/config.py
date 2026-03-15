"""Runtime configuration models for the knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_knb.errors import ValidationError

DEFAULT_DATA_DIR = Path(".rag-knb")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_BACKEND = "deterministic"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ANSWER_MODE = "extractive"
DEFAULT_ANSWER_VERBOSITY = "concise"
DEFAULT_RETRIEVAL_STRATEGY = "vector"
DEFAULT_VECTOR_BACKEND = "inmemory"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_QUESTION_LENGTH = 1000
DEFAULT_MAX_RETRIEVAL_LIMIT = 25
DEFAULT_MAX_DOCUMENTS_PER_INGEST = 25
DEFAULT_MAX_DOCUMENT_BYTES = 1_000_000
DEFAULT_MAX_CHUNKS_PER_INGEST = 5_000
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 60


class RuntimeConfig(BaseSettings):
    """Configuration values shared across CLI, services, and storage."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_KNB_",
        extra="ignore",
        frozen=True,
    )

    data_dir: Path = DEFAULT_DATA_DIR
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    answer_mode: str = DEFAULT_ANSWER_MODE
    answer_verbosity: str = DEFAULT_ANSWER_VERBOSITY
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY
    vector_backend: str = DEFAULT_VECTOR_BACKEND
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    allow_custom_llm_base_url: bool = True
    allowed_root: Path | None = None
    max_question_length: int = DEFAULT_MAX_QUESTION_LENGTH
    max_retrieval_limit: int = DEFAULT_MAX_RETRIEVAL_LIMIT
    max_documents_per_ingest: int = DEFAULT_MAX_DOCUMENTS_PER_INGEST
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES
    max_chunks_per_ingest: int = DEFAULT_MAX_CHUNKS_PER_INGEST
    llm_request_timeout_seconds: int = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS

    @classmethod
    def build(
        cls,
        *,
        data_dir: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_backend: str | None = None,
        embedding_model_name: str | None = None,
        answer_mode: str | None = None,
        answer_verbosity: str | None = None,
        retrieval_strategy: str | None = None,
        vector_backend: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        allow_custom_llm_base_url: bool | None = None,
        allowed_root: str | Path | None = None,
        max_question_length: int | None = None,
        max_retrieval_limit: int | None = None,
        max_documents_per_ingest: int | None = None,
        max_document_bytes: int | None = None,
        max_chunks_per_ingest: int | None = None,
        llm_request_timeout_seconds: int | None = None,
    ) -> RuntimeConfig:
        """Build and validate runtime configuration from optional overrides."""
        overrides = {
            "data_dir": data_dir,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_backend": embedding_backend,
            "embedding_model_name": embedding_model_name,
            "answer_mode": answer_mode,
            "answer_verbosity": answer_verbosity,
            "retrieval_strategy": retrieval_strategy,
            "vector_backend": vector_backend,
            "llm_model": llm_model,
            "llm_base_url": llm_base_url,
            "allow_custom_llm_base_url": allow_custom_llm_base_url,
            "allowed_root": allowed_root,
            "max_question_length": max_question_length,
            "max_retrieval_limit": max_retrieval_limit,
            "max_documents_per_ingest": max_documents_per_ingest,
            "max_document_bytes": max_document_bytes,
            "max_chunks_per_ingest": max_chunks_per_ingest,
            "llm_request_timeout_seconds": llm_request_timeout_seconds,
        }
        resolved_overrides = {
            key: value
            for key, value in overrides.items()
            if value is not None
        }
        try:
            return cls.model_validate(resolved_overrides)
        except PydanticValidationError as error:
            raise ValidationError(_validation_error_message(error)) from error

    @field_validator("chunk_size")
    @classmethod
    def _validate_chunk_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Chunk overlap cannot be negative.")
        return value

    @field_validator("embedding_backend")
    @classmethod
    def _validate_embedding_backend(cls, value: str) -> str:
        if value not in {"deterministic", "huggingface"}:
            raise ValueError("Embedding backend must be one of: deterministic, huggingface.")
        return value

    @field_validator("answer_mode")
    @classmethod
    def _validate_answer_mode(cls, value: str) -> str:
        if value not in {"extractive", "generative"}:
            raise ValueError("Answer mode must be one of: extractive, generative.")
        return value

    @field_validator("answer_verbosity")
    @classmethod
    def _validate_answer_verbosity(cls, value: str) -> str:
        if value not in {"concise", "verbose"}:
            raise ValueError("Answer verbosity must be one of: concise, verbose.")
        return value

    @field_validator("retrieval_strategy")
    @classmethod
    def _validate_retrieval_strategy(cls, value: str) -> str:
        if value not in {"vector", "hybrid"}:
            raise ValueError("Retrieval strategy must be one of: vector, hybrid.")
        return value

    @field_validator("vector_backend")
    @classmethod
    def _validate_vector_backend(cls, value: str) -> str:
        if value not in {"inmemory", "faiss"}:
            raise ValueError("Vector backend must be one of: inmemory, faiss.")
        return value

    @field_validator(
        "max_question_length",
        "max_retrieval_limit",
        "max_documents_per_ingest",
        "max_document_bytes",
        "max_chunks_per_ingest",
        "llm_request_timeout_seconds",
    )
    @classmethod
    def _validate_positive_limits(cls, value: int, info: Any) -> int:
        if value <= 0:
            messages = {
                "max_question_length": "Maximum question length must be greater than zero.",
                "max_retrieval_limit": "Maximum retrieval limit must be greater than zero.",
                "max_documents_per_ingest": "Maximum documents per ingest must be greater than zero.",
                "max_document_bytes": "Maximum document bytes must be greater than zero.",
                "max_chunks_per_ingest": "Maximum chunks per ingest must be greater than zero.",
                "llm_request_timeout_seconds": "LLM request timeout must be greater than zero.",
            }
            raise ValueError(messages[str(info.field_name)])
        return value

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> RuntimeConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        return self


def _validation_error_message(error: PydanticValidationError) -> str:
    """Extract the first stable user-facing validation message."""
    message = str(error.errors()[0]["msg"])
    if message.startswith("Value error, "):
        return message.removeprefix("Value error, ")
    return message
