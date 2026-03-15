"""Runtime configuration tests."""

from pathlib import Path
from typing import Any, cast

import pytest

from rag_knb.config import RuntimeConfig
from rag_knb.errors import ValidationError


def test_runtime_config_defaults_are_stable() -> None:
    """Runtime config should expose deterministic defaults."""
    config = RuntimeConfig.build()

    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.data_dir == Path(".rag-knb")
    assert config.answer_verbosity == "concise"
    assert config.retrieval_strategy == "vector"
    assert config.max_question_length == 1000
    assert config.max_retrieval_limit == 25
    assert config.max_documents_per_ingest == 25
    assert config.max_document_bytes == 1_000_000
    assert config.max_chunks_per_ingest == 5_000
    assert config.llm_request_timeout_seconds == 60


def test_runtime_config_accepts_overrides() -> None:
    """Runtime config should accept valid overrides."""
    config = RuntimeConfig.build(
        data_dir="custom-data",
        chunk_size=256,
        chunk_overlap=32,
        answer_verbosity="verbose",
        retrieval_strategy="hybrid",
        allow_custom_llm_base_url=False,
        allowed_root="sandbox",
        max_question_length=512,
        max_retrieval_limit=10,
        max_documents_per_ingest=5,
        max_document_bytes=4096,
        max_chunks_per_ingest=100,
        llm_request_timeout_seconds=30,
    )

    assert config.data_dir == Path("custom-data")
    assert config.chunk_size == 256
    assert config.chunk_overlap == 32
    assert config.answer_verbosity == "verbose"
    assert config.retrieval_strategy == "hybrid"
    assert config.allow_custom_llm_base_url is False
    assert config.allowed_root == Path("sandbox")
    assert config.max_question_length == 512
    assert config.max_retrieval_limit == 10
    assert config.max_documents_per_ingest == 5
    assert config.max_document_bytes == 4096
    assert config.max_chunks_per_ingest == 100
    assert config.llm_request_timeout_seconds == 30


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "Chunk size must be greater than zero."),
        ({"chunk_overlap": -1}, "Chunk overlap cannot be negative."),
        (
            {"chunk_size": 10, "chunk_overlap": 10},
            "Chunk overlap must be smaller than chunk size.",
        ),
        (
            {"answer_mode": "invented"},
            "Answer mode must be one of: extractive, generative.",
        ),
        (
            {"answer_verbosity": "chatty"},
            "Answer verbosity must be one of: concise, verbose.",
        ),
        (
            {"retrieval_strategy": "keyword"},
            "Retrieval strategy must be one of: vector, hybrid.",
        ),
        (
            {"max_question_length": 0},
            "Maximum question length must be greater than zero.",
        ),
        (
            {"max_retrieval_limit": 0},
            "Maximum retrieval limit must be greater than zero.",
        ),
        (
            {"max_documents_per_ingest": 0},
            "Maximum documents per ingest must be greater than zero.",
        ),
        (
            {"max_document_bytes": 0},
            "Maximum document bytes must be greater than zero.",
        ),
        (
            {"max_chunks_per_ingest": 0},
            "Maximum chunks per ingest must be greater than zero.",
        ),
        (
            {"llm_request_timeout_seconds": 0},
            "LLM request timeout must be greater than zero.",
        ),
    ],
)
def test_runtime_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Runtime config should fail fast on invalid values."""
    with pytest.raises(ValidationError) as error:
        RuntimeConfig.build(**cast(dict[str, Any], kwargs))

    assert str(error.value) == message
