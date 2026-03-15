"""Tests for shared service-construction helpers."""

from __future__ import annotations

from pathlib import Path

from rag_knb.config import DEFAULT_DATA_DIR
from rag_knb.service import KnowledgeBaseService
from rag_knb.service_factory import build_service_from_options


class _RuntimeOptionSource:
    """Simple object exposing runtime-option attributes."""

    data_dir = "custom-data"
    chunk_size = 128
    chunk_overlap = 16
    embedding_backend = None
    answer_mode = None
    answer_verbosity = "verbose"
    retrieval_strategy = "hybrid"
    vector_backend = None
    llm_model = None
    llm_base_url = None


class _EmptyRuntimeOptionSource:
    """Simple object without runtime overrides."""

    data_dir = None
    chunk_size = None
    chunk_overlap = None
    embedding_backend = None
    answer_mode = None
    answer_verbosity = None
    retrieval_strategy = None
    vector_backend = None
    llm_model = None
    llm_base_url = None


def test_build_service_from_options_reuses_existing_service_without_overrides() -> None:
    """The helper should reuse an existing service when no overrides are set."""
    service = KnowledgeBaseService()

    built_service = build_service_from_options(_EmptyRuntimeOptionSource(), existing_service=service)

    assert built_service is service


def test_build_service_from_options_builds_service_for_overrides() -> None:
    """The helper should build a configured service when overrides are present."""
    built_service = build_service_from_options(_RuntimeOptionSource())

    assert built_service is not None
    assert built_service.config.chunk_size == 128
    assert built_service.config.chunk_overlap == 16
    assert built_service.config.data_dir == Path("custom-data")
    assert built_service.config.answer_verbosity == "verbose"
    assert built_service.config.retrieval_strategy == "hybrid"


def test_build_service_from_options_creates_default_service_without_existing_one() -> None:
    """The helper should still build a default service when nothing is provided."""
    built_service = build_service_from_options(_EmptyRuntimeOptionSource())

    assert built_service.config.data_dir == DEFAULT_DATA_DIR
