"""Vector-store backend selection tests."""

from __future__ import annotations

import pytest

from rag_knb.config import RuntimeConfig
from rag_knb.errors import DependencyUnavailableError, ValidationError
from rag_knb.retrieval_engine.embeddings import DeterministicEmbedder
from rag_knb.retrieval_engine.vector_store import InMemoryVectorStore, build_vector_store


def test_inmemory_vector_backend_is_the_default() -> None:
    """The default vector backend should remain the local in-memory store."""
    backend = build_vector_store(RuntimeConfig.build(), DeterministicEmbedder())

    assert isinstance(backend, InMemoryVectorStore)


def test_faiss_backend_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting FAISS without the optional dependency should fail clearly."""
    monkeypatch.setattr("rag_knb.optional_dependencies.find_spec", lambda _: None)

    with pytest.raises(DependencyUnavailableError) as error:
        build_vector_store(RuntimeConfig.build(vector_backend="faiss"), DeterministicEmbedder())

    assert "optional 'faiss' dependency group" in str(error.value)


def test_runtime_config_rejects_unknown_vector_backend() -> None:
    """Unsupported vector backend names should fail in config validation."""
    with pytest.raises(ValidationError) as error:
        RuntimeConfig.build(vector_backend="custom")

    assert "Vector backend must be one of" in str(error.value)


def test_runtime_config_accepts_huggingface_plus_faiss_workflow() -> None:
    """The recommended semantic retrieval backend pair should remain a valid config."""
    config = RuntimeConfig.build(
        embedding_backend="huggingface",
        vector_backend="faiss",
    )

    assert config.embedding_backend == "huggingface"
    assert config.vector_backend == "faiss"
