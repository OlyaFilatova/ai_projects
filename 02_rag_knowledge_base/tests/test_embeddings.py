"""Embedding backend selection tests."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from rag_knb.config import RuntimeConfig
from rag_knb.errors import DependencyUnavailableError, ValidationError
from rag_knb.retrieval_engine.embeddings import DeterministicEmbedder, build_embedder


def test_deterministic_embedder_is_the_default_backend() -> None:
    """The offline deterministic embedder should remain the default."""
    embedder = build_embedder(RuntimeConfig.build())

    assert isinstance(embedder, DeterministicEmbedder)


def test_deterministic_embedder_normalizes_simple_query_variants() -> None:
    """The default embedder should ignore filler words and normalize basic variants."""
    embedder = DeterministicEmbedder()

    query_vector = embedder.embed("Which animal is energetic?")
    playful_vector = embedder.embed("playful animals")

    assert set(query_vector) == {"animal", "energetic"}
    assert set(playful_vector) == {"animal", "play", "playful"}


def test_deterministic_embedder_applies_small_safe_token_expansions() -> None:
    """The default embedder should add a small set of deterministic synonym hints."""
    embedder = DeterministicEmbedder()

    cat_vector = embedder.embed("cats")
    dog_vector = embedder.embed("dog")

    assert set(cat_vector) == {"cat", "feline", "pet"}
    assert set(dog_vector) == {"dog", "canine", "pet"}


def test_huggingface_backend_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting Hugging Face without the optional dependency should fail clearly."""
    monkeypatch.setattr("rag_knb.optional_dependencies.find_spec", lambda _: None)

    with pytest.raises(DependencyUnavailableError) as error:
        build_embedder(RuntimeConfig.build(embedding_backend="huggingface"))

    assert "LangChain dependencies" in str(error.value)


def test_huggingface_backend_suppresses_known_startup_noise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hugging Face backend startup should hide known dependency noise from CLI users."""

    class FakeEmbeddings:
        def __init__(self, model_name: str) -> None:
            print("BertModel LOAD REPORT from fake backend")
            print("embeddings.position_ids | UNEXPECTED", file=sys.stderr)
            import warnings

            warnings.warn(
                "Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
                UserWarning,
                stacklevel=1,
            )
            self.model_name = model_name

        def embed_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

    monkeypatch.setattr("rag_knb.retrieval_engine.embeddings.require_huggingface_langchain", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        types.SimpleNamespace(HuggingFaceEmbeddings=FakeEmbeddings),
    )

    embedder = build_embedder(RuntimeConfig.build(embedding_backend="huggingface"))

    assert embedder.workflow_metadata()["embedding_backend"] == "huggingface"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_runtime_config_rejects_unknown_embedding_backend() -> None:
    """Unsupported backend names should fail in config validation."""
    with pytest.raises(ValidationError) as error:
        RuntimeConfig.build(embedding_backend="custom")

    assert "Embedding backend must be one of" in str(error.value)


def test_semantic_retrieval_scenarios_fixture_documents_expected_tradeoffs() -> None:
    """Semantic workflow fixtures should document when semantic retrieval helps most."""
    fixture_path = Path("tests/fixtures/semantic_retrieval_scenarios.json")
    scenarios = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert scenarios
    assert all("question" in scenario for scenario in scenarios)
    assert all("deterministic_risk" in scenario for scenario in scenarios)
    assert all("semantic_expected_advantage" in scenario for scenario in scenarios)
