"""Embedding backends for retrieval."""

from __future__ import annotations

import io
import math
import re
import warnings
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from typing import Protocol, cast

from rag_knb.config import RuntimeConfig
from rag_knb.optional_dependencies import require_huggingface_langchain

TokenVector = dict[str, float] | list[float]
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "but",
    "by",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "say",
    "the",
    "their",
    "them",
    "to",
    "what",
    "which",
    "who",
    "with",
    "you",
}
TOKEN_EXPANSIONS = {
    "cat": ("feline", "pet"),
    "dog": ("canine", "pet"),
    "play": ("playful",),
    "nap": ("sleep",),
}


class EmbeddingBackend:
    """Interface for text embedding backends."""

    def embed(self, text: str) -> TokenVector:
        """Convert text into a vector representation."""
        raise NotImplementedError

    def workflow_metadata(self) -> dict[str, str]:
        """Describe the embedding workflow used to produce persisted vectors."""
        raise NotImplementedError


class LangChainEmbeddingsProvider(Protocol):
    """Protocol exposing a LangChain-compatible embeddings object."""

    @property
    def langchain_embeddings(self) -> object:
        """Return the underlying LangChain embeddings implementation."""


class QueryEmbeddingModel(Protocol):
    """Protocol for embeddings backends that can embed one query string."""

    def embed_query(self, text: str) -> list[float]:
        """Return one dense embedding for the provided query text."""


class DeterministicEmbedder(EmbeddingBackend):
    """A deterministic token-frequency embedder for offline use and tests."""

    def embed(self, text: str) -> TokenVector:
        """Create a normalized token-frequency vector."""
        tokens: list[str] = []
        for token in TOKEN_PATTERN.findall(text.lower()):
            normalized_token = _normalize_token(token)
            if normalized_token is None:
                continue
            tokens.extend(_expand_token(normalized_token))
        counts = Counter(tokens)
        norm = math.sqrt(sum(value * value for value in counts.values()))
        if norm == 0:
            return {}
        return {
            token: count / norm
            for token, count in sorted(counts.items())
        }

    def workflow_metadata(self) -> dict[str, str]:
        """Describe the deterministic embedding workflow."""
        return {
            "embedding_backend": "deterministic",
            "embedding_model_name": "deterministic-v1",
            "vector_shape": "sparse",
        }


class HuggingFaceEmbedder(EmbeddingBackend):
    """LangChain-backed Hugging Face embedder loaded only when explicitly requested."""

    def __init__(self, model_name: str) -> None:
        """Initialize the Hugging Face embedder lazily."""
        require_huggingface_langchain()
        self._model_name = model_name
        self._embeddings = self._build_embeddings(model_name)

    @property
    def langchain_embeddings(self) -> object:
        """Expose the LangChain embeddings implementation."""
        return self._embeddings

    def embed(self, text: str) -> TokenVector:
        """Convert text into a dense vector through LangChain Hugging Face embeddings."""
        dense_vector = self._embeddings.embed_query(text)
        return [float(value) for value in dense_vector]

    def workflow_metadata(self) -> dict[str, str]:
        """Describe the Hugging Face embedding workflow."""
        return {
            "embedding_backend": "huggingface",
            "embedding_model_name": self._model_name,
            "vector_shape": "dense",
        }

    def _build_embeddings(self, model_name: str) -> QueryEmbeddingModel:
        """Create the LangChain Hugging Face embeddings while hiding known startup noise."""
        with (
            warnings.catch_warnings(),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            warnings.filterwarnings(
                "ignore",
                message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
                category=UserWarning,
            )
            from langchain_huggingface import HuggingFaceEmbeddings

            return cast(QueryEmbeddingModel, HuggingFaceEmbeddings(model_name=model_name))


def cosine_similarity(left: TokenVector, right: TokenVector) -> float:
    """Compute cosine similarity between sparse token vectors."""
    left_values = _normalize_vector(left)
    right_values = _normalize_vector(right)
    if not left_values or not right_values:
        return 0.0
    if isinstance(left_values, dict) and isinstance(right_values, dict):
        shared_tokens = set(left_values).intersection(right_values)
        return sum(left_values[token] * right_values[token] for token in shared_tokens)

    left_dense = _to_dense_values(left_values)
    right_dense = _to_dense_values(right_values)
    dimensions = max(len(left_dense), len(right_dense))
    if dimensions == 0:
        return 0.0
    left_dense.extend([0.0] * (dimensions - len(left_dense)))
    right_dense.extend([0.0] * (dimensions - len(right_dense)))
    return sum(left_dense[index] * right_dense[index] for index in range(dimensions))


def build_embedder(config: RuntimeConfig) -> EmbeddingBackend:
    """Create an embedder from runtime configuration."""
    if config.embedding_backend == "huggingface":
        return HuggingFaceEmbedder(config.embedding_model_name)
    return DeterministicEmbedder()


def _normalize_token(token: str) -> str | None:
    """Reduce tokens to a small normalized form for offline lexical retrieval."""
    if token in STOPWORDS:
        return None

    normalized = token
    for suffix, replacement, minimum_length in (
        ("fully", "", 7),
        ("fulness", "", 8),
        ("fulness", "", 8),
        ("ful", "", 6),
        ("ingly", "", 7),
        ("ing", "", 5),
        ("edly", "", 6),
        ("ed", "", 4),
        ("ies", "y", 5),
        ("es", "", 5),
        ("s", "", 4),
    ):
        if normalized.endswith(suffix) and len(normalized) >= minimum_length:
            normalized = f"{normalized[:-len(suffix)]}{replacement}"
            break

    if len(normalized) >= 2 and normalized[-1] == normalized[-2] and normalized[-1] not in "aeiou":
        normalized = normalized[:-1]
    if not normalized or normalized in STOPWORDS:
        return None
    return normalized


def _expand_token(token: str) -> tuple[str, ...]:
    """Expand one normalized token into a small deterministic synonym family."""
    return (token, *TOKEN_EXPANSIONS.get(token, ()))


def _normalize_vector(vector: TokenVector) -> dict[str, float] | list[float]:
    """Normalize a vector into a supported runtime shape."""
    return vector


def _to_dense_values(vector: dict[str, float] | list[float]) -> list[float]:
    """Convert vector values into a dense list."""
    if isinstance(vector, list):
        return list(vector)
    resolved_dimensions = max((int(key) for key in vector), default=-1) + 1
    dense_values = [0.0] * resolved_dimensions
    for key, value in vector.items():
        dense_values[int(key)] = value
    return dense_values
