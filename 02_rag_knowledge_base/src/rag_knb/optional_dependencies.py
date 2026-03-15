"""Shared optional-dependency capability checks."""

from __future__ import annotations

import sys
from importlib.util import find_spec

from rag_knb.errors import DependencyUnavailableError


def has_langchain_text_splitters() -> bool:
    """Return whether LangChain text splitters are available."""
    # LangChain currently emits a runtime compatibility warning on Python 3.14+.
    # The built-in deterministic chunker keeps the default workflow stable there.
    if sys.version_info >= (3, 14):
        return False
    return _has_module("langchain_text_splitters")


def require_huggingface_langchain() -> None:
    """Require the optional LangChain Hugging Face embeddings package."""
    _require_module(
        "langchain_huggingface",
        "Hugging Face embeddings require the optional 'huggingface' LangChain dependencies.",
    )


def require_faiss() -> None:
    """Require the optional FAISS backend package."""
    _require_module(
        "faiss",
        "FAISS vector backend requires the optional 'faiss' dependency group.",
    )


def require_langchain_faiss() -> None:
    """Require the optional LangChain community FAISS integration."""
    _require_module(
        "langchain_community",
        "FAISS vector backend requires the optional LangChain community dependencies.",
    )


def _has_module(module_name: str) -> bool:
    """Return whether the importable module is installed."""
    return find_spec(module_name) is not None


def _require_module(module_name: str, error_message: str) -> None:
    """Fail with a stable user-facing error when an optional dependency is unavailable."""
    if not _has_module(module_name):
        raise DependencyUnavailableError(error_message)
