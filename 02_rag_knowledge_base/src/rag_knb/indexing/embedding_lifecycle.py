"""Helpers for persisted embedding workflow compatibility."""

from __future__ import annotations

from rag_knb.errors import PersistedStateError
from rag_knb.retrieval_engine.embeddings import EmbeddingBackend
from rag_knb.retrieval_engine.vector_store import IndexedChunk


def validate_embedding_workflow_compatibility(
    metadata: dict[str, object],
    indexed_chunks: list[IndexedChunk],
    embedder: EmbeddingBackend,
) -> None:
    """Reject persisted vectors that were produced by an incompatible embedder."""
    persisted_workflow = _resolve_persisted_embedding_workflow(metadata, indexed_chunks)
    current_workflow = embedder.workflow_metadata()
    if persisted_workflow["embedding_backend"] != current_workflow["embedding_backend"]:
        raise PersistedStateError(
            _embedding_workflow_error_message(persisted_workflow, current_workflow)
        )
    if (
        persisted_workflow["embedding_backend"] == "huggingface"
        and persisted_workflow["embedding_model_name"] != current_workflow["embedding_model_name"]
    ):
        raise PersistedStateError(
            _embedding_workflow_error_message(persisted_workflow, current_workflow)
        )


def _resolve_persisted_embedding_workflow(
    metadata: dict[str, object],
    indexed_chunks: list[IndexedChunk],
) -> dict[str, str]:
    """Resolve stored embedding workflow metadata or infer a safe legacy fallback."""
    workflow_payload = metadata.get("embedding_workflow")
    if isinstance(workflow_payload, dict):
        return {
            "embedding_backend": str(workflow_payload.get("embedding_backend", "unknown")),
            "embedding_model_name": str(workflow_payload.get("embedding_model_name", "unknown")),
            "vector_shape": str(workflow_payload.get("vector_shape", "unknown")),
        }
    if not indexed_chunks:
        return {
            "embedding_backend": "deterministic",
            "embedding_model_name": "deterministic-v1",
            "vector_shape": "sparse",
        }
    first_vector = indexed_chunks[0].vector
    if isinstance(first_vector, dict):
        return {
            "embedding_backend": "deterministic",
            "embedding_model_name": "deterministic-v1",
            "vector_shape": "sparse",
        }
    return {
        "embedding_backend": "huggingface",
        "embedding_model_name": "unknown-dense-model",
        "vector_shape": "dense",
    }


def _embedding_workflow_error_message(
    persisted_workflow: dict[str, str],
    current_workflow: dict[str, str],
) -> str:
    """Build the user-facing rebuild guidance for embedding workflow mismatches."""
    return (
        "Persisted vectors were built with embedding workflow "
        f"{persisted_workflow['embedding_backend']}:{persisted_workflow['embedding_model_name']}, "
        "but the current service expects "
        f"{current_workflow['embedding_backend']}:{current_workflow['embedding_model_name']}. "
        "Load the knowledge base with matching embedding settings or rebuild it with a fresh ingest and save."
    )
