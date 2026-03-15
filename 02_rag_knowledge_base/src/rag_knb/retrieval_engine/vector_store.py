"""Vector storage abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from rag_knb.config import RuntimeConfig
from rag_knb.models import Chunk, RetrievalResult
from rag_knb.optional_dependencies import require_faiss, require_langchain_faiss
from rag_knb.retrieval_engine.embeddings import (
    EmbeddingBackend,
    HuggingFaceEmbedder,
    TokenVector,
    cosine_similarity,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A chunk and its embedded vector representation."""

    chunk: Chunk
    vector: TokenVector


class VectorStore(Protocol):
    """Contract shared by vector-store backends."""

    def clear(self) -> None:
        """Remove all indexed chunks."""

    def add(self, chunk: Chunk, vector: TokenVector) -> None:
        """Add one chunk and its vector to the backend."""

    def replace(self, entries: list[IndexedChunk]) -> None:
        """Replace the backend state with indexed entries."""

    def search(
        self,
        query_text: str,
        query_vector: TokenVector,
        limit: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the backend for the closest chunk matches."""

    @property
    def entries(self) -> list[IndexedChunk]:
        """Expose entries for persistence workflows."""


class _EntryBackedVectorStore:
    """Shared entry management for vector-store backends."""

    def __init__(self) -> None:
        """Initialize the shared indexed-entry state."""
        self._entries: list[IndexedChunk] = []

    def clear(self) -> None:
        """Remove all indexed chunks."""
        self._entries.clear()
        self._on_entries_replaced()

    def add(self, chunk: Chunk, vector: TokenVector) -> None:
        """Add one chunk and its vector to the backend."""
        self._entries.append(IndexedChunk(chunk=chunk, vector=vector))
        self._on_entries_replaced()

    def replace(self, entries: list[IndexedChunk]) -> None:
        """Replace the backend state with indexed entries."""
        self._entries = list(entries)
        self._on_entries_replaced()

    @property
    def entries(self) -> list[IndexedChunk]:
        """Expose indexed chunks for persistence-oriented workflows."""
        return list(self._entries)

    def _on_entries_replaced(self) -> None:
        """Refresh any backend-specific index after entries change."""


class InMemoryVectorStore(_EntryBackedVectorStore):
    """A simple in-memory vector store for local retrieval."""

    def __init__(self) -> None:
        """Initialize an empty vector store."""
        super().__init__()

    def search(
        self,
        query_text: str,
        query_vector: TokenVector,
        limit: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-scoring chunks for the query vector."""
        del query_text
        return _score_indexed_chunks(self._entries, query_vector, limit, metadata_filters)


class FaissVectorStore(_EntryBackedVectorStore):
    """Optional FAISS-backed vector store."""

    def __init__(self) -> None:
        """Initialize the FAISS backend or fail with an explicit dependency error."""
        require_faiss()
        import faiss  # type: ignore

        super().__init__()
        self._faiss: Any = faiss
        self._index: Any = None

    def search(
        self,
        query_text: str,
        query_vector: TokenVector,
        limit: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the FAISS index and return the top matches."""
        del query_text
        if self._index is None or not self._entries:
            return []
        query_array = _vector_to_dense_matrix(query_vector)
        scores, indices = self._index.search(query_array, limit)
        return _build_index_search_results(self._entries, scores[0], indices[0], metadata_filters)

    def _on_entries_replaced(self) -> None:
        """Rebuild the FAISS index from the current sparse vectors."""
        import numpy

        if not self._entries:
            self._index = None
            return
        dimensions = _entry_vector_dimensions(self._entries)
        if dimensions <= 0:
            self._index = None
            return
        dense_vectors = [_dense_vector_values(entry.vector, dimensions=dimensions) for entry in self._entries]
        index = self._faiss.IndexFlatIP(dimensions)
        index.add(numpy.array(dense_vectors, dtype="float32"))
        self._index = index


class LangChainFaissVectorStore(_EntryBackedVectorStore):
    """LangChain FAISS backend using Hugging Face embeddings."""

    def __init__(self, embedder: HuggingFaceEmbedder) -> None:
        """Initialize the LangChain FAISS store."""
        require_langchain_faiss()
        super().__init__()
        self._embedder = embedder
        self._store: Any = None

    def search(
        self,
        query_text: str,
        query_vector: TokenVector,
        limit: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the LangChain FAISS store for matching chunks."""
        del query_vector
        if self._store is None:
            return []
        filter_payload = metadata_filters or None
        matches = self._store.similarity_search_with_relevance_scores(
            query_text,
            k=limit,
            filter=filter_payload,
        )
        return [
            RetrievalResult(
                chunk=_langchain_document_to_chunk(document),
                score=float(score),
            )
            for document, score in matches
            if score > 0
        ]

    def _on_entries_replaced(self) -> None:
        """Rebuild the LangChain FAISS store from indexed chunks."""
        from langchain_community.vectorstores import FAISS

        if not self._entries:
            self._store = None
            return
        texts = [entry.chunk.content for entry in self._entries]
        metadatas = [_chunk_to_langchain_metadata(entry.chunk) for entry in self._entries]
        ids = [entry.chunk.chunk_id for entry in self._entries]
        self._store = FAISS.from_texts(
            texts=texts,
            embedding=cast("Embeddings", self._embedder.langchain_embeddings),
            metadatas=metadatas,
            ids=ids,
        )


def build_vector_store(config: RuntimeConfig, embedder: EmbeddingBackend) -> VectorStore:
    """Create a vector-store backend from runtime configuration."""
    if config.vector_backend == "faiss":
        if isinstance(embedder, HuggingFaceEmbedder):
            return LangChainFaissVectorStore(embedder)
        return FaissVectorStore()
    return InMemoryVectorStore()


def _matches_metadata_filters(
    metadata: dict[str, object],
    metadata_filters: dict[str, str] | None,
) -> bool:
    """Check whether metadata satisfies exact-match filters."""
    if not metadata_filters:
        return True
    return all(str(metadata.get(key)) == value for key, value in metadata_filters.items())


def _score_indexed_chunks(
    entries: list[IndexedChunk],
    query_vector: TokenVector,
    limit: int,
    metadata_filters: dict[str, str] | None,
) -> list[RetrievalResult]:
    """Score indexed entries against the query vector and return top positive matches."""
    scored_matches = [
        RetrievalResult(chunk=entry.chunk, score=cosine_similarity(entry.vector, query_vector))
        for entry in entries
        if _matches_metadata_filters(entry.chunk.metadata, metadata_filters)
    ]
    ranked_matches = sorted(scored_matches, key=lambda item: item.score, reverse=True)
    return [match for match in ranked_matches[:limit] if match.score > 0]


def _build_index_search_results(
    entries: list[IndexedChunk],
    scores: Any,
    indices: Any,
    metadata_filters: dict[str, str] | None,
) -> list[RetrievalResult]:
    """Convert backend index results into filtered retrieval matches."""
    matches: list[RetrievalResult] = []
    for score, index in zip(scores, indices, strict=False):
        if index < 0 or score <= 0:
            continue
        chunk = entries[index].chunk
        if not _matches_metadata_filters(chunk.metadata, metadata_filters):
            continue
        matches.append(RetrievalResult(chunk=chunk, score=float(score)))
    return matches


def _entry_vector_dimensions(entries: list[IndexedChunk]) -> int:
    """Resolve the dense vector width required for the indexed entries."""
    dense_dimensions = [len(entry.vector) for entry in entries if isinstance(entry.vector, list)]
    sparse_dimensions = [
        max((int(key) for key in entry.vector), default=-1) + 1
        for entry in entries
        if isinstance(entry.vector, dict)
    ]
    return max([0, *dense_dimensions, *sparse_dimensions])


def _vector_to_dense_matrix(
    vector: TokenVector,
    *,
    dimensions: int | None = None,
) -> Any:
    """Convert a token vector into a dense float32 matrix for FAISS."""
    import numpy

    return numpy.array(
        [_dense_vector_values(vector, dimensions=dimensions)],
        dtype="float32",
    )


def _dense_vector_values(
    vector: TokenVector,
    *,
    dimensions: int | None = None,
) -> list[float]:
    """Convert a sparse or dense token vector into a dense Python list."""
    if isinstance(vector, list):
        resolved_dimensions = dimensions or len(vector)
        dense_values = list(vector)
        dense_values.extend([0.0] * (resolved_dimensions - len(dense_values)))
        return dense_values

    resolved_dimensions = dimensions or (max((int(key) for key in vector), default=-1) + 1)
    dense_vector: list[float] = [0.0] * resolved_dimensions
    for index, value in vector.items():
        dense_vector[int(index)] = value
    return dense_vector


def _chunk_to_langchain_metadata(chunk: Chunk) -> dict[str, object]:
    """Convert an internal chunk into LangChain document metadata."""
    return {
        **chunk.metadata,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
    }


def _langchain_document_to_chunk(document: Any) -> Chunk:
    """Convert a LangChain document back into the internal chunk model."""
    metadata = dict(document.metadata)
    return Chunk(
        chunk_id=str(metadata.pop("chunk_id")),
        document_id=str(metadata.pop("document_id")),
        content=document.page_content,
        start_offset=int(metadata.pop("start_offset")),
        end_offset=int(metadata.pop("end_offset")),
        metadata=metadata,
    )
