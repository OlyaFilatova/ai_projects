"""Persistence primitives for the local knowledge base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_knb.errors import PersistedStateError
from rag_knb.models import Chunk, Document
from rag_knb.retrieval_engine.vector_store import IndexedChunk

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PersistedKnowledgeBase:
    """Persisted knowledge-base state loaded from disk."""

    documents: list[Document]
    chunks: list[Chunk]
    indexed_chunks: list[IndexedChunk]
    metadata: dict[str, object]


class LocalKnowledgeBaseRepository:
    """Persist knowledge-base artifacts in a deterministic local layout."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository with a target data directory."""
        self._data_dir = data_dir

    @property
    def documents_path(self) -> Path:
        """Path to persisted documents."""
        return self._data_dir / "documents.json"

    @property
    def metadata_path(self) -> Path:
        """Path to persisted metadata."""
        return self._data_dir / "metadata.json"

    @property
    def chunks_path(self) -> Path:
        """Path to persisted chunks."""
        return self._data_dir / "chunks.json"

    @property
    def vectors_path(self) -> Path:
        """Path to persisted vectors."""
        return self._data_dir / "vectors.json"

    def save(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        indexed_chunks: list[IndexedChunk],
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Persist knowledge-base state to disk."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "format": "rag-knb-local",
                    **(metadata or {}),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.documents_path.write_text(
            json.dumps([document.to_dict() for document in documents], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.chunks_path.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.vectors_path.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": entry.chunk.chunk_id,
                        "vector": entry.vector,
                    }
                    for entry in indexed_chunks
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def load(self) -> PersistedKnowledgeBase:
        """Load persisted knowledge-base state from disk."""
        if not self._data_dir.exists():
            raise PersistedStateError(
                f"Knowledge-base data directory '{self._data_dir}' does not exist."
            )

        missing_paths = [
            path.name
            for path in (self.metadata_path, self.documents_path, self.chunks_path, self.vectors_path)
            if not path.exists()
        ]
        if missing_paths:
            missing_text = ", ".join(sorted(missing_paths))
            raise PersistedStateError(
                f"Knowledge-base data directory '{self._data_dir}' is incomplete. Missing: {missing_text}."
            )

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            documents_data = json.loads(self.documents_path.read_text(encoding="utf-8"))
            chunks_data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            vectors_data = json.loads(self.vectors_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PersistedStateError(
                f"Knowledge-base data directory '{self._data_dir}' contains invalid JSON: {exc.msg}."
            ) from exc
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise PersistedStateError(
                f"Knowledge-base data directory '{self._data_dir}' uses unsupported schema version "
                f"{metadata.get('schema_version')!r}."
            )

        documents = [Document(**payload) for payload in documents_data]
        chunks = [Chunk(**payload) for payload in chunks_data]
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        indexed_chunks: list[IndexedChunk] = []
        for vector_payload in vectors_data:
            chunk_id = vector_payload["chunk_id"]
            if chunk_id not in chunks_by_id:
                raise PersistedStateError(
                    f"Knowledge-base data directory '{self._data_dir}' references missing chunk '{chunk_id}'."
                )
            indexed_chunks.append(
                IndexedChunk(chunk=chunks_by_id[chunk_id], vector=vector_payload["vector"])
            )
        return PersistedKnowledgeBase(
            documents=documents,
            chunks=chunks,
            indexed_chunks=indexed_chunks,
            metadata=metadata,
        )
