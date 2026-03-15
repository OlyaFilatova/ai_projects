"""Persistence tests for the local knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_knb.errors import PersistedStateError
from rag_knb.indexing.storage import LocalKnowledgeBaseRepository
from rag_knb.service import KnowledgeBaseService
from tests.snapshot import assert_json_snapshot


def test_repository_persists_documents_chunks_and_vectors(tmp_path: Path) -> None:
    """Persistence should write the deterministic storage files."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text("Cats like windows. Dogs like parks.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"
    service = KnowledgeBaseService()
    service.ingest_paths([source_path])

    saved_path = service.save(data_dir)

    assert saved_path == data_dir
    assert (data_dir / "metadata.json").exists()
    assert (data_dir / "documents.json").exists()
    assert (data_dir / "chunks.json").exists()
    assert (data_dir / "vectors.json").exists()

    documents_payload = json.loads((data_dir / "documents.json").read_text(encoding="utf-8"))
    metadata_payload = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    assert documents_payload[0]["document_id"] == "notes"
    assert metadata_payload["embedding_workflow"]["embedding_backend"] == "deterministic"
    assert metadata_payload["embedding_workflow"]["embedding_model_name"] == "deterministic-v1"
    assert_json_snapshot("storage_metadata", metadata_payload)


def test_repository_loads_persisted_state_back_correctly(tmp_path: Path) -> None:
    """The persisted knowledge base should round-trip through the repository."""
    source_path = tmp_path / "science.md"
    source_path.write_text("# Space\n\nStars emit light.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"
    service = KnowledgeBaseService()
    service.ingest_paths([source_path])
    service.save(data_dir)
    repository = LocalKnowledgeBaseRepository(data_dir)

    loaded_state = repository.load()

    assert len(loaded_state.documents) == 1
    assert loaded_state.documents[0].metadata["format"] == "markdown"
    assert loaded_state.chunks
    assert loaded_state.indexed_chunks
    assert loaded_state.indexed_chunks[0].chunk.chunk_id == loaded_state.chunks[0].chunk_id


def test_service_can_reload_and_answer_without_source_documents(tmp_path: Path) -> None:
    """Reloaded state should answer questions without re-ingesting source files."""
    source_path = tmp_path / "pets.txt"
    source_path.write_text("Cats nap on warm chairs.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"
    writer = KnowledgeBaseService()
    writer.ingest_paths([source_path])
    writer.save(data_dir)
    reloaded_service = KnowledgeBaseService()

    reloaded_service.load(data_dir)
    answer = reloaded_service.ask("Where do cats nap?")

    assert answer.matched is True
    assert "warm chairs" in answer.answer_text


def test_repository_round_trips_mixed_plain_and_structured_corpora(tmp_path: Path) -> None:
    """Persistence should keep plain-text and structured records compatible in one corpus."""
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("Birds build nests.", encoding="utf-8")
    records_path = tmp_path / "pets.jsonl"
    records_path.write_text(
        '{"id": "cat-1", "animal": "cat", "trait": "playful", "favorite_food": "salmon"}\n',
        encoding="utf-8",
    )
    data_dir = tmp_path / "kb-data"
    writer = KnowledgeBaseService()
    writer.ingest_paths([notes_path, records_path])
    writer.save(data_dir)
    reloaded_service = KnowledgeBaseService()

    reloaded_service.load(data_dir)
    answer = reloaded_service.ask("Which animal likes salmon?")

    assert answer.reason == "matched"
    assert "animal: cat" in answer.answer_text
    assert any(document.metadata["format"] == "structured" for document in reloaded_service.documents)
    assert any(document.metadata["format"] == "text" for document in reloaded_service.documents)


def test_refresh_paths_can_incrementally_add_update_and_remove_sources(tmp_path: Path) -> None:
    """Incremental refresh should reuse persisted state while applying explicit source changes."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats nap in sunny spots.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"
    writer = KnowledgeBaseService()
    writer.ingest_paths([cats_path])
    writer.save(data_dir)

    cats_path.write_text("Cats nap on warm chairs.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs enjoy long walks.", encoding="utf-8")

    reloaded_service = KnowledgeBaseService()
    reloaded_service.load(data_dir)
    add_update_result = reloaded_service.refresh_paths([cats_path, dogs_path])

    assert str(dogs_path) in add_update_result.added_source_paths
    assert str(cats_path) in add_update_result.updated_source_paths
    assert add_update_result.removed_source_paths == []
    assert "warm chairs" in reloaded_service.ask("Where do cats nap?").answer_text
    assert "Dogs enjoy long walks." in reloaded_service.ask("What do dogs enjoy?").answer_text

    reloaded_service.save(data_dir)
    persisted_service = KnowledgeBaseService()
    persisted_service.load(data_dir)
    remove_result = persisted_service.refresh_paths([cats_path], remove_missing=True)

    assert str(dogs_path) in remove_result.removed_source_paths
    assert [document.document_id for document in persisted_service.documents] == ["cats"]
    assert all(match.chunk.document_id == "cats" for match in persisted_service.ask("What do dogs enjoy?").matches)


def test_loading_missing_data_directory_is_explicit(tmp_path: Path) -> None:
    """Missing persisted directories should raise a user-facing error."""
    service = KnowledgeBaseService()

    with pytest.raises(PersistedStateError) as error:
        service.load(tmp_path / "missing-data")

    assert "does not exist" in str(error.value)


def test_loading_incomplete_directory_is_explicit(tmp_path: Path) -> None:
    """Incomplete persisted directories should raise a user-facing error."""
    data_dir = tmp_path / "kb-data"
    data_dir.mkdir()
    (data_dir / "metadata.json").write_text('{"schema_version": 1}', encoding="utf-8")
    (data_dir / "documents.json").write_text("[]", encoding="utf-8")

    with pytest.raises(PersistedStateError) as error:
        LocalKnowledgeBaseRepository(data_dir).load()

    assert "is incomplete" in str(error.value)


def test_loading_invalid_json_is_explicit(tmp_path: Path) -> None:
    """Invalid JSON should surface a clear error."""
    data_dir = tmp_path / "kb-data"
    data_dir.mkdir()
    (data_dir / "metadata.json").write_text('{"schema_version": 1}', encoding="utf-8")
    (data_dir / "documents.json").write_text("[]", encoding="utf-8")
    (data_dir / "chunks.json").write_text("[]", encoding="utf-8")
    (data_dir / "vectors.json").write_text("{", encoding="utf-8")

    with pytest.raises(PersistedStateError) as error:
        LocalKnowledgeBaseRepository(data_dir).load()

    assert "contains invalid JSON" in str(error.value)


def test_loading_unsupported_schema_version_is_explicit(tmp_path: Path) -> None:
    """Unsupported storage schema versions should fail with a clear error."""
    data_dir = tmp_path / "kb-data"
    data_dir.mkdir()
    (data_dir / "metadata.json").write_text('{"schema_version": 999}', encoding="utf-8")
    (data_dir / "documents.json").write_text("[]", encoding="utf-8")
    (data_dir / "chunks.json").write_text("[]", encoding="utf-8")
    (data_dir / "vectors.json").write_text("[]", encoding="utf-8")

    with pytest.raises(PersistedStateError) as error:
        LocalKnowledgeBaseRepository(data_dir).load()

    assert "unsupported schema version" in str(error.value)


def test_loading_vectors_with_incompatible_embedding_workflow_is_explicit(tmp_path: Path) -> None:
    """Embedding workflow mismatches should require a rebuild instead of silent reuse."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text("Cats like windows.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"
    service = KnowledgeBaseService()
    service.ingest_paths([source_path])
    service.save(data_dir)

    metadata_payload = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata_payload["embedding_workflow"] = {
        "embedding_backend": "huggingface",
        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
        "vector_shape": "dense",
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata_payload), encoding="utf-8")

    with pytest.raises(PersistedStateError) as error:
        KnowledgeBaseService().load(data_dir)

    assert "Persisted vectors were built with embedding workflow" in str(error.value)
    assert "rebuild it with a fresh ingest and save" in str(error.value)
