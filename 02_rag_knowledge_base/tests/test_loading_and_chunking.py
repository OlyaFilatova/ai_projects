"""Tests for document loading and chunking primitives."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_knb.errors import DocumentLoadError, UnsupportedFileTypeError
from rag_knb.indexing.chunking import chunk_document
from rag_knb.indexing.loaders import load_document, load_documents


def test_load_txt_document(tmp_path: Path) -> None:
    """TXT files should load as plain-text documents."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("alpha beta gamma", encoding="utf-8")

    document = load_document(file_path)

    assert document.document_id == "notes"
    assert document.metadata["format"] == "text"
    assert document.content == "alpha beta gamma"


def test_load_markdown_document(tmp_path: Path) -> None:
    """Markdown files should load with markdown metadata."""
    file_path = tmp_path / "guide.md"
    file_path.write_text("# Title\n\nA paragraph.", encoding="utf-8")

    document = load_document(file_path)

    assert document.document_id == "guide"
    assert document.metadata["format"] == "markdown"
    assert document.content.startswith("# Title")


def test_load_jsonl_structured_records_as_multiple_documents(tmp_path: Path) -> None:
    """JSONL structured sources should expand into one field-aware document per record."""
    file_path = tmp_path / "pets.jsonl"
    file_path.write_text(
        '{"id": "cat-1", "animal": "cat", "trait": "playful", "favorite_food": "salmon"}\n'
        '{"id": "dog-1", "animal": "dog", "trait": "loyal", "favorite_food": "beef"}\n',
        encoding="utf-8",
    )

    documents = load_documents([file_path])

    assert [document.document_id for document in documents] == ["pets-cat_1", "pets-dog_1"]
    assert documents[0].metadata["format"] == "structured"
    assert documents[0].metadata["record_id"] == "cat-1"
    assert documents[0].metadata["field_animal"] == "cat"
    assert "animal: cat" in documents[0].content
    assert "favorite_food: salmon" in documents[0].content


def test_load_jsonl_rejects_invalid_line_with_clear_error(tmp_path: Path) -> None:
    """Malformed JSONL lines should still fail with a stable user-facing error."""
    file_path = tmp_path / "broken.jsonl"
    file_path.write_text('{"id": "cat-1", "animal": "cat"}\n{not valid json}\n', encoding="utf-8")

    with pytest.raises(DocumentLoadError) as error:
        load_documents([file_path])

    assert "contains invalid JSON" in str(error.value)


def test_unsupported_file_type_has_clear_error(tmp_path: Path) -> None:
    """Unsupported file types should raise a user-facing error."""
    file_path = tmp_path / "data.pdf"
    file_path.write_text("binary-ish", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError) as error:
        load_document(file_path)

    assert "Unsupported file type" in str(error.value)
    assert ".txt, .md, .markdown, .json, .jsonl" in str(error.value)


def test_empty_document_has_clear_error(tmp_path: Path) -> None:
    """Empty files should fail with an actionable error message."""
    file_path = tmp_path / "empty.md"
    file_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(DocumentLoadError) as error:
        load_document(file_path)

    assert "is empty and cannot be indexed" in str(error.value)


def test_load_document_rejects_files_larger_than_configured_limit(tmp_path: Path) -> None:
    """Source documents larger than the configured byte limit should fail clearly."""
    file_path = tmp_path / "large.txt"
    file_path.write_text("abcdefghij", encoding="utf-8")

    with pytest.raises(DocumentLoadError) as error:
        load_document(file_path, max_document_bytes=5)

    assert "configured maximum size of 5 bytes" in str(error.value)


def test_load_documents_accepts_in_range_document_sizes(tmp_path: Path) -> None:
    """Document loading should stay unchanged for documents inside the configured size limit."""
    file_path = tmp_path / "small.txt"
    file_path.write_text("alpha beta", encoding="utf-8")

    documents = load_documents([file_path], max_document_bytes=100)

    assert [document.document_id for document in documents] == ["small"]


def test_load_document_rejects_paths_outside_allowed_root(tmp_path: Path) -> None:
    """Allowed-root restrictions should reject source documents outside the configured root."""
    file_path = tmp_path / "outside.txt"
    file_path.write_text("alpha beta", encoding="utf-8")
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    with pytest.raises(DocumentLoadError) as error:
        load_document(file_path, allowed_root=allowed_root)

    assert "outside the configured allowed root" in str(error.value)


def test_chunk_document_uses_overlap_boundaries() -> None:
    """Chunking should preserve deterministic overlap across chunk boundaries."""
    document = load_document(_write_doc("abcdefghij"))

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.content for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]


def test_chunk_document_uses_paragraph_structure_when_available(tmp_path: Path) -> None:
    """Paragraph-aware chunking should keep related paragraphs together when possible."""
    file_path = tmp_path / "guide.md"
    file_path.write_text(
        "First paragraph.\n\nSecond paragraph is here.\n\nThird paragraph closes.",
        encoding="utf-8",
    )
    document = load_document(file_path)

    chunks = chunk_document(document, chunk_size=45, chunk_overlap=18)

    assert len(chunks) == 2
    assert chunks[0].content == "First paragraph.\n\nSecond paragraph is here."
    assert chunks[1].content == "Third paragraph closes."
    assert chunks[0].metadata["paragraph_count"] == 2


def test_chunk_document_prefers_sentence_boundaries_for_qa_friendly_chunks(tmp_path: Path) -> None:
    """Sentence-aware chunking should keep short related sentences together."""
    file_path = tmp_path / "animals.txt"
    file_path.write_text(
        "Cats nap in warm sunlight. Dogs patrol the yard. Cats stretch by the window.",
        encoding="utf-8",
    )
    document = load_document(file_path)

    chunks = chunk_document(document, chunk_size=55, chunk_overlap=18)

    assert [chunk.content for chunk in chunks] == [
        "Cats nap in warm sunlight. Dogs patrol the yard.",
        "Cats stretch by the window.",
    ]
    assert chunks[0].metadata["splitter"] == "sentence_aware"
    assert chunks[0].metadata["sentence_count"] == 2


def test_chunk_document_keeps_structured_records_intact(tmp_path: Path) -> None:
    """Structured records should stay as one field-aware chunk instead of sentence chunking."""
    file_path = tmp_path / "pets.json"
    file_path.write_text(
        '[{"id": "cat-1", "animal": "cat", "trait": "playful", "favorite_food": "salmon"}]',
        encoding="utf-8",
    )
    document = load_documents([file_path])[0]

    chunks = chunk_document(document, chunk_size=20, chunk_overlap=5)

    assert len(chunks) == 1
    assert chunks[0].metadata["splitter"] == "structured_record"
    assert chunks[0].metadata["field_favorite_food"] == "salmon"
    assert "trait: playful" in chunks[0].content


def test_chunk_document_falls_back_when_sentence_boundaries_are_too_large() -> None:
    """Sentence-aware chunking should safely fall back when one sentence is oversized."""
    document = load_document(_write_doc("abcdefghij"))

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.content for chunk in chunks] == ["abcd", "defg", "ghij"]


def _write_doc(content: str) -> Path:
    """Create a deterministic temporary document for simple chunking tests."""
    temporary_directory = tempfile.TemporaryDirectory()
    path = Path(temporary_directory.name) / "chunk-source.txt"
    path.write_text(content, encoding="utf-8")
    _TEMPORARY_DIRECTORIES.append(temporary_directory)
    return path


_TEMPORARY_DIRECTORIES: list[tempfile.TemporaryDirectory[str]] = []
