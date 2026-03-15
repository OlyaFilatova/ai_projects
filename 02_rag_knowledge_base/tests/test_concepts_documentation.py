"""Concept-documentation generator checks."""

from __future__ import annotations

from pathlib import Path

from rag_knb.concepts_documentation import (
    build_concept_mappings,
    render_concepts_document,
    write_concepts_document,
)


def test_render_concepts_document_maps_key_rag_areas() -> None:
    """The generated Markdown should map core concepts to code and test paths."""
    rendered = render_concepts_document()

    assert "# RAG concepts in this codebase" in rendered
    assert "## Retrieval backends, hybrid ranking, and reranking (covered)" in rendered
    assert "`src/rag_knb/retrieval_engine/retrieval.py::Retriever.search_with_plan`" in rendered
    assert "`tests/test_evaluation_harness.py`" in rendered
    assert "## Still-missing concepts (missing)" in rendered


def test_write_concepts_document_creates_markdown_file(tmp_path: Path) -> None:
    """The writer should persist the generated Markdown to the requested path."""
    output_path = tmp_path / "rag_concepts.md"

    written_path = write_concepts_document(output_path)

    assert written_path == output_path
    written_text = output_path.read_text(encoding="utf-8")
    assert "Context compression and answer planning (partial)" in written_text
    assert "Long-horizon dialogue memory and richer conversational state management." in written_text


def test_build_concept_mappings_exposes_honest_status_levels() -> None:
    """The concept catalog should include covered, partial, and missing concepts."""
    statuses = {concept.status for concept in build_concept_mappings()}

    assert statuses == {"covered", "missing", "partial"}
