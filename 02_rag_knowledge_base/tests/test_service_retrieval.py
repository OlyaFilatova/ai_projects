"""Service-layer retrieval tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rag_knb.answers.answer_results import build_clarification_needed_answer
from rag_knb.config import RuntimeConfig
from rag_knb.errors import DocumentLoadError, ValidationError
from rag_knb.models import Chunk, ConversationTurn, RetrievalResult
from rag_knb.service import KnowledgeBaseService


def test_service_can_ingest_and_answer_from_indexed_content(tmp_path: Path) -> None:
    """The service should answer from the most relevant indexed chunks."""
    source_path = tmp_path / "animals.txt"
    source_path.write_text(
        "Cats are curious animals.\nDogs enjoy walks.\nCats like sunny windows.",
        encoding="utf-8",
    )
    service = KnowledgeBaseService(config=RuntimeConfig.build(answer_verbosity="verbose"))

    ingest_result = service.ingest_paths([source_path])
    answer = service.ask("Which animals like sunny windows?")

    assert len(ingest_result.documents) == 1
    assert ingest_result.chunks
    assert answer.matched is True
    assert answer.reason == "matched"
    assert answer.diagnostics["match_count"] >= 1
    assert answer.diagnostics["matched_document_ids"] == ["animals"]
    assert answer.diagnostics["matches"][0]["document_id"] == "animals"
    assert "Cats like sunny windows." in answer.answer_text
    assert answer.matches[0].score > 0


def test_service_returns_explicit_no_match_response(tmp_path: Path) -> None:
    """Queries with no shared terms should produce a clear grounded failure response."""
    source_path = tmp_path / "science.txt"
    source_path.write_text("Stars emit light.", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(answer_verbosity="verbose"))
    service.ingest_paths([source_path])

    answer = service.ask("How do carrots grow underground?")

    assert answer.matched is False
    assert answer.reason == "no_match"
    assert answer.diagnostics["top_score"] == 0.0
    assert "No grounded answer found" in answer.answer_text


def test_service_can_filter_retrieval_by_metadata(tmp_path: Path) -> None:
    """Retrieval filters should restrict matches to the requested document metadata."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats sleep in sunny spots.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs sleep near the door.", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(answer_verbosity="verbose"))
    service.ingest_paths([cats_path, dogs_path])

    answer = service.ask("Cats sleep", metadata_filters={"file_name": "cats.txt"})

    assert answer.matched is True
    assert "Cats sleep in sunny spots." in answer.answer_text
    assert all(match.chunk.metadata["file_name"] == "cats.txt" for match in answer.matches)


def test_service_can_retrieve_structured_records_by_field_content(tmp_path: Path) -> None:
    """Structured record fields should participate in retrieval and metadata filtering."""
    records_path = tmp_path / "pets.jsonl"
    records_path.write_text(
        '{"id": "cat-1", "animal": "cat", "trait": "playful", "favorite_food": "salmon"}\n'
        '{"id": "dog-1", "animal": "dog", "trait": "loyal", "favorite_food": "beef"}\n',
        encoding="utf-8",
    )
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("Birds build nests in trees.", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(answer_verbosity="verbose"))
    service.ingest_paths([records_path, notes_path])

    answer = service.ask("Which animal likes salmon?")
    filtered_answer = service.ask("playful", metadata_filters={"field_animal": "cat"})

    assert answer.reason == "matched"
    assert "animal: cat" in answer.answer_text
    assert "favorite_food: salmon" in answer.answer_text
    assert filtered_answer.reason == "matched"
    assert all(match.chunk.metadata["field_animal"] == "cat" for match in filtered_answer.matches)


def test_service_can_apply_metadata_aware_ranking_in_mixed_corpora(tmp_path: Path) -> None:
    """Metadata-aware ranking should help structured records win when the query asks for one."""
    records_path = tmp_path / "animals.jsonl"
    records_path.write_text(
        '{"id": "cat-record", "animal": "cat", "trait": "playful"}\n',
        encoding="utf-8",
    )
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("The cat is playful.", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(answer_verbosity="verbose"))
    service.ingest_paths([records_path, notes_path])

    answer = service.ask("Which structured cat record is playful?")

    assert answer.reason == "matched"
    assert answer.matches[0].chunk.metadata["format"] == "structured"
    assert answer.matches[0].chunk.metadata["record_id"] == "cat-record"


def test_service_deduplicates_duplicate_evidence_in_results(tmp_path: Path) -> None:
    """Near-duplicate documents should not overcount identical evidence in retrieval results."""
    first_path = tmp_path / "first.txt"
    first_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    second_path = tmp_path / "second.txt"
    second_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([first_path, second_path])

    answer = service.ask("Where do cats nap?", limit=3)

    assert answer.reason == "matched"
    assert len(answer.matches) == 1
    assert answer.answer_text.count("Cats nap in warm sunlight.") == 1


def test_service_can_use_optional_source_weight_when_scores_are_close(tmp_path: Path) -> None:
    """Explicit source weights should break close ties without changing exact filter behavior."""
    first_path = tmp_path / "first.txt"
    first_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    second_path = tmp_path / "second.txt"
    second_path.write_text("Cats nap in warm sunlight near the window.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([first_path, second_path])

    service._chunks[0].metadata["source_weight"] = 2.0
    service._vector_store.entries[0].chunk.metadata["source_weight"] = 2.0

    answer = service.ask("Cats nap in warm sunlight", limit=2)

    assert answer.reason == "matched"
    assert answer.matches[0].chunk.metadata["file_name"] == "first.txt"


def test_service_matches_natural_animal_queries_with_default_embedder(tmp_path: Path) -> None:
    """The default offline retrieval should handle simple natural-language variants."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats are energetic during the day.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs are loyal and like to play.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([cats_path, dogs_path])

    energetic_answer = service.ask("Which animal is energetic?")
    playful_answer = service.ask("playful animal")

    assert energetic_answer.reason == "matched"
    assert "Cats are energetic during the day." in energetic_answer.answer_text
    assert playful_answer.reason == "matched"
    assert "Dogs are loyal and like to play." in playful_answer.answer_text


def test_service_supports_small_synonym_style_query_expansions(tmp_path: Path) -> None:
    """The default offline retrieval should support a few safe synonym-style variants."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs like to play fetch.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([cats_path, dogs_path])

    feline_answer = service.ask("Which feline likes sunlight?")
    canine_answer = service.ask("Which canine is playful?")

    assert feline_answer.reason == "matched"
    assert "Cats nap in warm sunlight." in feline_answer.answer_text
    assert canine_answer.reason == "matched"
    assert "Dogs like to play fetch." in canine_answer.answer_text


def test_service_can_use_hybrid_retrieval_strategy(tmp_path: Path) -> None:
    """Hybrid retrieval should combine lexical and embedding signals deterministically."""
    source_path = tmp_path / "animals.txt"
    source_path.write_text(
        "Cats nap in warm sunlight.\nDogs are loyal and energetic playmates.",
        encoding="utf-8",
    )
    service = KnowledgeBaseService(config=RuntimeConfig.build(retrieval_strategy="hybrid"))
    service.ingest_paths([source_path])

    answer = service.ask("energetic playmate")

    assert answer.reason == "matched"
    assert "Dogs are loyal and energetic playmates." in answer.answer_text


def test_hybrid_retrieval_keeps_exact_match_documents_ranked_ahead_of_partial_noise(tmp_path: Path) -> None:
    """Hybrid retrieval should still prefer the exact lexical match in a noisy small corpus."""
    source_path = tmp_path / "animals.txt"
    source_path.write_text(
        (
            "Cats nap in warm sunlight.\n"
            "Dogs nap near the door.\n"
            "Lizards rest on heated rocks."
        ),
        encoding="utf-8",
    )
    service = KnowledgeBaseService(config=RuntimeConfig.build(retrieval_strategy="hybrid"))
    service.ingest_paths([source_path])

    answer = service.ask("Cats nap in warm sunlight")

    assert answer.reason == "matched"
    assert answer.matches[0].chunk.content.startswith("Cats nap in warm sunlight.")


def test_service_can_route_ambiguous_questions_to_clarification(tmp_path: Path) -> None:
    """Ambiguous top matches should return a clarification-needed response."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats are energetic and playful.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs are energetic and playful.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([cats_path, dogs_path])

    answer = service.ask("Which pet is playful?")

    assert answer.reason == "clarification_needed"
    assert "Do you mean document 'cats.txt' or document 'dogs.txt'?" in answer.answer_text


def test_clarification_questions_can_distinguish_mixed_corpora_sources() -> None:
    """Clarification prompts should distinguish structured records from text documents when both match."""
    answer = build_clarification_needed_answer(
        "Which cat is playful?",
        [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="pets-cat_1:0",
                    document_id="pets-cat_1",
                    content="animal: cat\ntrait: playful",
                    start_offset=0,
                    end_offset=26,
                    metadata={"record_id": "cat-1", "format": "structured"},
                ),
                score=0.8,
            ),
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="notes:0",
                    document_id="notes",
                    content="Cats are playful.",
                    start_offset=0,
                    end_offset=18,
                    metadata={"file_name": "notes.txt", "format": "text"},
                ),
                score=0.79,
            ),
        ],
    )

    assert answer.reason == "clarification_needed"
    assert "record 'cat-1'" in answer.answer_text
    assert "document 'notes.txt'" in answer.answer_text


def test_service_can_aggregate_multi_hop_evidence_across_documents(tmp_path: Path) -> None:
    """Explicit multi-part questions should merge grounded evidence from multiple documents."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs enjoy energetic play.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([cats_path, dogs_path])

    answer = service.ask("What do cats do and what do dogs do?")

    assert answer.reason == "matched"
    assert "Cats nap in warm sunlight. [cats:0]" in answer.answer_text
    assert "Dogs enjoy energetic play. [dogs:0]" in answer.answer_text
    assert [alignment["chunk_id"] for alignment in answer.diagnostics["claim_alignments"]] == ["cats:0", "dogs:0"]


def test_service_exposes_parent_context_for_child_matches(tmp_path: Path) -> None:
    """Diagnostics should expose parent document context alongside child chunk matches."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text(
        "Cats nap in warm sunlight. Dogs patrol the yard. Cats scratch old sofas.",
        encoding="utf-8",
    )
    service = KnowledgeBaseService(config=RuntimeConfig.build(chunk_size=40, chunk_overlap=5))
    service.ingest_paths([source_path])

    answer = service.ask("Where do cats nap?")

    assert answer.reason == "matched"
    assert answer.diagnostics["parent_context"][0]["parent_document_id"] == "notes"
    assert "Cats nap in warm sunlight." in answer.diagnostics["parent_context"][0]["parent_excerpt"]


def test_service_can_use_conversation_turns_for_follow_up_questions(tmp_path: Path) -> None:
    """Optional conversation state should help resolve follow-up questions."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs enjoy energetic play.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([cats_path, dogs_path])

    answer = service.ask(
        "Where do they nap?",
        conversation_turns=[ConversationTurn(question="Tell me about cats")],
    )

    assert answer.reason == "matched"
    assert "Cats nap in warm sunlight." in answer.answer_text
    assert answer.diagnostics["conversation_plan"]["mode"] == "continuation"


def test_service_can_plan_comparison_follow_ups_from_conversation(tmp_path: Path) -> None:
    """Comparison follow-ups should use conversation-aware planning and aggregate both subjects."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs enjoy energetic play.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([cats_path, dogs_path])

    answer = service.ask(
        "Compare them.",
        conversation_turns=[
            ConversationTurn(question="Tell me about cats"),
            ConversationTurn(question="What about dogs?"),
        ],
    )

    assert answer.reason == "matched"
    assert "Cats nap in warm sunlight. [cats:0]" in answer.answer_text
    assert "Dogs enjoy energetic play. [dogs:0]" in answer.answer_text
    assert answer.diagnostics["conversation_plan"]["mode"] == "comparison"


def test_service_can_remove_documents_from_the_knowledge_base(tmp_path: Path) -> None:
    """Removing a document should also remove its chunks from retrieval."""
    keep_path = tmp_path / "keep.txt"
    keep_path.write_text("Keep this document.", encoding="utf-8")
    drop_path = tmp_path / "drop.txt"
    drop_path.write_text("Drop this document.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([keep_path, drop_path])

    remaining_documents = service.remove_documents(["drop"])

    assert [document.document_id for document in remaining_documents] == ["keep"]
    answer = service.ask("Drop")
    assert answer.reason in {"no_match", "low_confidence"}


def test_service_emits_query_logs_and_timing_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Query execution should emit structured logs and timing diagnostics."""
    source_path = tmp_path / "timing.txt"
    source_path.write_text("Mercury is the closest planet to the Sun.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([source_path])

    with caplog.at_level(logging.INFO):
        answer = service.ask("Which planet is closest?")

    assert answer.diagnostics["retrieval_duration_ms"] >= 0.0
    assert '"event": "query_completed"' in caplog.text


def test_service_rewrites_filler_heavy_questions_in_diagnostics(tmp_path: Path) -> None:
    """Query diagnostics should expose deterministic rewriting for filler-heavy questions."""
    source_path = tmp_path / "cats.txt"
    source_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([source_path])

    answer = service.ask("Can you tell me what cats nap in?")

    assert answer.reason == "matched"
    assert answer.diagnostics["original_question"] == "Can you tell me what cats nap in?"
    assert answer.diagnostics["rewritten_question"] == "what cats nap in"
    assert answer.diagnostics["retrieval_queries"] == ["what cats nap in"]


def test_service_can_decompose_simple_two_part_questions(tmp_path: Path) -> None:
    """Simple two-part questions should search via a small decomposed query plan."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats nap in warm sunlight.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs enjoy energetic play.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([cats_path, dogs_path])

    answer = service.ask("Cats nap and dogs play")

    assert answer.reason == "matched"
    assert answer.diagnostics["retrieval_queries"] == ["Cats nap", "dogs play"]
    assert {match.chunk.document_id for match in answer.matches} == {"cats", "dogs"}


def test_service_rejects_questions_longer_than_configured_limit(tmp_path: Path) -> None:
    """Question-length guardrails should fail clearly through the shared service layer."""
    source_path = tmp_path / "limits.txt"
    source_path.write_text("Cats nap in sunny spots.", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(max_question_length=10))
    service.ingest_paths([source_path])

    with pytest.raises(ValidationError) as error:
        service.ask("This question is definitely too long.")

    assert "configured maximum of 10 characters" in str(error.value)


def test_service_rejects_retrieval_limit_above_configured_maximum(tmp_path: Path) -> None:
    """Retrieval-limit guardrails should fail clearly through the shared service layer."""
    source_path = tmp_path / "limits.txt"
    source_path.write_text("Cats nap in sunny spots.", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(max_retrieval_limit=2))
    service.ingest_paths([source_path])

    with pytest.raises(ValidationError) as error:
        service.ask("Cats nap", limit=3)

    assert "configured maximum of 2" in str(error.value)


def test_service_rejects_too_many_documents_in_one_ingest_call(tmp_path: Path) -> None:
    """Document-count ingest guardrails should fail clearly through the service layer."""
    first_path = tmp_path / "first.txt"
    first_path.write_text("alpha", encoding="utf-8")
    second_path = tmp_path / "second.txt"
    second_path.write_text("beta", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(max_documents_per_ingest=1))

    with pytest.raises(ValidationError) as error:
        service.ingest_paths([first_path, second_path])

    assert "configured maximum of 1" in str(error.value)


def test_service_rejects_documents_larger_than_configured_byte_limit(tmp_path: Path) -> None:
    """Document-size ingest guardrails should fail clearly through the service layer."""
    source_path = tmp_path / "large.txt"
    source_path.write_text("abcdefghij", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(max_document_bytes=5))

    with pytest.raises(DocumentLoadError) as error:
        service.ingest_paths([source_path])

    assert "configured maximum size of 5 bytes" in str(error.value)


def test_service_rejects_data_dirs_outside_allowed_root(tmp_path: Path) -> None:
    """Persistence targets outside the configured root should fail clearly."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source_path = allowed_root / "notes.txt"
    source_path.write_text("Cats nap in sunny spots.", encoding="utf-8")
    service = KnowledgeBaseService(config=RuntimeConfig.build(allowed_root=allowed_root))
    service.ingest_paths([source_path])

    with pytest.raises(ValidationError) as error:
        service.save(tmp_path / "outside-kb")

    assert "outside the configured allowed root" in str(error.value)


def test_service_rejects_chunk_counts_above_configured_ingest_budget(tmp_path: Path) -> None:
    """Chunk-budget ingest guardrails should fail clearly through the service layer."""
    source_path = tmp_path / "many-chunks.txt"
    source_path.write_text("abcdefghij", encoding="utf-8")
    service = KnowledgeBaseService(
        config=RuntimeConfig.build(chunk_size=4, chunk_overlap=1, max_chunks_per_ingest=2)
    )

    with pytest.raises(ValidationError) as error:
        service.ingest_paths([source_path])

    assert "configured maximum of 2" in str(error.value)
