"""Deterministic retrieval regression checks."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from rag_knb.service import KnowledgeBaseService
from tests.snapshot import assert_json_snapshot


def test_retrieval_returns_expected_top_documents_for_reference_queries(tmp_path: Path) -> None:
    """Reference queries should keep returning the expected top document IDs."""
    astronomy_path = tmp_path / "astronomy.txt"
    astronomy_path.write_text("Jupiter is the largest planet in the solar system.", encoding="utf-8")
    botany_path = tmp_path / "botany.txt"
    botany_path.write_text("Sunflowers turn toward bright sunlight.", encoding="utf-8")
    history_path = tmp_path / "history.txt"
    history_path.write_text("The printing press changed how books spread.", encoding="utf-8")
    service = KnowledgeBaseService()
    service.ingest_paths([astronomy_path, botany_path, history_path])

    astronomy_answer = service.ask("Which planet is the largest?")
    botany_answer = service.ask("Which flowers turn toward sunlight?")
    history_answer = service.ask("What changed how books spread?")

    assert_json_snapshot(
        "retrieval_top_documents",
        [
            astronomy_answer.matches[0].chunk.document_id,
            botany_answer.matches[0].chunk.document_id,
            history_answer.matches[0].chunk.document_id,
        ],
    )


def test_retrieval_latency_stays_within_local_regression_budget(tmp_path: Path) -> None:
    """Small deterministic workloads should stay within a generous local timing budget."""
    source_paths: list[Path] = []
    for index in range(20):
        source_path = tmp_path / f"doc-{index}.txt"
        source_path.write_text(
            f"Document {index} contains repeated topic tokens about planets and stars {index}.",
            encoding="utf-8",
        )
        source_paths.append(source_path)
    service = KnowledgeBaseService()
    service.ingest_paths(source_paths)

    started_at = perf_counter()
    for _ in range(25):
        answer = service.ask("Which document talks about planets and stars?")
        assert answer.reason in {"matched", "low_confidence"}
    average_query_ms = ((perf_counter() - started_at) * 1000) / 25

    assert average_query_ms < 50.0


def test_reranking_prioritizes_the_most_direct_question_match(tmp_path: Path) -> None:
    """A wider candidate pool should be reranked toward the most direct answer chunk."""
    broad_path = tmp_path / "broad.txt"
    broad_path.write_text(
        "Cats are energetic and curious animals that nap in sunlight.",
        encoding="utf-8",
    )
    direct_path = tmp_path / "direct.txt"
    direct_path.write_text(
        "Dogs are loyal and energetic playmates.",
        encoding="utf-8",
    )
    service = KnowledgeBaseService()
    service.ingest_paths([broad_path, direct_path])

    answer = service.ask("energetic playmate")

    assert answer.reason == "matched"
    assert answer.matches[0].chunk.document_id == "direct"
