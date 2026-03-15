"""Fixture-driven evaluation harness checks."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from rag_knb.errors import DependencyUnavailableError
from rag_knb.models import AnswerResult, Chunk, RetrievalResult
from rag_knb.retrieval_engine import evaluation as evaluation_module
from rag_knb.retrieval_engine.evaluation import (
    EvaluationCase,
    compare_retrieval_strategies,
    evaluate_answer,
    summarize_results,
    summarize_results_by_group,
)
from rag_knb.service import KnowledgeBaseService


def test_evaluation_harness_scores_local_quality_signals(tmp_path: Path) -> None:
    """The lightweight evaluation harness should score broader retrieval and answer traits."""
    fixture_path = Path("tests/fixtures/evaluation_cases.json")
    fixture_cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    evaluation_cases: list[EvaluationCase] = []
    evaluation_results = []

    for fixture_case in fixture_cases:
        document_paths: list[Path] = []
        for file_name, content in fixture_case["documents"].items():
            path = tmp_path / fixture_case["name"] / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            document_paths.append(path)

        service = KnowledgeBaseService()
        service.ingest_paths(document_paths)
        answer = service.ask(fixture_case["question"])
        evaluation_case = EvaluationCase(
            name=fixture_case["name"],
            question=fixture_case["question"],
            expected_document_ids=fixture_case["expected_document_ids"],
            expected_answer_substrings=fixture_case["expected_answer_substrings"],
            relevant_document_ids=fixture_case["relevant_document_ids"],
            expected_reason=fixture_case.get("expected_reason", "matched"),
            expected_support_substrings=fixture_case.get("expected_support_substrings", []),
            benchmark_group=fixture_case.get("benchmark_group", "general"),
        )
        evaluation_cases.append(evaluation_case)
        evaluation_results.append(evaluate_answer(evaluation_case, answer))

    summary = summarize_results(evaluation_results)
    grouped_summary = summarize_results_by_group(evaluation_cases, evaluation_results)

    assert summary["case_count"] == 7.0
    assert summary["retrieval_relevance"] == 1.0
    assert summary["answer_focus"] == 1.0
    assert summary["support_visibility"] == pytest.approx(5 / 7)
    assert summary["citation_quality"] == 1.0
    assert summary["reason_accuracy"] == 1.0
    assert summary["support_coverage"] == 1.0
    assert summary["clarification_accuracy"] == 1.0
    assert summary["low_confidence_accuracy"] == 1.0
    assert summary["precision_at_k"] == pytest.approx(11 / 14)
    assert summary["recall_at_k"] == pytest.approx(6 / 7)
    assert summary["mrr"] == pytest.approx(6 / 7)
    assert grouped_summary["retrieval"]["case_count"] == 3.0
    assert grouped_summary["routing"]["reason_accuracy"] == 1.0
    assert grouped_summary["routing"]["mrr"] == 0.5
    assert grouped_summary["safety"]["citation_quality"] == 1.0


def test_evaluation_metrics_handle_partial_rankings_and_reason_routing() -> None:
    """Evaluation metrics should expose ranking quality and routed-answer correctness."""
    summary = summarize_results(
        [
            evaluate_answer(
                EvaluationCase(
                    name="partial",
                    question="Q",
                    expected_document_ids=["dogs"],
                    relevant_document_ids=["dogs", "cats"],
                    expected_answer_substrings=["dogs"],
                    expected_support_substrings=["dogs"],
                    expected_reason="low_confidence",
                ),
                AnswerResult(
                    answer_text="Low-confidence retrieval for 'Q'. Review the supporting chunks before trusting an answer.",
                    matches=[
                        RetrievalResult(
                            chunk=Chunk(
                                chunk_id="dogs:0",
                                document_id="dogs",
                                content="dogs",
                                start_offset=0,
                                end_offset=4,
                            ),
                            score=1.0,
                        ),
                        RetrievalResult(
                            chunk=Chunk(
                                chunk_id="birds:0",
                                document_id="birds",
                                content="birds",
                                start_offset=0,
                                end_offset=5,
                            ),
                            score=0.5,
                        ),
                    ],
                    matched=False,
                    reason="low_confidence",
                    diagnostics={"claim_alignments": []},
                ),
            )
        ]
    )

    assert summary["reason_accuracy"] == 1.0
    assert summary["low_confidence_accuracy"] == 1.0
    assert summary["precision_at_k"] == 0.5
    assert summary["recall_at_k"] == 0.5
    assert summary["mrr"] == 1.0


def test_retrieval_strategy_comparison_harness_surfaces_strategy_tradeoffs(tmp_path: Path) -> None:
    """The comparison harness should expose raw-vs-reranked tradeoffs across strategies."""
    fixture_cases = [
        {
            "name": "pet_record_tradeoff",
            "question": "pet record",
            "documents": {
                "animals.jsonl": (
                    '{"id": "cat-record", "animal": "cat", "trait": "playful"}\n'
                    '{"id": "dog-record", "animal": "dog", "trait": "loyal"}\n'
                ),
                "notes.txt": "The household pet sleeps indoors.",
            },
            "expected_document_ids": ["animals-dog_record"],
            "expected_answer_substrings": [],
            "relevant_document_ids": ["animals-dog_record", "animals-cat_record"],
            "expected_reason": "clarification_needed",
            "benchmark_group": "retrieval",
        }
    ]

    comparisons = compare_retrieval_strategies(
        fixture_cases,
        workspace=tmp_path,
        strategy_overrides={
            "vector": {"retrieval_strategy": "vector"},
            "hybrid": {"retrieval_strategy": "hybrid"},
        },
    )

    by_name = {comparison.strategy_name: comparison for comparison in comparisons}

    assert sorted(by_name) == ["hybrid", "vector"]
    assert by_name["vector"].status == "available"
    assert by_name["hybrid"].status == "available"
    assert by_name["vector"].summary["case_count"] == 1.0
    assert by_name["hybrid"].summary["case_count"] == 1.0
    assert len(by_name["vector"].case_results) == 1
    assert len(by_name["hybrid"].case_results) == 1

    vector_case = by_name["vector"].case_results[0]
    hybrid_case = by_name["hybrid"].case_results[0]

    assert vector_case.case_name == "pet_record_tradeoff"
    assert vector_case.answer_reason == "low_confidence"
    assert vector_case.candidate_document_ids == [
        "animals-dog_record",
        "animals-cat_record",
        "notes",
    ]
    assert vector_case.reranked_document_ids == [
        "animals-dog_record",
        "animals-cat_record",
        "notes",
    ]
    assert hybrid_case.candidate_document_ids == [
        "notes",
        "animals-dog_record",
        "animals-cat_record",
    ]
    assert hybrid_case.answer_reason == "matched"
    assert hybrid_case.reranked_document_ids == [
        "animals-dog_record",
        "animals-cat_record",
        "notes",
    ]


def test_retrieval_strategy_comparison_marks_unavailable_optional_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional backends should be surfaced as unavailable without breaking the comparison run."""
    original_builder = cast(
        Callable[[dict[str, Any]], KnowledgeBaseService],
        evaluation_module._build_service_for_strategy,
    )

    def fake_builder(overrides: dict[str, object]) -> KnowledgeBaseService:
        if overrides.get("embedding_backend") == "huggingface":
            raise DependencyUnavailableError("missing optional semantic backend")
        return original_builder(overrides)

    monkeypatch.setattr(
        evaluation_module,
        "_build_service_for_strategy",
        fake_builder,
    )

    comparisons = compare_retrieval_strategies(
        [
            {
                "name": "simple",
                "question": "Cats nap",
                "documents": {"cats.txt": "Cats nap in warm sunlight."},
                "expected_document_ids": ["cats"],
                "expected_answer_substrings": ["Cats nap"],
                "relevant_document_ids": ["cats"],
                "benchmark_group": "retrieval",
            }
        ],
        workspace=tmp_path,
        strategy_overrides={
            "vector": {"retrieval_strategy": "vector"},
            "semantic_hybrid": {
                "retrieval_strategy": "hybrid",
                "embedding_backend": "huggingface",
                "vector_backend": "faiss",
            },
        },
    )

    by_name = {comparison.strategy_name: comparison for comparison in comparisons}

    assert by_name["vector"].status == "available"
    assert by_name["vector"].summary["case_count"] == 1.0
    assert len(by_name["vector"].case_results) == 1
    assert by_name["semantic_hybrid"].status == "unavailable"
    assert by_name["semantic_hybrid"].summary == {}
    assert by_name["semantic_hybrid"].case_results == []
