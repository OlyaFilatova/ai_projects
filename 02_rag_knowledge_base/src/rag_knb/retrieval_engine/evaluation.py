"""Lightweight evaluation helpers for local retrieval and answer quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_knb.config import RuntimeConfig
from rag_knb.errors import DependencyUnavailableError
from rag_knb.models import AnswerResult, RetrievalResult
from rag_knb.retrieval_engine.query_rewriting import build_query_plan

if TYPE_CHECKING:
    from rag_knb.service import KnowledgeBaseService


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One local evaluation case and its expected answer traits."""

    name: str
    question: str
    expected_document_ids: list[str]
    expected_answer_substrings: list[str]
    relevant_document_ids: list[str]
    expected_reason: str = "matched"
    expected_support_substrings: list[str] = field(default_factory=list)
    benchmark_group: str = "general"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Scored result for one evaluation case."""

    name: str
    retrieval_relevant: bool
    answer_focused: bool
    support_visible: bool
    citation_quality: bool
    reason_correct: bool
    support_coverage: float
    clarification_correct: bool
    low_confidence_correct: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalStrategyCaseComparison:
    """Per-case retrieval comparison details for one strategy."""

    case_name: str
    answer_reason: str
    candidate_document_ids: list[str]
    reranked_document_ids: list[str]


@dataclass(frozen=True, slots=True)
class RetrievalStrategyComparison:
    """One local comparison summary for a retrieval strategy."""

    strategy_name: str
    status: str
    summary: dict[str, float]
    case_results: list[RetrievalStrategyCaseComparison]


def evaluate_answer(case: EvaluationCase, answer: AnswerResult) -> EvaluationResult:
    """Score one answer against a small set of local quality checks."""
    top_document_id = answer.matches[0].chunk.document_id if answer.matches else None
    ranked_document_ids = list(dict.fromkeys(match.chunk.document_id for match in answer.matches))
    retrieval_relevant = (
        top_document_id in case.expected_document_ids if case.expected_document_ids else not answer.matches
    )
    answer_focused = all(
        expected_snippet.lower() in answer.answer_text.lower()
        for expected_snippet in case.expected_answer_substrings
    ) and answer.answer_text.count("\n- ") <= 3
    support_visible = _support_visible(answer)
    citation_quality = _citation_quality(answer, case.expected_support_substrings)
    reason_correct = answer.reason == case.expected_reason
    support_coverage = _support_coverage(answer, case.expected_support_substrings)
    clarification_correct = (
        True if case.expected_reason != "clarification_needed" else answer.reason == "clarification_needed"
    )
    low_confidence_correct = (
        True if case.expected_reason != "low_confidence" else answer.reason == "low_confidence"
    )
    precision_at_k = _precision_at_k(ranked_document_ids, case.relevant_document_ids)
    recall_at_k = _recall_at_k(ranked_document_ids, case.relevant_document_ids)
    reciprocal_rank = _reciprocal_rank(ranked_document_ids, case.relevant_document_ids)
    return EvaluationResult(
        name=case.name,
        retrieval_relevant=retrieval_relevant,
        answer_focused=answer_focused,
        support_visible=support_visible,
        citation_quality=citation_quality,
        reason_correct=reason_correct,
        support_coverage=support_coverage,
        clarification_correct=clarification_correct,
        low_confidence_correct=low_confidence_correct,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
    )


def summarize_results(results: list[EvaluationResult]) -> dict[str, float]:
    """Summarize a batch of local evaluation results."""
    if not results:
        return {
            "retrieval_relevance": 0.0,
            "answer_focus": 0.0,
            "support_visibility": 0.0,
            "citation_quality": 0.0,
            "reason_accuracy": 0.0,
            "support_coverage": 0.0,
            "clarification_accuracy": 0.0,
            "low_confidence_accuracy": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
        }
    total = len(results)
    return {
        "case_count": float(total),
        "retrieval_relevance": sum(result.retrieval_relevant for result in results) / total,
        "answer_focus": sum(result.answer_focused for result in results) / total,
        "support_visibility": sum(result.support_visible for result in results) / total,
        "citation_quality": sum(result.citation_quality for result in results) / total,
        "reason_accuracy": sum(result.reason_correct for result in results) / total,
        "support_coverage": sum(result.support_coverage for result in results) / total,
        "clarification_accuracy": sum(result.clarification_correct for result in results) / total,
        "low_confidence_accuracy": sum(result.low_confidence_correct for result in results) / total,
        "precision_at_k": sum(result.precision_at_k for result in results) / total,
        "recall_at_k": sum(result.recall_at_k for result in results) / total,
        "mrr": sum(result.reciprocal_rank for result in results) / total,
    }


def summarize_results_by_group(
    cases: list[EvaluationCase],
    results: list[EvaluationResult],
) -> dict[str, dict[str, float]]:
    """Summarize evaluation results by benchmark group."""
    results_by_name = {result.name: result for result in results}
    grouped_results: dict[str, list[EvaluationResult]] = {}
    for case in cases:
        result = results_by_name.get(case.name)
        if result is None:
            continue
        grouped_results.setdefault(case.benchmark_group, []).append(result)
    return {
        group_name: summarize_results(group_results)
        for group_name, group_results in grouped_results.items()
    }


def compare_retrieval_strategies(
    fixture_cases: list[dict[str, Any]],
    *,
    workspace: Path,
    strategy_overrides: dict[str, dict[str, Any]],
) -> list[RetrievalStrategyComparison]:
    """Run the local fixture cases across multiple retrieval strategy configurations."""
    comparisons: list[RetrievalStrategyComparison] = []
    for strategy_name, overrides in strategy_overrides.items():
        try:
            evaluation_results: list[EvaluationResult] = []
            case_results: list[RetrievalStrategyCaseComparison] = []
            for fixture_case in fixture_cases:
                document_paths = _materialize_fixture_case(workspace / strategy_name, fixture_case)
                service = _build_service_for_strategy(overrides)
                service.ingest_paths(document_paths)
                raw_matches = _candidate_matches_for_case(service, fixture_case["question"], limit=3)
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
                evaluation_results.append(evaluate_answer(evaluation_case, answer))
                case_results.append(
                    RetrievalStrategyCaseComparison(
                        case_name=fixture_case["name"],
                        answer_reason=answer.reason,
                        candidate_document_ids=_ordered_document_ids(raw_matches),
                        reranked_document_ids=_ordered_document_ids(answer.matches),
                    )
                )
            comparisons.append(
                RetrievalStrategyComparison(
                    strategy_name=strategy_name,
                    status="available",
                    summary=summarize_results(evaluation_results),
                    case_results=case_results,
                )
            )
        except DependencyUnavailableError:
            comparisons.append(
                RetrievalStrategyComparison(
                    strategy_name=strategy_name,
                    status="unavailable",
                    summary={},
                    case_results=[],
                )
            )
    return comparisons


def _support_visible(answer: AnswerResult) -> bool:
    """Return whether the answer exposes visible grounded support."""
    if not answer.matches:
        return False
    return "[" in answer.answer_text or bool(answer.diagnostics.get("claim_alignments"))


def _citation_quality(answer: AnswerResult, expected_support_substrings: list[str]) -> bool:
    """Return whether visible citations align with the expected support snippets."""
    if not expected_support_substrings:
        return True
    if "[" not in answer.answer_text and not answer.diagnostics.get("claim_alignments"):
        return False
    return _support_coverage(answer, expected_support_substrings) == 1.0


def _support_coverage(answer: AnswerResult, expected_support_substrings: list[str]) -> float:
    """Calculate how much expected support is exposed in the answer or diagnostics."""
    if not expected_support_substrings:
        return 1.0
    support_text = answer.answer_text.lower()
    claim_alignments = answer.diagnostics.get("claim_alignments", [])
    if isinstance(claim_alignments, list):
        support_text = " ".join(
            [
                support_text,
                *[
                    str(alignment.get("support_sentence", "")).lower()
                    for alignment in claim_alignments
                    if isinstance(alignment, dict)
                ],
            ]
        )
    supported = sum(expected.lower() in support_text for expected in expected_support_substrings)
    return supported / len(expected_support_substrings)


def _precision_at_k(ranked_document_ids: list[str], relevant_document_ids: list[str]) -> float:
    """Calculate precision at k over ranked unique document ids."""
    if not ranked_document_ids:
        return 0.0
    relevant_hits = sum(document_id in relevant_document_ids for document_id in ranked_document_ids)
    return relevant_hits / len(ranked_document_ids)


def _recall_at_k(ranked_document_ids: list[str], relevant_document_ids: list[str]) -> float:
    """Calculate recall at k over ranked unique document ids."""
    if not relevant_document_ids:
        return 0.0
    relevant_hits = sum(document_id in ranked_document_ids for document_id in relevant_document_ids)
    return relevant_hits / len(relevant_document_ids)


def _reciprocal_rank(ranked_document_ids: list[str], relevant_document_ids: list[str]) -> float:
    """Calculate reciprocal rank for the first relevant document."""
    for index, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant_document_ids:
            return 1 / index
    return 0.0


def _materialize_fixture_case(base_dir: Path, fixture_case: dict[str, Any]) -> list[Path]:
    """Write one fixture case corpus to disk and return the materialized paths."""
    document_paths: list[Path] = []
    for file_name, content in fixture_case["documents"].items():
        path = base_dir / fixture_case["name"] / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
        document_paths.append(path)
    return document_paths


def _build_service_for_strategy(overrides: dict[str, Any]) -> KnowledgeBaseService:
    """Build one service instance for a strategy comparison run."""
    from rag_knb.service import KnowledgeBaseService

    return KnowledgeBaseService(config=RuntimeConfig.build(**overrides))


def _candidate_matches_for_case(
    service: KnowledgeBaseService,
    question: str,
    *,
    limit: int,
) -> list[RetrievalResult]:
    """Return the merged candidate matches before question-focused reranking."""
    query_plan = build_query_plan(question)
    retriever = service._retriever
    merged_by_chunk_id: dict[str, RetrievalResult] = {}
    candidate_limit = max(limit * 3, limit)
    for retrieval_query in query_plan.retrieval_queries:
        query_vector = retriever._embedder.embed(retrieval_query)
        if retriever._retrieval_strategy == "hybrid":
            candidates = retriever._search_hybrid(retrieval_query, query_vector, candidate_limit, None)
        else:
            candidates = retriever._vector_store.search(
                retrieval_query,
                query_vector,
                limit=candidate_limit,
                metadata_filters=None,
            )
        for candidate in candidates:
            existing_candidate = merged_by_chunk_id.get(candidate.chunk.chunk_id)
            if existing_candidate is None or candidate.score > existing_candidate.score:
                merged_by_chunk_id[candidate.chunk.chunk_id] = candidate
    ranked_candidates = sorted(
        merged_by_chunk_id.values(),
        key=lambda match: match.score,
        reverse=True,
    )
    return ranked_candidates[:candidate_limit]


def _ordered_document_ids(matches: list[RetrievalResult]) -> list[str]:
    """Return unique document ids in first-seen ranking order."""
    return list(dict.fromkeys(match.chunk.document_id for match in matches))
