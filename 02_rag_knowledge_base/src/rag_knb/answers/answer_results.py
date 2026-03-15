"""Shared helpers for grounded answer result construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rag_knb.models import AnswerResult, RetrievalResult

DIAGNOSTIC_SNIPPET_LIMIT = 80


def build_empty_answer() -> AnswerResult:
    """Build the explicit empty-knowledge-base answer."""
    return _build_answer_result(
        answer_text=(
            "The knowledge base is empty. Ingest one or more supported documents before asking questions."
        ),
        matches=[],
        matched=False,
        reason="empty",
        retrieval_duration_ms=0.0,
    )


def build_no_match_answer(question: str) -> AnswerResult:
    """Build a deterministic no-match answer."""
    return _build_answer_result(
        answer_text=(
            f"No grounded answer found for '{question}'. Add relevant documents or ask a narrower question."
        ),
        matches=[],
        matched=False,
        reason="no_match",
    )


def build_low_confidence_answer(
    question: str,
    matches: list[RetrievalResult],
) -> AnswerResult:
    """Build an explicit low-confidence answer."""
    return _build_answer_result(
        answer_text=(
            f"Low-confidence retrieval for '{question}'. Review the supporting chunks before trusting an answer."
        ),
        matches=matches,
        matched=False,
        reason="low_confidence",
    )


def build_clarification_needed_answer(
    question: str,
    matches: list[RetrievalResult],
) -> AnswerResult:
    """Build an explicit clarification-needed answer for ambiguous retrieval."""
    document_ids = ", ".join(dict.fromkeys(match.chunk.document_id for match in matches[:2]))
    clarification_options = _clarification_options(matches[:2])
    return _build_answer_result(
        answer_text=(
            f"The question '{question}' matches multiple possible sources ({document_ids}). "
            f"Do you mean {clarification_options}?"
        ),
        matches=matches,
        matched=False,
        reason="clarification_needed",
    )


def build_matched_answer(
    answer_text: str,
    matches: list[RetrievalResult],
) -> AnswerResult:
    """Build a matched grounded answer."""
    return _build_answer_result(
        answer_text=answer_text,
        matches=matches,
        matched=True,
        reason="matched",
    )


def with_retrieval_duration(answer: AnswerResult, retrieval_duration_ms: float) -> AnswerResult:
    """Return a new answer result with retrieval timing included in diagnostics."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["retrieval_duration_ms"] = retrieval_duration_ms
    return replace(answer, diagnostics=diagnostics)


def with_query_plan(
    answer: AnswerResult,
    *,
    original_question: str,
    rewritten_question: str,
    retrieval_queries: list[str],
) -> AnswerResult:
    """Return a new answer result with query-plan diagnostics attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["original_question"] = original_question
    diagnostics["rewritten_question"] = rewritten_question
    diagnostics["retrieval_queries"] = retrieval_queries
    return replace(answer, diagnostics=diagnostics)


def with_conversation_plan(answer: AnswerResult, conversation_plan: dict[str, object]) -> AnswerResult:
    """Return a new answer result with conversation-aware planning diagnostics attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["conversation_plan"] = conversation_plan
    return replace(answer, diagnostics=diagnostics)


def with_claim_alignments(
    answer: AnswerResult,
    claim_alignments: list[dict[str, object]],
) -> AnswerResult:
    """Return a new answer result with claim-to-evidence alignments attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["claim_alignments"] = claim_alignments
    return replace(answer, diagnostics=diagnostics)


def with_semantic_verification(
    answer: AnswerResult,
    semantic_verification: list[dict[str, object]],
) -> AnswerResult:
    """Return a new answer result with semantic-verification diagnostics attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["semantic_verification"] = semantic_verification
    return replace(answer, diagnostics=diagnostics)


def with_context_window(answer: AnswerResult, context_window: list[dict[str, object]]) -> AnswerResult:
    """Return a new answer result with the compact context window attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["context_window"] = context_window
    return replace(answer, diagnostics=diagnostics)


def with_evidence_set(answer: AnswerResult, evidence_set: list[dict[str, object]]) -> AnswerResult:
    """Return a new answer result with the compact evidence set attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["evidence_set"] = evidence_set
    return replace(answer, diagnostics=diagnostics)


def with_answer_plan(answer: AnswerResult, answer_plan: dict[str, object]) -> AnswerResult:
    """Return a new answer result with answer-planning diagnostics attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["answer_plan"] = answer_plan
    return replace(answer, diagnostics=diagnostics)


def with_parent_context(answer: AnswerResult, parent_context: list[dict[str, object]]) -> AnswerResult:
    """Return a new answer result with parent context expansions attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["parent_context"] = parent_context
    return replace(answer, diagnostics=diagnostics)


def with_prompt_injection_policy(
    answer: AnswerResult,
    *,
    blocked_sentences: list[dict[str, str]],
    blocked_chunk_ids: list[str],
    downgraded_chunk_ids: list[str],
) -> AnswerResult:
    """Return a new answer result with prompt-injection policy diagnostics attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["prompt_injection_policy"] = {
        "trust_boundary": "retrieved_context_is_untrusted_evidence",
        "blocked_sentence_count": len(blocked_sentences),
        "blocked_chunk_ids": blocked_chunk_ids,
        "downgraded_chunk_ids": downgraded_chunk_ids,
        "downgraded_chunk_count": len(downgraded_chunk_ids),
        "blocked_sentences": blocked_sentences,
    }
    return replace(answer, diagnostics=diagnostics)


def with_confidence_policy(answer: AnswerResult, confidence_policy: dict[str, object]) -> AnswerResult:
    """Return a new answer result with confidence-routing diagnostics attached."""
    diagnostics = dict(answer.diagnostics)
    diagnostics["confidence_policy"] = confidence_policy
    return replace(answer, diagnostics=diagnostics)


def _build_answer_result(
    answer_text: str,
    matches: list[RetrievalResult],
    matched: bool,
    reason: str,
    retrieval_duration_ms: float | None = None,
) -> AnswerResult:
    """Build an answer result with consistent diagnostics."""
    diagnostics = _build_diagnostics(matches)
    if retrieval_duration_ms is not None:
        diagnostics["retrieval_duration_ms"] = retrieval_duration_ms
    return AnswerResult(
        answer_text=answer_text,
        matches=matches,
        matched=matched,
        reason=reason,
        diagnostics=diagnostics,
    )


def _build_diagnostics(matches: list[RetrievalResult]) -> dict[str, Any]:
    """Build the shared diagnostics shape from retrieval matches."""
    return {
        "match_count": len(matches),
        "matched_chunk_ids": [match.chunk.chunk_id for match in matches],
        "matched_document_ids": list(dict.fromkeys(match.chunk.document_id for match in matches)),
        "top_score": matches[0].score if matches else 0.0,
        "matches": [
            {
                "chunk_id": match.chunk.chunk_id,
                "document_id": match.chunk.document_id,
                "score": round(match.score, 6),
                "snippet": _build_snippet(match.chunk.content),
            }
            for match in matches
        ],
    }


def _build_snippet(content: str, limit: int = DIAGNOSTIC_SNIPPET_LIMIT) -> str:
    """Build a deterministic short snippet for diagnostics output."""
    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def _clarification_options(matches: list[RetrievalResult]) -> str:
    """Build a short deterministic clarification question from top matches."""
    labels = [_clarification_label(match) for match in matches]
    if not labels:
        return "which source you mean"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _clarification_label(match: RetrievalResult) -> str:
    """Build one small human-facing clarification label for a retrieved match."""
    file_name = match.chunk.metadata.get("file_name")
    record_id = match.chunk.metadata.get("record_id")
    if isinstance(record_id, str) and record_id:
        return f"record '{record_id}'"
    if isinstance(file_name, str) and file_name:
        return f"document '{file_name}'"
    return f"document '{match.chunk.document_id}'"
