"""Deterministic query rewriting helpers for local retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

FILLER_PHRASES = (
    "can you tell me",
    "could you tell me",
    "please tell me",
    "i want to know",
    "do you know",
)
CONJUNCTION_PATTERN = re.compile(r"\s+(?:and|also)\s+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Resolved retrieval queries derived from one user question."""

    original_question: str
    rewritten_question: str
    retrieval_queries: list[str]


def build_query_plan(question: str) -> QueryPlan:
    """Build a small deterministic query plan from one user question."""
    rewritten_question = _rewrite_question(question)
    retrieval_queries = _decompose_question(rewritten_question)
    return QueryPlan(
        original_question=question,
        rewritten_question=rewritten_question,
        retrieval_queries=retrieval_queries or [rewritten_question],
    )


def _rewrite_question(question: str) -> str:
    """Strip filler phrasing and normalize whitespace."""
    rewritten = question.strip()
    lowered = rewritten.lower()
    for filler in FILLER_PHRASES:
        if lowered.startswith(filler):
            rewritten = rewritten[len(filler):].strip(" ,:.!?")
            break
    rewritten = rewritten.rstrip(" ?")
    return WHITESPACE_PATTERN.sub(" ", rewritten).strip()


def _decompose_question(question: str) -> list[str]:
    """Split a simple multi-part question into a small number of retrieval queries."""
    parts = [part.strip(" ,:.!?") for part in CONJUNCTION_PATTERN.split(question) if part.strip(" ,:.!?")]
    if len(parts) <= 1 or len(parts) > 2:
        return [question]
    return parts
