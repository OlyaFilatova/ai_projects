"""Deterministic context-building helpers for answer generation."""

from __future__ import annotations

import re
from dataclasses import replace

from rag_knb.models import RetrievalResult

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
QUESTION_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
QUESTION_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "do",
    "document",
    "documents",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "say",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
EVIDENCE_SNIPPET_LIMIT = 3
EVIDENCE_SCORE_SCALE = 1000
SUMMARY_QUERY_PREFIXES = ("summarize", "summary", "give a summary")
COMPOSITION_QUERY_PREFIXES = ("why", "what makes", "how does", "how do")


def build_evidence_set(
    question: str,
    matches: list[RetrievalResult],
    *,
    limit: int = EVIDENCE_SNIPPET_LIMIT,
) -> list[dict[str, object]]:
    """Build a compact question-aware evidence set from retrieved matches."""
    question_tokens = {
        token
        for token in QUESTION_TOKEN_PATTERN.findall(question.lower())
        if token not in QUESTION_STOPWORDS
    }
    ranked_entries: list[tuple[tuple[int, int, int], dict[str, object]]] = []
    for match_index, match in enumerate(matches):
        for sentence_index, sentence in enumerate(_split_sentences(match.chunk.content)):
            ranked_entries.append(
                (
                    _evidence_rank_key(sentence, question_tokens, match.score, match_index, sentence_index),
                    {
                        "chunk_id": match.chunk.chunk_id,
                        "document_id": match.chunk.document_id,
                        "sentence": sentence,
                        "score": round(match.score, 6),
                    },
                )
            )

    ranked_entries.sort(key=lambda item: item[0], reverse=True)
    evidence_entries: list[dict[str, object]] = []
    seen_sentences: set[str] = set()
    for _, entry in ranked_entries:
        sentence = str(entry["sentence"])
        normalized_sentence = " ".join(sentence.lower().split())
        if normalized_sentence in seen_sentences:
            continue
        seen_sentences.add(normalized_sentence)
        evidence_entries.append(entry)
        if len(evidence_entries) >= limit:
            break
    return evidence_entries


def build_evidence_matches(
    question: str,
    matches: list[RetrievalResult],
    *,
    limit: int = EVIDENCE_SNIPPET_LIMIT,
) -> list[RetrievalResult]:
    """Compress retrieved matches into sentence-level evidence matches."""
    evidence_matches: list[RetrievalResult] = []
    source_matches = {match.chunk.chunk_id: match for match in matches}
    for entry in build_evidence_set(question, matches, limit=limit):
        chunk_id = str(entry["chunk_id"])
        source_match = source_matches[chunk_id]
        sentence = str(entry["sentence"])
        evidence_chunk = replace(
            source_match.chunk,
            content=sentence,
            start_offset=0,
            end_offset=len(sentence),
            metadata={
                **source_match.chunk.metadata,
                "source_chunk_id": source_match.chunk.chunk_id,
                "source_content": source_match.chunk.content,
            },
        )
        evidence_matches.append(
            RetrievalResult(
                chunk=evidence_chunk,
                score=float(str(entry["score"])),
            )
        )
    return evidence_matches


def build_answer_plan(question: str, evidence_set: list[dict[str, object]]) -> dict[str, object]:
    """Build a lightweight answer plan from compressed evidence."""
    mode = _answer_mode(question)
    evidence_budget = 1
    if mode in {"summary", "comparison", "composition"}:
        evidence_budget = min(max(len(evidence_set), 2), 3)
    supporting_documents = list(dict.fromkeys(str(entry["document_id"]) for entry in evidence_set))
    return {
        "mode": mode,
        "evidence_budget": evidence_budget,
        "compressed_evidence_count": len(evidence_set),
        "supporting_documents": supporting_documents,
    }


def build_context_window(matches: list[RetrievalResult], limit: int = 3) -> list[dict[str, object]]:
    """Build a compact sentence-level context window from retrieved matches."""
    entries: list[dict[str, object]] = []
    seen_sentences: set[tuple[str, str]] = set()
    for match in matches:
        for sentence in _split_sentences(match.chunk.content):
            dedupe_key = (sentence, match.chunk.chunk_id)
            if dedupe_key in seen_sentences:
                continue
            seen_sentences.add(dedupe_key)
            entries.append(
                {
                    "chunk_id": match.chunk.chunk_id,
                    "document_id": match.chunk.document_id,
                    "sentence": sentence,
                    "score": round(match.score, 6),
                }
            )
            if len(entries) >= limit:
                return entries
    return entries


def render_context_window(matches: list[RetrievalResult], limit: int = 3) -> str:
    """Render a compact prompt-ready context window."""
    return "\n".join(
        f"[{entry['chunk_id']}] {entry['sentence']}" for entry in build_context_window(matches, limit=limit)
    )


def render_evidence_set(question: str, matches: list[RetrievalResult], limit: int = 3) -> str:
    """Render the compressed evidence set for prompt construction."""
    return "\n".join(
        f"[{entry['chunk_id']}] {entry['sentence']}" for entry in build_evidence_set(question, matches, limit=limit)
    )


def _split_sentences(content: str) -> list[str]:
    """Split content into compact sentence-like segments."""
    sentences = [segment.strip() for segment in SENTENCE_PATTERN.split(content) if segment.strip()]
    return sentences or [content.strip()]


def _evidence_rank_key(
    sentence: str,
    question_tokens: set[str],
    score: float,
    match_index: int,
    sentence_index: int,
) -> tuple[int, int, int]:
    """Build a deterministic ranking key for evidence compression."""
    sentence_tokens = set(QUESTION_TOKEN_PATTERN.findall(sentence.lower()))
    overlap = len(question_tokens.intersection(sentence_tokens))
    return (
        overlap,
        int(score * EVIDENCE_SCORE_SCALE),
        -((match_index * EVIDENCE_SCORE_SCALE) + sentence_index),
    )


def _answer_mode(question: str) -> str:
    """Infer one small deterministic answer mode from the question text."""
    normalized = question.lower().strip()
    if normalized.startswith(SUMMARY_QUERY_PREFIXES):
        return "summary"
    if normalized.startswith("compare") or " and " in normalized:
        return "comparison"
    if normalized.startswith(COMPOSITION_QUERY_PREFIXES):
        return "composition"
    return "direct"
