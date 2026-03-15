"""Retrieval orchestration helpers."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from rag_knb.models import Chunk, RetrievalResult
from rag_knb.retrieval_engine.embeddings import EmbeddingBackend, TokenVector, cosine_similarity
from rag_knb.retrieval_engine.query_rewriting import QueryPlan, build_query_plan
from rag_knb.retrieval_engine.vector_store import IndexedChunk, VectorStore

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CANDIDATE_MULTIPLIER = 3
HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_LEXICAL_WEIGHT = 0.3
EXACT_PHRASE_BONUS = 1.0
METADATA_MATCH_WEIGHT = 0.15
STRUCTURED_INTENT_BONUS = 0.2
STRUCTURED_QUERY_TOKENS = {"record", "records", "structured"}
SOURCE_WEIGHT_SCALE = 0.05


class Retriever:
    """Retrieve the most relevant chunks for a query."""

    def __init__(
        self,
        embedder: EmbeddingBackend,
        vector_store: VectorStore,
        retrieval_strategy: str = "vector",
    ) -> None:
        """Initialize the retriever."""
        self._embedder = embedder
        self._vector_store = vector_store
        self._retrieval_strategy = retrieval_strategy

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and index chunks for later retrieval."""
        self._vector_store.clear()
        for chunk in chunks:
            self._vector_store.add(chunk, self._embedder.embed(chunk.content))

    def search(
        self,
        query: str,
        limit: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the indexed chunks for a query."""
        query_plan = build_query_plan(query)
        merged_matches = self.search_with_plan(query_plan, limit=limit, metadata_filters=metadata_filters)
        return merged_matches

    def search_with_plan(
        self,
        query_plan: QueryPlan,
        limit: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the indexed chunks for a precomputed query plan."""
        merged_by_chunk_id: dict[str, RetrievalResult] = {}
        candidate_limit = max(limit * CANDIDATE_MULTIPLIER, limit)
        for retrieval_query in query_plan.retrieval_queries:
            query_vector = self._embedder.embed(retrieval_query)
            if self._retrieval_strategy == "hybrid":
                candidates = self._search_hybrid(retrieval_query, query_vector, candidate_limit, metadata_filters)
            else:
                candidates = self._vector_store.search(
                    retrieval_query,
                    query_vector,
                    limit=candidate_limit,
                    metadata_filters=metadata_filters,
                )
            for candidate in candidates:
                existing_candidate = merged_by_chunk_id.get(candidate.chunk.chunk_id)
                if existing_candidate is None or candidate.score > existing_candidate.score:
                    merged_by_chunk_id[candidate.chunk.chunk_id] = candidate
        return _rerank_matches(query_plan.rewritten_question, list(merged_by_chunk_id.values()), limit)

    def _search_hybrid(
        self,
        query: str,
        query_vector: TokenVector,
        limit: int,
        metadata_filters: dict[str, str] | None,
    ) -> list[RetrievalResult]:
        """Combine vector and lexical signals with BM25-backed lexical scoring."""
        query_tokens = TOKEN_PATTERN.findall(query.lower())
        filtered_entries = [
            entry
            for entry in self._vector_store.entries
            if not metadata_filters
            or all(str(entry.chunk.metadata.get(key)) == value for key, value in metadata_filters.items())
        ]
        lexical_scores = _bm25_scores(query_tokens, filtered_entries)
        scored_matches: list[RetrievalResult] = []
        for entry in filtered_entries:
            vector_score = cosine_similarity(entry.vector, query_vector)
            lexical_score = lexical_scores.get(entry.chunk.chunk_id, 0.0)
            combined_score = (
                (HYBRID_VECTOR_WEIGHT * vector_score)
                + (HYBRID_LEXICAL_WEIGHT * lexical_score)
            )
            if combined_score <= 0:
                continue
            scored_matches.append(RetrievalResult(chunk=entry.chunk, score=combined_score))
        return sorted(scored_matches, key=lambda item: item.score, reverse=True)[:limit]


def _bm25_scores(query_tokens: list[str], entries: list[IndexedChunk]) -> dict[str, float]:
    """Return normalized BM25 scores for the provided entries."""
    if not query_tokens or not entries:
        return {}
    normalized_query_tokens = [_normalize_bm25_token(token) for token in query_tokens]
    normalized_query_tokens = [token for token in normalized_query_tokens if token]
    if not normalized_query_tokens:
        return {}
    tokenized_corpus = [
        [token for token in (_normalize_bm25_token(token) for token in TOKEN_PATTERN.findall(entry.chunk.content.lower())) if token]
        for entry in entries
    ]
    if not any(tokenized_corpus):
        return {}
    bm25 = BM25Okapi(tokenized_corpus)
    raw_scores = bm25.get_scores(normalized_query_tokens)
    max_score = max(raw_scores, default=0.0)
    if max_score <= 0:
        return {}
    return {
        entry.chunk.chunk_id: float(score / max_score)
        for entry, score in zip(entries, raw_scores, strict=True)
        if score > 0
    }


def _normalize_bm25_token(token: str) -> str:
    """Apply light deterministic normalization so BM25 stays friendly to simple variants."""
    normalized = token.lower()
    for suffix, replacement, minimum_length in (
        ("ies", "y", 5),
        ("es", "", 5),
        ("s", "", 4),
    ):
        if normalized.endswith(suffix) and len(normalized) >= minimum_length:
            normalized = f"{normalized[:-len(suffix)]}{replacement}"
            break
    return normalized


def _lexical_overlap_score(query_tokens: set[str], content: str) -> float:
    """Return a simple normalized lexical overlap score for hybrid retrieval."""
    if not query_tokens:
        return 0.0
    content_tokens = set(TOKEN_PATTERN.findall(content.lower()))
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(content_tokens))
    return overlap / len(query_tokens)


def _rerank_matches(query: str, matches: list[RetrievalResult], limit: int) -> list[RetrievalResult]:
    """Rerank a broader candidate set with a lightweight question-focused heuristic."""
    query_tokens = set(TOKEN_PATTERN.findall(query.lower()))
    deduplicated_matches = _deduplicate_matches(matches)
    reranked_matches = sorted(
        deduplicated_matches,
        key=lambda match: _rerank_key(query, query_tokens, match),
        reverse=True,
    )
    return reranked_matches[:limit]


def _rerank_key(
    query: str,
    query_tokens: set[str],
    match: RetrievalResult,
) -> tuple[float, float, float]:
    """Build a stable reranking key for one candidate match."""
    lexical_score = _lexical_overlap_score(query_tokens, match.chunk.content)
    structured_score = _structured_field_overlap_score(query_tokens, match.chunk)
    metadata_bonus = _metadata_rank_bonus(query_tokens, match.chunk)
    source_weight_bonus = _source_weight_bonus(match.chunk)
    exact_phrase_bonus = EXACT_PHRASE_BONUS if query.lower() in match.chunk.content.lower() else 0.0
    return (
        lexical_score + exact_phrase_bonus + structured_score + metadata_bonus + source_weight_bonus,
        match.score,
        -float(match.chunk.start_offset),
    )


def _deduplicate_matches(matches: list[RetrievalResult]) -> list[RetrievalResult]:
    """Keep only the strongest retrieval result for each normalized content fingerprint."""
    deduplicated: dict[str, RetrievalResult] = {}
    for match in matches:
        fingerprint = _content_fingerprint(match.chunk.content)
        existing_match = deduplicated.get(fingerprint)
        if existing_match is None or match.score > existing_match.score:
            deduplicated[fingerprint] = match
    return list(deduplicated.values())


def _structured_field_overlap_score(query_tokens: set[str], chunk: Chunk) -> float:
    """Return a small bonus for structured field names and values that match the query."""
    if chunk.metadata.get("format") != "structured":
        return 0.0
    if not query_tokens:
        return 0.0
    structured_tokens: set[str] = set()
    field_names = chunk.metadata.get("structured_fields", [])
    if isinstance(field_names, list):
        for field_name in field_names:
            structured_tokens.update(TOKEN_PATTERN.findall(str(field_name).lower()))
            value = chunk.metadata.get(f"field_{field_name}")
            if isinstance(value, str):
                structured_tokens.update(TOKEN_PATTERN.findall(value.lower()))
    if not structured_tokens:
        return 0.0
    return len(query_tokens.intersection(structured_tokens)) / len(query_tokens)


def _metadata_rank_bonus(query_tokens: set[str], chunk: Chunk) -> float:
    """Return a small explainable ranking bonus for metadata that matches the query."""
    if not query_tokens:
        return 0.0
    metadata_tokens: set[str] = set()
    for key in ("document_id", "file_name", "source_path", "format", "record_id"):
        value = chunk.document_id if key == "document_id" else chunk.metadata.get(key)
        if value is not None:
            metadata_tokens.update(TOKEN_PATTERN.findall(str(value).lower()))
    overlap_bonus = 0.0
    if metadata_tokens:
        overlap_bonus = (len(query_tokens.intersection(metadata_tokens)) / len(query_tokens)) * METADATA_MATCH_WEIGHT
    structured_intent_bonus = 0.0
    if chunk.metadata.get("format") == "structured" and query_tokens.intersection(STRUCTURED_QUERY_TOKENS):
        structured_intent_bonus = STRUCTURED_INTENT_BONUS
    return overlap_bonus + structured_intent_bonus


def _source_weight_bonus(chunk: Chunk) -> float:
    """Return a small optional ranking bonus from explicit source metadata."""
    source_weight = chunk.metadata.get("source_weight")
    if not isinstance(source_weight, (int, float)):
        return 0.0
    return float(source_weight) * SOURCE_WEIGHT_SCALE


def _content_fingerprint(content: str) -> str:
    """Build a deterministic normalized content fingerprint for deduplication."""
    return " ".join(content.lower().split())
