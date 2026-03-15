"""Service-layer primitives shared by future interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from rag_knb.answers.answer_results import (
    build_empty_answer,
    with_conversation_plan,
    with_query_plan,
    with_retrieval_duration,
)
from rag_knb.answers.answering import TextGenerator, build_answerer
from rag_knb.config import RuntimeConfig
from rag_knb.errors import ValidationError
from rag_knb.indexing.chunking import chunk_documents
from rag_knb.indexing.embedding_lifecycle import validate_embedding_workflow_compatibility
from rag_knb.indexing.loaders import load_documents
from rag_knb.indexing.storage import LocalKnowledgeBaseRepository
from rag_knb.library_policies import (
    validate_max_count,
    validate_positive_request_limit,
    validate_text_length,
)
from rag_knb.models import (
    AnswerResult,
    Chunk,
    ConversationTurn,
    Document,
    IngestResult,
    RefreshResult,
)
from rag_knb.observability import get_logger, log_event
from rag_knb.pathing import is_path_within_allowed_root, resolve_data_dir
from rag_knb.retrieval_engine.embeddings import EmbeddingBackend, build_embedder
from rag_knb.retrieval_engine.query_rewriting import build_query_plan
from rag_knb.retrieval_engine.retrieval import Retriever
from rag_knb.retrieval_engine.vector_store import IndexedChunk, build_vector_store

CONVERSATION_TOKEN_STOPWORDS = {
    "about",
    "compare",
    "me",
    "tell",
    "the",
    "them",
    "they",
    "what",
}


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Current high-level knowledge-base status."""

    summary: str


@dataclass(frozen=True, slots=True)
class _RefreshInputs:
    """Grouped inputs used during one incremental refresh run."""

    refreshed_documents_by_source: dict[str, list[Document]]
    existing_documents_by_source: dict[str, list[Document]]
    existing_chunks_by_source: dict[str, list[Chunk]]
    existing_entries_by_source: dict[str, list[IndexedChunk]]
    removed_source_paths: set[str]


@dataclass(frozen=True, slots=True)
class _RefreshArtifacts:
    """Merged state artifacts produced by an incremental refresh."""

    documents: list[Document]
    chunks: list[Chunk]
    entries: list[IndexedChunk]
    added_source_paths: list[str]
    updated_source_paths: list[str]
    removed_source_paths: list[str]
    unchanged_source_paths: list[str]


class KnowledgeBaseService:
    """Small service facade that later prompts can extend."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        text_generator: TextGenerator | None = None,
    ) -> None:
        """Initialize the service with runtime configuration."""
        self._config = config or RuntimeConfig()
        self._documents: list[Document] = []
        self._chunks: list[Chunk] = []
        self._embedder = build_embedder(self._config)
        self._vector_store = build_vector_store(self._config, self._embedder)
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            retrieval_strategy=self._config.retrieval_strategy,
        )
        self._answerer = build_answerer(self._config, generator=text_generator)
        self._logger = get_logger(__name__)

    @property
    def config(self) -> RuntimeConfig:
        """Expose the current runtime configuration."""
        return self._config

    def status(self) -> ServiceStatus:
        """Return a bootstrap-stage status summary."""
        if not self._chunks:
            return ServiceStatus(
                summary=(
                    "RAG KnB is ready but empty. Ingest documents to enable grounded retrieval."
                )
            )
        return ServiceStatus(
            summary=f"RAG KnB has {len(self._documents)} documents and {len(self._chunks)} chunks indexed."
        )

    def ingest_paths(self, paths: list[Path]) -> IngestResult:
        """Load documents from disk, chunk them, and index them in memory."""
        started_at = perf_counter()
        validate_max_count(
            len(paths),
            self._config.max_documents_per_ingest,
            "Document count exceeds the configured maximum of "
            f"{self._config.max_documents_per_ingest} for one ingest call.",
        )
        documents = load_documents(
            paths,
            max_document_bytes=self._config.max_document_bytes,
            allowed_root=self._config.allowed_root,
        )
        chunks = chunk_documents(
            documents,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        validate_max_count(
            len(chunks),
            self._config.max_chunks_per_ingest,
            "Chunk count exceeds the configured maximum of "
            f"{self._config.max_chunks_per_ingest} for one ingest call.",
        )
        if not chunks:
            raise ValidationError("No indexable content was found in the provided documents.")
        self._replace_state(documents, chunks)
        self._log_duration(
            "ingest_completed",
            started_at,
            document_count=len(documents),
            chunk_count=len(chunks),
        )
        return IngestResult(documents=documents, chunks=chunks)

    def refresh_paths(
        self,
        paths: list[Path],
        *,
        remove_missing: bool = False,
    ) -> RefreshResult:
        """Incrementally refresh indexed state from current source files."""
        if not self._documents:
            return self._refresh_via_full_ingest(paths)

        started_at = perf_counter()
        refreshed_documents, refreshed_chunks = self._load_refresh_documents_and_chunks(paths)
        refresh_inputs = self._build_refresh_inputs(refreshed_documents, remove_missing=remove_missing)
        refresh_artifacts = self._merge_refresh_artifacts(refresh_inputs)
        self._replace_loaded_state(
            refresh_artifacts.documents,
            refresh_artifacts.chunks,
            refresh_artifacts.entries,
        )
        self._log_refresh_duration(started_at, refresh_artifacts)
        return RefreshResult(
            documents=refresh_artifacts.documents,
            chunks=refresh_artifacts.chunks,
            added_source_paths=refresh_artifacts.added_source_paths,
            updated_source_paths=refresh_artifacts.updated_source_paths,
            removed_source_paths=refresh_artifacts.removed_source_paths,
            unchanged_source_paths=refresh_artifacts.unchanged_source_paths,
        )

    def ask(
        self,
        question: str,
        limit: int = 3,
        metadata_filters: dict[str, str] | None = None,
        conversation_turns: list[ConversationTurn] | None = None,
    ) -> AnswerResult:
        """Return an answer grounded in the indexed chunks."""
        if not question.strip():
            raise ValidationError("Question cannot be empty.")
        validate_text_length(
            question,
            self._config.max_question_length,
            f"Question length exceeds the configured maximum of {self._config.max_question_length} characters.",
        )
        validate_positive_request_limit(
            limit,
            self._config.max_retrieval_limit,
            f"Retrieval limit exceeds the configured maximum of {self._config.max_retrieval_limit}.",
        )
        if not self._chunks:
            return build_empty_answer()
        started_at = perf_counter()
        retrieval_question = _build_conversation_aware_question(question, conversation_turns)
        conversation_plan = _build_conversation_answer_plan(question, conversation_turns)
        query_plan = build_query_plan(retrieval_question)
        matches = self._retriever.search_with_plan(query_plan, limit=limit, metadata_filters=metadata_filters)
        answer = self._answerer.answer(question, matches)
        answer = with_query_plan(
            answer,
            original_question=query_plan.original_question,
            rewritten_question=query_plan.rewritten_question,
            retrieval_queries=query_plan.retrieval_queries,
        )
        answer = with_conversation_plan(answer, conversation_plan)
        retrieval_duration_ms = round((perf_counter() - started_at) * 1000, 3)
        answer = with_retrieval_duration(answer, retrieval_duration_ms)
        log_event(
            self._logger,
            "query_completed",
            matched=answer.matched,
            reason=answer.reason,
            match_count=len(answer.matches),
            retrieval_duration_ms=retrieval_duration_ms,
        )
        return answer

    def save(self, data_dir: Path | None = None) -> Path:
        """Persist the current knowledge-base state to disk."""
        started_at = perf_counter()
        target_dir = self._resolve_data_dir(data_dir)
        repository = self._build_repository(target_dir)
        repository.save(
            documents=self._documents,
            chunks=self._chunks,
            indexed_chunks=self._vector_store.entries,
            metadata={"embedding_workflow": self._embedder.workflow_metadata()},
        )
        self._log_duration(
            "save_completed",
            started_at,
            data_dir=str(target_dir),
            document_count=len(self._documents),
        )
        return target_dir

    def load(self, data_dir: Path | None = None) -> Path:
        """Load persisted knowledge-base state from disk."""
        started_at = perf_counter()
        target_dir = self._resolve_data_dir(data_dir)
        repository = self._build_repository(target_dir)
        persisted_state = repository.load()
        validate_embedding_workflow_compatibility(
            persisted_state.metadata,
            persisted_state.indexed_chunks,
            self._embedder,
        )
        self._replace_loaded_state(persisted_state.documents, persisted_state.chunks, persisted_state.indexed_chunks)
        self._log_duration(
            "load_completed",
            started_at,
            data_dir=str(target_dir),
            document_count=len(self._documents),
        )
        return target_dir

    @property
    def documents(self) -> list[Document]:
        """Expose the loaded documents."""
        return list(self._documents)

    @property
    def chunks(self) -> list[Chunk]:
        """Expose the indexed chunks."""
        return list(self._chunks)

    def list_documents(self) -> list[Document]:
        """List indexed documents."""
        return list(self._documents)

    def remove_documents(self, document_ids: list[str]) -> list[Document]:
        """Remove documents and their chunks from the knowledge base."""
        started_at = perf_counter()
        document_id_set = set(document_ids)
        remaining_documents = [
            document for document in self._documents if document.document_id not in document_id_set
        ]
        remaining_chunks = [
            chunk for chunk in self._chunks if chunk.document_id not in document_id_set
        ]
        self._replace_state(remaining_documents, remaining_chunks)
        self._log_duration(
            "documents_removed",
            started_at,
            removed_document_ids=sorted(document_ids),
            remaining_document_count=len(self._documents),
        )
        return self.list_documents()

    def _replace_state(self, documents: list[Document], chunks: list[Chunk]) -> None:
        """Replace in-memory documents and rebuild the retrieval index."""
        self._documents = documents
        self._chunks = chunks
        self._reindex_chunks(chunks)

    def _replace_loaded_state(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        indexed_chunks: list[IndexedChunk],
    ) -> None:
        """Replace service state from persisted artifacts."""
        self._documents = documents
        self._chunks = chunks
        self._vector_store.replace(indexed_chunks)

    def _reindex_chunks(self, chunks: list[Chunk]) -> None:
        """Rebuild retrieval state from the current chunks."""
        self._retriever.index_chunks(chunks)

    def _resolve_data_dir(self, data_dir: Path | None) -> Path:
        """Resolve the persistence directory for a save/load operation."""
        target_dir = resolve_data_dir(data_dir, self._config.data_dir)
        if not is_path_within_allowed_root(target_dir, self._config.allowed_root):
            raise ValidationError(
                f"Path '{target_dir}' is outside the configured allowed root '{self._config.allowed_root}'."
            )
        return target_dir

    def _build_repository(self, data_dir: Path) -> LocalKnowledgeBaseRepository:
        """Create the local repository for the target data directory."""
        return LocalKnowledgeBaseRepository(data_dir)

    def _refresh_via_full_ingest(self, paths: list[Path]) -> RefreshResult:
        """Refresh through the existing full-ingest path when no state is loaded yet."""
        ingest_result = self.ingest_paths(paths)
        return RefreshResult(
            documents=ingest_result.documents,
            chunks=ingest_result.chunks,
            added_source_paths=sorted({document.source_path for document in ingest_result.documents}),
            updated_source_paths=[],
            removed_source_paths=[],
            unchanged_source_paths=[],
        )

    def _load_refresh_documents_and_chunks(self, paths: list[Path]) -> tuple[list[Document], list[Chunk]]:
        """Load and validate the refreshed source documents and chunks."""
        refreshed_documents = load_documents(
            paths,
            max_document_bytes=self._config.max_document_bytes,
            allowed_root=self._config.allowed_root,
        )
        refreshed_chunks = chunk_documents(
            refreshed_documents,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        validate_max_count(
            len(refreshed_documents),
            self._config.max_documents_per_ingest,
            "Document count exceeds the configured maximum of "
            f"{self._config.max_documents_per_ingest} for one refresh call.",
        )
        validate_max_count(
            len(refreshed_chunks),
            self._config.max_chunks_per_ingest,
            "Chunk count exceeds the configured maximum of "
            f"{self._config.max_chunks_per_ingest} for one refresh call.",
        )
        return refreshed_documents, refreshed_chunks

    def _build_refresh_inputs(
        self,
        refreshed_documents: list[Document],
        *,
        remove_missing: bool,
    ) -> _RefreshInputs:
        """Collect grouped refresh inputs from the current and refreshed state."""
        refreshed_documents_by_source = _group_documents_by_source_path(refreshed_documents)
        existing_documents_by_source = _group_documents_by_source_path(self._documents)
        refreshed_source_paths = set(refreshed_documents_by_source)
        existing_source_paths = set(existing_documents_by_source)
        return _RefreshInputs(
            refreshed_documents_by_source=refreshed_documents_by_source,
            existing_documents_by_source=existing_documents_by_source,
            existing_chunks_by_source=_group_chunks_by_source_path(self._chunks),
            existing_entries_by_source=_group_entries_by_source_path(self._vector_store.entries),
            removed_source_paths=(
                existing_source_paths.difference(refreshed_source_paths) if remove_missing else set()
            ),
        )

    def _merge_refresh_artifacts(self, refresh_inputs: _RefreshInputs) -> _RefreshArtifacts:
        """Merge unchanged and refreshed source groups into one deterministic state payload."""
        added_source_paths: list[str] = []
        updated_source_paths: list[str] = []
        unchanged_source_paths: list[str] = []
        combined_documents: list[Document] = []
        combined_chunks: list[Chunk] = []
        combined_entries: list[IndexedChunk] = []

        for source_path in _all_refresh_source_paths(refresh_inputs):
            refreshed_group = refresh_inputs.refreshed_documents_by_source.get(source_path)
            existing_group = refresh_inputs.existing_documents_by_source.get(source_path)
            if _is_unchanged_existing_source(source_path, refreshed_group, existing_group, refresh_inputs):
                combined_documents.extend(existing_group or [])
                combined_chunks.extend(refresh_inputs.existing_chunks_by_source.get(source_path, []))
                combined_entries.extend(refresh_inputs.existing_entries_by_source.get(source_path, []))
                unchanged_source_paths.append(source_path)
                continue
            if refreshed_group is None:
                continue
            _record_refresh_change(source_path, existing_group, added_source_paths, updated_source_paths)
            new_chunks = chunk_documents(
                refreshed_group,
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )
            combined_documents.extend(refreshed_group)
            combined_chunks.extend(new_chunks)
            combined_entries.extend(_build_indexed_entries(new_chunks, self._embedder))

        return _RefreshArtifacts(
            documents=_sort_documents(combined_documents),
            chunks=_sort_chunks(combined_chunks),
            entries=_sort_entries(combined_entries),
            added_source_paths=added_source_paths,
            updated_source_paths=updated_source_paths,
            removed_source_paths=sorted(refresh_inputs.removed_source_paths),
            unchanged_source_paths=unchanged_source_paths,
        )

    def _log_refresh_duration(self, started_at: float, refresh_artifacts: _RefreshArtifacts) -> None:
        """Log one completed refresh with explicit change categories."""
        self._log_duration(
            "refresh_completed",
            started_at,
            added_source_paths=refresh_artifacts.added_source_paths,
            updated_source_paths=refresh_artifacts.updated_source_paths,
            removed_source_paths=refresh_artifacts.removed_source_paths,
            unchanged_source_paths=refresh_artifacts.unchanged_source_paths,
        )

    def _log_duration(self, event: str, started_at: float, **fields: object) -> None:
        """Emit one structured log event with a consistent duration field."""
        log_event(
            self._logger,
            event,
            duration_ms=round((perf_counter() - started_at) * 1000, 3),
            **fields,
        )


def _build_conversation_aware_question(
    question: str,
    conversation_turns: list[ConversationTurn] | None,
) -> str:
    """Build a short retrieval question that includes only recent conversation context."""
    if not conversation_turns:
        return question
    recent_questions = [turn.question.strip() for turn in conversation_turns[-2:] if turn.question.strip()]
    if not recent_questions:
        return question
    if _is_comparison_follow_up(question) and len(recent_questions) >= 2:
        topics = [topic for topic in (_extract_conversation_topic(item) for item in recent_questions[-2:]) if topic]
        if len(topics) >= 2:
            return " and ".join(topics)
        return " and ".join(recent_questions[-2:])
    return " ".join([*recent_questions, question])


def _build_conversation_answer_plan(
    question: str,
    conversation_turns: list[ConversationTurn] | None,
) -> dict[str, object]:
    """Build a lightweight answer-planning summary for conversational follow-ups."""
    mode = "direct"
    if _is_comparison_follow_up(question):
        mode = "comparison"
    elif question.lower().startswith("what about ") or _is_continuation_follow_up(question):
        mode = "continuation"
    elif question.lower().startswith("summarize"):
        mode = "summary"
    elif question.lower().startswith("which ") or question.lower().startswith("what ") and "?" in question:
        mode = "clarification_or_direct"
    return {
        "mode": mode,
        "conversation_turn_count": len(conversation_turns or []),
        "uses_conversation_context": bool(conversation_turns),
    }


def _is_comparison_follow_up(question: str) -> bool:
    """Return whether one follow-up question asks for a comparison."""
    normalized_question = question.lower().strip()
    return normalized_question.startswith("compare") or "compare them" in normalized_question


def _is_continuation_follow_up(question: str) -> bool:
    """Return whether one short follow-up depends on recent conversation context."""
    normalized_question = question.lower()
    return " they " in f" {normalized_question} " or " them" in normalized_question


def _extract_conversation_topic(question: str) -> str:
    """Extract one lightweight topic token from a recent conversation question."""
    tokens = [
        token
        for token in question.lower().replace("?", "").split()
        if token not in CONVERSATION_TOKEN_STOPWORDS
    ]
    return tokens[-1] if tokens else ""


def _group_documents_by_source_path(documents: list[Document]) -> dict[str, list[Document]]:
    """Group documents by source path while preserving deterministic ordering."""
    grouped: dict[str, list[Document]] = {}
    for document in documents:
        grouped.setdefault(document.source_path, []).append(document)
    return {key: _sort_documents(value) for key, value in grouped.items()}


def _group_chunks_by_source_path(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by their originating source path."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        source_path = str(chunk.metadata.get("source_path", ""))
        grouped.setdefault(source_path, []).append(chunk)
    return {key: _sort_chunks(value) for key, value in grouped.items()}


def _group_entries_by_source_path(entries: list[IndexedChunk]) -> dict[str, list[IndexedChunk]]:
    """Group indexed chunks by their originating source path."""
    grouped: dict[str, list[IndexedChunk]] = {}
    for entry in entries:
        source_path = str(entry.chunk.metadata.get("source_path", ""))
        grouped.setdefault(source_path, []).append(entry)
    return {key: _sort_entries(value) for key, value in grouped.items()}


def _all_refresh_source_paths(refresh_inputs: _RefreshInputs) -> list[str]:
    """Return the deterministic source-path iteration order for one refresh run."""
    existing_source_paths = set(refresh_inputs.existing_documents_by_source)
    refreshed_source_paths = set(refresh_inputs.refreshed_documents_by_source)
    return sorted((existing_source_paths - refresh_inputs.removed_source_paths).union(refreshed_source_paths))


def _is_unchanged_existing_source(
    source_path: str,
    refreshed_group: list[Document] | None,
    existing_group: list[Document] | None,
    refresh_inputs: _RefreshInputs,
) -> bool:
    """Return whether one source path should reuse its current indexed artifacts."""
    if refreshed_group is None:
        return existing_group is not None
    if existing_group is None:
        return False
    return _document_groups_match(existing_group, refreshed_group)


def _record_refresh_change(
    source_path: str,
    existing_group: list[Document] | None,
    added_source_paths: list[str],
    updated_source_paths: list[str],
) -> None:
    """Record whether one refreshed source path is new or updated."""
    if existing_group is None:
        added_source_paths.append(source_path)
        return
    updated_source_paths.append(source_path)


def _document_groups_match(existing: list[Document], refreshed: list[Document]) -> bool:
    """Return whether two source-path document groups are content-identical."""
    return _document_group_signature(existing) == _document_group_signature(refreshed)


def _document_group_signature(documents: list[Document]) -> list[tuple[str, str]]:
    """Build a deterministic signature for one source-path document group."""
    return [
        (document.document_id, _document_content_fingerprint(document))
        for document in _sort_documents(documents)
    ]


def _document_content_fingerprint(document: Document) -> str:
    """Build a stable content fingerprint for change detection."""
    metadata_without_fingerprint = {
        key: value
        for key, value in document.metadata.items()
        if key != "content_fingerprint"
    }
    payload = f"{document.document_id}\n{document.content}\n{metadata_without_fingerprint!r}"
    return _content_fingerprint(payload)


def _build_indexed_entries(
    chunks: list[Chunk],
    embedder: EmbeddingBackend,
) -> list[IndexedChunk]:
    """Embed only the provided chunks for incremental refresh workflows."""
    return [IndexedChunk(chunk=chunk, vector=embedder.embed(chunk.content)) for chunk in chunks]


def _sort_documents(documents: list[Document]) -> list[Document]:
    """Sort documents into a deterministic persisted order."""
    return sorted(documents, key=lambda document: (document.source_path, document.document_id))


def _sort_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Sort chunks into a deterministic persisted order."""
    return sorted(chunks, key=lambda chunk: (str(chunk.metadata.get("source_path", "")), chunk.chunk_id))


def _sort_entries(entries: list[IndexedChunk]) -> list[IndexedChunk]:
    """Sort indexed entries into a deterministic persisted order."""
    return sorted(entries, key=lambda entry: (str(entry.chunk.metadata.get("source_path", "")), entry.chunk.chunk_id))


def _content_fingerprint(payload: str) -> str:
    """Hash one change-detection payload deterministically."""
    from hashlib import sha256

    return sha256(payload.encode("utf-8")).hexdigest()
