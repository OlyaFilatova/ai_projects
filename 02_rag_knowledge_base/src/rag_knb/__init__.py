"""RAG KnB package."""

from rag_knb.answers.answering import ExtractiveAnswerer
from rag_knb.config import RuntimeConfig
from rag_knb.errors import DocumentLoadError, RagKnbError, UnsupportedFileTypeError
from rag_knb.indexing.chunking import chunk_document, chunk_documents
from rag_knb.indexing.loaders import load_document, load_documents
from rag_knb.models import (
    AnswerResult,
    Chunk,
    ConversationTurn,
    Document,
    IngestResult,
    RefreshResult,
    RetrievalResult,
)
from rag_knb.retrieval_engine.embeddings import DeterministicEmbedder
from rag_knb.retrieval_engine.retrieval import Retriever
from rag_knb.retrieval_engine.vector_store import InMemoryVectorStore
from rag_knb.service import KnowledgeBaseService, ServiceStatus

__all__ = [
    "AnswerResult",
    "Chunk",
    "ConversationTurn",
    "DeterministicEmbedder",
    "Document",
    "DocumentLoadError",
    "ExtractiveAnswerer",
    "InMemoryVectorStore",
    "IngestResult",
    "KnowledgeBaseService",
    "RagKnbError",
    "RefreshResult",
    "Retriever",
    "RetrievalResult",
    "RuntimeConfig",
    "ServiceStatus",
    "UnsupportedFileTypeError",
    "chunk_document",
    "chunk_documents",
    "load_document",
    "load_documents",
]
