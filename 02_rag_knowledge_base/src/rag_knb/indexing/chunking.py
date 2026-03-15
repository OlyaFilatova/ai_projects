"""Deterministic text chunking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module

from rag_knb.errors import RagKnbError
from rag_knb.models import Chunk, Document
from rag_knb.optional_dependencies import has_langchain_text_splitters


class ChunkingError(RagKnbError):
    """Raised when chunking configuration is invalid."""


PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n+")
SENTENCE_PATTERN = re.compile(r".+?(?:[.!?](?=\s|$)|\n{2,}|$)", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """One sentence-like span with offsets inside the stripped document content."""

    text: str
    start_offset: int
    end_offset: int


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split a document into deterministic overlap-aware chunks."""
    if chunk_size <= 0:
        raise ChunkingError("Chunk size must be greater than zero.")
    if chunk_overlap < 0:
        raise ChunkingError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("Chunk overlap must be smaller than chunk size.")

    content = document.content.strip()
    if not content:
        return []
    if document.metadata.get("format") == "structured":
        return _chunk_structured_record(document, content)

    sentences = _split_sentences(content)
    if len(sentences) > 1:
        sentence_chunks = _chunk_by_sentences(document, sentences, chunk_size, chunk_overlap)
        if sentence_chunks:
            return sentence_chunks

    if has_langchain_text_splitters():
        return _chunk_with_langchain(document, content, chunk_size, chunk_overlap)

    paragraphs = _split_paragraphs(content)
    if len(paragraphs) > 1:
        return _chunk_by_paragraphs(document, paragraphs, chunk_size, chunk_overlap)
    return _chunk_by_fixed_width(document, content, chunk_size, chunk_overlap)


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk multiple documents with the same chunking configuration."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return chunks


def _split_paragraphs(content: str) -> list[str]:
    """Split content into normalized paragraphs."""
    return [paragraph.strip() for paragraph in PARAGRAPH_BREAK_PATTERN.split(content) if paragraph.strip()]


def _split_sentences(content: str) -> list[SentenceSpan]:
    """Split content into deterministic sentence-like spans with stable offsets."""
    sentences: list[SentenceSpan] = []
    search_start = 0
    for match in SENTENCE_PATTERN.finditer(content):
        raw_text = match.group(0)
        stripped_text = raw_text.strip()
        if not stripped_text:
            continue
        start_offset = content.find(stripped_text, search_start)
        if start_offset < 0:
            start_offset = content.find(stripped_text)
        end_offset = start_offset + len(stripped_text)
        sentences.append(
            SentenceSpan(
                text=stripped_text,
                start_offset=max(start_offset, 0),
                end_offset=max(end_offset, 0),
            )
        )
        search_start = max(end_offset, 0)
    return sentences


def _chunk_with_langchain(
    document: Document,
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk content with LangChain's recursive splitter when available."""
    splitter_module = import_module("langchain_text_splitters")
    splitter_class = splitter_module.RecursiveCharacterTextSplitter

    splitter = splitter_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )
    chunk_texts = splitter.split_text(content)
    chunks: list[Chunk] = []
    search_start = 0
    for chunk_index, chunk_text in enumerate(chunk_texts):
        start_offset = content.find(chunk_text, search_start)
        if start_offset < 0:
            start_offset = content.find(chunk_text)
        end_offset = start_offset + len(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{chunk_index}",
                document_id=document.document_id,
                content=chunk_text,
                start_offset=max(start_offset, 0),
                end_offset=max(end_offset, 0),
                metadata={
                    "chunk_index": chunk_index,
                    "paragraph_count": chunk_text.count("\n\n") + 1,
                    "parent_document_id": document.document_id,
                    "parent_content": document.content.strip(),
                    "source_path": document.source_path,
                    "splitter": "langchain_recursive_character",
                    **document.metadata,
                },
            )
        )
        search_start = max(start_offset, 0) + 1
    return chunks


def _chunk_structured_record(document: Document, content: str) -> list[Chunk]:
    """Keep small structured records intact as one field-aware chunk."""
    return [
        Chunk(
            chunk_id=f"{document.document_id}:0",
            document_id=document.document_id,
            content=content,
            start_offset=0,
            end_offset=len(content),
            metadata={
                "chunk_index": 0,
                "paragraph_count": content.count("\n\n") + 1,
                "parent_document_id": document.document_id,
                "parent_content": document.content.strip(),
                "source_path": document.source_path,
                "splitter": "structured_record",
                **document.metadata,
            },
        )
    ]


def _chunk_by_paragraphs(
    document: Document,
    paragraphs: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Build deterministic paragraph-aware chunks when structure is available."""
    oversized_paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > chunk_size]
    if oversized_paragraphs:
        return _chunk_by_fixed_width(document, document.content.strip(), chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    paragraph_index = 0
    chunk_index = 0
    while paragraph_index < len(paragraphs):
        chunk_paragraphs: list[str] = []
        consumed_length = 0
        next_index = paragraph_index
        while next_index < len(paragraphs):
            paragraph = paragraphs[next_index]
            candidate_length = consumed_length + len(paragraph) + (2 if chunk_paragraphs else 0)
            if chunk_paragraphs and candidate_length > chunk_size:
                break
            chunk_paragraphs.append(paragraph)
            consumed_length = candidate_length
            next_index += 1

        chunk_text = "\n\n".join(chunk_paragraphs)
        start_offset = document.content.find(chunk_paragraphs[0])
        end_offset = start_offset + len(chunk_text)
        chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{chunk_index}",
                    document_id=document.document_id,
                    content=chunk_text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata={
                        "chunk_index": chunk_index,
                        "paragraph_count": len(chunk_paragraphs),
                        "parent_document_id": document.document_id,
                        "parent_content": document.content.strip(),
                        "source_path": document.source_path,
                        **document.metadata,
                    },
                )
        )
        chunk_index += 1
        next_paragraph_index = _next_paragraph_start(next_index, chunk_paragraphs, chunk_overlap)
        paragraph_index = max(next_paragraph_index, paragraph_index + 1)
    return chunks


def _chunk_by_sentences(
    document: Document,
    sentences: list[SentenceSpan],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Build deterministic sentence-aware chunks when sentence boundaries are reliable."""
    if any(len(sentence.text) > chunk_size for sentence in sentences):
        return []

    chunks: list[Chunk] = []
    sentence_index = 0
    chunk_index = 0
    while sentence_index < len(sentences):
        chunk_sentences: list[SentenceSpan] = []
        consumed_length = 0
        next_index = sentence_index
        while next_index < len(sentences):
            sentence = sentences[next_index]
            candidate_length = consumed_length + len(sentence.text) + (1 if chunk_sentences else 0)
            if chunk_sentences and candidate_length > chunk_size:
                break
            chunk_sentences.append(sentence)
            consumed_length = candidate_length
            next_index += 1

        start_offset = chunk_sentences[0].start_offset
        end_offset = chunk_sentences[-1].end_offset
        chunk_text = document.content.strip()[start_offset:end_offset].strip()
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{chunk_index}",
                document_id=document.document_id,
                content=chunk_text,
                start_offset=start_offset,
                end_offset=end_offset,
                metadata={
                    "chunk_index": chunk_index,
                    "paragraph_count": chunk_text.count("\n\n") + 1,
                    "sentence_count": len(chunk_sentences),
                    "parent_document_id": document.document_id,
                    "parent_content": document.content.strip(),
                    "source_path": document.source_path,
                    "splitter": "sentence_aware",
                    **document.metadata,
                },
            )
        )
        chunk_index += 1
        next_sentence_index = _next_sentence_start(next_index, chunk_sentences, chunk_overlap)
        sentence_index = max(next_sentence_index, sentence_index + 1)
    return chunks


def _next_paragraph_start(
    next_index: int,
    chunk_paragraphs: list[str],
    chunk_overlap: int,
) -> int:
    """Determine the next paragraph start index using overlap-aware carryover."""
    if chunk_overlap <= 0:
        return next_index
    carried_length = 0
    carried_count = 0
    for paragraph in reversed(chunk_paragraphs):
        paragraph_length = len(paragraph) + (2 if carried_count else 0)
        if carried_length + paragraph_length > chunk_overlap:
            break
        carried_length += paragraph_length
        carried_count += 1
    if carried_count == 0:
        return next_index
    return max(next_index - carried_count, 0)


def _next_sentence_start(
    next_index: int,
    chunk_sentences: list[SentenceSpan],
    chunk_overlap: int,
) -> int:
    """Determine the next sentence start index using overlap-aware carryover."""
    if chunk_overlap <= 0:
        return next_index
    carried_length = 0
    carried_count = 0
    for sentence in reversed(chunk_sentences):
        sentence_length = len(sentence.text) + (1 if carried_count else 0)
        if carried_length + sentence_length > chunk_overlap:
            break
        carried_length += sentence_length
        carried_count += 1
    if carried_count == 0:
        return next_index
    return max(next_index - carried_count, 0)


def _chunk_by_fixed_width(
    document: Document,
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split content into fixed-width chunks as a deterministic fallback."""
    chunks: list[Chunk] = []
    start_offset = 0
    step = chunk_size - chunk_overlap
    chunk_index = 0
    while start_offset < len(content):
        end_offset = min(len(content), start_offset + chunk_size)
        chunk_text = content[start_offset:end_offset].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{chunk_index}",
                    document_id=document.document_id,
                    content=chunk_text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata={
                        "chunk_index": chunk_index,
                        "paragraph_count": 0,
                        "parent_document_id": document.document_id,
                        "parent_content": document.content.strip(),
                        "source_path": document.source_path,
                        **document.metadata,
                    },
                )
            )
            chunk_index += 1
        if end_offset >= len(content):
            break
        start_offset += step
    return chunks
