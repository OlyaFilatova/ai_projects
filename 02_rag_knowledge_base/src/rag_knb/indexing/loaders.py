"""Document loading primitives for supported local files."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable
from pathlib import Path

import jsonlines

from rag_knb.errors import DocumentLoadError, UnsupportedFileTypeError
from rag_knb.library_policies import validate_document_file_size
from rag_knb.models import Document, StructuredRecord
from rag_knb.pathing import is_path_within_allowed_root

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".jsonl"}
FIELD_NAME_PATTERN = re.compile(r"[^a-z0-9]+")


def load_document(
    path: Path,
    max_document_bytes: int | None = None,
    allowed_root: Path | None = None,
) -> Document:
    """Load one supported document from disk."""
    documents = _load_path_documents(
        path,
        max_document_bytes=max_document_bytes,
        allowed_root=allowed_root,
    )
    if len(documents) != 1:
        raise DocumentLoadError(
            f"Structured source '{path.expanduser()}' expands to {len(documents)} records. "
            "Use load_documents(...) or ingest_paths(...) for multi-record files."
        )
    return documents[0]


def load_documents(
    paths: list[Path],
    max_document_bytes: int | None = None,
    allowed_root: Path | None = None,
) -> list[Document]:
    """Load multiple supported documents."""
    documents: list[Document] = []
    for path in paths:
        documents.extend(
            _load_path_documents(
                path,
                max_document_bytes=max_document_bytes,
                allowed_root=allowed_root,
            )
        )
    return documents


def _load_path_documents(
    path: Path,
    max_document_bytes: int | None = None,
    allowed_root: Path | None = None,
) -> list[Document]:
    """Load one path into one or more indexable document payloads."""
    resolved_path = path.expanduser()
    if not is_path_within_allowed_root(resolved_path, allowed_root):
        raise DocumentLoadError(
            f"Document '{resolved_path}' is outside the configured allowed root '{allowed_root}'."
        )
    suffix = resolved_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type for '{resolved_path}'. Supported types: .txt, .md, .markdown, .json, .jsonl."
        )

    if max_document_bytes is not None:
        file_size = _with_document_io_errors(resolved_path, lambda: resolved_path.stat().st_size)
        validate_document_file_size(resolved_path, file_size, max_document_bytes)

    content = _with_document_io_errors(
        resolved_path,
        lambda: resolved_path.read_text(encoding="utf-8"),
    )

    if not content.strip():
        raise DocumentLoadError(f"Document '{resolved_path}' is empty and cannot be indexed.")
    if suffix in {".json", ".jsonl"}:
        return _load_structured_documents(resolved_path, suffix, content)
    return [
        Document(
            document_id=resolved_path.stem,
            source_path=str(resolved_path),
            content=content,
            metadata=_build_document_metadata(resolved_path, suffix),
        )
    ]


def _load_structured_documents(resolved_path: Path, suffix: str, content: str) -> list[Document]:
    """Load a JSON or JSONL file into one document per structured record."""
    records = _parse_structured_records(resolved_path, suffix, content)
    if not records:
        raise DocumentLoadError(f"Structured source '{resolved_path}' contains no indexable records.")
    documents: list[Document] = []
    for record_index, record_payload in enumerate(records):
        fields = _coerce_record_fields(record_payload)
        if not fields:
            continue
        raw_record_id = str(record_payload.get("id") or record_payload.get("record_id") or record_index)
        normalized_record_id = _normalize_identifier(raw_record_id)
        metadata = _build_structured_metadata(resolved_path, suffix, record_index, raw_record_id, fields)
        record = StructuredRecord(
            record_id=raw_record_id,
            source_path=str(resolved_path),
            fields=fields,
            metadata=metadata,
        )
        documents.append(record.to_document(document_id=f"{resolved_path.stem}-{normalized_record_id}"))
    if not documents:
        raise DocumentLoadError(f"Structured source '{resolved_path}' contains no scalar record fields.")
    return documents


def _parse_structured_records(resolved_path: Path, suffix: str, content: str) -> list[dict[str, object]]:
    """Parse structured JSON content into a list of object records."""
    try:
        if suffix == ".jsonl":
            parsed_records = _parse_jsonl_records(content)
        else:
            parsed_records = _normalize_structured_payload(json.loads(content))
    except (json.JSONDecodeError, jsonlines.InvalidLineError) as exc:
        raise DocumentLoadError(
            f"Structured source '{resolved_path}' contains invalid JSON: {_json_error_message(exc)}."
        ) from exc
    return _validated_structured_record_objects(resolved_path, parsed_records)


def _parse_jsonl_records(content: str) -> list[object]:
    """Parse one JSON object per non-empty line."""
    records: list[object] = []
    with jsonlines.Reader(io.StringIO(content)) as reader:
        for record in reader.iter(skip_empty=True):
            records.append(record)
    return records


def _normalize_structured_payload(parsed_payload: object) -> list[object]:
    """Normalize one parsed JSON payload into a record list."""
    if isinstance(parsed_payload, list):
        return parsed_payload
    return [parsed_payload]


def _json_error_message(error: json.JSONDecodeError | jsonlines.InvalidLineError) -> str:
    """Normalize JSON and JSONL parser errors into one stable user-facing message."""
    if isinstance(error, json.JSONDecodeError):
        return error.msg
    return str(error)


def _validated_structured_record_objects(
    resolved_path: Path,
    parsed_records: list[object],
) -> list[dict[str, object]]:
    """Validate that all parsed structured records are JSON objects."""
    if all(isinstance(item, dict) for item in parsed_records):
        return [item for item in parsed_records if isinstance(item, dict)]
    raise DocumentLoadError(
        f"Structured source '{resolved_path}' must contain JSON objects, not arrays or scalars alone."
    )


def _coerce_record_fields(record_payload: dict[str, object]) -> dict[str, str]:
    """Coerce one JSON object into deterministic string fields."""
    fields: dict[str, str] = {}
    for field_name, value in record_payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            fields[str(field_name)] = str(value)
            continue
        fields[str(field_name)] = json.dumps(value, sort_keys=True)
    return fields


def _build_structured_metadata(
    resolved_path: Path,
    suffix: str,
    record_index: int,
    record_id: str,
    fields: dict[str, str],
) -> dict[str, str | int | list[str]]:
    """Build deterministic metadata for one structured record document."""
    metadata: dict[str, str | int | list[str]] = {
        **_build_document_metadata(resolved_path, suffix, format_override="structured"),
        "record_id": record_id,
        "record_index": record_index,
        "structured_fields": [_normalize_identifier(field_name) for field_name in fields],
    }
    for field_name, value in fields.items():
        metadata[f"field_{_normalize_identifier(field_name)}"] = value
    return metadata


def _with_document_io_errors[T](resolved_path: Path, operation: Callable[[], T]) -> T:
    """Run one document filesystem operation with stable user-facing errors."""
    try:
        return operation()
    except FileNotFoundError as exc:
        raise DocumentLoadError(f"Document not found: '{resolved_path}'.") from exc
    except OSError as exc:
        raise DocumentLoadError(f"Failed to read document '{resolved_path}': {exc}.") from exc


def _build_document_metadata(
    resolved_path: Path,
    suffix: str,
    *,
    format_override: str | None = None,
) -> dict[str, str]:
    """Build the deterministic document metadata payload."""
    return {
        "file_name": resolved_path.name,
        "file_type": suffix.lstrip("."),
        "format": format_override or ("markdown" if suffix in {".md", ".markdown"} else "text"),
    }


def _normalize_identifier(value: str) -> str:
    """Normalize one free-form identifier into a stable lowercase token."""
    normalized = FIELD_NAME_PATTERN.sub("_", value.lower()).strip("_")
    return normalized or "record"
