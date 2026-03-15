"""Command-line interface for RAG KnB."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Annotated, Any, Protocol, cast

import typer
from rich.console import Console
from rich.table import Table
from typer.testing import CliRunner

from rag_knb.errors import RagKnbError, ValidationError
from rag_knb.pathing import coerce_paths, resolve_data_dir
from rag_knb.service import KnowledgeBaseService
from rag_knb.service_factory import build_service_from_options

app = typer.Typer(
    add_completion=False,
    help="Local retrieval-augmented knowledge base.",
    no_args_is_help=True,
)
_ACTIVE_SERVICE: KnowledgeBaseService | None = None
_OUTPUT_CONSOLE = Console(color_system=None, highlight=False, markup=False, width=240)


class _IngestArgs(Protocol):
    paths: list[str]
    data_dir: str | None


class _ListArgs(Protocol):
    data_dir: str | None


class _RemoveArgs(Protocol):
    data_dir: str | None
    document_ids: list[str]


class _AskArgs(Protocol):
    document_paths: list[str]
    data_dir: str | None
    question: str
    metadata_filters: list[str]
    show_diagnostics: bool


def _version_callback(value: bool) -> None:
    """Print the CLI version when requested."""
    if value:
        typer.echo("rag-knb 0.1.0")
        raise typer.Exit()


def _runtime_args(**kwargs: object) -> SimpleNamespace:
    """Build one attribute-based runtime options object."""
    return SimpleNamespace(**kwargs)


def _build_service_from_args(
    args: object,
    service: KnowledgeBaseService | None,
) -> KnowledgeBaseService:
    """Resolve the active service, applying config overrides when provided."""
    return build_service_from_options(args, existing_service=service)


def _run_with_service(
    command: Any,
    args: object,
) -> None:
    """Run one CLI command with shared error handling and service construction."""
    if command is handle_status:
        raise typer.Exit(handle_status(args))
    try:
        active_service = _build_service_from_args(args, _ACTIVE_SERVICE)
        raise typer.Exit(command(args, active_service))
    except RagKnbError as error:
        _OUTPUT_CONSOLE.print(f"Error: {error}")
        raise typer.Exit(1) from error


def _parse_filters(filter_values: list[str]) -> dict[str, str]:
    """Parse repeated key=value filters into a metadata filter dictionary."""
    parsed_filters: dict[str, str] = {}
    for filter_value in filter_values:
        if "=" not in filter_value:
            raise ValidationError("Metadata filters must use key=value form.")
        key, value = filter_value.split("=", maxsplit=1)
        if not key or not value:
            raise ValidationError("Metadata filters must use key=value form.")
        parsed_filters[key] = value
    return parsed_filters


def handle_status(_: object) -> int:
    """Render the current high-level service status."""
    status = KnowledgeBaseService().status()
    _OUTPUT_CONSOLE.print(status.summary)
    return 0


def handle_ingest(args: object, service: KnowledgeBaseService) -> int:
    """Ingest one or more supported documents."""
    typed_args = cast(_IngestArgs, args)
    ingest_result = service.ingest_paths(coerce_paths(typed_args.paths))
    data_dir = typed_args.data_dir
    if data_dir:
        service.save(resolve_data_dir(data_dir, service.config.data_dir))
    _OUTPUT_CONSOLE.print(
        f"Ingested {len(ingest_result.documents)} document(s) into {len(ingest_result.chunks)} chunk(s)."
    )
    return 0


def handle_list_documents(args: object, service: KnowledgeBaseService) -> int:
    """List indexed documents, optionally loading a persisted knowledge base first."""
    typed_args = cast(_ListArgs, args)
    data_dir = typed_args.data_dir
    if data_dir:
        service.load(resolve_data_dir(data_dir, service.config.data_dir))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Document ID")
    table.add_column("Metadata")
    for document in service.list_documents():
        table.add_row(document.document_id, str(document.metadata))
    _OUTPUT_CONSOLE.print(table)
    return 0


def handle_remove_document(args: object, service: KnowledgeBaseService) -> int:
    """Remove documents by id, optionally persisting the updated state back to disk."""
    typed_args = cast(_RemoveArgs, args)
    data_dir = typed_args.data_dir
    if not data_dir:
        raise ValidationError(
            "remove-document requires --data-dir so the updated knowledge base can be saved."
        )
    resolved_data_dir = resolve_data_dir(data_dir, service.config.data_dir)
    service.load(resolved_data_dir)
    remaining_documents = service.remove_documents(typed_args.document_ids)
    service.save(resolved_data_dir)
    _OUTPUT_CONSOLE.print(f"Remaining documents: {len(remaining_documents)}")
    return 0


def handle_ask(args: object, service: KnowledgeBaseService) -> int:
    """Ask a grounded question against the indexed chunks."""
    typed_args = cast(_AskArgs, args)
    document_paths = typed_args.document_paths
    data_dir = typed_args.data_dir
    if document_paths and data_dir:
        raise ValidationError("Choose either --document or --data-dir for ask, not both.")
    if data_dir:
        service.load(resolve_data_dir(data_dir, service.config.data_dir))
    if document_paths:
        service.ingest_paths(coerce_paths(document_paths))
    answer = service.ask(
        typed_args.question,
        metadata_filters=_parse_filters(typed_args.metadata_filters),
    )
    _OUTPUT_CONSOLE.print(answer.answer_text)
    if answer.reason == "matched":
        _OUTPUT_CONSOLE.print(f"Matches: {len(answer.matches)}")
    if typed_args.show_diagnostics:
        _OUTPUT_CONSOLE.print("Diagnostics:")
        _OUTPUT_CONSOLE.print_json(data=answer.diagnostics)
    if answer.reason == "matched":
        return 0
    if answer.reason == "empty":
        return 2
    return 1


@app.callback()
def main_callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Show the CLI version and exit.",
            is_eager=True,
            callback=_version_callback,
        ),
    ] = None,
) -> None:
    """Top-level CLI callback."""
    del version


@app.command("status")
def status_command() -> None:
    """Show project bootstrap status."""
    args = _runtime_args()
    _run_with_service(handle_status, args)


@app.command("ingest")
def ingest_command(
    paths: Annotated[list[str], typer.Argument(..., help="Paths to TXT or Markdown documents.")],
    data_dir: Annotated[str | None, typer.Option(help="Local data directory for persisted knowledge-base state.")] = None,
    chunk_size: Annotated[int | None, typer.Option(help="Chunk size to use for ingestion and indexing.")] = None,
    chunk_overlap: Annotated[int | None, typer.Option(help="Chunk overlap to use for ingestion and indexing.")] = None,
    embedding_backend: Annotated[
        str | None,
        typer.Option(help="Embedding backend to use for retrieval.", case_sensitive=False),
    ] = None,
    answer_mode: Annotated[
        str | None,
        typer.Option(help="Answer mode to use after retrieval.", case_sensitive=False),
    ] = None,
    answer_verbosity: Annotated[
        str | None,
        typer.Option(help="How much grounded supporting text to include in matched answers.", case_sensitive=False),
    ] = None,
    retrieval_strategy: Annotated[
        str | None,
        typer.Option(help="Retrieval strategy to use for candidate scoring.", case_sensitive=False),
    ] = None,
    vector_backend: Annotated[
        str | None,
        typer.Option(help="Vector-store backend to use for retrieval.", case_sensitive=False),
    ] = None,
    llm_model: Annotated[str | None, typer.Option(help="Model name for built-in generative answering.")] = None,
    llm_base_url: Annotated[str | None, typer.Option(help="Base URL for the built-in OpenAI-compatible chat API.")] = None,
    allowed_root: Annotated[str | None, typer.Option(help="Optional root directory restriction for source documents and persisted data.")] = None,
    max_question_length: Annotated[int | None, typer.Option(help="Maximum accepted question length for library query validation.")] = None,
    max_retrieval_limit: Annotated[int | None, typer.Option(help="Maximum retrieval limit accepted by library query validation.")] = None,
    max_documents_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of documents accepted in one ingest call.")] = None,
    max_document_bytes: Annotated[int | None, typer.Option(help="Maximum source-document size in bytes accepted by the library.")] = None,
    max_chunks_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of chunks accepted from one ingest call.")] = None,
    llm_request_timeout_seconds: Annotated[int | None, typer.Option(help="Timeout in seconds for built-in LLM HTTP requests.")] = None,
) -> None:
    """Ingest one or more documents."""
    args = _runtime_args(
        paths=paths,
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_backend=embedding_backend,
        answer_mode=answer_mode,
        answer_verbosity=answer_verbosity,
        retrieval_strategy=retrieval_strategy,
        vector_backend=vector_backend,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        allowed_root=allowed_root,
        max_question_length=max_question_length,
        max_retrieval_limit=max_retrieval_limit,
        max_documents_per_ingest=max_documents_per_ingest,
        max_document_bytes=max_document_bytes,
        max_chunks_per_ingest=max_chunks_per_ingest,
        llm_request_timeout_seconds=llm_request_timeout_seconds,
    )
    _run_with_service(handle_ingest, args)


@app.command("list-documents")
def list_documents_command(
    data_dir: Annotated[str | None, typer.Option(help="Local data directory for persisted knowledge-base state.")] = None,
    chunk_size: Annotated[int | None, typer.Option(help="Chunk size to use for ingestion and indexing.")] = None,
    chunk_overlap: Annotated[int | None, typer.Option(help="Chunk overlap to use for ingestion and indexing.")] = None,
    embedding_backend: Annotated[
        str | None,
        typer.Option(help="Embedding backend to use for retrieval.", case_sensitive=False),
    ] = None,
    answer_mode: Annotated[
        str | None,
        typer.Option(help="Answer mode to use after retrieval.", case_sensitive=False),
    ] = None,
    answer_verbosity: Annotated[
        str | None,
        typer.Option(help="How much grounded supporting text to include in matched answers.", case_sensitive=False),
    ] = None,
    retrieval_strategy: Annotated[
        str | None,
        typer.Option(help="Retrieval strategy to use for candidate scoring.", case_sensitive=False),
    ] = None,
    vector_backend: Annotated[
        str | None,
        typer.Option(help="Vector-store backend to use for retrieval.", case_sensitive=False),
    ] = None,
    llm_model: Annotated[str | None, typer.Option(help="Model name for built-in generative answering.")] = None,
    llm_base_url: Annotated[str | None, typer.Option(help="Base URL for the built-in OpenAI-compatible chat API.")] = None,
    allowed_root: Annotated[str | None, typer.Option(help="Optional root directory restriction for source documents and persisted data.")] = None,
    max_question_length: Annotated[int | None, typer.Option(help="Maximum accepted question length for library query validation.")] = None,
    max_retrieval_limit: Annotated[int | None, typer.Option(help="Maximum retrieval limit accepted by library query validation.")] = None,
    max_documents_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of documents accepted in one ingest call.")] = None,
    max_document_bytes: Annotated[int | None, typer.Option(help="Maximum source-document size in bytes accepted by the library.")] = None,
    max_chunks_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of chunks accepted from one ingest call.")] = None,
    llm_request_timeout_seconds: Annotated[int | None, typer.Option(help="Timeout in seconds for built-in LLM HTTP requests.")] = None,
) -> None:
    """List indexed documents."""
    args = _runtime_args(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_backend=embedding_backend,
        answer_mode=answer_mode,
        answer_verbosity=answer_verbosity,
        retrieval_strategy=retrieval_strategy,
        vector_backend=vector_backend,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        allowed_root=allowed_root,
        max_question_length=max_question_length,
        max_retrieval_limit=max_retrieval_limit,
        max_documents_per_ingest=max_documents_per_ingest,
        max_document_bytes=max_document_bytes,
        max_chunks_per_ingest=max_chunks_per_ingest,
        llm_request_timeout_seconds=llm_request_timeout_seconds,
    )
    _run_with_service(handle_list_documents, args)


@app.command("remove-document")
def remove_document_command(
    document_ids: Annotated[list[str], typer.Argument(..., help="Document IDs to remove.")],
    data_dir: Annotated[str | None, typer.Option(help="Local data directory for persisted knowledge-base state.")] = None,
    chunk_size: Annotated[int | None, typer.Option(help="Chunk size to use for ingestion and indexing.")] = None,
    chunk_overlap: Annotated[int | None, typer.Option(help="Chunk overlap to use for ingestion and indexing.")] = None,
    embedding_backend: Annotated[
        str | None,
        typer.Option(help="Embedding backend to use for retrieval.", case_sensitive=False),
    ] = None,
    answer_mode: Annotated[
        str | None,
        typer.Option(help="Answer mode to use after retrieval.", case_sensitive=False),
    ] = None,
    answer_verbosity: Annotated[
        str | None,
        typer.Option(help="How much grounded supporting text to include in matched answers.", case_sensitive=False),
    ] = None,
    retrieval_strategy: Annotated[
        str | None,
        typer.Option(help="Retrieval strategy to use for candidate scoring.", case_sensitive=False),
    ] = None,
    vector_backend: Annotated[
        str | None,
        typer.Option(help="Vector-store backend to use for retrieval.", case_sensitive=False),
    ] = None,
    llm_model: Annotated[str | None, typer.Option(help="Model name for built-in generative answering.")] = None,
    llm_base_url: Annotated[str | None, typer.Option(help="Base URL for the built-in OpenAI-compatible chat API.")] = None,
    allowed_root: Annotated[str | None, typer.Option(help="Optional root directory restriction for source documents and persisted data.")] = None,
    max_question_length: Annotated[int | None, typer.Option(help="Maximum accepted question length for library query validation.")] = None,
    max_retrieval_limit: Annotated[int | None, typer.Option(help="Maximum retrieval limit accepted by library query validation.")] = None,
    max_documents_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of documents accepted in one ingest call.")] = None,
    max_document_bytes: Annotated[int | None, typer.Option(help="Maximum source-document size in bytes accepted by the library.")] = None,
    max_chunks_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of chunks accepted from one ingest call.")] = None,
    llm_request_timeout_seconds: Annotated[int | None, typer.Option(help="Timeout in seconds for built-in LLM HTTP requests.")] = None,
) -> None:
    """Remove documents by id."""
    args = _runtime_args(
        document_ids=document_ids,
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_backend=embedding_backend,
        answer_mode=answer_mode,
        answer_verbosity=answer_verbosity,
        retrieval_strategy=retrieval_strategy,
        vector_backend=vector_backend,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        allowed_root=allowed_root,
        max_question_length=max_question_length,
        max_retrieval_limit=max_retrieval_limit,
        max_documents_per_ingest=max_documents_per_ingest,
        max_document_bytes=max_document_bytes,
        max_chunks_per_ingest=max_chunks_per_ingest,
        llm_request_timeout_seconds=llm_request_timeout_seconds,
    )
    _run_with_service(handle_remove_document, args)


@app.command("ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="Question to answer from indexed chunks.")],
    document_paths: Annotated[
        list[str] | None,
        typer.Option("--document", help="Optional document to ingest before answering. Repeat for multiple files."),
    ] = None,
    show_diagnostics: Annotated[
        bool,
        typer.Option(help="Show retrieval diagnostics for the answer."),
    ] = False,
    metadata_filters: Annotated[
        list[str] | None,
        typer.Option("--filter", help="Exact metadata filter in key=value form. Repeat for multiple filters."),
    ] = None,
    data_dir: Annotated[str | None, typer.Option(help="Local data directory for persisted knowledge-base state.")] = None,
    chunk_size: Annotated[int | None, typer.Option(help="Chunk size to use for ingestion and indexing.")] = None,
    chunk_overlap: Annotated[int | None, typer.Option(help="Chunk overlap to use for ingestion and indexing.")] = None,
    embedding_backend: Annotated[
        str | None,
        typer.Option(help="Embedding backend to use for retrieval.", case_sensitive=False),
    ] = None,
    answer_mode: Annotated[
        str | None,
        typer.Option(help="Answer mode to use after retrieval.", case_sensitive=False),
    ] = None,
    answer_verbosity: Annotated[
        str | None,
        typer.Option(help="How much grounded supporting text to include in matched answers.", case_sensitive=False),
    ] = None,
    retrieval_strategy: Annotated[
        str | None,
        typer.Option(help="Retrieval strategy to use for candidate scoring.", case_sensitive=False),
    ] = None,
    vector_backend: Annotated[
        str | None,
        typer.Option(help="Vector-store backend to use for retrieval.", case_sensitive=False),
    ] = None,
    llm_model: Annotated[str | None, typer.Option(help="Model name for built-in generative answering.")] = None,
    llm_base_url: Annotated[str | None, typer.Option(help="Base URL for the built-in OpenAI-compatible chat API.")] = None,
    allowed_root: Annotated[str | None, typer.Option(help="Optional root directory restriction for source documents and persisted data.")] = None,
    max_question_length: Annotated[int | None, typer.Option(help="Maximum accepted question length for library query validation.")] = None,
    max_retrieval_limit: Annotated[int | None, typer.Option(help="Maximum retrieval limit accepted by library query validation.")] = None,
    max_documents_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of documents accepted in one ingest call.")] = None,
    max_document_bytes: Annotated[int | None, typer.Option(help="Maximum source-document size in bytes accepted by the library.")] = None,
    max_chunks_per_ingest: Annotated[int | None, typer.Option(help="Maximum number of chunks accepted from one ingest call.")] = None,
    llm_request_timeout_seconds: Annotated[int | None, typer.Option(help="Timeout in seconds for built-in LLM HTTP requests.")] = None,
) -> None:
    """Ask a grounded question."""
    resolved_document_paths = document_paths or []
    resolved_metadata_filters = metadata_filters or []
    args = _runtime_args(
        question=question,
        document_paths=resolved_document_paths,
        show_diagnostics=show_diagnostics,
        metadata_filters=resolved_metadata_filters,
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_backend=embedding_backend,
        answer_mode=answer_mode,
        answer_verbosity=answer_verbosity,
        retrieval_strategy=retrieval_strategy,
        vector_backend=vector_backend,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        allowed_root=allowed_root,
        max_question_length=max_question_length,
        max_retrieval_limit=max_retrieval_limit,
        max_documents_per_ingest=max_documents_per_ingest,
        max_document_bytes=max_document_bytes,
        max_chunks_per_ingest=max_chunks_per_ingest,
        llm_request_timeout_seconds=llm_request_timeout_seconds,
    )
    _run_with_service(handle_ask, args)


def run_cli(argv: list[str] | None = None, service: KnowledgeBaseService | None = None) -> int:
    """Execute the CLI against an optional shared service instance."""
    global _ACTIVE_SERVICE
    previous_service = _ACTIVE_SERVICE
    _ACTIVE_SERVICE = service
    try:
        runner = CliRunner()
        result = runner.invoke(app, argv or [])
    finally:
        _ACTIVE_SERVICE = previous_service

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.exit_code


def main() -> None:
    """Execute the Typer application."""
    app()
