# RAG KnB

RAG KnB is a local RAG concepts lab with a CLI. It is designed to explore, test, and demonstrate core retrieval-augmented generation patterns such as document loading, chunking, retrieval, grounding, answer routing, evaluation, conversational follow-ups, and trust-aware handling of retrieved content.

It is intentionally educational and inspectable rather than production-optimized. The goal of the project is to make RAG behavior visible and easy to evolve, not to present itself as a full production RAG platform.

## Current scope

- Installable Python package via `pyproject.toml`
- CLI entry point exposed as `rag-knb`
- Reusable library entry point via `rag_knb.KnowledgeBaseService`
- TXT, Markdown, and simple JSON/JSONL record loading
- Deterministic sentence-aware chunking with paragraph and fixed-width fallbacks
- Deterministic local retrieval by default, plus opt-in hybrid retrieval
- BM25-backed lexical scoring inside the hybrid retrieval path
- Lightweight second-stage reranking over retrieved candidates
- Focused grounded answering with cited extractive responses by default
- Built-in `generative` mode with concise grounded post-processing
- Deterministic prompt-injection filtering for hostile retrieved instructions
- Configurable answer verbosity for extractive answers
- Local persistence to versioned JSON files
- Reloading persisted knowledge bases from disk
- Metadata filters during retrieval
- Document listing and document removal for persisted knowledge bases
- Lightweight structured logging, retrieval timing, and richer diagnostics
- Local evaluation harness with grouped benchmark fixtures for retrieval, routing, reasoning, and safety behavior
- Retrieval-strategy comparison helper for side-by-side local strategy tradeoff checks
- Generated concept-to-code documentation for the current RAG concepts covered by the repo
- LangChain, Hugging Face, and FAISS-backed workflows available in the current package install
- Typer CLI, Rich output formatting, Pydantic-backed runtime config, and mypy-clean source files

## Source layout

Implementation modules are bundled under logical packages:

- `rag_knb.answers`: answer construction, context shaping, prompt injection, and built-in LLM integration
- `rag_knb.indexing`: document loading, chunking, persistence, and embedding lifecycle compatibility
- `rag_knb.retrieval_engine`: embeddings, query rewriting, retrieval, vector stores, and evaluation helpers

The public top-level package API remains available through `rag_knb.__init__`, but the internal source of truth lives in those package directories instead of a flat module layout.

## What does not work yet

- The CLI can persist via `ingest --data-dir` and reload existing knowledge bases, but it still does not expose a dedicated standalone `save` command.
- Optional backends such as Hugging Face embeddings and FAISS require extra dependencies and are not installed by default.
- The evaluation harness is intentionally small and should be extended before using it to justify bigger default-backend changes.
- Observability is intentionally lightweight: structured logs and diagnostics only, with no tracing or metrics backend.

## Base RAG capabilities matrix

This project now covers a fairly broad local baseline RAG workflow, but it still stops short of a production-grade retrieval and reasoning stack.

| Area | Covered now | Still missing / intentionally limited |
| --- | --- | --- |
| Retrieval | Deterministic vector retrieval, optional hybrid retrieval, BM25 lexical scoring, lightweight reranking, metadata filters, metadata-aware ranking bonuses, duplicate-result suppression | No full faceted search, no learned retriever, no large-scale distributed indexing |
| Grounding | Extractive citations, claim-to-support diagnostics, sentence-span support metadata, prompt-injection filtering, trust-boundary diagnostics | No token-level attribution, no formal entailment checker, no external moderation pipeline |
| Answering | Focused extractive answers, optional generative mode with grounding fallback, yes/no formatting, multi-hop aggregation for explicit multi-part questions, and small supported-composition examples | No deep chain-of-thought reasoning, no long-form synthesis planner, no tool-using agent behavior |
| Evaluation | Local fixture-driven harness with retrieval, support, citation, routing, and ranking metrics | Still small and local; not a benchmark-grade evaluation suite |
| Conversation | Conversation-aware retrieval question building, lightweight answer planning for continuation and comparison follow-ups | No full dialogue memory, no long-horizon state management, no user-profile personalization |

## Recent additions

The later stages extended the baseline in a few important ways:

- broader local evaluation coverage for routing, support, citations, and ranking behavior
- explicit compressed evidence selection between retrieval and answering
- sentence-span claim grounding in diagnostics
- deterministic confidence-policy routing with visible reasons
- simple multi-hop evidence aggregation for explicit multi-part questions
- lightweight supported composition for explanation-style questions, without claiming deeper reasoning
- metadata-aware ranking bonuses for mixed corpora
- duplicate-evidence suppression and optional source weighting
- richer trust-boundary diagnostics for hostile retrieved content
- lightweight conversation-aware answer planning for follow-up questions
- retrieval-strategy comparison summaries with raw candidate vs reranked document order
- a generated `docs/rag_concepts_in_codebase.md` map from RAG concepts to code paths and tests

## Remaining gaps

The project still does not claim:

- full semantic verification of every generated claim
- robust long-context synthesis across many documents
- production observability, tracing, or cost controls
- permission-aware multi-tenant retrieval
- sophisticated conversational memory or dialogue policy
- benchmark-scale evaluation strong enough to justify major default-backend decisions on its own

## Install

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

The default install already includes the libraries used by the current local workflow:

- `Typer` for the CLI command surface
- `Rich` for formatted CLI rendering
- `pydantic-settings` for runtime config validation
- `jsonlines` for JSONL parsing
- `RapidFuzz` for narrow typo-tolerant answer heuristics
- `rank-bm25` for the lexical side of hybrid retrieval
- LangChain adapters plus the current Hugging Face and FAISS workflow dependencies

Built-in generative answering uses an OpenAI-compatible chat-completions API. Set `OPENAI_API_KEY` before using `--answer-mode generative`.
The library also adds an explicit safety policy so retrieved documents are treated as evidence, not executable instructions.

## CLI usage

Inspect the CLI:

```bash
rag-knb --help
```

Ask a grounded question from source files:

```bash
rag-knb ask --document ./cats.txt "What does the document say about cats?"
```

The default extractive path returns a short grounded answer with chunk citations instead of dumping whole chunks.
Retrieved context is treated as untrusted data: hostile instructions such as attempts to override policies or reveal secrets are filtered before extractive or generative answering uses them.

Repeat `--document` for multiple files:

```bash
rag-knb ask --document ./cats.txt --document ./dogs.txt "Which animal is energetic?"
```

Show retrieval diagnostics:

```bash
rag-knb ask --document ./cats.txt --show-diagnostics "What does the document say about cats?"
```

Diagnostics are rendered as formatted JSON in the CLI.

Request a more detailed grounded answer:

```bash
rag-knb ask --answer-verbosity verbose --document ./cats.txt "What does the document say about cats?"
```

Try the hybrid retrieval strategy:

```bash
rag-knb ask --retrieval-strategy hybrid --document ./cats.txt --document ./dogs.txt "playful animal"
```

Filter retrieval to matching document metadata:

```bash
rag-knb ask --document ./cats.txt --document ./dogs.txt --filter file_name=cats.txt "Cats nap"
```

Ask from a simple structured JSONL source:

```bash
rag-knb ask --document ./pets.jsonl "Which animal likes salmon?"
```

Use the built-in LLM-backed answer path:

```bash
OPENAI_API_KEY=test rag-knb ask \
  --document ./cats.txt \
  --answer-mode generative \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model llama3.2:3b \
  "Cats nap"
```

Ingest files into the current process:

```bash
rag-knb ingest ./cats.txt ./dogs.txt
```

Persist a knowledge base to disk:

```bash
rag-knb ingest --data-dir ./.rag-knb ./cats.txt ./dogs.txt
```

Query a persisted knowledge base:

```bash
rag-knb ask --data-dir ./.rag-knb "What does the saved knowledge base say about cats?"
```

List persisted documents:

```bash
rag-knb list-documents --data-dir ./.rag-knb
```

Remove a persisted document and save the updated state:

```bash
rag-knb remove-document --data-dir ./.rag-knb cats
```

## Library usage

The main library entry point is `KnowledgeBaseService`.

Example:

```python
from pathlib import Path

from rag_knb import KnowledgeBaseService, RuntimeConfig

service = KnowledgeBaseService(
    config=RuntimeConfig.build(
        data_dir=".rag-knb",
        answer_mode="extractive",
        answer_verbosity="concise",
        retrieval_strategy="vector",
    )
)

service.ingest_paths([Path("cats.txt"), Path("dogs.txt")])
answer = service.ask("Which animal is energetic?")

print(answer.answer_text)
print(answer.diagnostics)
service.save()
```

Incremental refresh for persisted knowledge bases:

```python
service = KnowledgeBaseService(config=RuntimeConfig.build(data_dir=".rag-knb"))
service.load()
refresh_result = service.refresh_paths([Path("cats.txt"), Path("dogs.txt")], remove_missing=True)
service.save()
print(refresh_result.to_dict())
```

Structured sources:

- `.jsonl` files load one JSON object per line as one internal record-document each
- `.json` files may contain one object or a list of objects
- scalar record fields are rendered into field-aware text such as `animal: cat`
- structured fields are also copied into chunk metadata as `field_<name>` so retrieval filters can target them in mixed corpora
- `refresh_paths(...)` can reuse unchanged documents and vectors for persisted corpora, while `ingest_paths(...)` remains the safe full rebuild path

## Enabling Hugging Face and FAISS

The default setup uses the built-in deterministic embedder and in-memory vector store. To switch to Hugging Face embeddings with the FAISS backend, enable them through runtime options:

Enable them in the CLI with runtime options:

```bash
rag-knb ask \
  --document ./cats.txt \
  --embedding-backend huggingface \
  --vector-backend faiss \
  --retrieval-strategy hybrid \
  "What does the document say about cats?"
```

Enable them in library code through `RuntimeConfig`:

```python
from pathlib import Path

from rag_knb import KnowledgeBaseService, RuntimeConfig

service = KnowledgeBaseService(
    config=RuntimeConfig.build(
        embedding_backend="huggingface",
        vector_backend="faiss",
        retrieval_strategy="hybrid",
    )
)

service.ingest_paths([Path("cats.txt")])
answer = service.ask("What does the document say about cats?")
```

Important notes:

- Hugging Face embeddings may download model files on first use.
- FAISS is only used when `vector_backend="faiss"` is selected.
- The workflows are still treated as optional at runtime even though the package metadata currently includes the underlying libraries.
- The recommended semantic workflow in this repo is `embedding_backend="huggingface"` plus `vector_backend="faiss"` and `retrieval_strategy="hybrid"`.
- Semantic retrieval is usually most helpful when questions paraphrase the source text rather than repeating its exact keywords.
- The hybrid path now uses `rank-bm25` for lexical scoring before the lightweight reranker runs.

## Evaluation harness

This repo includes a small local evaluation harness for regression-friendly quality checks. The fixture cases live in `tests/fixtures/evaluation_cases.json` and currently score:

- retrieval relevance
- answer focus
- support visibility
- citation quality
- reason accuracy
- support coverage
- clarification accuracy
- low-confidence accuracy
- precision@k
- recall@k
- mean reciprocal rank (MRR)

The library also includes a reusable retrieval-strategy comparison helper in `rag_knb.retrieval_engine.evaluation.compare_retrieval_strategies(...)`. It records both raw candidate ranking and final reranked document order so you can compare vector, hybrid, and optional semantic workflows on the same fixture cases without turning the repo into a benchmark runner.

Run it locally with:

```bash
./.venv/bin/python -m pytest tests/test_evaluation_harness.py
```

Add future cases by extending the fixture file with representative documents, a question, expected top documents, and expected answer substrings.

Generate the concept-to-code map document locally with:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/generate_rag_concepts_doc.py
```

By default this writes `docs/rag_concepts_in_codebase.md`.

## Default retrieval recommendation

Current recommendation based on the local evaluation harness and the Stage 8 retrieval work:

- Keep the deterministic embedder with the in-memory vector store as the default baseline.
- Use the semantic workflow (`huggingface` + `faiss` + `hybrid`) when your queries paraphrase source text or use broader conceptual wording.

Why the default stays unchanged for now:

- the deterministic path remains the safest zero-download local workflow
- the current evaluation harness is still intentionally small, so it is not yet strong enough evidence for a default migration
- semantic retrieval adds optional dependency and model-download cost that not every local user wants

If the default changes later, the main migration impacts will be:

- heavier first-run setup
- less deterministic behavior across environments
- more need for model caching and dependency guidance

## Integrating into an API

This project no longer ships a built-in FastAPI interface. The intended model is:

- keep `rag_knb` as the library layer
- build your own API around `KnowledgeBaseService`
- map your API request models to `RuntimeConfig.build(...)` and service calls

Minimal example with FastAPI:

```python
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_knb import KnowledgeBaseService, RuntimeConfig


class AskRequest(BaseModel):
    question: str
    document_paths: list[str] = Field(default_factory=list)
    data_dir: str | None = None


app = FastAPI()


@app.post("/ask")
async def ask_question(request: AskRequest) -> dict[str, object]:
    service = KnowledgeBaseService(
        config=RuntimeConfig.build(data_dir=request.data_dir)
    )
    if request.data_dir:
        service.load()
    if request.document_paths:
        service.ingest_paths([Path(path) for path in request.document_paths])
    answer = service.ask(request.question)
    return answer.to_dict()
```

Practical integration guidance:

- keep request validation in your API framework
- keep business logic in `KnowledgeBaseService`
- treat `RuntimeConfig` as the boundary for per-request runtime options
- use `service.save()` and `service.load()` explicitly when your API needs persistence
- add your own auth, rate limiting, and deployment concerns in the API layer

## App/API responsibilities

`rag_knb` is the library layer. An embedding application or API is still responsible for the surrounding operational and security controls.

Library responsibilities:

- document loading, chunking, embeddings, retrieval, persistence, and grounded-answer logic
- runtime config validation and explicit domain errors
- deterministic default behavior and optional backend selection
- library-owned safety limits for question length, retrieval limits, ingest size, chunk budgets, and built-in LLM request timeouts
- optional allowed-root restrictions for source documents and persistence paths
- optional policy control for custom `llm_base_url` values

App/API responsibilities:

- authentication and authorization
- rate limiting and abuse prevention
- tenant isolation and per-user access control
- request-size limits and traffic shaping
- secret management
- deployment/network controls
- deciding whether untrusted users may select paths, `data_dir`, generative mode, or custom LLM endpoints

## Library safety controls

The library enforces these controls when configured through `RuntimeConfig`:

- `max_question_length`: maximum accepted question length
- `max_retrieval_limit`: maximum retrieval limit accepted by the shared service
- `max_documents_per_ingest`: maximum number of documents accepted by one ingest call
- `max_document_bytes`: maximum bytes accepted for one source document
- `max_chunks_per_ingest`: maximum chunk output accepted from one ingest call
- `llm_request_timeout_seconds`: timeout for built-in LLM HTTP requests
- `allowed_root`: optional root restriction for source documents and persistence targets
- `allow_custom_llm_base_url`: policy switch for whether non-default `llm_base_url` values are allowed

Example:

```python
from rag_knb import RuntimeConfig

config = RuntimeConfig.build(
    allowed_root="./sandbox",
    max_question_length=500,
    max_retrieval_limit=5,
    max_documents_per_ingest=3,
    max_document_bytes=200_000,
    max_chunks_per_ingest=500,
    llm_request_timeout_seconds=20,
    allow_custom_llm_base_url=False,
)
```

## Configuration

The Typer-based CLI supports these runtime options on `ingest` and `ask`:

- `--data-dir`: directory used for persisted knowledge-base files
- `--chunk-size`: chunk size used during ingestion
- `--chunk-overlap`: overlap used during ingestion
- `--embedding-backend`: `deterministic` or `huggingface`
- `--answer-mode`: `extractive` or `generative`
- `--answer-verbosity`: `concise` or `verbose`
- `--retrieval-strategy`: `vector` or `hybrid`
- `--vector-backend`: `inmemory` or `faiss`
- `--llm-model`: model name for the built-in OpenAI-compatible generator
- `--llm-base-url`: base URL for the built-in OpenAI-compatible generator
- `--allowed-root`: optional root restriction for source documents and persisted data
- `--max-question-length`: maximum accepted question length
- `--max-retrieval-limit`: maximum retrieval limit accepted by the service
- `--max-documents-per-ingest`: maximum number of documents accepted by one ingest call
- `--max-document-bytes`: maximum bytes accepted for one source document
- `--max-chunks-per-ingest`: maximum chunk output accepted from one ingest call
- `--llm-request-timeout-seconds`: timeout for built-in LLM HTTP requests

Validation rules:

- `chunk_size` must be greater than zero
- `chunk_overlap` cannot be negative
- `chunk_overlap` must be smaller than `chunk_size`
- `embedding_backend` must be `deterministic` or `huggingface`
- `answer_mode` must be `extractive` or `generative`
- `answer_verbosity` must be `concise` or `verbose`
- `retrieval_strategy` must be `vector` or `hybrid`
- `vector_backend` must be `inmemory` or `faiss`
- `max_question_length`, `max_retrieval_limit`, `max_documents_per_ingest`, `max_document_bytes`, `max_chunks_per_ingest`, and `llm_request_timeout_seconds` must be greater than zero

Built-in generative mode requirements:

- `OPENAI_API_KEY` must be set in the environment
- the selected endpoint must implement an OpenAI-compatible `POST /chat/completions` API
- answers are still post-processed to include supporting chunk citations
- custom `llm_base_url` values can be disabled through `allow_custom_llm_base_url=False`

## Persistence layout

Persisted knowledge bases use a simple local JSON layout inside the selected data directory:

- `metadata.json`
- `documents.json`
- `chunks.json`
- `vectors.json`

`metadata.json` stores the storage schema version and the embedding workflow that produced the persisted vectors. The default data directory is `.rag-knb`. Create or update that persisted directory with `rag-knb ingest --data-dir ./.rag-knb ...`.

If you change embedding backends or Hugging Face model names, rebuild and save the knowledge base again. The loader rejects incompatible persisted vectors instead of silently mixing embeddings from different workflows.

## Dependency model

- `.[dev]`: pytest and local development tools
- the base package install currently includes the libraries used for deterministic retrieval, hybrid retrieval, LangChain-backed chunking, Hugging Face embeddings, and FAISS-backed vector storage

The default workflow still behaves as local-first and deterministic even though the package metadata currently includes the optional-backend libraries. The built-in generative path uses the Python standard library and an OpenAI-compatible HTTP API, so it does not require an additional SDK package.

The deterministic offline embedder intentionally uses light normalization plus a very small expansion map for common local-demo queries. The hybrid path now uses `rank-bm25` for lexical scoring, and the retriever applies a lightweight reranking pass before answering.

Required-technology mapping:

- LangChain: recursive text splitter and optional backend adapters in `src/rag_knb/indexing/chunking.py`, `src/rag_knb/retrieval_engine/embeddings.py`, and `src/rag_knb/retrieval_engine/vector_store.py`
- vector database: FAISS backend in `src/rag_knb/retrieval_engine/vector_store.py`
- Hugging Face embeddings: LangChain Hugging Face embeddings in `src/rag_knb/retrieval_engine/embeddings.py`

Internal structure:

- `src/rag_knb/runtime_options.py`: shared runtime override normalization for interface layers
- `src/rag_knb/pathing.py`: shared path coercion and persistence-target resolution
- `src/rag_knb/service_factory.py`: shared service construction helpers for interface layers
- `src/rag_knb/optional_dependencies.py`: centralized optional dependency guards
- `src/rag_knb/cli.py`: Typer command surface and Rich-backed CLI rendering
- `src/rag_knb/answers/`: grounded answer construction, context shaping, prompt-injection filtering, and built-in LLM support
- `src/rag_knb/indexing/`: document loading, chunking, persistence, and embedding workflow compatibility
- `src/rag_knb/retrieval_engine/`: embeddings, query rewriting, retrieval, vector stores, and local evaluation helpers

What remains intentionally custom:

- CLI orchestration and service integration boundaries
- deterministic fallback chunking, retrieval, persistence, and grounded-answer rules

What is delegated to third-party libraries:

- LangChain adapters for recursive text splitting, Hugging Face embeddings, and FAISS integration
- FAISS itself for the optional vector index backend

## Testing

Run the full local suite with:

```bash
./.venv/bin/python -m pytest
```

Run static type checking with:

```bash
./.venv/bin/python -m mypy .
```

Current tests cover:

- TXT and Markdown loading
- unsupported files and empty-document validation
- paragraph-aware chunk boundaries
- service-layer ingest and grounded retrieval
- CLI ingest, ask, filtering, document management, reload, and config flows
- persistence round-trips and load-time failures
- structured query logs and retrieval timing diagnostics
- snapshot tests for stable CLI outputs
- deterministic retrieval regression checks for relevance and local latency
- static typing with mypy

## Known limitations

- The default embedder is deterministic and offline-friendly, but semantic quality is limited.
- Optional backends such as Hugging Face embeddings and FAISS require extra dependencies and, for Hugging Face, potentially model downloads.
- The CLI can query and manage a persisted knowledge base, but it still does not expose a dedicated standalone `save` command outside the `ingest --data-dir` flow.
- Retrieval remains intentionally lightweight and local-first. It includes deterministic query rewriting, hybrid scoring, and lightweight reranking, but it does not yet include learned rerankers, distributed storage backends, or large-scale retrieval infrastructure.
- The built-in generative path currently targets OpenAI-compatible chat-completions APIs only.
- Diagnostics are intentionally lightweight: retrieval timing, query-plan details, parent-context expansions, and match metadata are available, but there is no full tracing, metrics backend, or dashboarding.
