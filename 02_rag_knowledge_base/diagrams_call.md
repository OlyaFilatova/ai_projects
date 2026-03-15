# Call Diagrams

This file focuses on current call relationships between the main functions and classes in `rag_knb`.

## CLI Dispatch And Service Creation

```mermaid
flowchart TD
    A[User runs rag-knb command] --> B[cli.main / __main__]
    B --> C[cli.run_cli]
    C --> D[Typer app / CliRunner]
    D --> E[Typer option parsing]
    E --> F{Command selected}
    F -->|status| G[cli.handle_status]
    F -->|ingest ask list remove| H[cli._build_service_from_args]
    H --> I[service_factory.build_service_from_options]
    I --> J[RuntimeOptionValues.from_object]
    J --> K{Overrides present}
    K -->|no| L[Reuse existing service or create KnowledgeBaseService]
    K -->|yes| M[RuntimeOptionValues.to_runtime_config]
    M --> N[RuntimeConfig.build]
    N --> O[KnowledgeBaseService.__init__]
    L --> P[Selected CLI handler]
    O --> P
```

## KnowledgeBaseService Construction

```mermaid
flowchart LR
    A[KnowledgeBaseService.__init__] --> B[retrieval_engine.build_embedder]
    A --> C[retrieval_engine.build_vector_store]
    A --> D[Retriever.__init__]
    A --> E[answers.build_answerer]
    A --> F[observability.get_logger]
    B --> G{Embedding backend}
    G -->|deterministic| H[DeterministicEmbedder]
    G -->|huggingface| I[HuggingFaceEmbedder]
    C --> J{Vector backend}
    J -->|inmemory| K[InMemoryVectorStore]
    J -->|faiss| L[FAISS-backed VectorStore]
    E --> M{Answer mode}
    M -->|extractive| N[ExtractiveAnswerer]
    M -->|generative| O[build_builtin_text_generator or custom generator]
    O --> P[GenerativeAnswerer]
```

## Ingest Call Flow

```mermaid
flowchart TD
    A[cli.handle_ingest or caller] --> B[KnowledgeBaseService.ingest_paths]
    B --> C[library_policies.validate_max_count for documents]
    B --> D[indexing.loaders.load_documents]
    D --> E[load_document per path]
    E --> F[pathing / allowed-root check]
    E --> G[file-size validation]
    E --> H[read text or parse JSON / JSONL via jsonlines]
    E --> I[build Document records]
    B --> J[indexing.chunking.chunk_documents]
    J --> K[chunk_document]
    K --> L{Structured record?}
    L -->|yes| M[_chunk_structured_record]
    L -->|no| N{LangChain splitter available?}
    N -->|yes| O[_chunk_with_langchain]
    N -->|no| P[_chunk_by_sentences]
    P --> Q{Sentence chunks available?}
    Q -->|no| R[_chunk_by_paragraphs or _chunk_by_fixed_width]
    B --> S[library_policies.validate_max_count for chunks]
    B --> T[_replace_state]
    T --> U[Retriever.index_chunks]
    U --> V[VectorStore.clear]
    U --> W[EmbeddingBackend.embed per chunk]
    W --> X[VectorStore.add]
    B --> Y[_log_duration]
    Y --> Z[observability.log_event ingest_completed]
```

## Refresh Call Flow

```mermaid
flowchart LR
    A[Caller requests refresh_paths] --> B[KnowledgeBaseService.refresh_paths]
    B --> C{Existing indexed state?}
    C -->|no| D[_refresh_via_full_ingest]
    C -->|yes| E[_load_refresh_documents_and_chunks]
    E --> F[indexing.loaders.load_documents]
    E --> G[indexing.chunking.chunk_documents]
    B --> H[_build_refresh_inputs]
    H --> I[group refreshed and existing state by source_path]
    B --> J[_merge_refresh_artifacts]
    J --> K[reuse unchanged documents chunks and vectors]
    J --> L[embed only changed chunks]
    B --> M[_replace_loaded_state]
    M --> N[VectorStore.replace]
    B --> O[_log_refresh_duration]
    O --> P[observability.log_event refresh_completed]
```

## Query And Answer Call Flow

```mermaid
flowchart TD
    A[cli.handle_ask or caller] --> B[KnowledgeBaseService.ask]
    B --> C[validate_text_length]
    B --> D[validate_positive_request_limit]
    B --> E{Any indexed chunks?}
    E -->|no| F[build_empty_answer]
    E -->|yes| G[_build_conversation_aware_question]
    G --> GP[conversation answer planning]
    GP --> H[query_rewriting.build_query_plan]
    H --> I[Retriever.search_with_plan]
    I --> J[Embed each retrieval query]
    J --> K{retrieval_strategy}
    K -->|vector| L[VectorStore.search]
    K -->|hybrid| M[Retriever._search_hybrid]
    L --> N[Merge best candidate per chunk]
    M --> N
    N --> ND[deduplicate near-identical evidence]
    ND --> O[_rerank_matches with metadata and source-weight hints]
    O --> P[Answerer.answer]
    P --> Q{Answerer type}
    Q -->|extractive| R[ExtractiveAnswerer.answer]
    Q -->|generative| S[GenerativeAnswerer.answer]
    R --> T[apply_prompt_injection_policy]
    S --> T
    T --> U{safe matches + confidence + clarification checks}
    U -->|extractive path| V[select supporting sentences or multi-hop evidence plus typo-tolerant yes/no formatting]
    U -->|generative path| W[TextGenerator.generate]
    W --> X[validate semantic support sentence by sentence]
    X --> Y{Unsupported claims present?}
    Y -->|yes| Z[fallback_answerer.answer]
    Y -->|no| AA[build matched generative answer]
    V --> AB[build matched extractive answer]
    Z --> AC[attach query plan, conversation plan, evidence set, trust diagnostics]
    AA --> AC
    AB --> AC
    F --> AC
    AC --> AD[observability.log_event query_completed]
```

## Evaluation And Strategy Comparison Call Flow

```mermaid
flowchart TD
    A[Test or local script] --> B[retrieval_engine.evaluation.compare_retrieval_strategies]
    B --> C[_materialize_fixture_case]
    B --> D[_build_service_for_strategy]
    D --> E[KnowledgeBaseService.__init__]
    B --> F[_candidate_matches_for_case]
    F --> G[query_rewriting.build_query_plan]
    G --> H[Retriever._search_hybrid or VectorStore.search]
    H --> I[_ordered_document_ids for raw candidate order]
    B --> J[KnowledgeBaseService.ask]
    J --> K[evaluate_answer]
    K --> L[summarize_results]
    I --> M[RetrievalStrategyCaseComparison]
    J --> M
    L --> N[RetrievalStrategyComparison]
    M --> N
    B --> O{Optional backend available?}
    O -->|no| P[status = unavailable]
    O -->|yes| N
```

## Generative LLM Call Flow

```mermaid
flowchart TD
    A[answers.build_answerer] --> B{answer_mode generative}
    B --> C[answers.llm.build_builtin_text_generator]
    C --> D[Read OPENAI_API_KEY]
    C --> E[Check allow_custom_llm_base_url policy]
    C --> F[Create OpenAIChatTextGenerator]
    F --> G[GenerativeAnswerer.answer]
    G --> H[context_building.build_context_window]
    H --> I[context_building.render_context_window]
    I --> J[OpenAIChatTextGenerator.generate]
    J --> K[_build_user_prompt]
    J --> L[_post_json]
    L --> M[urllib.request.Request]
    M --> N[urllib.request.urlopen]
    N --> O[json.loads]
    O --> P[Validate choices and content]
    P --> Q[Return generated text]
    Q --> R[grounding validation or extractive fallback]
```

## Save, Load, And Remove Call Flow

```mermaid
flowchart LR
    A[Caller requests save] --> B[KnowledgeBaseService.save]
    B --> C[_resolve_data_dir]
    C --> D[pathing.resolve_data_dir]
    C --> E[pathing.is_path_within_allowed_root]
    B --> F[_build_repository]
    F --> G[indexing.storage.LocalKnowledgeBaseRepository]
    B --> H[repository.save]
    H --> I[write metadata.json]
    H --> J[write documents.json]
    H --> K[write chunks.json]
    H --> L[write vectors.json]
    B --> M[_log_duration]

    N[Caller requests load] --> O[KnowledgeBaseService.load]
    O --> P[_resolve_data_dir]
    O --> Q[_build_repository]
    O --> R[repository.load]
    R --> S[read and parse JSON files]
    R --> T[rebuild persisted state]
    O --> U[validate_embedding_workflow_compatibility]
    O --> V[_replace_loaded_state]
    V --> W[VectorStore.replace]
    O --> X[_log_duration]

    Y[Caller requests remove_documents] --> Z[KnowledgeBaseService.remove_documents]
    Z --> ZA[filter documents and chunks by document_id]
    ZA --> ZB[_replace_state]
```

## Concepts Documentation Generation Flow

```mermaid
flowchart TD
    A[User runs scripts/generate_rag_concepts_doc.py] --> B[concepts_documentation.write_concepts_document]
    B --> C[render_concepts_document]
    C --> D[build_concept_mappings]
    D --> E[Map concepts to src/rag_knb modules and tests]
    C --> F[Markdown document text]
    F --> G[docs/rag_concepts_in_codebase.md]
```

## Class Collaboration Overview

```mermaid
classDiagram
    direction LR
    class KnowledgeBaseService {
        +ingest_paths(paths)
        +refresh_paths(paths, remove_missing)
        +ask(question, limit, metadata_filters, conversation_turns)
        +save(data_dir)
        +load(data_dir)
        +list_documents()
        +remove_documents(document_ids)
    }

    class Retriever {
        +index_chunks(chunks)
        +search(query, limit, metadata_filters)
        +search_with_plan(query_plan, limit, metadata_filters)
    }

    class RuntimeConfig
    class EmbeddingBackend {
        <<interface>>
        +embed(text)
    }
    class DeterministicEmbedder
    class HuggingFaceEmbedder
    class VectorStore {
        <<interface>>
        +clear()
        +add(chunk, vector)
        +replace(entries)
        +search(query_text, query_vector, limit, metadata_filters)
    }
    class InMemoryVectorStore
    class FaissVectorStore
    class ExtractiveAnswerer {
        +answer(question, matches)
    }
    class GenerativeAnswerer {
        +answer(question, matches)
    }
    class OpenAIChatTextGenerator {
        +generate(question, context)
    }
    class LocalKnowledgeBaseRepository {
        +save(documents, chunks, indexed_chunks)
        +load()
    }
    class ConceptDocumentation {
        +build_concept_mappings()
        +render_concepts_document()
        +write_concepts_document()
    }

    KnowledgeBaseService --> RuntimeConfig
    KnowledgeBaseService --> Retriever
    KnowledgeBaseService --> ExtractiveAnswerer
    KnowledgeBaseService --> GenerativeAnswerer
    KnowledgeBaseService --> LocalKnowledgeBaseRepository
    Retriever --> EmbeddingBackend
    Retriever --> VectorStore
    DeterministicEmbedder --|> EmbeddingBackend
    HuggingFaceEmbedder --|> EmbeddingBackend
    InMemoryVectorStore --|> VectorStore
    FaissVectorStore --|> VectorStore
    GenerativeAnswerer --> OpenAIChatTextGenerator
```

## End-To-End Timing View

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Factory as Service Factory
    participant Service as KnowledgeBaseService
    participant Loaders as indexing.loaders
    participant Chunking as indexing.chunking
    participant Retriever
    participant Store as VectorStore
    participant Answerer
    participant LLM

    User->>CLI: rag-knb ingest files
    CLI->>Factory: build_service_from_options
    Factory->>Service: create or reuse
    CLI->>Service: ingest_paths
    Service->>Loaders: load_documents
    Service->>Chunking: chunk_documents
    Service->>Retriever: index_chunks
    Retriever->>Store: clear and add

    User->>CLI: rag-knb ask question
    CLI->>Factory: build_service_from_options
    Factory->>Service: create or reuse
    CLI->>Service: ask
    Service->>Retriever: search_with_plan
    Retriever->>Store: search
    Service->>Answerer: answer
    Answerer->>LLM: generate only in generative mode
    Answerer-->>Service: AnswerResult
    Service-->>CLI: grounded answer + diagnostics
    CLI-->>User: print result via Rich console
```
