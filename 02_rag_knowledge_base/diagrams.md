# RAG KnB Diagrams

This file explains the current package layout and the main runtime flows in `rag_knb`.

## 1. High-Level Structure

```mermaid
flowchart LR
    User[User / Calling App]
    CLI[CLI<br/>rag_knb.cli<br/>Typer + Rich]
    API[Consumer API / App]
    Factory[Service Factory<br/>rag_knb.service_factory]
    Service[KnowledgeBaseService<br/>rag_knb.service]

    Answers[answers package<br/>answering, llm,<br/>context_building, prompt_injection]
    Indexing[indexing package<br/>loaders, chunking,<br/>storage, embedding_lifecycle]
    Retrieval[retrieval_engine package<br/>embeddings, query_rewriting,<br/>retrieval, vector_store, evaluation]
    Concepts[concepts_documentation<br/>generated concept-to-code map]
    Core[core modules<br/>config and runtime_options via Pydantic,<br/>models, errors, pathing, observability, library_policies]

    User --> CLI
    User --> API
    CLI --> Factory
    API --> Factory
    Factory --> Service

    Service --> Answers
    Service --> Indexing
    Service --> Retrieval
    Service --> Concepts
    Service --> Core
```

## 2. Package Layout

```mermaid
flowchart LR
    subgraph Public["Public Entry Points"]
        Init[rag_knb.__init__]
        CLIEntry[rag_knb.cli<br/>Typer + Rich]
        Factory[rag_knb.service_factory]
        Service[rag_knb.service]
    end

    subgraph Answers["rag_knb.answers"]
        Answering[answering]
        Results[answer_results]
        Context[context_building]
        LLM[llm]
        Injection[prompt_injection]
    end

    subgraph Indexing["rag_knb.indexing"]
        Loaders[loaders]
        Chunking[chunking]
        Storage[storage]
        Lifecycle[embedding_lifecycle]
    end

    subgraph Retrieval["rag_knb.retrieval_engine"]
        Embeddings[embeddings]
        Rewrite[query_rewriting]
        Retrieve[retrieval]
        Store[vector_store]
        Evaluate[evaluation]
    end

    subgraph Shared["Shared Core Modules"]
        Config[config<br/>Pydantic Settings]
        Options[runtime_options<br/>Pydantic model]
        Concepts[concepts_documentation]
        Models[models]
        Errors[errors]
        Pathing[pathing]
        Policies[library_policies]
        Obs[observability]
        OptionalDeps[optional_dependencies]
    end

    CLIEntry --> Factory
    Factory --> Options
    Options --> Config
    Init --> Service

    Service --> Loaders
    Service --> Chunking
    Service --> Storage
    Service --> Lifecycle
    Service --> Embeddings
    Service --> Rewrite
    Service --> Retrieve
    Service --> Store
    Service --> Answering
    Service --> Obs
    Service --> Policies
    Service --> Pathing
    Service --> Concepts
    Service --> Models

    Answering --> Results
    Answering --> Context
    Answering --> LLM
    Answering --> Injection

    Chunking --> OptionalDeps
    Embeddings --> OptionalDeps
    Store --> OptionalDeps
    Evaluate --> Models
    Concepts --> Evaluate
```

## 3. CLI Usage Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI
    participant F as rag_knb.service_factory
    participant S as KnowledgeBaseService

    U->>C: rag-knb status / ingest / ask / list-documents / remove-document
    C->>F: build_service_from_options(args)
    F->>S: create configured service

    alt status
        C->>S: status()
        S-->>C: ServiceStatus
        C-->>U: summary text
    else ingest
        C->>S: ingest_paths(paths)
        opt with --data-dir
            C->>S: save(data_dir)
        end
        S-->>C: IngestResult
        C-->>U: status text
    else ask with --document
        C->>S: ingest_paths(document_paths)
        C->>S: ask(question, metadata_filters)
        S-->>C: AnswerResult
        C-->>U: answer text + optional diagnostics
    else ask with --data-dir
        C->>S: load(data_dir)
        C->>S: ask(question, metadata_filters)
        S-->>C: AnswerResult
        C-->>U: answer text + optional diagnostics
    else list-documents
        opt with --data-dir
            C->>S: load(data_dir)
        end
        C->>S: list_documents()
        S-->>C: Document list
        C-->>U: printed ids and metadata
    else remove-document
        C->>S: load(data_dir)
        C->>S: remove_documents(ids)
        C->>S: save(data_dir)
        C-->>U: remaining count
    end
```

## 4. Ingest And Refresh Flow

```mermaid
flowchart TD
    Start["ingest_paths(paths) or refresh_paths(paths)"] --> Limits1[Validate document-count limit]
    Limits1 --> Load[indexing.loaders.load_documents]
    Load --> PathCheck[Optional allowed-root check]
    PathCheck --> SizeCheck[Optional per-file byte limit]
    SizeCheck --> Parse[Load TXT / Markdown / JSON / JSONL]
    Parse --> Chunk[indexing.chunking.chunk_documents]
    Chunk --> ParentMeta[Attach parent-document metadata]
    ParentMeta --> Limits2[Validate chunk-budget limit]
    Limits2 --> Reindex[Retriever.index_chunks or partial refresh embed]
    Reindex --> Store[retrieval_engine.vector_store add / replace]
    Store --> State[Service replaces in-memory documents chunks and entries]
    State --> Log[observability.log_event ingest_completed / refresh_completed]
    Log --> Result[Return IngestResult / RefreshResult]
```

## 5. Query Flow

```mermaid
flowchart TD
    Ask["ask(question, limit, filters, conversation_turns)"] --> Q1[Reject blank question]
    Q1 --> Q2[Validate max_question_length]
    Q2 --> Q3[Validate max_retrieval_limit]
    Q3 --> Empty{Indexed chunks exist?}

    Empty -- No --> EmptyResult[build_empty_answer]
    Empty -- Yes --> Conv[Build conversation-aware retrieval question]
    Conv --> Plan[retrieval_engine.query_rewriting.build_query_plan]
    Plan --> Search[Retriever.search_with_plan]

    Search --> QueryLoop[Run one or more retrieval queries]
    QueryLoop --> EmbedQuery[embed query text]
    EmbedQuery --> Strategy{retrieval_strategy}
    Strategy -- vector --> VectorSearch[vector_store.search]
    Strategy -- hybrid --> HybridSearch[vector plus BM25 lexical candidate scoring]
    VectorSearch --> Merge[Merge best candidate per chunk]
    HybridSearch --> Merge
    Merge --> Dedupe[Deduplicate near-identical retrieved evidence]
    Dedupe --> Rerank[Lightweight reranking on rewritten question plus metadata and source-weight hints]
    Rerank --> Answer["answers.answering.build_answerer().answer"]

    Answer --> Policy[prompt_injection policy filters hostile instructions and downgrades unsafe evidence]
    Policy --> Mode{answer_mode}
    Mode -- extractive --> Extractive[Select supporting sentences, evidence sets, and citations]
    Mode -- generative --> Context[Build compact evidence set]
    Context --> LLM[answers.llm OpenAI-compatible generator]
    LLM --> Grounding[Validate sentence-level support or fall back]

    Extractive --> Finalize[Attach query plan, conversation plan, evidence set, context window, parent context, trust diagnostics]
    Grounding --> Finalize
    EmptyResult --> Finalize
    Finalize --> Log[observability.log_event query_completed]
    Log --> Return[Return AnswerResult]
```

## 6. Evaluation And Comparison Flow

```mermaid
flowchart TD
    Fixtures[tests/fixtures/evaluation_cases.json or ad hoc fixture cases] --> Eval[evaluation.evaluate_answer]
    Eval --> Summary[summarize_results]
    Eval --> Grouped[summarize_results_by_group]
    Fixtures --> Compare[compare_retrieval_strategies]
    Compare --> Strategies[Build one service per strategy override]
    Strategies --> Materialize[Materialize fixture documents in a temp workspace]
    Materialize --> Ask[Ingest and ask per strategy]
    Ask --> CandidateOrder[Record raw candidate document order]
    Ask --> FinalOrder[Record final reranked document order]
    CandidateOrder --> Comparison[RetrievalStrategyComparison]
    FinalOrder --> Comparison
    Summary --> Comparison
```

## 7. Coverage Summary

```mermaid
flowchart LR
    Covered[Covered local baseline<br/>retrieval, grounding, focused answering,<br/>evaluation, conversation follow-ups]
    Partial[Partially covered<br/>multi-hop aggregation, semantic verification,<br/>trust diagnostics, conversation planning]
    Missing[Still outside scope<br/>benchmark-scale eval, deep reasoning,<br/>full dialogue memory, production observability]

    Covered --> Partial --> Missing
```

## 8. Persistence Flow

```mermaid
flowchart LR
    subgraph Save["service.save(data_dir)"]
        S1[Resolve data_dir]
        S2[Optional allowed-root check]
        S3[indexing.storage.LocalKnowledgeBaseRepository.save]
        S4[Write metadata.json]
        S5[Write documents.json]
        S6[Write chunks.json]
        S7[Write vectors.json]
    end

    subgraph Load["service.load(data_dir)"]
        L1[Resolve data_dir]
        L2[Optional allowed-root check]
        L3[indexing.storage.LocalKnowledgeBaseRepository.load]
        L4[Validate directory and required files]
        L5[Validate JSON payloads and schema version]
        L6[Validate embedding workflow compatibility]
        L7[Rebuild documents chunks and indexed entries]
        L8[Replace in-memory state]
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7 --> L8
```

## 9. Optional Backend Paths

```mermaid
flowchart TD
    Config[RuntimeConfig] --> EmbedChoice{embedding_backend}
    Config --> VectorChoice{vector_backend}
    Config --> AnswerChoice{answer_mode}

    EmbedChoice -- deterministic --> DetEmbed[DeterministicEmbedder]
    EmbedChoice -- huggingface --> HFEmbed[HuggingFaceEmbedder via LangChain]

    VectorChoice -- inmemory --> InMem[InMemoryVectorStore]
    VectorChoice -- faiss --> Faiss[FAISS-backed vector store]

    AnswerChoice -- extractive --> Extractive[ExtractiveAnswerer]
    AnswerChoice -- generative --> Generative[GenerativeAnswerer + OpenAI-compatible HTTP client]

    Optional[optional_dependencies checks]
    HFEmbed --> Optional
    Faiss --> Optional
```

## 10. Concepts Documentation Flow

```mermaid
flowchart LR
    Script[scripts/generate_rag_concepts_doc.py] --> Module[concepts_documentation.write_concepts_document]
    Module --> Catalog[build_concept_mappings]
    Catalog --> CodeMap[Implementation paths in src/rag_knb]
    Catalog --> TestMap[Related tests]
    Module --> Output[docs/rag_concepts_in_codebase.md]
```

## 11. Integration Model

```mermaid
flowchart TD
    Consumer[Consumer App / API]
    RequestModel[Transport-layer request model]
    Runtime[RuntimeConfig.build]
    Service[KnowledgeBaseService]
    Persistence[Optional local persisted KB]
    Response[AnswerResult / IngestResult / RefreshResult]

    Consumer --> RequestModel
    RequestModel --> Runtime
    Runtime --> Service
    Service --> Persistence
    Service --> Response
    Response --> Consumer
```
