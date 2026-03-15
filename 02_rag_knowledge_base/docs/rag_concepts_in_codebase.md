# RAG concepts in this codebase

This generated document maps the main RAG concepts explored by the project to concrete code paths and tests.
Status values are `covered`, `partial`, and `missing` so the document stays honest about current scope.

## Document loading and chunking (covered)

The repo can load local text, markdown, and JSONL sources, then chunk them into retrieval-friendly units.

Implementation paths:
- `src/rag_knb/indexing/loaders.py::load_documents`
- `src/rag_knb/indexing/chunking.py::chunk_document`
- `src/rag_knb/indexing/chunking.py::chunk_documents`

Tests:
- `tests/test_loading_and_chunking.py`

## Retrieval backends, hybrid ranking, and reranking (covered)

The project compares deterministic embeddings, hybrid retrieval, vector stores, and final reranking for grounded local search.

Implementation paths:
- `src/rag_knb/retrieval_engine/embeddings.py::build_embedder`
- `src/rag_knb/retrieval_engine/vector_store.py::build_vector_store`
- `src/rag_knb/retrieval_engine/retrieval.py::Retriever.search_with_plan`
- `src/rag_knb/retrieval_engine/evaluation.py::compare_retrieval_strategies`

Tests:
- `tests/test_service_retrieval.py`
- `tests/test_embeddings.py`
- `tests/test_vector_store.py`
- `tests/test_evaluation_harness.py`

## Query rewriting and conversation-aware retrieval input (partial)

The service can rewrite queries and fold recent turns into the retrieval question, but it is still lightweight conversation state rather than full dialogue memory.

Implementation paths:
- `src/rag_knb/retrieval_engine/query_rewriting.py::build_query_plan`
- `src/rag_knb/service.py::KnowledgeBaseService.ask`
- `src/rag_knb/service.py::_build_conversation_aware_question`

Tests:
- `tests/test_service_retrieval.py`
- `tests/test_cli.py`

Notes and remaining gaps:
- No long-horizon memory management.
- No user-profile personalization or topic stack.

## Context compression and answer planning (partial)

Retrieved matches are compressed into evidence sentences and a lightweight answer plan before answer generation.

Implementation paths:
- `src/rag_knb/answers/context_building.py::build_evidence_set`
- `src/rag_knb/answers/context_building.py::build_answer_plan`
- `src/rag_knb/answers/answering.py::ExtractiveAnswerer.answer`
- `src/rag_knb/answers/answering.py::GenerativeAnswerer.answer`

Tests:
- `tests/test_answering.py`

Notes and remaining gaps:
- No long-form synthesis planner for very large contexts.
- No adaptive token-budgeting against real model context windows.

## Grounding, citations, and semantic verification (partial)

Answers expose citations, sentence-level support, prompt-injection filtering, and semantic-verification style checks before generative output is accepted.

Implementation paths:
- `src/rag_knb/answers/answer_results.py::with_claim_alignments`
- `src/rag_knb/answers/answer_results.py::with_semantic_verification`
- `src/rag_knb/answers/prompt_injection.py::apply_prompt_injection_policy`
- `src/rag_knb/answers/answering.py::GenerativeAnswerer.answer`

Tests:
- `tests/test_answering.py`
- `tests/test_cli.py`

Notes and remaining gaps:
- No full entailment model for claim verification.

## Clarification routing and conversational answer planning (partial)

Ambiguous questions can route to clarification prompts and conversation-aware answer-plan diagnostics.

Implementation paths:
- `src/rag_knb/answers/answer_results.py::build_clarification_needed_answer`
- `src/rag_knb/service.py::_build_conversation_answer_plan`
- `src/rag_knb/answers/answering.py::_route_reason`

Tests:
- `tests/test_service_retrieval.py`
- `tests/test_cli.py`

Notes and remaining gaps:
- Clarification is still deterministic and short-form.
- The project does not run multi-turn clarification loops automatically.

## Multi-hop aggregation and supported composition (partial)

The answerer can aggregate support across multiple documents for explicit multi-part or composition-style questions.

Implementation paths:
- `src/rag_knb/answers/answering.py::_select_multi_hop_supporting_sentences`
- `src/rag_knb/answers/context_building.py::build_answer_plan`

Tests:
- `tests/test_answering.py`
- `tests/test_service_retrieval.py`

Notes and remaining gaps:
- No deeper chain-of-thought style reasoning.
- No planner that decomposes complex questions into separate retrieval hops.

## Evaluation and comparison harnesses (covered)

The repo includes local fixture-driven evaluation, grouped summaries, and side-by-side retrieval-strategy comparison helpers.

Implementation paths:
- `src/rag_knb/retrieval_engine/evaluation.py::evaluate_answer`
- `src/rag_knb/retrieval_engine/evaluation.py::summarize_results`
- `src/rag_knb/retrieval_engine/evaluation.py::summarize_results_by_group`
- `src/rag_knb/retrieval_engine/evaluation.py::compare_retrieval_strategies`

Tests:
- `tests/test_evaluation_harness.py`

## Still-missing concepts (missing)

Some important RAG-lab topics are intentionally still outside the current codebase.

Notes and remaining gaps:
- Benchmark-scale evaluation strong enough to justify major backend changes by itself.
- Full semantic entailment verification for every generated claim.
- Long-horizon dialogue memory and richer conversational state management.
