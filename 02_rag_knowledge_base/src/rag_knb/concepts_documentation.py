"""Generate concise concept-to-code documentation for the local RAG lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONCEPTS_DOC_PATH = Path("docs/rag_concepts_in_codebase.md")


@dataclass(frozen=True, slots=True)
class ConceptMapping:
    """One RAG concept mapped to code paths and tests."""

    name: str
    status: str
    summary: str
    implementation_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    remaining_gaps: tuple[str, ...] = field(default_factory=tuple)


def build_concept_mappings() -> list[ConceptMapping]:
    """Return the current concept map for this codebase."""
    return [
        ConceptMapping(
            name="Document loading and chunking",
            status="covered",
            summary="The repo can load local text, markdown, and JSONL sources, then chunk them into retrieval-friendly units.",
            implementation_paths=(
                "src/rag_knb/indexing/loaders.py::load_documents",
                "src/rag_knb/indexing/chunking.py::chunk_document",
                "src/rag_knb/indexing/chunking.py::chunk_documents",
            ),
            test_paths=("tests/test_loading_and_chunking.py",),
        ),
        ConceptMapping(
            name="Retrieval backends, hybrid ranking, and reranking",
            status="covered",
            summary="The project compares deterministic embeddings, hybrid retrieval, vector stores, and final reranking for grounded local search.",
            implementation_paths=(
                "src/rag_knb/retrieval_engine/embeddings.py::build_embedder",
                "src/rag_knb/retrieval_engine/vector_store.py::build_vector_store",
                "src/rag_knb/retrieval_engine/retrieval.py::Retriever.search_with_plan",
                "src/rag_knb/retrieval_engine/evaluation.py::compare_retrieval_strategies",
            ),
            test_paths=(
                "tests/test_service_retrieval.py",
                "tests/test_embeddings.py",
                "tests/test_vector_store.py",
                "tests/test_evaluation_harness.py",
            ),
        ),
        ConceptMapping(
            name="Query rewriting and conversation-aware retrieval input",
            status="partial",
            summary="The service can rewrite queries and fold recent turns into the retrieval question, but it is still lightweight conversation state rather than full dialogue memory.",
            implementation_paths=(
                "src/rag_knb/retrieval_engine/query_rewriting.py::build_query_plan",
                "src/rag_knb/service.py::KnowledgeBaseService.ask",
                "src/rag_knb/service.py::_build_conversation_aware_question",
            ),
            test_paths=("tests/test_service_retrieval.py", "tests/test_cli.py"),
            remaining_gaps=(
                "No long-horizon memory management.",
                "No user-profile personalization or topic stack.",
            ),
        ),
        ConceptMapping(
            name="Context compression and answer planning",
            status="partial",
            summary="Retrieved matches are compressed into evidence sentences and a lightweight answer plan before answer generation.",
            implementation_paths=(
                "src/rag_knb/answers/context_building.py::build_evidence_set",
                "src/rag_knb/answers/context_building.py::build_answer_plan",
                "src/rag_knb/answers/answering.py::ExtractiveAnswerer.answer",
                "src/rag_knb/answers/answering.py::GenerativeAnswerer.answer",
            ),
            test_paths=("tests/test_answering.py",),
            remaining_gaps=(
                "No long-form synthesis planner for very large contexts.",
                "No adaptive token-budgeting against real model context windows.",
            ),
        ),
        ConceptMapping(
            name="Grounding, citations, and semantic verification",
            status="partial",
            summary="Answers expose citations, sentence-level support, prompt-injection filtering, and semantic-verification style checks before generative output is accepted.",
            implementation_paths=(
                "src/rag_knb/answers/answer_results.py::with_claim_alignments",
                "src/rag_knb/answers/answer_results.py::with_semantic_verification",
                "src/rag_knb/answers/prompt_injection.py::apply_prompt_injection_policy",
                "src/rag_knb/answers/answering.py::GenerativeAnswerer.answer",
            ),
            test_paths=("tests/test_answering.py", "tests/test_cli.py"),
            remaining_gaps=("No full entailment model for claim verification.",),
        ),
        ConceptMapping(
            name="Clarification routing and conversational answer planning",
            status="partial",
            summary="Ambiguous questions can route to clarification prompts and conversation-aware answer-plan diagnostics.",
            implementation_paths=(
                "src/rag_knb/answers/answer_results.py::build_clarification_needed_answer",
                "src/rag_knb/service.py::_build_conversation_answer_plan",
                "src/rag_knb/answers/answering.py::_route_reason",
            ),
            test_paths=("tests/test_service_retrieval.py", "tests/test_cli.py"),
            remaining_gaps=(
                "Clarification is still deterministic and short-form.",
                "The project does not run multi-turn clarification loops automatically.",
            ),
        ),
        ConceptMapping(
            name="Multi-hop aggregation and supported composition",
            status="partial",
            summary="The answerer can aggregate support across multiple documents for explicit multi-part or composition-style questions.",
            implementation_paths=(
                "src/rag_knb/answers/answering.py::_select_multi_hop_supporting_sentences",
                "src/rag_knb/answers/context_building.py::build_answer_plan",
            ),
            test_paths=("tests/test_answering.py", "tests/test_service_retrieval.py"),
            remaining_gaps=(
                "No deeper chain-of-thought style reasoning.",
                "No planner that decomposes complex questions into separate retrieval hops.",
            ),
        ),
        ConceptMapping(
            name="Evaluation and comparison harnesses",
            status="covered",
            summary="The repo includes local fixture-driven evaluation, grouped summaries, and side-by-side retrieval-strategy comparison helpers.",
            implementation_paths=(
                "src/rag_knb/retrieval_engine/evaluation.py::evaluate_answer",
                "src/rag_knb/retrieval_engine/evaluation.py::summarize_results",
                "src/rag_knb/retrieval_engine/evaluation.py::summarize_results_by_group",
                "src/rag_knb/retrieval_engine/evaluation.py::compare_retrieval_strategies",
            ),
            test_paths=("tests/test_evaluation_harness.py",),
        ),
        ConceptMapping(
            name="Still-missing concepts",
            status="missing",
            summary="Some important RAG-lab topics are intentionally still outside the current codebase.",
            implementation_paths=(),
            test_paths=(),
            remaining_gaps=(
                "Benchmark-scale evaluation strong enough to justify major backend changes by itself.",
                "Full semantic entailment verification for every generated claim.",
                "Long-horizon dialogue memory and richer conversational state management.",
            ),
        ),
    ]


def render_concepts_document() -> str:
    """Render the current concept map as concise Markdown."""
    lines = [
        "# RAG concepts in this codebase",
        "",
        "This generated document maps the main RAG concepts explored by the project to concrete code paths and tests.",
        "Status values are `covered`, `partial`, and `missing` so the document stays honest about current scope.",
        "",
    ]
    for concept in build_concept_mappings():
        lines.extend(
            [
                f"## {concept.name} ({concept.status})",
                "",
                concept.summary,
                "",
            ]
        )
        if concept.implementation_paths:
            lines.append("Implementation paths:")
            lines.extend(f"- `{path}`" for path in concept.implementation_paths)
            lines.append("")
        if concept.test_paths:
            lines.append("Tests:")
            lines.extend(f"- `{path}`" for path in concept.test_paths)
            lines.append("")
        if concept.remaining_gaps:
            lines.append("Notes and remaining gaps:")
            lines.extend(f"- {gap}" for gap in concept.remaining_gaps)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_concepts_document(output_path: Path = DEFAULT_CONCEPTS_DOC_PATH) -> Path:
    """Write the rendered concept map to disk and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_concepts_document(), encoding="utf-8")
    return output_path
