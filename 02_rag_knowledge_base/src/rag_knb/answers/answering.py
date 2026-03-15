"""Grounded answer generation primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from rapidfuzz import process

from rag_knb.answers.answer_results import (
    build_clarification_needed_answer,
    build_low_confidence_answer,
    build_matched_answer,
    build_no_match_answer,
    with_answer_plan,
    with_claim_alignments,
    with_confidence_policy,
    with_context_window,
    with_evidence_set,
    with_parent_context,
    with_prompt_injection_policy,
    with_semantic_verification,
)
from rag_knb.answers.context_building import (
    build_answer_plan,
    build_context_window,
    build_evidence_matches,
    build_evidence_set,
)
from rag_knb.answers.llm import build_builtin_text_generator
from rag_knb.answers.prompt_injection import (
    PromptInjectionPolicyResult,
    apply_prompt_injection_policy,
)
from rag_knb.config import RuntimeConfig
from rag_knb.models import AnswerResult, RetrievalResult

# The default deterministic embedder produces modest cosine scores for natural-language
# questions. Keep the threshold low enough to answer simple docs-style prompts while
# still flagging very weak matches.
LOW_CONFIDENCE_SCORE = 0.15
LOW_CONFIDENCE_SUPPORT_COVERAGE = 0.75
LOW_CONFIDENCE_SCORE_GAP = 0.1
CONCISE_SUPPORTING_SENTENCE_LIMIT = 1
VERBOSE_SUPPORTING_SENTENCE_LIMIT = 3
GENERATIVE_SENTENCE_LIMIT = 2
GROUNDING_COVERAGE_THRESHOLD = 0.75
STRONG_GROUNDING_COVERAGE_THRESHOLD = 0.85
WEAK_GROUNDING_COVERAGE_THRESHOLD = 0.6
MAX_UNCOVERED_GROUNDING_TOKENS = 1
SENTENCE_RANK_STRIDE = 1000
PARENT_EXCERPT_LIMIT = 160
CLARIFICATION_SCORE_DELTA = 0.05
FUZZY_TOKEN_MATCH_THRESHOLD = 90.0
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
QUESTION_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
LEADING_YES_NO_VERBS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "should",
    "was",
    "were",
    "will",
    "would",
}
QUESTION_STOPWORDS = {
    "a",
    "an",
    "about",
    "and",
    "are",
    "do",
    "document",
    "documents",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "say",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
STRUCTURED_QUERY_TOKENS = {"record", "records", "structured"}
GROUNDING_STOPWORDS = QUESTION_STOPWORDS | {
    "all",
    "also",
    "an",
    "about",
    "be",
    "compare",
    "briefly",
    "can",
    "companions",
    "direct",
    "document",
    "documents",
    "from",
    "if",
    "into",
    "more",
    "on",
    "only",
    "or",
    "that",
    "then",
    "they",
    "this",
    "through",
    "typically",
    "up",
    "with",
}
GROUNDING_EQUIVALENTS = {
    "active": {"energetic"},
    "animal": {"pet"},
    "companion": {"friend"},
    "companions": {"friend"},
    "day": {"daytime"},
    "energetic": {"active"},
    "feline": {"cat"},
    "friend": {"companion", "companions"},
    "nap": {"sleep", "rest"},
    "night": {"nighttime"},
    "pet": {"animal"},
    "playful": {"play"},
    "rest": {"nap", "sleep"},
    "sleep": {"nap", "rest"},
}
YES_NO_ANTONYMS = {
    "allowed": {"blocked", "disallowed", "forbidden"},
    "blocked": {"allowed"},
    "dependent": {"independent"},
    "disallowed": {"allowed"},
    "forbidden": {"allowed"},
    "independent": {"dependent"},
    "safe": {"unsafe"},
    "unsafe": {"safe"},
}


class TextGenerator(Protocol):
    """Protocol for optional grounded text generators."""

    def generate(self, question: str, context: list[RetrievalResult]) -> str:
        """Generate an answer from grounded context."""


class Answerer(Protocol):
    """Protocol for answer generation strategies."""

    def answer(self, question: str, matches: list[RetrievalResult]) -> AnswerResult:
        """Build an answer from grounded matches."""


class ExtractiveAnswerer:
    """Create deterministic answers directly from retrieved chunks."""

    def __init__(self, answer_verbosity: str = "concise") -> None:
        """Initialize the extractive answerer with one verbosity setting."""
        self._answer_verbosity = answer_verbosity

    def answer(self, question: str, matches: list[RetrievalResult]) -> AnswerResult:
        """Build an answer that cites the retrieved context."""
        if not matches:
            return build_no_match_answer(question)
        policy_result = apply_prompt_injection_policy(matches)
        safe_matches = policy_result.matches
        if not safe_matches:
            return _with_prompt_injection_result(build_no_match_answer(question), policy_result)
        route_reason, confidence_policy = _route_reason(question, safe_matches)
        if route_reason == "low_confidence":
            return _attach_confidence_policy(
                _with_prompt_injection_result(build_low_confidence_answer(question, safe_matches), policy_result),
                confidence_policy,
            )
        if route_reason == "clarification_needed":
            return _attach_confidence_policy(
                _with_prompt_injection_result(
                    build_clarification_needed_answer(question, safe_matches),
                    policy_result,
                ),
                confidence_policy,
            )
        evidence_matches = build_evidence_matches(
            question,
            safe_matches,
            limit=_evidence_limit(question, self._answer_verbosity),
        )
        evidence_set = build_evidence_set(
            question,
            safe_matches,
            limit=_evidence_limit(question, self._answer_verbosity),
        )
        answer_plan = build_answer_plan(question, evidence_set)

        answer_lines = ["Grounded answer:"]
        supporting_sentences = _select_answer_supporting_sentences(
            question,
            evidence_matches,
            answer_verbosity=_effective_answer_verbosity(question, self._answer_verbosity),
        )
        if not supporting_sentences:
            return _with_prompt_injection_result(build_no_match_answer(question), policy_result)
        direct_yes_no_answer = _build_direct_yes_no_answer(question, supporting_sentences)
        if direct_yes_no_answer is None:
            for sentence, chunk_id in supporting_sentences:
                answer_lines.append(f"- {sentence} [{chunk_id}]")
            answer_text = "\n".join(answer_lines)
        else:
            answer_text = direct_yes_no_answer

        answer = build_matched_answer(answer_text, safe_matches)
        answer = with_claim_alignments(answer, _build_claim_alignments(supporting_sentences))
        answer = with_semantic_verification(
            answer,
            [
                _build_verification_entry(sentence, sentence, "strong", 1.0)
                for sentence, _ in supporting_sentences
            ],
        )
        answer = with_answer_plan(answer, answer_plan)
        answer = with_evidence_set(answer, evidence_set)
        answer = with_context_window(answer, build_context_window(evidence_matches))
        answer = with_parent_context(answer, _build_parent_context(safe_matches))
        answer = _attach_confidence_policy(answer, confidence_policy)
        return _with_prompt_injection_result(answer, policy_result)


@dataclass(frozen=True, slots=True)
class GenerativeAnswerer:
    """Optional generator-backed answerer that preserves grounded citations."""

    generator: TextGenerator
    fallback_answerer: ExtractiveAnswerer = field(default_factory=ExtractiveAnswerer)

    def answer(self, question: str, matches: list[RetrievalResult]) -> AnswerResult:
        """Generate a cited answer from supporting chunks."""
        if not matches:
            return build_no_match_answer(question)
        policy_result = apply_prompt_injection_policy(matches)
        safe_matches = policy_result.matches
        if not safe_matches:
            return _with_prompt_injection_result(build_no_match_answer(question), policy_result)
        route_reason, confidence_policy = _route_reason(question, safe_matches)
        if route_reason == "low_confidence":
            return _attach_confidence_policy(
                _with_prompt_injection_result(build_low_confidence_answer(question, safe_matches), policy_result),
                confidence_policy,
            )
        if route_reason == "clarification_needed":
            return _attach_confidence_policy(
                _with_prompt_injection_result(
                    build_clarification_needed_answer(question, safe_matches),
                    policy_result,
                ),
                confidence_policy,
            )
        evidence_matches = build_evidence_matches(
            question,
            safe_matches,
            limit=_evidence_limit(question),
        )
        evidence_set = build_evidence_set(
            question,
            safe_matches,
            limit=_evidence_limit(question),
        )
        answer_plan = build_answer_plan(question, evidence_set)

        generated_text = _compress_generated_text(self.generator.generate(question, evidence_matches))
        grounded, semantic_verification = _verify_generated_text(generated_text, evidence_matches)
        if not grounded:
            fallback_answer = self.fallback_answerer.answer(question, safe_matches)
            fallback_answer = with_semantic_verification(fallback_answer, semantic_verification)
            return _with_prompt_injection_result(fallback_answer, policy_result)
        support_lines = [
            f"- {sentence} [{chunk_id}]"
            for sentence, chunk_id in _select_answer_supporting_sentences(
                question,
                evidence_matches,
                answer_verbosity="concise",
            )
        ]
        citations = " ".join(f"[{match.chunk.chunk_id}]" for match in evidence_matches)
        answer_parts = [f"{generated_text} {citations}".strip()]
        if support_lines:
            answer_parts.append("Support:")
            answer_parts.extend(support_lines)
        answer_text = "\n".join(answer_parts)
        answer = build_matched_answer(answer_text, safe_matches)
        claim_sentences = [
            (sentence, chunk_id)
            for sentence, chunk_id in _select_answer_supporting_sentences(
                question,
                evidence_matches,
                answer_verbosity="concise",
            )
        ]
        generated_sentences = [
            (
                _strip_inline_citations(sentence),
                claim_sentences[0][1] if claim_sentences else evidence_matches[0].chunk.chunk_id,
            )
            for sentence in _split_sentences(generated_text)
            if sentence.strip()
        ]
        answer = with_claim_alignments(
            answer,
            _build_claim_alignments(
                generated_sentences,
                support_sentence=claim_sentences[0][0] if claim_sentences else None,
            ),
        )
        answer = with_semantic_verification(answer, semantic_verification)
        answer = with_answer_plan(answer, answer_plan)
        answer = with_evidence_set(answer, evidence_set)
        answer = with_context_window(answer, build_context_window(evidence_matches))
        answer = with_parent_context(answer, _build_parent_context(safe_matches))
        answer = _attach_confidence_policy(answer, confidence_policy)
        return _with_prompt_injection_result(answer, policy_result)


def build_answerer(
    config: RuntimeConfig,
    generator: TextGenerator | None = None,
) -> Answerer:
    """Create an answerer from runtime configuration."""
    if config.answer_mode == "generative":
        active_generator = generator or build_builtin_text_generator(config)
        return GenerativeAnswerer(generator=active_generator)
    return ExtractiveAnswerer(answer_verbosity=config.answer_verbosity)


def _select_supporting_sentences(
    question: str,
    matches: list[RetrievalResult],
    limit: int = 2,
) -> list[tuple[str, str]]:
    """Select the most relevant short supporting sentences from retrieved matches."""
    question_tokens = {
        token
        for token in QUESTION_TOKEN_PATTERN.findall(question.lower())
        if token not in QUESTION_STOPWORDS
    }
    ranked_sentences: list[tuple[tuple[int, float, int], str, str]] = []
    for match_index, match in enumerate(matches):
        sentences = _split_sentences(match.chunk.content)
        for sentence_index, sentence in enumerate(sentences):
            ranked_sentences.append(
                (
                    _sentence_rank_key(sentence, question_tokens, match.score, match_index, sentence_index),
                    sentence,
                    match.chunk.chunk_id,
                )
            )

    ranked_sentences.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, str]] = []
    seen_sentences: set[tuple[str, str]] = set()
    for _, sentence, chunk_id in ranked_sentences:
        dedupe_key = (sentence, chunk_id)
        if dedupe_key in seen_sentences:
            continue
        seen_sentences.add(dedupe_key)
        selected.append((sentence, chunk_id))
        if len(selected) >= limit:
            break
    return selected


def _sentence_limit_for_verbosity(answer_verbosity: str) -> int:
    """Return how many supporting sentences to include for one verbosity mode."""
    if answer_verbosity == "verbose":
        return VERBOSE_SUPPORTING_SENTENCE_LIMIT
    return CONCISE_SUPPORTING_SENTENCE_LIMIT


def _effective_answer_verbosity(question: str, answer_verbosity: str) -> str:
    """Use a slightly richer answer shape for summary-style questions."""
    normalized_question = question.lower().strip()
    if normalized_question.startswith(("summarize", "summary", "give a summary")):
        return "verbose"
    if normalized_question.startswith(("why", "what makes", "how does", "how do")):
        return "verbose"
    return answer_verbosity


def _evidence_limit(question: str, answer_verbosity: str = "concise") -> int:
    """Return how much compressed evidence to keep for one question."""
    limit = _sentence_limit_for_verbosity(_effective_answer_verbosity(question, answer_verbosity))
    if _should_aggregate_multi_hop(question):
        return max(limit, 3)
    return limit


def _select_answer_supporting_sentences(
    question: str,
    matches: list[RetrievalResult],
    *,
    answer_verbosity: str,
) -> list[tuple[str, str]]:
    """Select either focused or aggregated supporting sentences for one answer."""
    if _should_aggregate_multi_hop(question):
        return _select_multi_hop_supporting_sentences(question, matches, limit=2)
    return _select_supporting_sentences(
        question,
        matches,
        limit=_sentence_limit_for_verbosity(answer_verbosity),
    )


def _select_multi_hop_supporting_sentences(
    question: str,
    matches: list[RetrievalResult],
    *,
    limit: int,
) -> list[tuple[str, str]]:
    """Select one high-value supporting sentence per document for simple multi-hop answers."""
    ranked_sentences = _select_supporting_sentences(question, matches, limit=len(matches))
    selected: list[tuple[str, str]] = []
    seen_documents: set[str] = set()
    chunk_to_document_id = {match.chunk.chunk_id: match.chunk.document_id for match in matches}
    for sentence, chunk_id in ranked_sentences:
        document_id = chunk_to_document_id.get(chunk_id, chunk_id)
        if document_id in seen_documents:
            continue
        seen_documents.add(document_id)
        selected.append((sentence, chunk_id))
        if len(selected) >= limit:
            break
    return selected


def _build_direct_yes_no_answer(
    question: str,
    supporting_sentences: list[tuple[str, str]],
) -> str | None:
    """Render a direct grounded yes/no answer for simple binary questions."""
    predicate_token = _extract_yes_no_predicate_token(question)
    if predicate_token is None or not supporting_sentences:
        return None
    sentence, chunk_id = supporting_sentences[0]
    sentence_tokens = _grounding_token_set(sentence)
    if predicate_token in sentence_tokens:
        return f"Yes. The document says {sentence} [{chunk_id}]"
    if YES_NO_ANTONYMS.get(predicate_token, set()).intersection(sentence_tokens):
        return f"No. The document says {sentence} [{chunk_id}]"
    return None


def _extract_yes_no_predicate_token(question: str) -> str | None:
    """Return the main predicate token for a simple yes/no question when possible."""
    question_tokens = QUESTION_TOKEN_PATTERN.findall(question.lower())
    if not question_tokens or question_tokens[0] not in LEADING_YES_NO_VERBS:
        return None
    predicate_candidates = [
        _normalize_grounding_token(token)
        for token in question_tokens[1:]
        if token not in {"that", "there", "these", "those"}
    ]
    normalized_candidates = [token for token in predicate_candidates if token is not None]
    if not normalized_candidates:
        return None
    return _resolve_known_yes_no_token(normalized_candidates[-1])


def _resolve_known_yes_no_token(token: str) -> str:
    """Resolve one near-match token to a known yes/no predicate family when possible."""
    resolved_match = process.extractOne(
        token,
        list(YES_NO_ANTONYMS),
        score_cutoff=FUZZY_TOKEN_MATCH_THRESHOLD,
    )
    if resolved_match is None:
        return token
    return str(resolved_match[0])


def _compress_generated_text(generated_text: str, limit: int = GENERATIVE_SENTENCE_LIMIT) -> str:
    """Trim verbose generated output to a small number of sentences."""
    sentences = [segment.strip() for segment in SENTENCE_PATTERN.split(generated_text) if segment.strip()]
    if not sentences:
        return generated_text.strip()
    return " ".join(sentences[:limit]).strip()


def _verify_generated_text(
    generated_text: str,
    matches: list[RetrievalResult],
) -> tuple[bool, list[dict[str, object]]]:
    """Verify generated claims against retrieved evidence and return diagnostics."""
    generated_sentences = [_strip_inline_citations(sentence) for sentence in _split_sentences(generated_text)]
    generated_sentences = [sentence for sentence in generated_sentences if sentence.strip()]
    if not generated_sentences:
        return False, []

    context_sentences = [
        sentence
        for match in matches
        for sentence in _split_sentences(match.chunk.content)
    ]
    context_sentences = [sentence for sentence in context_sentences if sentence.strip()]
    if not context_sentences:
        return False, []

    verification_results: list[dict[str, object]] = []
    for generated_sentence in generated_sentences:
        best_support_sentence = ""
        best_coverage = 0.0
        best_status = "unsupported"
        for context_sentence in context_sentences:
            coverage, status = _sentence_support_profile(generated_sentence, context_sentence)
            if coverage > best_coverage:
                best_support_sentence = context_sentence
                best_coverage = coverage
                best_status = status
        verification_results.append(
            _build_verification_entry(
                generated_sentence,
                best_support_sentence,
                best_status,
                best_coverage,
            )
        )
    return all(result["status"] == "strong" for result in verification_results), verification_results


def _strip_inline_citations(text: str) -> str:
    """Remove inline chunk citations before grounding checks."""
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def _sentence_is_supported(generated_sentence: str, context_sentence: str) -> bool:
    """Return whether one generated sentence is supported by one context sentence."""
    coverage, status = _sentence_support_profile(generated_sentence, context_sentence)
    return status == "strong" and coverage >= GROUNDING_COVERAGE_THRESHOLD


def _sentence_support_profile(generated_sentence: str, context_sentence: str) -> tuple[float, str]:
    """Return support coverage plus one coarse semantic support status."""
    generated_tokens = _grounding_token_set(generated_sentence)
    context_tokens = _grounding_token_set(context_sentence)
    if not generated_tokens or not context_tokens:
        return 0.0, "unsupported"
    uncovered_tokens = generated_tokens.difference(context_tokens)
    supported_token_count = len(generated_tokens.intersection(context_tokens))
    coverage = supported_token_count / len(generated_tokens)
    if coverage >= STRONG_GROUNDING_COVERAGE_THRESHOLD and len(uncovered_tokens) <= MAX_UNCOVERED_GROUNDING_TOKENS:
        return coverage, "strong"
    if coverage >= WEAK_GROUNDING_COVERAGE_THRESHOLD:
        return coverage, "weak"
    return coverage, "unsupported"


def _build_verification_entry(
    claim: str,
    support_sentence: str,
    status: str,
    coverage: float,
) -> dict[str, object]:
    """Build one semantic-verification diagnostics entry."""
    return {
        "claim": claim,
        "support_sentence": support_sentence,
        "status": status,
        "coverage": round(coverage, 6),
    }


def _grounding_token_set(text: str) -> set[str]:
    """Build a small normalized semantic token set for grounding checks."""
    grounded_tokens: set[str] = set()
    for token in QUESTION_TOKEN_PATTERN.findall(text.lower()):
        normalized_token = _normalize_grounding_token(token)
        if normalized_token is None:
            continue
        grounded_tokens.add(normalized_token)
        grounded_tokens.update(GROUNDING_EQUIVALENTS.get(normalized_token, set()))
    return grounded_tokens


def _normalize_grounding_token(token: str) -> str | None:
    """Normalize one token for sentence-level grounding checks."""
    if token in GROUNDING_STOPWORDS:
        return None
    normalized = token
    if normalized.endswith("mates") and len(normalized) >= 6:
        normalized = normalized[:-1]
    for suffix, replacement, minimum_length in (
        ("ingly", "", 7),
        ("ing", "", 5),
        ("edly", "", 6),
        ("ed", "", 4),
        ("ies", "y", 5),
        ("es", "", 5),
        ("s", "", 4),
    ):
        if normalized.endswith(suffix) and len(normalized) >= minimum_length:
            normalized = f"{normalized[:-len(suffix)]}{replacement}"
            break
    if len(normalized) >= 2 and normalized[-1] == normalized[-2] and normalized[-1] not in "aeiou":
        normalized = normalized[:-1]
    if not normalized or normalized in GROUNDING_STOPWORDS:
        return None
    return normalized


def _split_sentences(content: str) -> list[str]:
    """Split chunk text into compact sentence-like segments."""
    sentences = [segment.strip() for segment in SENTENCE_PATTERN.split(content) if segment.strip()]
    return sentences or [content.strip()]


def _sentence_rank_key(
    sentence: str,
    question_tokens: set[str],
    match_score: float,
    match_index: int,
    sentence_index: int,
) -> tuple[int, float, int]:
    """Build a ranking key for one candidate supporting sentence."""
    sentence_tokens = set(QUESTION_TOKEN_PATTERN.findall(sentence.lower()))
    token_overlap = len(question_tokens.intersection(sentence_tokens))
    return (
        token_overlap,
        match_score,
        -((match_index * SENTENCE_RANK_STRIDE) + sentence_index),
    )


def _build_claim_alignments(
    claims: list[tuple[str, str]],
    *,
    support_sentence: str | None = None,
) -> list[dict[str, object]]:
    """Build claim-to-evidence mappings for diagnostics."""
    alignments: list[dict[str, object]] = []
    for claim, chunk_id in claims:
        resolved_support_sentence = support_sentence or claim
        alignments.append(
            {
                "claim": claim,
                "chunk_id": chunk_id,
                "support_sentence": resolved_support_sentence,
                "support_span": {
                    "text": resolved_support_sentence,
                    "start_offset": 0,
                    "end_offset": len(resolved_support_sentence),
                },
            }
        )
    return alignments


def _with_prompt_injection_result(
    answer: AnswerResult,
    policy_result: PromptInjectionPolicyResult,
) -> AnswerResult:
    """Attach one prompt-injection policy result to an answer consistently."""
    return with_prompt_injection_policy(
        answer,
        blocked_sentences=policy_result.blocked_sentences,
        blocked_chunk_ids=policy_result.blocked_chunk_ids,
        downgraded_chunk_ids=policy_result.downgraded_chunk_ids,
    )


def _build_parent_context(matches: list[RetrievalResult]) -> list[dict[str, object]]:
    """Build parent-document expansions for matched child chunks."""
    parent_entries: list[dict[str, object]] = []
    seen_parent_ids: set[str] = set()
    for match in matches:
        parent_document_id = str(match.chunk.metadata.get("parent_document_id", match.chunk.document_id))
        if parent_document_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_document_id)
        parent_entries.append(
            {
                "parent_document_id": parent_document_id,
                "child_chunk_id": match.chunk.chunk_id,
                "parent_excerpt": str(match.chunk.metadata.get("parent_content", match.chunk.content))[
                    :PARENT_EXCERPT_LIMIT
                ],
            }
        )
    return parent_entries


def _route_reason(question: str, matches: list[RetrievalResult]) -> tuple[str, dict[str, object]]:
    """Return one deterministic routing reason plus supporting policy diagnostics."""
    evidence_set = build_evidence_set(question, matches, limit=2)
    support_coverage = _support_coverage(question, evidence_set)
    score_gap = matches[0].score - matches[1].score if len(matches) > 1 else matches[0].score
    evidence_agreement = _evidence_agreement(question, evidence_set, score_gap)
    if matches[0].score < LOW_CONFIDENCE_SCORE:
        return "low_confidence", _build_confidence_policy(
            matches[0].score,
            score_gap,
            support_coverage,
            evidence_agreement,
            "top_score_below_threshold",
        )
    if _needs_clarification(question, matches):
        return "clarification_needed", _build_confidence_policy(
            matches[0].score,
            score_gap,
            support_coverage,
            evidence_agreement,
            "ambiguous_top_matches",
        )
    if (
        not _should_aggregate_multi_hop(question)
        and support_coverage < LOW_CONFIDENCE_SUPPORT_COVERAGE
        and score_gap < LOW_CONFIDENCE_SCORE_GAP
    ):
        return "low_confidence", _build_confidence_policy(
            matches[0].score,
            score_gap,
            support_coverage,
            evidence_agreement,
            "weak_support_coverage",
        )
    return "matched", _build_confidence_policy(
        matches[0].score,
        score_gap,
        support_coverage,
        evidence_agreement,
        "matched",
    )


def _build_confidence_policy(
    top_score: float,
    score_gap: float,
    support_coverage: float,
    evidence_agreement: bool,
    route_basis: str,
) -> dict[str, object]:
    """Build one explicit confidence-policy diagnostics payload."""
    return {
        "top_score": round(top_score, 6),
        "score_gap": round(score_gap, 6),
        "support_coverage": round(support_coverage, 6),
        "evidence_agreement": evidence_agreement,
        "route_basis": route_basis,
    }


def _support_coverage(question: str, evidence_set: list[dict[str, object]]) -> float:
    """Estimate how much of the question is reflected in the selected evidence."""
    question_tokens = _support_token_set(question)
    if not question_tokens or not evidence_set:
        return 0.0
    evidence_tokens: set[str] = set()
    for entry in evidence_set:
        evidence_tokens.update(_support_token_set(str(entry["sentence"])))
    return len(question_tokens.intersection(evidence_tokens)) / len(question_tokens)


def _evidence_agreement(question: str, evidence_set: list[dict[str, object]], score_gap: float) -> bool:
    """Return whether the top evidence agrees enough for one confident answer."""
    if len(evidence_set) < 2 or _is_explicit_multi_document_question(question):
        return True
    first_document_id = str(evidence_set[0]["document_id"])
    second_document_id = str(evidence_set[1]["document_id"])
    return first_document_id == second_document_id or score_gap > CLARIFICATION_SCORE_DELTA


def _support_token_set(text: str) -> set[str]:
    """Build a normalized token set for routing coverage without synonym expansion."""
    tokens: set[str] = set()
    for token in QUESTION_TOKEN_PATTERN.findall(text.lower()):
        normalized_token = _normalize_grounding_token(token)
        if normalized_token is not None:
            tokens.add(normalized_token)
    return tokens


def _is_explicit_multi_document_question(question: str) -> bool:
    """Return whether the question intentionally asks across multiple topics."""
    normalized_question = question.lower()
    return " and " in normalized_question or " also " in normalized_question


def _prefers_structured_match(question: str, matches: list[RetrievalResult]) -> bool:
    """Return whether the question explicitly prefers a structured record-style answer."""
    if not matches:
        return False
    question_tokens = set(QUESTION_TOKEN_PATTERN.findall(question.lower()))
    return (
        bool(question_tokens.intersection(STRUCTURED_QUERY_TOKENS))
        and matches[0].chunk.metadata.get("format") == "structured"
    )


def _should_aggregate_multi_hop(question: str) -> bool:
    """Return whether the answer should merge evidence from multiple documents."""
    normalized_question = question.lower()
    return _is_explicit_multi_document_question(question) or normalized_question.startswith("compare")


def _attach_confidence_policy(answer: AnswerResult, confidence_policy: dict[str, object]) -> AnswerResult:
    """Attach confidence-policy diagnostics to one answer result."""
    return with_confidence_policy(answer, confidence_policy)


def _needs_clarification(question: str, matches: list[RetrievalResult]) -> bool:
    """Return whether the top matches are too ambiguous to answer confidently."""
    if _should_aggregate_multi_hop(question):
        return False
    if _prefers_structured_match(question, matches):
        return False
    if len(matches) < 2:
        return False
    first_match, second_match = matches[0], matches[1]
    if first_match.chunk.document_id == second_match.chunk.document_id:
        return False
    return abs(first_match.score - second_match.score) <= CLARIFICATION_SCORE_DELTA
