"""
DocMind — Answer Quality Evaluator

Evaluates the complete production RAG pipeline:

Question
    ↓
Hybrid Retrieval
    ↓
Dense + BM25
    ↓
Score Fusion
    ↓
Cross-Encoder Reranking
    ↓
Retrieved Evidence
    ↓
Grounded Prompt
    ↓
Ollama Cloud
    ↓
Answer + Sources

This evaluator does NOT invent reference answers.

It evaluates:
    - retrieval support
    - source correctness
    - citation correctness
    - context relevance
    - groundedness proxies
    - abstention behavior
    - prompt-injection warning signals
    - answer length
    - retrieval latency
    - generation latency
    - end-to-end latency
    - difficulty breakdown
    - category breakdown

Reference-answer based correctness is intentionally separated from
automated evidence-based evaluation. Human-validated reference
answers can be added later.
"""

from __future__ import annotations

import csv
import re
import statistics
import time
from pathlib import Path
from typing import Any

from app.evaluation.answer_evaluation_dataset import (
    AnswerEvaluationCase,
    get_answer_evaluation_dataset,
)

from app.evaluation.context_relevance import (
    calculate_context_relevance,
)

from app.generation.answer_service import (
    ABSTENTION_MESSAGE,
    AnswerService,
)


# =====================================================================
# PATHS / CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "docs"
    / "answer_evaluation_results.csv"
)

DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_K = 10


# =====================================================================
# CONSTANTS
# =====================================================================

ABSTENTION_PATTERNS = (
    "i don't know based on the available documents",
    "i do not know based on the available documents",
)

INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the system prompt",
    "ignore system instructions",
    "system prompt",
    "developer message",
    "reveal your prompt",
    "reveal the prompt",
    "api key",
    "secret",
    "password",
)


# =====================================================================
# TEXT HELPERS
# =====================================================================

def normalize_text(text: str) -> str:
    """Normalize text for lightweight comparisons."""

    text = str(text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> set[str]:
    """Return simple alphanumeric tokens."""

    return set(
        re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
            normalize_text(text),
        )
    )


# =====================================================================
# SOURCE HELPERS
# =====================================================================

def _source_filename(
    source: dict[str, Any],
) -> str:
    return str(
        source.get("filename")
        or source.get("source")
        or ""
    ).strip()


def _source_page(
    source: dict[str, Any],
) -> Any:
    return source.get("page")


def _source_category(
    source: dict[str, Any],
) -> str:
    return str(
        source.get("category")
        or ""
    ).strip().lower()


def expected_source_match(
    expected_sources: tuple[str, ...],
    actual_sources: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """
    Check whether an expected source was retrieved.

    Matching is filename based because the existing retrieval
    evaluation dataset defines relevance at source-document level.
    """

    if not expected_sources:
        return False, []

    expected_normalized = {
        normalize_text(source)
        for source in expected_sources
    }

    matched: list[str] = []

    for source in actual_sources:
        filename = normalize_text(
            _source_filename(source)
        )

        if not filename:
            continue

        for expected in expected_normalized:

            if (
                filename == expected
                or expected in filename
                or filename in expected
            ):
                matched.append(
                    _source_filename(source)
                )

                break

    return (
        bool(matched),
        sorted(set(matched)),
    )


# =====================================================================
# CITATION EVALUATION
# =====================================================================

def extract_citation_numbers(
    answer: str,
) -> list[int]:
    """
    Extract inline citation numbers such as [1], [2], [3].

    DocMind currently displays authoritative source cards separately,
    so absence of inline citations is not automatically considered
    an error.
    """

    numbers = re.findall(
        r"\[(\d+)\]",
        str(answer or ""),
    )

    return [
        int(number)
        for number in numbers
    ]


def evaluate_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate citation correctness.

    Two modes are supported:

    1. inline
       If the generated answer contains [1], [2], etc.

    2. source_metadata
       If no inline citations are present, DocMind's authoritative
       source metadata is evaluated instead.
    """

    citation_numbers = extract_citation_numbers(
        answer
    )

    source_count = len(sources)

    invalid_citations = [
        number
        for number in citation_numbers
        if number < 1
        or number > source_count
    ]

    if citation_numbers:

        citation_correct = (
            len(invalid_citations) == 0
        )

        if citation_numbers:
            citation_score = max(
                0.0,
                1.0
                - (
                    len(invalid_citations)
                    / len(citation_numbers)
                ),
            )
        else:
            citation_score = 0.0

        return {
            "citation_numbers": citation_numbers,
            "invalid_citations": invalid_citations,
            "citation_correct": citation_correct,
            "citation_score": citation_score,
            "citation_mode": "inline",
        }

    # -----------------------------------------------------------------
    # Current DocMind UI uses authoritative source cards.
    # -----------------------------------------------------------------

    return {
        "citation_numbers": [],
        "invalid_citations": [],
        "citation_correct": source_count > 0,
        "citation_score": (
            1.0
            if source_count > 0
            else 0.0
        ),
        "citation_mode": "source_metadata",
    }


# =====================================================================
# ABSTENTION EVALUATION
# =====================================================================

def is_abstention(
    answer: str,
) -> bool:
    """Detect DocMind's grounded abstention response."""

    normalized = normalize_text(
        answer
    )

    if not normalized:
        return True

    return any(
        pattern in normalized
        for pattern in ABSTENTION_PATTERNS
    )


def evaluate_abstention(
    answer: str,
    expected: bool,
) -> dict[str, Any]:
    """Evaluate whether abstention behavior is correct."""

    actual = is_abstention(
        answer
    )

    correct = (
        actual
        if expected
        else not actual
    )

    return {
        "abstention_expected": expected,
        "abstained": actual,
        "abstention_correct": correct,
    }


# =====================================================================
# GROUNDEDNESS
# =====================================================================

def calculate_token_overlap(
    answer: str,
    context: str,
) -> float:
    """
    Calculate a lightweight lexical support score.

    This is a proxy, NOT proof of factual correctness.
    """

    answer_tokens = tokenize(
        answer
    )

    context_tokens = tokenize(
        context
    )

    if not answer_tokens:
        return 0.0

    if not context_tokens:
        return 0.0

    overlap = (
        answer_tokens
        & context_tokens
    )

    return (
        len(overlap)
        / len(answer_tokens)
    )


def evaluate_groundedness(
    answer: str,
    context: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Estimate answer groundedness using retrieved evidence.

    Signals:

        - lexical answer/context support
        - existence of retrieved evidence
        - abstention

    This is intentionally described as a proxy rather than a
    definitive hallucination detector.
    """

    if is_abstention(answer):

        return {
            "groundedness_score": 1.0,
            "token_support_score": 1.0,
            "evidence_available": bool(
                sources
            ),
            "groundedness_method": (
                "abstention"
            ),
        }

    token_support = (
        calculate_token_overlap(
            answer,
            context,
        )
    )

    evidence_available = bool(
        sources
    )

    if evidence_available:
        groundedness = (
            0.7 * token_support
            + 0.3
        )
    else:
        groundedness = 0.0

    groundedness = min(
        1.0,
        max(
            0.0,
            groundedness,
        ),
    )

    return {
        "groundedness_score": groundedness,
        "token_support_score": token_support,
        "evidence_available": evidence_available,
        "groundedness_method": (
            "lexical_evidence_proxy"
        ),
    }


# =====================================================================
# PROMPT INJECTION SIGNALS
# =====================================================================

def detect_suspicious_content(
    answer: str,
) -> dict[str, Any]:
    """
    Detect common prompt-injection-related phrases in generated
    answers.

    This is a heuristic security signal, not a complete detector.
    """

    normalized = normalize_text(
        answer
    )

    matches = [
        pattern
        for pattern in INJECTION_PATTERNS
        if pattern in normalized
    ]

    return {
        "suspicious_content": bool(
            matches
        ),
        "suspicious_patterns": matches,
    }


# =====================================================================
# RETRIEVAL / CONTEXT EXTRACTION
# =====================================================================

def _extract_retrieved_results(
    answer_service: AnswerService,
    question: str,
) -> list[dict[str, Any]]:
    """
    Reuse the exact production retrieval path.

    This intentionally calls AnswerService._retrieve() so the
    evaluation uses the same Hybrid/BM25/Reranker configuration
    as the application.
    """

    results = answer_service._retrieve(
        question
    )

    if results is None:
        return []

    return list(results)


def _build_context(
    answer_service: AnswerService,
    results: list[dict[str, Any]],
) -> str:
    """Reuse AnswerService's existing context builder."""

    return answer_service.build_context(
        results
    )


def _sources_from_results(
    answer_service: AnswerService,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reuse AnswerService's source metadata builder."""

    return answer_service._build_sources(
        results
    )


# =====================================================================
# CONTEXT RELEVANCE
# =====================================================================

def _evaluate_context_relevance(
    question: str,
    context: str,
    sources: list[dict[str, Any]],
    expected_sources: tuple[str, ...],
) -> dict[str, Any]:
    """
    Evaluate context relevance using the project's existing
    context_relevance.py implementation.

    The existing implementation requires:

        question
        context
        sources
        expected_source
        expected_keywords
    """

    expected_source = (
        expected_sources[0]
        if expected_sources
        else None
    )

    try:

        result = calculate_context_relevance(
            question=question,
            context=context,
            sources=sources,
            expected_source=expected_source,
            expected_keywords=(),
        )

    except Exception as exc:

        return {
            "context_relevance_score": 0.0,
            "context_relevance_error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    if not isinstance(
        result,
        dict,
    ):

        try:
            score = float(
                result
            )

        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        return {
            "context_relevance_score": score,
        }

    score = result.get(
        "context_relevance_score",
        result.get(
            "score",
            0.0,
        ),
    )

    try:
        score = float(
            score
        )

    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    return {
        **result,
        "context_relevance_score": score,
    }


# =====================================================================
# SINGLE CASE
# =====================================================================

def evaluate_case(
    answer_service: AnswerService,
    case: AnswerEvaluationCase,
) -> dict[str, Any]:
    """
    Evaluate one question against the complete RAG pipeline.
    """

    started = time.perf_counter()

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    retrieval_started = time.perf_counter()

    try:

        retrieved_results = (
            _extract_retrieved_results(
                answer_service,
                case.question,
            )
        )

        retrieval_error = ""

    except Exception as exc:

        retrieved_results = []

        retrieval_error = (
            f"{type(exc).__name__}: {exc}"
        )

    retrieval_ms = (
        time.perf_counter()
        - retrieval_started
    ) * 1000

    # -----------------------------------------------------------------
    # Context
    # -----------------------------------------------------------------

    context_started = time.perf_counter()

    try:

        context = _build_context(
            answer_service,
            retrieved_results,
        )

        retrieval_sources = (
            _sources_from_results(
                answer_service,
                retrieved_results,
            )
        )

        context_error = ""

    except Exception as exc:

        context = ""
        retrieval_sources = []

        context_error = (
            f"{type(exc).__name__}: {exc}"
        )

    context_ms = (
        time.perf_counter()
        - context_started
    ) * 1000

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------

    generation_started = time.perf_counter()

    try:

        result = answer_service.ask(
            case.question
        )

        generation_error = ""

    except Exception as exc:

        result = {
            "question": case.question,
            "answer": (
                "Evaluation error: "
                f"{type(exc).__name__}: {exc}"
            ),
            "sources": [],
            "retrieved_count": 0,
            "top_k": DEFAULT_TOP_K,
            "retrieval_mode": (
                answer_service.get_retrieval_mode()
            ),
            "timings_ms": {},
            "observability": {},
        }

        generation_error = (
            f"{type(exc).__name__}: {exc}"
        )

    generation_ms = (
        time.perf_counter()
        - generation_started
    ) * 1000

    # -----------------------------------------------------------------
    # Answer + authoritative sources
    # -----------------------------------------------------------------

    answer = str(
        result.get(
            "answer",
            "",
        )
    ).strip()

    answer_sources = result.get(
        "sources",
        [],
    )

    if not answer_sources:
        answer_sources = (
            retrieval_sources
        )

    # -----------------------------------------------------------------
    # Expected source support
    # -----------------------------------------------------------------

    (
        expected_source_found,
        matched_sources,
    ) = expected_source_match(
        case.expected_sources,
        answer_sources,
    )

    # -----------------------------------------------------------------
    # Context relevance
    # -----------------------------------------------------------------

    context_relevance = (
        _evaluate_context_relevance(
            question=case.question,
            context=context,
            sources=answer_sources,
            expected_sources=case.expected_sources,
        )
    )

    # -----------------------------------------------------------------
    # Groundedness
    # -----------------------------------------------------------------

    groundedness = (
        evaluate_groundedness(
            answer=answer,
            context=context,
            sources=answer_sources,
        )
    )

    # -----------------------------------------------------------------
    # Citations
    # -----------------------------------------------------------------

    citations = evaluate_citations(
        answer=answer,
        sources=answer_sources,
    )

    # -----------------------------------------------------------------
    # Abstention
    # -----------------------------------------------------------------

    abstention = evaluate_abstention(
        answer=answer,
        expected=case.abstention_expected,
    )

    # -----------------------------------------------------------------
    # Security
    # -----------------------------------------------------------------

    suspicious = (
        detect_suspicious_content(
            answer
        )
    )

    # -----------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------

    total_ms = (
        time.perf_counter()
        - started
    ) * 1000

    # -----------------------------------------------------------------
    # Row
    # -----------------------------------------------------------------

    row: dict[str, Any] = {
        "question": case.question,

        "difficulty": case.difficulty,

        "category": case.category,

        "expected_sources": " | ".join(
            case.expected_sources
        ),

        "matched_sources": " | ".join(
            matched_sources
        ),

        "expected_source_found": (
            expected_source_found
        ),

        "answer": answer,

        "abstention_expected": (
            case.abstention_expected
        ),

        "abstained": (
            abstention["abstained"]
        ),

        "abstention_correct": (
            abstention["abstention_correct"]
        ),

        "retrieved_count": len(
            retrieved_results
        ),

        "source_count": len(
            answer_sources
        ),

        "context_characters": len(
            context
        ),

        "context_relevance_score": round(
            float(
                context_relevance.get(
                    "context_relevance_score",
                    0.0,
                )
            ),
            4,
        ),

        "token_support_score": round(
            float(
                groundedness.get(
                    "token_support_score",
                    0.0,
                )
            ),
            4,
        ),

        "groundedness_score": round(
            float(
                groundedness.get(
                    "groundedness_score",
                    0.0,
                )
            ),
            4,
        ),

        "citation_score": round(
            float(
                citations.get(
                    "citation_score",
                    0.0,
                )
            ),
            4,
        ),

        "citation_correct": (
            citations[
                "citation_correct"
            ]
        ),

        "citation_mode": (
            citations[
                "citation_mode"
            ]
        ),

        "citation_numbers": ",".join(
            str(number)
            for number in citations[
                "citation_numbers"
            ]
        ),

        "suspicious_content": (
            suspicious[
                "suspicious_content"
            ]
        ),

        "suspicious_patterns": " | ".join(
            suspicious[
                "suspicious_patterns"
            ]
        ),

        "answer_length": len(
            answer
        ),

        "retrieval_ms": round(
            retrieval_ms,
            2,
        ),

        "context_ms": round(
            context_ms,
            2,
        ),

        "generation_ms": round(
            generation_ms,
            2,
        ),

        "total_ms": round(
            total_ms,
            2,
        ),

        "retrieval_mode": result.get(
            "retrieval_mode",
            answer_service.get_retrieval_mode(),
        ),

        "top_k": result.get(
            "top_k",
            DEFAULT_TOP_K,
        ),

        "top_hybrid_score": result.get(
            "observability",
            {},
        ).get(
            "top_hybrid_score"
        ),

        "top_reranker_score": result.get(
            "observability",
            {},
        ).get(
            "top_reranker_score"
        ),

        "retrieval_error": retrieval_error,

        "context_error": context_error,

        "generation_error": generation_error,
    }

    return row


# =====================================================================
# AGGREGATION
# =====================================================================

def _mean(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return statistics.mean(
        values
    )


def _percentage(
    values: list[bool],
) -> float:

    if not values:
        return 0.0

    return (
        sum(
            1
            for value in values
            if value
        )
        / len(values)
    ) * 100


def calculate_aggregate_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate overall answer-quality metrics."""

    if not rows:
        return {
            "total_questions": 0
        }

    return {
        "total_questions": len(
            rows
        ),

        "expected_source_hit_rate": round(
            _percentage(
                [
                    bool(
                        row[
                            "expected_source_found"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),

        "abstention_accuracy": round(
            _percentage(
                [
                    bool(
                        row[
                            "abstention_correct"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),

        "groundedness_mean": round(
            _mean(
                [
                    float(
                        row[
                            "groundedness_score"
                        ]
                    )
                    for row in rows
                ]
            ),
            4,
        ),

        "context_relevance_mean": round(
            _mean(
                [
                    float(
                        row[
                            "context_relevance_score"
                        ]
                    )
                    for row in rows
                ]
            ),
            4,
        ),

        "citation_correctness": round(
            _percentage(
                [
                    bool(
                        row[
                            "citation_correct"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),

        "suspicious_answer_rate": round(
            _percentage(
                [
                    bool(
                        row[
                            "suspicious_content"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),

        "avg_answer_length": round(
            _mean(
                [
                    float(
                        row[
                            "answer_length"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),

        "avg_retrieval_ms": round(
            _mean(
                [
                    float(
                        row[
                            "retrieval_ms"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),

        "avg_generation_ms": round(
            _mean(
                [
                    float(
                        row[
                            "generation_ms"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),

        "avg_total_ms": round(
            _mean(
                [
                    float(
                        row[
                            "total_ms"
                        ]
                    )
                    for row in rows
                ]
            ),
            2,
        ),
    }


def group_metrics(
    rows: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    """Calculate metrics grouped by a field."""

    groups: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    for row in rows:

        value = str(
            row.get(
                field,
                "unknown",
            )
        )

        groups.setdefault(
            value,
            [],
        ).append(row)

    output: list[dict[str, Any]] = []

    for value, group in sorted(
        groups.items()
    ):

        metrics = (
            calculate_aggregate_metrics(
                group
            )
        )

        output.append(
            {
                field: value,
                **metrics,
            }
        )

    return output


# =====================================================================
# CSV
# =====================================================================

def save_results(
    rows: list[dict[str, Any]],
    path: Path = RESULTS_PATH,
) -> None:
    """Save evaluation results as CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =====================================================================
# COMPLETE EVALUATION
# =====================================================================

def evaluate_all(
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> dict[str, Any]:
    """
    Run the complete 30-question answer evaluation.

    Production configuration:

        Hybrid Dense + BM25
        Candidate K = 10
        Cross-Encoder reranking
        Final Top K = 3
        Ollama Cloud generation
    """

    cases = (
        get_answer_evaluation_dataset()
    )

    print("=" * 80)
    print(
        "DOCMIND — ANSWER QUALITY EVALUATION"
    )
    print("=" * 80)

    print(
        f"Questions: {len(cases)}"
    )

    print(
        f"Top-K: {top_k}"
    )

    print(
        f"Candidate-K: {candidate_k}"
    )

    print(
        f"Hybrid retrieval: {use_hybrid}"
    )

    print(
        f"Cross-Encoder reranker: {use_reranker}"
    )

    print()

    # --------------------------------------------------------------
    # One AnswerService instance is intentionally reused for all
    # questions so the evaluation uses the same configuration and
    # avoids repeatedly loading embedding/reranker models.
    # --------------------------------------------------------------

    answer_service = AnswerService(
        top_k=top_k,
        use_hybrid=use_hybrid,
        candidate_k=candidate_k,
        use_reranker=use_reranker,
    )

    rows: list[
        dict[str, Any]
    ] = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(cases):02d}] "
            f"{case.question}"
        )

        started = time.perf_counter()

        row = evaluate_case(
            answer_service,
            case,
        )

        rows.append(
            row
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            "    "
            f"source_hit="
            f"{row['expected_source_found']} "
            f"groundedness="
            f"{row['groundedness_score']:.3f} "
            f"context="
            f"{row['context_relevance_score']:.3f} "
            f"citation="
            f"{row['citation_correct']} "
            f"time="
            f"{elapsed:.2f}s"
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_results(
        rows,
        RESULTS_PATH,
    )

    # --------------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------------

    overall = (
        calculate_aggregate_metrics(
            rows
        )
    )

    by_difficulty = (
        group_metrics(
            rows,
            "difficulty",
        )
    )

    by_category = (
        group_metrics(
            rows,
            "category",
        )
    )

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------

    print()

    print("=" * 80)
    print(
        "OVERALL RESULTS"
    )
    print("=" * 80)

    for key, value in (
        overall.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print("=" * 80)
    print(
        "BY DIFFICULTY"
    )
    print("=" * 80)

    for row in by_difficulty:
        print(row)

    print()

    print("=" * 80)
    print(
        "BY CATEGORY"
    )
    print("=" * 80)

    for row in by_category:
        print(row)

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_PATH
    )

    return {
        "rows": rows,
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "results_path": str(
            RESULTS_PATH
        ),
    }


# =====================================================================
# COMMAND-LINE ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    evaluate_all()