"""
DocMind - Normal LLM vs RAG Evaluation

Compares:
1. Normal Ollama Cloud LLM without knowledge-base retrieval.
2. DocMind RAG using retrieval, grounded context, and Ollama Cloud.

Measures:
- Normal LLM generation latency
- RAG retrieval latency
- RAG context-building latency
- RAG generation latency
- RAG total latency
- Normal evidence-support proxy
- RAG groundedness proxy
- RAG context relevance
- Citation availability
- Expected-source hit rate
- Abstention behavior

Results:
docs/normal_vs_rag_results.csv
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.evaluation.answer_evaluation_dataset import (
    get_answer_evaluation_dataset,
)
from app.evaluation.context_relevance import (
    calculate_context_relevance,
)
from app.generation.answer_service import (
    ABSTENTION_MESSAGE,
    AnswerService,
)
from app.generation.llm import OllamaCloud


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "docs"
    / "normal_vs_rag_results.csv"
)


# ============================================================
# NORMAL LLM PROMPT
# ============================================================

NORMAL_LLM_SYSTEM_PROMPT = """
You are a general-purpose assistant.

Answer the user's question using your general language-model
knowledge.

You do NOT have access to DocMind's private knowledge base.

Do not claim that you searched a knowledge base.
Do not invent citations.
Do not pretend that sources were retrieved.

If you are uncertain, clearly say that you are uncertain.
""".strip()


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for lexical comparison.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> set[str]:
    """
    Convert text into normalized lexical tokens.

    This is intentionally simple because these metrics are
    evidence-support proxies rather than factuality detectors.
    """

    normalized = normalize_text(text)

    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "as",
        "at",
        "from",
        "what",
        "which",
        "how",
        "does",
        "do",
        "can",
        "this",
        "that",
        "it",
        "they",
        "their",
        "its",
        "about",
    }

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
        and token not in stopwords
    }


# ============================================================
# EVIDENCE SUPPORT PROXY
# ============================================================

def evidence_support_proxy(
    question: str,
    answer: str,
    context: str,
) -> float:
    """
    Estimate how much of the answer vocabulary is supported
    by the available evidence.

    IMPORTANT:
    This is a lexical proxy.

    It is NOT a factuality detector and should not be presented
    as a definitive correctness score.
    """

    answer_tokens = tokenize(answer)

    if not answer_tokens:
        return 0.0

    evidence_tokens = tokenize(
        f"{question} {context}"
    )

    supported_tokens = (
        answer_tokens.intersection(
            evidence_tokens
        )
    )

    score = (
        len(supported_tokens)
        / len(answer_tokens)
    )

    return round(
        score,
        4,
    )


# ============================================================
# SOURCE MATCHING
# ============================================================

def source_matches(
    expected_source: str,
    sources: list[dict[str, Any]],
) -> bool:
    """
    Check whether the expected source appears in
    the retrieved source metadata.
    """

    expected = normalize_text(
        expected_source
    )

    if not expected:
        return False

    for source in sources:

        filename = normalize_text(
            str(
                source.get(
                    "filename",
                    "",
                )
            )
        )

        source_name = normalize_text(
            str(
                source.get(
                    "source",
                    "",
                )
            )
        )

        if (
            expected in filename
            or expected in source_name
        ):
            return True

    return False


def citation_available(
    sources: list[dict[str, Any]],
) -> bool:
    """
    RAG citation availability is true when source metadata
    is returned by the RAG pipeline.
    """

    return bool(sources)


# ============================================================
# NORMAL LLM
# ============================================================

def run_normal_llm(
    llm: OllamaCloud,
    question: str,
) -> tuple[str, float, str | None]:
    """
    Run the question against Ollama Cloud without RAG.
    """

    started = time.perf_counter()

    try:

        answer = llm.chat(
            system_prompt=(
                NORMAL_LLM_SYSTEM_PROMPT
            ),
            user_prompt=question,
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return (
            answer,
            round(
                elapsed_ms,
                2,
            ),
            None,
        )

    except Exception as exc:

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return (
            "",
            round(
                elapsed_ms,
                2,
            ),
            str(exc),
        )


# ============================================================
# RAG
# ============================================================

def run_rag(
    answer_service: AnswerService,
    question: str,
) -> dict[str, Any]:
    """
    Run the question through the complete DocMind RAG pipeline.
    """

    try:

        result = answer_service.ask(
            question
        )

        return {
            "result": result,
            "error": None,
        }

    except Exception as exc:

        return {
            "result": {},
            "error": str(exc),
        }


# ============================================================
# RETRIEVE ACTUAL RAG CONTEXT
# ============================================================

def get_actual_rag_context(
    answer_service: AnswerService,
    question: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve the same chunks used by the production RAG
    pipeline and build the actual context string.

    This avoids calculating groundedness from source metadata
    alone.
    """

    retrieved_results = (
        answer_service._retrieve(
            question
        )
    )

    context = (
        answer_service.build_context(
            retrieved_results
        )
    )

    sources = (
        answer_service._build_sources(
            retrieved_results
        )
    )

    return (
        context,
        sources,
    )


# ============================================================
# EVALUATE ONE CASE
# ============================================================

def evaluate_case(
    case: Any,
    llm: OllamaCloud,
    answer_service: AnswerService,
) -> dict[str, Any]:

    question = case.question

    print()
    print("=" * 70)
    print(
        f"Question: {question}"
    )
    print(
        f"Difficulty: {case.difficulty}"
    )
    print(
        f"Category: {case.category}"
    )

    # --------------------------------------------------------
    # NORMAL LLM
    # --------------------------------------------------------

    (
        normal_answer,
        normal_generation_ms,
        normal_error,
    ) = run_normal_llm(
        llm=llm,
        question=question,
    )

    normal_support = (
        evidence_support_proxy(
            question=question,
            answer=normal_answer,
            context="",
        )
    )

    normal_abstained = (
        ABSTENTION_MESSAGE.lower()
        in normal_answer.lower()
        if normal_answer
        else False
    )

    print(
        f"Normal generation: "
        f"{normal_generation_ms:.2f} ms"
    )

    if normal_error:
        print(
            f"Normal LLM error: "
            f"{normal_error}"
        )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag_output = run_rag(
        answer_service=answer_service,
        question=question,
    )

    rag_result = rag_output["result"]
    rag_error = rag_output["error"]

    rag_answer = ""
    rag_sources: list[
        dict[str, Any]
    ] = []

    rag_context = ""

    rag_retrieval_ms = 0.0
    rag_context_ms = 0.0
    rag_generation_ms = 0.0
    rag_total_ms = 0.0

    rag_groundedness = 0.0
    rag_context_relevance = 0.0

    expected_source_found = False
    rag_abstained = False

    if rag_error:

        print(
            f"RAG error: "
            f"{rag_error}"
        )

    else:

        # ----------------------------------------------------
        # Production answer
        # ----------------------------------------------------

        rag_answer = str(
            rag_result.get(
                "answer",
                "",
            )
        )

        rag_sources = list(
            rag_result.get(
                "sources",
                [],
            )
        )

        # ----------------------------------------------------
        # EXACT PRODUCTION TIMING KEYS
        # ----------------------------------------------------

        timings_ms = dict(
            rag_result.get(
                "timings_ms",
                {},
            )
        )

        rag_retrieval_ms = float(
            timings_ms.get(
                "retrieval",
                0.0,
            )
        )

        rag_context_ms = float(
            timings_ms.get(
                "context_build",
                0.0,
            )
        )

        rag_generation_ms = float(
            timings_ms.get(
                "generation",
                0.0,
            )
        )

        rag_total_ms = float(
            timings_ms.get(
                "total",
                0.0,
            )
        )

        # ----------------------------------------------------
        # ACTUAL RETRIEVED CONTEXT
        # ----------------------------------------------------
        #
        # AnswerService.ask() already performed retrieval.
        #
        # We use the same production retrieval pipeline again
        # only for evaluation metrics.
        #
        # This is deliberately kept separate from the answer
        # generation timing.
        # ----------------------------------------------------

        try:

            (
                actual_context,
                actual_sources,
            ) = get_actual_rag_context(
                answer_service=answer_service,
                question=question,
            )

            if actual_context:
                rag_context = (
                    actual_context
                )

            if actual_sources:
                rag_sources = (
                    actual_sources
                )

        except Exception as exc:

            print(
                "Warning: unable to rebuild "
                f"evaluation context: {exc}"
            )

        # ----------------------------------------------------
        # GROUNDEDNESS
        # ----------------------------------------------------

        rag_groundedness = (
            evidence_support_proxy(
                question=question,
                answer=rag_answer,
                context=rag_context,
            )
        )

        # ----------------------------------------------------
        # CONTEXT RELEVANCE
        # ----------------------------------------------------

        expected_source = (
            case.expected_sources[0]
            if case.expected_sources
            else None
        )

        try:

            relevance_result = (
                calculate_context_relevance(
                    question=question,
                    context=rag_context,
                    sources=rag_sources,
                    expected_source=expected_source,
                )
            )

            rag_context_relevance = float(
                relevance_result.get(
                    "context_relevance_score",
                    0.0,
                )
            )

        except Exception as exc:

            print(
                "Warning: context relevance "
                f"calculation failed: {exc}"
            )

            rag_context_relevance = 0.0

        # ----------------------------------------------------
        # EXPECTED SOURCE
        # ----------------------------------------------------

        expected_source_found = any(
            source_matches(
                expected_source,
                rag_sources,
            )
            for expected_source
            in case.expected_sources
        )

        # ----------------------------------------------------
        # ABSTENTION
        # ----------------------------------------------------

        rag_abstained = (
            ABSTENTION_MESSAGE.lower()
            in rag_answer.lower()
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"RAG retrieval: "
            f"{rag_retrieval_ms:.2f} ms"
        )

        print(
            f"RAG context build: "
            f"{rag_context_ms:.2f} ms"
        )

        print(
            f"RAG generation: "
            f"{rag_generation_ms:.2f} ms"
        )

        print(
            f"RAG total: "
            f"{rag_total_ms:.2f} ms"
        )

    # --------------------------------------------------------
    # RESULT DISPLAY
    # --------------------------------------------------------

    print(
        f"Normal evidence proxy: "
        f"{normal_support:.4f}"
    )

    print(
        f"RAG groundedness proxy: "
        f"{rag_groundedness:.4f}"
    )

    print(
        f"RAG context relevance: "
        f"{rag_context_relevance:.4f}"
    )

    print(
        f"Expected source found: "
        f"{expected_source_found}"
    )

    # --------------------------------------------------------
    # RESULT ROW
    # --------------------------------------------------------

    return {
        "question": question,
        "difficulty": case.difficulty,
        "category": case.category,

        "expected_sources": "|".join(
            case.expected_sources
        ),

        "normal_answer": normal_answer,
        "rag_answer": rag_answer,

        "normal_generation_ms": (
            normal_generation_ms
        ),

        "rag_retrieval_ms": (
            rag_retrieval_ms
        ),

        "rag_context_ms": (
            rag_context_ms
        ),

        "rag_generation_ms": (
            rag_generation_ms
        ),

        "rag_total_ms": (
            rag_total_ms
        ),

        "normal_evidence_support_proxy": (
            normal_support
        ),

        "rag_groundedness_proxy": (
            rag_groundedness
        ),

        "rag_context_relevance": (
            rag_context_relevance
        ),

        "normal_citation_available": False,

        "rag_citation_available": (
            citation_available(
                rag_sources
            )
        ),

        "expected_source_found": (
            expected_source_found
        ),

        "normal_abstained": (
            normal_abstained
        ),

        "rag_abstained": (
            rag_abstained
        ),

        "rag_source_count": (
            len(rag_sources)
        ),

        "normal_error": (
            normal_error or ""
        ),

        "rag_error": (
            rag_error or ""
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, float]:

    if not rows:
        return {}

    frame = pd.DataFrame(
        rows
    )

    return {
        "normal_generation_ms": float(
            frame[
                "normal_generation_ms"
            ].mean()
        ),

        "rag_retrieval_ms": float(
            frame[
                "rag_retrieval_ms"
            ].mean()
        ),

        "rag_context_ms": float(
            frame[
                "rag_context_ms"
            ].mean()
        ),

        "rag_generation_ms": float(
            frame[
                "rag_generation_ms"
            ].mean()
        ),

        "rag_total_ms": float(
            frame[
                "rag_total_ms"
            ].mean()
        ),

        "normal_evidence_support_proxy": (
            float(
                frame[
                    "normal_evidence_support_proxy"
                ].mean()
            )
        ),

        "rag_groundedness_proxy": (
            float(
                frame[
                    "rag_groundedness_proxy"
                ].mean()
            )
        ),

        "rag_context_relevance": (
            float(
                frame[
                    "rag_context_relevance"
                ].mean()
            )
        ),

        "normal_citation_rate": (
            float(
                frame[
                    "normal_citation_available"
                ].mean()
            )
        ),

        "rag_citation_rate": (
            float(
                frame[
                    "rag_citation_available"
                ].mean()
            )
        ),

        "expected_source_hit_rate": (
            float(
                frame[
                    "expected_source_found"
                ].mean()
            )
        ),

        "normal_abstention_rate": (
            float(
                frame[
                    "normal_abstained"
                ].mean()
            )
        ),

        "rag_abstention_rate": (
            float(
                frame[
                    "rag_abstained"
                ].mean()
            )
        ),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    rows: list[dict[str, Any]],
) -> None:

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        rows
    )

    # Explicit column order prevents accidental CSV
    # schema/order problems.

    column_order = [
        "question",
        "difficulty",
        "category",
        "expected_sources",
        "normal_answer",
        "rag_answer",
        "normal_generation_ms",
        "rag_retrieval_ms",
        "rag_context_ms",
        "rag_generation_ms",
        "rag_total_ms",
        "normal_evidence_support_proxy",
        "rag_groundedness_proxy",
        "rag_context_relevance",
        "normal_citation_available",
        "rag_citation_available",
        "expected_source_found",
        "normal_abstained",
        "rag_abstained",
        "rag_source_count",
        "normal_error",
        "rag_error",
    ]

    dataframe = dataframe[
        column_order
    ]

    dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary: dict[str, float],
) -> None:

    print()
    print("=" * 70)
    print(
        "NORMAL LLM VS RAG SUMMARY"
    )
    print("=" * 70)

    print(
        f"Normal LLM generation: "
        f"{summary['normal_generation_ms']:.2f} ms"
    )

    print(
        f"RAG retrieval: "
        f"{summary['rag_retrieval_ms']:.2f} ms"
    )

    print(
        f"RAG context build: "
        f"{summary['rag_context_ms']:.2f} ms"
    )

    print(
        f"RAG generation: "
        f"{summary['rag_generation_ms']:.2f} ms"
    )

    print(
        f"RAG total: "
        f"{summary['rag_total_ms']:.2f} ms"
    )

    print(
        f"Normal evidence-support proxy: "
        f"{summary['normal_evidence_support_proxy']:.4f}"
    )

    print(
        f"RAG groundedness proxy: "
        f"{summary['rag_groundedness_proxy']:.4f}"
    )

    print(
        f"RAG context relevance: "
        f"{summary['rag_context_relevance']:.4f}"
    )

    evidence_difference = (
        summary["rag_groundedness_proxy"]
        - summary[
            "normal_evidence_support_proxy"
        ]
    )

    print(
        f"Evidence-support difference: "
        f"{evidence_difference * 100:.2f} "
        f"percentage points"
    )

    print(
        f"Normal citation availability: "
        f"{summary['normal_citation_rate'] * 100:.2f}%"
    )

    print(
        f"RAG citation availability: "
        f"{summary['rag_citation_rate'] * 100:.2f}%"
    )

    print(
        f"RAG expected-source hit rate: "
        f"{summary['expected_source_hit_rate'] * 100:.2f}%"
    )

    print(
        f"Normal abstention rate: "
        f"{summary['normal_abstention_rate'] * 100:.2f}%"
    )

    print(
        f"RAG abstention rate: "
        f"{summary['rag_abstention_rate'] * 100:.2f}%"
    )

    print()
    print(
        "Detailed results saved to:"
    )
    print(
        RESULTS_PATH
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Compare normal Ollama Cloud "
            "LLM responses against DocMind RAG."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Number of evaluation questions "
            "to run. Default: all 30."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Number of RAG chunks retrieved."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    cases = (
        get_answer_evaluation_dataset()
    )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        cases = cases[
            :args.limit
        ]

    if not cases:
        raise RuntimeError(
            "No answer-evaluation cases found."
        )

    print("=" * 70)
    print(
        "DOCMIND - NORMAL LLM VS RAG"
    )
    print("=" * 70)

    print(
        f"Evaluation questions: "
        f"{len(cases)}"
    )

    print(
        f"RAG top-K: "
        f"{args.top_k}"
    )

    print(
        "Ollama model: "
        f"{os.getenv(
            'OLLAMA_MODEL',
            'gpt-oss:120b-cloud'
        )}"
    )

    # --------------------------------------------------------
    # SHARED CLIENTS
    # --------------------------------------------------------

    llm = OllamaCloud()

    answer_service = AnswerService(
        top_k=args.top_k,
        use_hybrid=True,
        candidate_k=10,
        dense_weight=0.6,
        bm25_weight=0.4,
        use_reranker=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(cases)}]"
        )

        row = evaluate_case(
            case=case,
            llm=llm,
            answer_service=answer_service,
        )

        rows.append(row)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        rows
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize(
        rows
    )

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()