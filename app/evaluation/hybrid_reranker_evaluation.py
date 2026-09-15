"""
DocMind RAG - Hybrid + Reranker Evaluation

Compares:

    Baseline Dense Retrieval
        vs
    Hybrid Dense + BM25 + Cross-Encoder

using the project's existing 30-question evaluation dataset.

This is an evaluation experiment only.
It does not modify the production AnswerService.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.evaluation.dataset import get_evaluation_questions
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.retriever import Retriever


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = (
    PROJECT_DIR
    / "docs"
    / "hybrid_reranker_evaluation.csv"
)

TOP_K = 3
CANDIDATE_K = 10


def normalize_source(source: str | None) -> str:
    """
    Normalize a source filename/path for comparison.
    """

    if not source:
        return ""

    return Path(str(source)).name.lower().strip()


def expected_sources(item: dict[str, Any]) -> set[str]:
    """
    Return normalized expected source filenames.
    """

    return {
        normalize_source(source)
        for source in item.get(
            "expected_sources",
            [],
        )
    }


def retrieved_sources(
    results: list[dict[str, Any]],
) -> list[str]:
    """
    Return normalized source filenames in retrieval order.
    """

    return [
        normalize_source(
            result.get("filename")
            or result.get("source")
        )
        for result in results
    ]


def hit_at_1(
    results: list[dict[str, Any]],
    expected: set[str],
) -> float:
    """
    Hit@1 = 1 when the first result is relevant.
    """

    if not results:
        return 0.0

    return float(
        normalize_source(
            results[0].get("filename")
            or results[0].get("source")
        )
        in expected
    )


def hit_at_k(
    results: list[dict[str, Any]],
    expected: set[str],
    k: int,
) -> float:
    """
    Hit@K = 1 when any expected source appears in top-K.
    """

    sources = set(
        retrieved_sources(results[:k])
    )

    return float(
        bool(sources & expected)
    )


def precision_at_k(
    results: list[dict[str, Any]],
    expected: set[str],
    k: int,
) -> float:
    """
    Source-level Precision@K.

    Duplicate chunks from the same source are counted as the
    same source rather than independent relevant documents.
    """

    top_sources = retrieved_sources(
        results[:k]
    )

    if not top_sources:
        return 0.0

    relevant = sum(
        1
        for source in top_sources
        if source in expected
    )

    return relevant / len(top_sources)


def recall_at_k(
    results: list[dict[str, Any]],
    expected: set[str],
    k: int,
) -> float:
    """
    Recall@K over expected source files.
    """

    if not expected:
        return 0.0

    retrieved = set(
        retrieved_sources(results[:k])
    )

    return len(
        retrieved & expected
    ) / len(expected)


def reciprocal_rank(
    results: list[dict[str, Any]],
    expected: set[str],
) -> float:
    """
    Reciprocal rank of the first relevant source.
    """

    for index, result in enumerate(
        results,
        start=1,
    ):

        source = normalize_source(
            result.get("filename")
            or result.get("source")
        )

        if source in expected:
            return 1.0 / index

    return 0.0


def all_expected_at_k(
    results: list[dict[str, Any]],
    expected: set[str],
    k: int,
) -> float:
    """
    1 when every expected source is present in top-K.
    """

    if not expected:
        return 0.0

    retrieved = set(
        retrieved_sources(results[:k])
    )

    return float(
        expected.issubset(retrieved)
    )


def evaluate_retriever(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
]:
    """
    Evaluate one retrieval strategy.
    """

    rows: list[dict[str, Any]] = []

    for item in questions:

        question_id = item["id"]
        question = item["question"]
        difficulty = item["difficulty"]
        category = item["category"]

        expected = expected_sources(item)

        results = retriever.retrieve(
            question,
            top_k=TOP_K,
        )

        sources = retrieved_sources(
            results
        )

        row = {
            "question_id": question_id,
            "question": question,
            "difficulty": difficulty,
            "category": category,
            "retriever": name,
            "expected_sources": "|".join(
                sorted(expected)
            ),
            "retrieved_sources": "|".join(
                sources
            ),
            "hit_at_1": hit_at_1(
                results,
                expected,
            ),
            "hit_at_3": hit_at_k(
                results,
                expected,
                TOP_K,
            ),
            "precision_at_3": precision_at_k(
                results,
                expected,
                TOP_K,
            ),
            "recall_at_3": recall_at_k(
                results,
                expected,
                TOP_K,
            ),
            "mrr": reciprocal_rank(
                results,
                expected,
            ),
            "source_coverage_at_3": recall_at_k(
                results,
                expected,
                TOP_K,
            ),
            "all_expected_at_3": all_expected_at_k(
                results,
                expected,
                TOP_K,
            ),
        }

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    metrics = {
        "Hit@1": dataframe[
            "hit_at_1"
        ].mean(),
        "Hit@3": dataframe[
            "hit_at_3"
        ].mean(),
        "Precision@3": dataframe[
            "precision_at_3"
        ].mean(),
        "Recall@3": dataframe[
            "recall_at_3"
        ].mean(),
        "MRR": dataframe[
            "mrr"
        ].mean(),
        "Source Coverage@3": dataframe[
            "source_coverage_at_3"
        ].mean(),
        "All Expected@3": dataframe[
            "all_expected_at_3"
        ].mean(),
    }

    return metrics, rows


def print_metrics(
    name: str,
    metrics: dict[str, float],
) -> None:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    for metric, value in metrics.items():

        print(
            f"{metric:<22}: "
            f"{value:.3f}"
        )


def main() -> None:

    print("=" * 70)
    print("DOCMIND — HYBRID + RERANKER EVALUATION")
    print("=" * 70)

    print()
    print(f"Candidate K : {CANDIDATE_K}")
    print(f"Final K     : {TOP_K}")

    questions = get_evaluation_questions()

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    print()
    print("Loading baseline dense retriever...")

    dense_retriever = Retriever()

    print(
        "Loading hybrid + reranker retriever..."
    )

    hybrid_retriever = HybridRetriever()

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    dense_metrics, dense_rows = evaluate_retriever(
        name="Dense",
        retriever=dense_retriever,
        questions=questions,
    )

    # --------------------------------------------------------
    # HYBRID + RERANKER
    # --------------------------------------------------------

    hybrid_metrics, hybrid_rows = (
        evaluate_retriever(
            name="Hybrid+Reranker",
            retriever=hybrid_retriever,
            questions=questions,
        )
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_metrics(
        "DENSE RETRIEVAL",
        dense_metrics,
    )

    print_metrics(
        "HYBRID + RERANKER",
        hybrid_metrics,
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for metric in dense_metrics:

        dense_value = dense_metrics[
            metric
        ]

        hybrid_value = hybrid_metrics[
            metric
        ]

        change = (
            hybrid_value
            - dense_value
        )

        print(
            f"{metric:<22}: "
            f"Dense={dense_value:.3f}  "
            f"Hybrid+Reranker={hybrid_value:.3f}  "
            f"Change={change:+.3f}"
        )

    # --------------------------------------------------------
    # QUESTION-LEVEL COMPARISON
    # --------------------------------------------------------

    dense_by_id = {
        row["question_id"]: row
        for row in dense_rows
    }

    hybrid_by_id = {
        row["question_id"]: row
        for row in hybrid_rows
    }

    comparison_rows = []

    for item in questions:

        question_id = item["id"]

        dense_row = dense_by_id[
            question_id
        ]

        hybrid_row = hybrid_by_id[
            question_id
        ]

        comparison_rows.append(
            {
                "question_id": question_id,
                "question": item[
                    "question"
                ],
                "difficulty": item[
                    "difficulty"
                ],
                "category": item[
                    "category"
                ],
                "dense_hit_at_1": dense_row[
                    "hit_at_1"
                ],
                "hybrid_hit_at_1": hybrid_row[
                    "hit_at_1"
                ],
                "dense_mrr": dense_row[
                    "mrr"
                ],
                "hybrid_mrr": hybrid_row[
                    "mrr"
                ],
                "dense_recall_at_3": dense_row[
                    "recall_at_3"
                ],
                "hybrid_recall_at_3": hybrid_row[
                    "recall_at_3"
                ],
                "dense_all_expected_at_3": dense_row[
                    "all_expected_at_3"
                ],
                "hybrid_all_expected_at_3": hybrid_row[
                    "all_expected_at_3"
                ],
                "dense_sources": dense_row[
                    "retrieved_sources"
                ],
                "hybrid_sources": hybrid_row[
                    "retrieved_sources"
                ],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Detailed results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()