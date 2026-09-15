"""
DocMind RAG - Retrieval Evaluation

Evaluates retrieval quality using:
- Hit Rate@K
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

Evaluation is performed for:
- Easy questions
- Medium questions
- Difficult questions
- Overall dataset

The evaluation dataset contains expected source filenames.
A retrieved chunk is considered relevant when its source filename
matches one of the expected source filenames.
"""

from pathlib import Path
import csv
import sys
from collections import defaultdict
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT COMPONENTS
# ============================================================

from app.evaluation.dataset import EVALUATION_QUESTIONS
from app.retrieval.retriever import Retriever


# ============================================================
# CONFIGURATION
# ============================================================

K_VALUES = [1, 3, 5, 10]

RESULTS_FILE = (
    PROJECT_ROOT
    / "docs"
    / "retrieval_evaluation_results.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_source(source: str) -> str:
    """
    Normalize a source filename for comparison.
    """
    return Path(source).name.strip().lower()


def get_expected_sources(question: dict[str, Any]) -> set[str]:
    """
    Return normalized expected source filenames.
    """
    return {
        normalize_source(source)
        for source in question.get("expected_sources", [])
    }


def get_retrieved_sources(results: list[dict[str, Any]]) -> list[str]:
    """
    Extract normalized source filenames from retrieval results.
    """
    sources = []

    for result in results:
        metadata = result.get("metadata", {})

        source = (
            result.get("filename")
            or metadata.get("filename")
            or result.get("source")
            or metadata.get("source")
            or ""
        )

        if source:
            sources.append(normalize_source(source))

    return sources


# ============================================================
# METRICS
# ============================================================

def hit_rate_at_k(
    retrieved_sources: list[str],
    expected_sources: set[str],
    k: int,
) -> float:
    """
    Hit Rate@K

    Returns 1 when at least one expected source is retrieved
    within the first K results, otherwise 0.
    """

    top_k = retrieved_sources[:k]

    return float(
        any(source in expected_sources for source in top_k)
    )


def precision_at_k(
    retrieved_sources: list[str],
    expected_sources: set[str],
    k: int,
) -> float:
    """
    Precision@K

    A retrieved chunk is considered relevant when its source
    filename is one of the expected sources.

    Formula:

        relevant retrieved chunks / K
    """

    top_k = retrieved_sources[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        1
        for source in top_k
        if source in expected_sources
    )

    return relevant / len(top_k)


def recall_at_k(
    retrieved_sources: list[str],
    expected_sources: set[str],
    k: int,
) -> float:
    """
    Recall@K

    Measures how many of the expected source documents were
    retrieved within the first K results.

    Duplicate chunks from the same document count only once.

    Formula:

        unique expected sources retrieved / total expected sources
    """

    if not expected_sources:
        return 0.0

    top_k_unique = set(retrieved_sources[:k])

    found = top_k_unique.intersection(expected_sources)

    return len(found) / len(expected_sources)


def reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: set[str],
) -> float:
    """
    Reciprocal Rank

    Finds the first relevant retrieved result.

    Formula:

        1 / rank

    Returns 0 when no relevant result is found.
    """

    for index, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0 / index

    return 0.0


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    retriever: Retriever,
    question: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate one question for every configured K value.
    """

    question_text = question["question"]
    difficulty = question["difficulty"]
    category = question.get("category", "unknown")

    expected_sources = get_expected_sources(question)

    # Retrieve the maximum required number once.
    results = retriever.retrieve(
        question_text,
        top_k=max(K_VALUES),
    )

    retrieved_sources = get_retrieved_sources(results)

    row: dict[str, Any] = {
        "question_id": question.get("id", ""),
        "question": question_text,
        "difficulty": difficulty,
        "category": category,
        "expected_sources": "; ".join(
            sorted(expected_sources)
        ),
    }

    # --------------------------------------------------------
    # Metrics for every K
    # --------------------------------------------------------

    for k in K_VALUES:

        row[f"hit_rate@{k}"] = hit_rate_at_k(
            retrieved_sources,
            expected_sources,
            k,
        )

        row[f"precision@{k}"] = precision_at_k(
            retrieved_sources,
            expected_sources,
            k,
        )

        row[f"recall@{k}"] = recall_at_k(
            retrieved_sources,
            expected_sources,
            k,
        )

    row["reciprocal_rank"] = reciprocal_rank(
        retrieved_sources,
        expected_sources,
    )

    row["retrieved_sources"] = "; ".join(
        retrieved_sources
    )

    return row


# ============================================================
# AGGREGATE METRICS
# ============================================================

def average_metric(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> float:
    """
    Calculate the average value of a metric.
    """

    if not rows:
        return 0.0

    values = [
        float(row[metric_name])
        for row in rows
    ]

    return sum(values) / len(values)


def calculate_summary(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Calculate aggregate retrieval metrics.
    """

    summary: dict[str, float] = {}

    for k in K_VALUES:

        summary[f"hit_rate@{k}"] = average_metric(
            rows,
            f"hit_rate@{k}",
        )

        summary[f"precision@{k}"] = average_metric(
            rows,
            f"precision@{k}",
        )

        summary[f"recall@{k}"] = average_metric(
            rows,
            f"recall@{k}",
        )

    summary["mrr"] = average_metric(
        rows,
        "reciprocal_rank",
    )

    return summary


# ============================================================
# GROUP RESULTS BY DIFFICULTY
# ============================================================

def group_by_difficulty(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group evaluation results into Easy / Medium / Difficult.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        groups[row["difficulty"]].append(row)

    return dict(groups)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    rows: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """
    Save per-question evaluation results as CSV.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "question_id",
        "question",
        "difficulty",
        "category",
        "expected_sources",
    ]

    for k in K_VALUES:
        fieldnames.extend(
            [
                f"hit_rate@{k}",
                f"precision@{k}",
                f"recall@{k}",
            ]
        )

    fieldnames.extend(
        [
            "reciprocal_rank",
            "retrieved_sources",
        ]
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    title: str,
    rows: list[dict[str, Any]],
) -> None:
    """
    Print a readable metric summary.
    """

    summary = calculate_summary(rows)

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Questions: {len(rows)}"
    )

    print()

    print(
        f"{'K':<8}"
        f"{'Hit Rate':<15}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
    )

    print("-" * 53)

    for k in K_VALUES:

        print(
            f"{k:<8}"
            f"{summary[f'hit_rate@{k}']:<15.3f}"
            f"{summary[f'precision@{k}']:<15.3f}"
            f"{summary[f'recall@{k}']:<15.3f}"
        )

    print()

    print(
        f"MRR: {summary['mrr']:.3f}"
    )


# ============================================================
# PRINT FAILED / WEAK QUESTIONS
# ============================================================

def print_weak_results(
    rows: list[dict[str, Any]],
) -> None:
    """
    Print questions where the expected source was not found
    in Top-3 retrieval results.
    """

    weak = [
        row
        for row in rows
        if row["hit_rate@3"] == 0
    ]

    print()
    print("=" * 72)
    print("QUESTIONS WITH NO EXPECTED SOURCE IN TOP-3")
    print("=" * 72)

    if not weak:
        print("None. Every question retrieved an expected source in Top-3.")
        return

    for row in weak:

        print()
        print(
            f"[{row['question_id']}] "
            f"{row['difficulty']} - "
            f"{row['category']}"
        )

        print(
            f"Question: {row['question']}"
        )

        print(
            f"Expected: {row['expected_sources']}"
        )

        print(
            f"Retrieved: {row['retrieved_sources']}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("DOCMIND RAG - RETRIEVAL EVALUATION")
    print("=" * 72)

    print()
    print(
        f"Evaluation questions: "
        f"{len(EVALUATION_QUESTIONS)}"
    )

    print(
        f"K values: {K_VALUES}"
    )

    print()
    print("Loading retrieval system...")

    retriever = Retriever()

    print(
        f"Vector store chunks: "
        f"{retriever.count()}"
    )

    print()
    print("Running evaluation...")
    print()

    rows: list[dict[str, Any]] = []

    for index, question in enumerate(
    EVALUATION_QUESTIONS,
    start=1,
):

        print(
            f"[{index:02d}/{len(EVALUATION_QUESTIONS)}] "
            f"{question['question']}"
        )

        result = evaluate_question(
            retriever,
            question,
        )

        rows.append(result)

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    save_results(
        rows,
        RESULTS_FILE,
    )

    print()
    print(
        f"Detailed results saved to:"
    )

    print(
        f"  {RESULTS_FILE}"
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print_summary(
        "OVERALL RETRIEVAL PERFORMANCE",
        rows,
    )

    # --------------------------------------------------------
    # Difficulty summaries
    # --------------------------------------------------------

    groups = group_by_difficulty(rows)

    for difficulty in [
        "easy",
        "medium",
        "difficult",
    ]:

        if difficulty in groups:

            print_summary(
                f"{difficulty.upper()} QUESTIONS",
                groups[difficulty],
            )

    # --------------------------------------------------------
    # Weak retrieval cases
    # --------------------------------------------------------

    print_weak_results(rows)

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RETRIEVAL EVALUATION COMPLETE")
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()