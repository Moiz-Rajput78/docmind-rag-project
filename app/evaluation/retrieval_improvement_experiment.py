# ============================================================
# DOCMIND — RETRIEVAL IMPROVEMENT EXPERIMENT
# ============================================================

from pathlib import Path
import sys
import csv

# Add project root to Python import path
PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.retrieval.retriever import Retriever
from app.evaluation.dataset import get_evaluation_questions


OUTPUT_DIR = PROJECT_DIR / "docs"
OUTPUT_FILE = OUTPUT_DIR / "retrieval_distance_experiment.csv"


def normalize_source(source: str | None) -> str:
    """Normalize a source filename for comparison."""

    if not source:
        return ""

    return Path(source).name.lower()


def evaluate_threshold(
    retriever: Retriever,
    evaluation_questions: list[dict],
    threshold: float | None,
    top_k: int = 10,
) -> dict:
    """
    Evaluate retrieval while optionally filtering results
    using a Chroma distance threshold.
    """

    total = len(evaluation_questions)

    hit_count = 0
    relevant_retrieved = 0
    total_relevant = 0
    retrieved_result_count = 0

    for item in evaluation_questions:

        question = item["question"]

        expected_sources = {
            normalize_source(source)
            for source in item.get("expected_sources", [])
        }

        results = retriever.retrieve(
            question,
            top_k=top_k,
        )

        filtered_results = []

        for result in results:

            distance = result.get("distance")

            if threshold is None:
                filtered_results.append(result)

            elif distance is not None and distance <= threshold:
                filtered_results.append(result)

        retrieved_result_count += len(filtered_results)

        retrieved_sources = {
            normalize_source(result.get("source"))
            for result in filtered_results
        }

        retrieved_sources.discard("")

        if expected_sources & retrieved_sources:
            hit_count += 1

        relevant = len(expected_sources & retrieved_sources)

        relevant_retrieved += relevant
        total_relevant += len(expected_sources)

    hit_rate = hit_count / total if total else 0

    precision = (
        relevant_retrieved / retrieved_result_count
        if retrieved_result_count
        else 0
    )

    recall = (
        relevant_retrieved / total_relevant
        if total_relevant
        else 0
    )

    average_results = (
        retrieved_result_count / total
        if total
        else 0
    )

    return {
        "threshold": (
            "none" if threshold is None else threshold
        ),
        "top_k": top_k,
        "hit_rate": round(hit_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "average_results": round(average_results, 2),
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_questions = get_evaluation_questions()

    print("=" * 70)
    print("DOCMIND — RETRIEVAL DISTANCE EXPERIMENT")
    print("=" * 70)

    retriever = Retriever()

    print(f"\nChroma chunks: {retriever.count()}")
    print(
        f"Evaluation questions: "
        f"{len(evaluation_questions)}"
    )

    # --------------------------------------------------------
    # Distance thresholds
    # --------------------------------------------------------

    thresholds = [
        None,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
    ]

    rows = []

    print("\nRunning experiments...\n")

    for threshold in thresholds:

        result = evaluate_threshold(
            retriever=retriever,
            evaluation_questions=evaluation_questions,
            threshold=threshold,
            top_k=10,
        )

        rows.append(result)

        print(
            f"Threshold={str(result['threshold']):>5} | "
            f"Hit={result['hit_rate']:.3f} | "
            f"Precision={result['precision']:.3f} | "
            f"Recall={result['recall']:.3f} | "
            f"Avg Results={result['average_results']:.2f}"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "top_k",
                "hit_rate",
                "precision",
                "recall",
                "average_results",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()