# ============================================================
# DOCMIND — CROSS-ENCODER RERANKER EXPERIMENT
# ============================================================

from pathlib import Path
import sys
import time
import csv

# Add project root to Python import path
PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sentence_transformers import CrossEncoder

from app.retrieval.retriever import Retriever
from app.evaluation.dataset import get_evaluation_questions


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_K = 10
FINAL_K = 3

OUTPUT_DIR = PROJECT_DIR / "docs"
OUTPUT_FILE = OUTPUT_DIR / "reranker_experiment.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_source(source: str | None) -> str:
    """Normalize source filename for comparison."""

    if not source:
        return ""

    return Path(source).name.lower()


def rerank_results(
    reranker: CrossEncoder,
    question: str,
    results: list[dict],
    final_k: int = 3,
) -> list[dict]:
    """
    Score question/chunk pairs using the cross-encoder
    and return the highest-scoring results.
    """

    if not results:
        return []

    pairs = [
        [question, result["text"]]
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for result, score in zip(results, scores):

        item = dict(result)

        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    selected = reranked[:final_k]

    for index, result in enumerate(selected):
        result["rank"] = index + 1

    return selected


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_reranker(
    retriever: Retriever,
    reranker: CrossEncoder,
    evaluation_questions: list[dict],
) -> dict:

    total_questions = len(evaluation_questions)

    hit_count = 0
    relevant_retrieved = 0
    total_relevant = 0

    total_retrieved = 0

    for item in evaluation_questions:

        question = item["question"]

        expected_sources = {
            normalize_source(source)
            for source in item.get("expected_sources", [])
        }

        # ----------------------------------------------------
        # Step 1: Retrieve broad candidate set
        # ----------------------------------------------------

        candidates = retriever.retrieve(
            question,
            top_k=INITIAL_K,
        )

        # ----------------------------------------------------
        # Step 2: Rerank candidates
        # ----------------------------------------------------

        final_results = rerank_results(
            reranker=reranker,
            question=question,
            results=candidates,
            final_k=FINAL_K,
        )

        total_retrieved += len(final_results)

        retrieved_sources = {
            normalize_source(result.get("source"))
            for result in final_results
        }

        retrieved_sources.discard("")

        # ----------------------------------------------------
        # Hit
        # ----------------------------------------------------

        if expected_sources & retrieved_sources:
            hit_count += 1

        # ----------------------------------------------------
        # Relevant source overlap
        # ----------------------------------------------------

        relevant = len(
            expected_sources & retrieved_sources
        )

        relevant_retrieved += relevant
        total_relevant += len(expected_sources)

    hit_rate = (
        hit_count / total_questions
        if total_questions
        else 0
    )

    precision = (
        relevant_retrieved / total_retrieved
        if total_retrieved
        else 0
    )

    recall = (
        relevant_retrieved / total_relevant
        if total_relevant
        else 0
    )

    return {
        "method": "retrieve_10_rerank_3",
        "initial_k": INITIAL_K,
        "final_k": FINAL_K,
        "hit_rate": round(hit_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DOCMIND — CROSS-ENCODER RERANKER EXPERIMENT")
    print("=" * 70)

    print(f"\nReranker model:")
    print(RERANKER_MODEL)

    print("\nLoading reranker...")

    start_time = time.time()

    reranker = CrossEncoder(
        RERANKER_MODEL,
    )

    load_time = time.time() - start_time

    print(
        f"Reranker loaded in {load_time:.2f} seconds."
    )

    print("\nLoading retriever...")

    retriever = Retriever()

    evaluation_questions = get_evaluation_questions()

    print(f"Chroma chunks: {retriever.count()}")
    print(
        f"Evaluation questions: "
        f"{len(evaluation_questions)}"
    )

    print(
        f"\nRetrieval: Top-{INITIAL_K}"
        f" → Rerank → Top-{FINAL_K}"
    )

    print("\nRunning experiment...\n")

    start_time = time.time()

    result = evaluate_reranker(
        retriever=retriever,
        reranker=reranker,
        evaluation_questions=evaluation_questions,
    )

    elapsed = time.time() - start_time

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Hit Rate : {result['hit_rate']:.3f}"
    )

    print(
        f"Precision: {result['precision']:.3f}"
    )

    print(
        f"Recall   : {result['recall']:.3f}"
    )

    print(
        f"\nEvaluation time: {elapsed:.2f} seconds"
    )

    # --------------------------------------------------------
    # Save result
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
                "method",
                "initial_k",
                "final_k",
                "hit_rate",
                "precision",
                "recall",
            ],
        )

        writer.writeheader()
        writer.writerow(result)

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()