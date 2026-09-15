"""
Hybrid Retrieval Experiment
============================

Compares three retrieval strategies on the existing 30-question
DocMind evaluation dataset:

1. Dense retrieval
2. BM25 keyword retrieval
3. Hybrid retrieval (dense + BM25)

This is an EXPERIMENT ONLY.
It does not modify the production Retriever or AnswerService.

Results are saved to:
    docs/hybrid_retrieval_experiment.csv
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.evaluation.dataset import get_evaluation_questions
from app.retrieval.retriever import Retriever


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "docs"
OUTPUT_FILE = OUTPUT_DIR / "hybrid_retrieval_experiment.csv"

CHROMA_DIR = PROJECT_DIR / "data" / "chroma"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Retrieve more candidates from each method before fusion.
CANDIDATE_K = 10

# Final number of chunks evaluated.
FINAL_K = 3

# Weight given to dense retrieval in the hybrid score.
# BM25 receives (1 - DENSE_WEIGHT).
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Simple lowercase word tokenizer for BM25.

    Keeps words and common alphanumeric terms such as:
        NovaDesk
        Node.js
        5-working-days
    """
    if not text:
        return []

    return re.findall(r"[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*", text.lower())


# ============================================================
# BM25 INDEX
# ============================================================

class BM25Index:
    """BM25 index built from the same chunks stored in ChromaDB."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

        # Fetch all indexed chunks from ChromaDB.
        collection_data = retriever.store.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents: list[str] = collection_data.get("documents") or []
        self.metadatas: list[dict[str, Any]] = (
            collection_data.get("metadatas") or []
        )
        self.ids: list[str] = collection_data.get("ids") or []

        tokenized_documents = [
            tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        print(f"BM25 indexed chunks: {len(self.documents)}")

    def search(
        self,
        query: str,
        top_k: int = CANDIDATE_K,
    ) -> list[dict[str, Any]]:
        """Return the top BM25 matches."""

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for rank, index in enumerate(
            ranked_indexes[:top_k],
            start=1,
        ):
            metadata = (
                self.metadatas[index]
                if index < len(self.metadatas)
                else {}
            )

            document = self.documents[index]

            results.append(
                {
                    "rank": rank,
                    "id": (
                        self.ids[index]
                        if index < len(self.ids)
                        else None
                    ),
                    "text": document,
                    "bm25_score": float(scores[index]),
                    "metadata": metadata,
                    "source": metadata.get("source"),
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "category": metadata.get("category"),
                    "chunk_number": metadata.get("chunk_number"),
                }
            )

        return results


# ============================================================
# SOURCE NORMALIZATION
# ============================================================

def source_name(result: dict[str, Any]) -> str:
    """Return the filename/source identifier used for evaluation."""

    filename = result.get("filename")

    if filename:
        return Path(str(filename)).name

    source = result.get("source")

    if source:
        return Path(str(source)).name

    return ""


# ============================================================
# EXPECTED SOURCE MATCHING
# ============================================================

def normalize_filename(filename: str) -> str:
    return Path(str(filename)).name.lower().strip()


def result_matches_expected(
    result: dict[str, Any],
    expected_sources: list[str],
) -> bool:
    actual = normalize_filename(source_name(result))

    expected = {
        normalize_filename(source)
        for source in expected_sources
    }

    return actual in expected


def source_coverage(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    """
    Fraction of expected source files represented in the results.
    """

    if not expected_sources:
        return 0.0

    expected = {
        normalize_filename(source)
        for source in expected_sources
    }

    retrieved = {
        normalize_filename(source_name(result))
        for result in results
    }

    matched = expected.intersection(retrieved)

    return len(matched) / len(expected)


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_search(
    retriever: Retriever,
    query: str,
    top_k: int = CANDIDATE_K,
) -> list[dict[str, Any]]:
    """Run the existing dense retriever."""

    return retriever.retrieve(
        query=query,
        top_k=top_k,
    )


# ============================================================
# DENSE SCORE NORMALIZATION
# ============================================================

def min_max_normalize(values: list[float]) -> list[float]:
    """
    Normalize values to [0, 1].

    Used so dense and BM25 scores can be combined.
    """

    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [1.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_search(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    final_k: int = FINAL_K,
) -> list[dict[str, Any]]:
    """
    Combine dense and BM25 results using weighted reciprocal
    rank fusion.

    We use Reciprocal Rank Fusion (RRF) rather than directly
    combining raw Chroma distances and BM25 scores because the
    two retrieval systems produce scores on different scales.

    Each result receives:

        RRF contribution = 1 / (60 + rank)

    Then the dense and BM25 contributions are weighted.
    """

    fused: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------
    # Dense contribution
    # --------------------------------------------------------

    for result in dense_results:
        result_id = result.get("id")

        if not result_id:
            continue

        if result_id not in fused:
            fused[result_id] = {
                "id": result_id,
                "text": result.get("text"),
                "metadata": result.get("metadata", {}),
                "source": result.get("source"),
                "filename": result.get("filename"),
                "page": result.get("page"),
                "category": result.get("category"),
                "chunk_number": result.get("chunk_number"),
                "dense_rank": None,
                "bm25_rank": None,
                "dense_rrf": 0.0,
                "bm25_rrf": 0.0,
            }

        rank = result.get("rank", 999)

        fused[result_id]["dense_rank"] = rank
        fused[result_id]["dense_rrf"] = (
            DENSE_WEIGHT / (60 + rank)
        )

    # --------------------------------------------------------
    # BM25 contribution
    # --------------------------------------------------------

    for result in bm25_results:
        result_id = result.get("id")

        if not result_id:
            continue

        if result_id not in fused:
            fused[result_id] = {
                "id": result_id,
                "text": result.get("text"),
                "metadata": result.get("metadata", {}),
                "source": result.get("source"),
                "filename": result.get("filename"),
                "page": result.get("page"),
                "category": result.get("category"),
                "chunk_number": result.get("chunk_number"),
                "dense_rank": None,
                "bm25_rank": None,
                "dense_rrf": 0.0,
                "bm25_rrf": 0.0,
            }

        rank = result.get("rank", 999)

        fused[result_id]["bm25_rank"] = rank
        fused[result_id]["bm25_rrf"] = (
            BM25_WEIGHT / (60 + rank)
        )

    # --------------------------------------------------------
    # Calculate final hybrid score
    # --------------------------------------------------------

    for result in fused.values():
        result["hybrid_score"] = (
            result["dense_rrf"]
            + result["bm25_rrf"]
        )

    ranked = sorted(
        fused.values(),
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )

    # Add final ranks.
    for rank, result in enumerate(
        ranked[:final_k],
        start=1,
    ):
        result["rank"] = rank

    return ranked[:final_k]


# ============================================================
# METRIC CALCULATIONS
# ============================================================

def hit_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> bool:
    return any(
        result_matches_expected(result, expected_sources)
        for result in results
    )


def reciprocal_rank(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    for rank, result in enumerate(results, start=1):
        if result_matches_expected(result, expected_sources):
            return 1.0 / rank

    return 0.0


def precision_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    if not results:
        return 0.0

    relevant = sum(
        result_matches_expected(result, expected_sources)
        for result in results
    )

    return relevant / len(results)


def recall_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    if not expected_sources:
        return 0.0

    expected = {
        normalize_filename(source)
        for source in expected_sources
    }

    retrieved = {
        normalize_filename(source_name(result))
        for result in results
    }

    relevant = len(expected.intersection(retrieved))

    return relevant / len(expected)


def all_expected_retrieved(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> bool:
    if not expected_sources:
        return False

    expected = {
        normalize_filename(source)
        for source in expected_sources
    }

    retrieved = {
        normalize_filename(source_name(result))
        for result in results
    }

    return expected.issubset(retrieved)


# ============================================================
# EVALUATE ONE METHOD
# ============================================================

def evaluate_method(
    name: str,
    question_results: list[dict[str, Any]],
) -> dict[str, float]:
    total = len(question_results)

    if total == 0:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "precision_at_3": 0.0,
            "recall_at_3": 0.0,
            "mrr": 0.0,
            "coverage_at_3": 0.0,
            "all_expected_at_3": 0.0,
        }

    hit1 = 0
    hit3 = 0
    precision3 = 0.0
    recall3 = 0.0
    mrr = 0.0
    coverage3 = 0.0
    all_expected3 = 0

    for item in question_results:
        results = item["results"]
        expected = item["expected_sources"]

        top1 = results[:1]
        top3 = results[:3]

        if hit_at_k(top1, expected):
            hit1 += 1

        if hit_at_k(top3, expected):
            hit3 += 1

        precision3 += precision_at_k(top3, expected)
        recall3 += recall_at_k(top3, expected)
        mrr += reciprocal_rank(top3, expected)
        coverage3 += source_coverage(top3, expected)

        if all_expected_retrieved(top3, expected):
            all_expected3 += 1

    metrics = {
        "hit_at_1": hit1 / total,
        "hit_at_3": hit3 / total,
        "precision_at_3": precision3 / total,
        "recall_at_3": recall3 / total,
        "mrr": mrr / total,
        "coverage_at_3": coverage3 / total,
        "all_expected_at_3": all_expected3 / total,
    }

    print(f"\n{name}")
    print("-" * 50)
    print(f"Hit@1             : {metrics['hit_at_1']:.3f}")
    print(f"Hit@3             : {metrics['hit_at_3']:.3f}")
    print(f"Precision@3       : {metrics['precision_at_3']:.3f}")
    print(f"Recall@3          : {metrics['recall_at_3']:.3f}")
    print(f"MRR               : {metrics['mrr']:.3f}")
    print(f"Source Coverage@3 : {metrics['coverage_at_3']:.3f}")
    print(f"All Expected@3    : {metrics['all_expected_at_3']:.3f}")

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    rows: list[dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question",
        "difficulty",
        "category",
        "expected_sources",

        "dense_sources",
        "bm25_sources",
        "hybrid_sources",

        "dense_hit_at_3",
        "bm25_hit_at_3",
        "hybrid_hit_at_3",

        "dense_precision_at_3",
        "bm25_precision_at_3",
        "hybrid_precision_at_3",

        "dense_recall_at_3",
        "bm25_recall_at_3",
        "hybrid_recall_at_3",

        "dense_mrr",
        "bm25_mrr",
        "hybrid_mrr",

        "dense_coverage_at_3",
        "bm25_coverage_at_3",
        "hybrid_coverage_at_3",
    ]

    with OUTPUT_FILE.open(
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

    print(f"\nDetailed results saved to:")
    print(OUTPUT_FILE)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main() -> None:
    print("=" * 70)
    print("HYBRID RETRIEVAL EXPERIMENT")
    print("=" * 70)

    print(f"Candidate K       : {CANDIDATE_K}")
    print(f"Final K           : {FINAL_K}")
    print(f"Dense weight      : {DENSE_WEIGHT}")
    print(f"BM25 weight       : {BM25_WEIGHT}")
    print(f"Chroma directory  : {CHROMA_DIR}")

    # --------------------------------------------------------
    # Initialize existing dense retriever.
    # --------------------------------------------------------

    retriever = Retriever(
        chroma_dir=CHROMA_DIR,
        collection_name="docmind_documents",
    )

    print(f"Dense indexed chunks: {retriever.count()}")

    # --------------------------------------------------------
    # Build BM25 index from the same Chroma chunks.
    # --------------------------------------------------------

    bm25_index = BM25Index(retriever)

    # --------------------------------------------------------
    # Load existing evaluation dataset.
    # --------------------------------------------------------

    questions = get_evaluation_questions()

    print(f"Evaluation questions: {len(questions)}")

    dense_evaluation = []
    bm25_evaluation = []
    hybrid_evaluation = []

    detailed_rows = []

    start_time = time.time()

    # --------------------------------------------------------
    # Evaluate every question.
    # --------------------------------------------------------

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        expected_sources = item["expected_sources"]

        print(
            f"\nQuestion {question_id}: {question}"
        )

        # Dense
        dense_results = dense_search(
            retriever,
            question,
            CANDIDATE_K,
        )

        # BM25
        bm25_results = bm25_index.search(
            question,
            CANDIDATE_K,
        )

        # Hybrid
        hybrid_results = hybrid_search(
            dense_results,
            bm25_results,
            FINAL_K,
        )

        # Final K for dense/BM25 comparison.
        dense_top3 = dense_results[:FINAL_K]
        bm25_top3 = bm25_results[:FINAL_K]

        dense_evaluation.append(
            {
                "id": question_id,
                "results": dense_top3,
                "expected_sources": expected_sources,
            }
        )

        bm25_evaluation.append(
            {
                "id": question_id,
                "results": bm25_top3,
                "expected_sources": expected_sources,
            }
        )

        hybrid_evaluation.append(
            {
                "id": question_id,
                "results": hybrid_results,
                "expected_sources": expected_sources,
            }
        )

        # ----------------------------------------------------
        # Detailed row
        # ----------------------------------------------------

        dense_sources = [
            source_name(result)
            for result in dense_top3
        ]

        bm25_sources = [
            source_name(result)
            for result in bm25_top3
        ]

        hybrid_sources = [
            source_name(result)
            for result in hybrid_results
        ]

        detailed_rows.append(
            {
                "id": question_id,
                "question": question,
                "difficulty": item["difficulty"],
                "category": item["category"],
                "expected_sources": "|".join(
                    expected_sources
                ),

                "dense_sources": "|".join(
                    dense_sources
                ),
                "bm25_sources": "|".join(
                    bm25_sources
                ),
                "hybrid_sources": "|".join(
                    hybrid_sources
                ),

                "dense_hit_at_3": int(
                    hit_at_k(
                        dense_top3,
                        expected_sources,
                    )
                ),
                "bm25_hit_at_3": int(
                    hit_at_k(
                        bm25_top3,
                        expected_sources,
                    )
                ),
                "hybrid_hit_at_3": int(
                    hit_at_k(
                        hybrid_results,
                        expected_sources,
                    )
                ),

                "dense_precision_at_3": round(
                    precision_at_k(
                        dense_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "bm25_precision_at_3": round(
                    precision_at_k(
                        bm25_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "hybrid_precision_at_3": round(
                    precision_at_k(
                        hybrid_results,
                        expected_sources,
                    ),
                    4,
                ),

                "dense_recall_at_3": round(
                    recall_at_k(
                        dense_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "bm25_recall_at_3": round(
                    recall_at_k(
                        bm25_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "hybrid_recall_at_3": round(
                    recall_at_k(
                        hybrid_results,
                        expected_sources,
                    ),
                    4,
                ),

                "dense_mrr": round(
                    reciprocal_rank(
                        dense_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "bm25_mrr": round(
                    reciprocal_rank(
                        bm25_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "hybrid_mrr": round(
                    reciprocal_rank(
                        hybrid_results,
                        expected_sources,
                    ),
                    4,
                ),

                "dense_coverage_at_3": round(
                    source_coverage(
                        dense_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "bm25_coverage_at_3": round(
                    source_coverage(
                        bm25_top3,
                        expected_sources,
                    ),
                    4,
                ),
                "hybrid_coverage_at_3": round(
                    source_coverage(
                        hybrid_results,
                        expected_sources,
                    ),
                    4,
                ),
            }
        )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Calculate aggregate metrics.
    # --------------------------------------------------------

    dense_metrics = evaluate_method(
        "DENSE RETRIEVAL",
        dense_evaluation,
    )

    bm25_metrics = evaluate_method(
        "BM25 RETRIEVAL",
        bm25_evaluation,
    )

    hybrid_metrics = evaluate_method(
        "HYBRID RETRIEVAL",
        hybrid_evaluation,
    )

    # --------------------------------------------------------
    # Comparison table.
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RETRIEVAL COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<22}"
        f"{'Dense':>12}"
        f"{'BM25':>12}"
        f"{'Hybrid':>12}"
    )

    print("-" * 70)

    metric_labels = [
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("precision_at_3", "Precision@3"),
        ("recall_at_3", "Recall@3"),
        ("mrr", "MRR"),
        ("coverage_at_3", "Coverage@3"),
        ("all_expected_at_3", "All Expected@3"),
    ]

    for key, label in metric_labels:
        print(
            f"{label:<22}"
            f"{dense_metrics[key]:>12.3f}"
            f"{bm25_metrics[key]:>12.3f}"
            f"{hybrid_metrics[key]:>12.3f}"
        )

    print("-" * 70)
    print(f"Evaluation time: {elapsed:.2f} seconds")

    # --------------------------------------------------------
    # Save detailed CSV.
    # --------------------------------------------------------

    save_results(detailed_rows)

    print("\nExperiment complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()