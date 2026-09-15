"""
Hybrid Retrieval Weight Experiment
===================================

Tests different Dense/BM25 weights using the existing 30-question
DocMind evaluation dataset.

This experiment does NOT modify the production retrieval pipeline.

It compares:

    Dense 1.0 / BM25 0.0
    Dense 0.7 / BM25 0.3
    Dense 0.6 / BM25 0.4
    Dense 0.5 / BM25 0.5
    Dense 0.4 / BM25 0.6
    Dense 0.3 / BM25 0.7
    Dense 0.0 / BM25 1.0

Results are saved to:

    docs/hybrid_weight_experiment.csv
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
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_DIR / "data" / "chroma"

OUTPUT_DIR = PROJECT_DIR / "docs"
OUTPUT_FILE = OUTPUT_DIR / "hybrid_weight_experiment.csv"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

CANDIDATE_K = 10
FINAL_K = 3

WEIGHTS = [
    (1.0, 0.0),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.0, 1.0),
]


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    """

    if not text:
        return []

    return re.findall(
        r"[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*",
        text.lower(),
    )


# ============================================================
# BM25 INDEX
# ============================================================

class BM25Index:
    """
    BM25 index built from the same ChromaDB chunks used by
    dense retrieval.
    """

    def __init__(self, retriever: Retriever):
        collection_data = retriever.store.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents: list[str] = (
            collection_data.get("documents") or []
        )

        self.metadatas: list[dict[str, Any]] = (
            collection_data.get("metadatas") or []
        )

        self.ids: list[str] = (
            collection_data.get("ids") or []
        )

        tokenized_documents = [
            tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        print(
            f"BM25 indexed chunks: {len(self.documents)}"
        )

    def search(
        self,
        query: str,
        top_k: int = CANDIDATE_K,
    ) -> list[dict[str, Any]]:
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

            results.append(
                {
                    "rank": rank,
                    "id": (
                        self.ids[index]
                        if index < len(self.ids)
                        else None
                    ),
                    "text": self.documents[index],
                    "metadata": metadata,
                    "source": metadata.get("source"),
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "category": metadata.get("category"),
                    "chunk_number": metadata.get(
                        "chunk_number"
                    ),
                }
            )

        return results


# ============================================================
# SOURCE HELPERS
# ============================================================

def source_name(result: dict[str, Any]) -> str:
    filename = result.get("filename")

    if filename:
        return Path(str(filename)).name

    source = result.get("source")

    if source:
        return Path(str(source)).name

    return ""


def normalize_filename(filename: str) -> str:
    return Path(str(filename)).name.lower().strip()


# ============================================================
# EXPECTED SOURCE METRICS
# ============================================================

def result_matches_expected(
    result: dict[str, Any],
    expected_sources: list[str],
) -> bool:
    actual = normalize_filename(
        source_name(result)
    )

    expected = {
        normalize_filename(source)
        for source in expected_sources
    }

    return actual in expected


def hit_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> bool:
    return any(
        result_matches_expected(
            result,
            expected_sources,
        )
        for result in results
    )


def precision_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    if not results:
        return 0.0

    relevant = sum(
        result_matches_expected(
            result,
            expected_sources,
        )
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

    return len(
        expected.intersection(retrieved)
    ) / len(expected)


def reciprocal_rank(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result_matches_expected(
            result,
            expected_sources,
        ):
            return 1.0 / rank

    return 0.0


def source_coverage(
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

    return len(
        expected.intersection(retrieved)
    ) / len(expected)


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
# HYBRID RRF
# ============================================================

def hybrid_search(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    dense_weight: float,
    bm25_weight: float,
    final_k: int = FINAL_K,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion.

    Dense and BM25 ranks are combined using:

        dense_weight / (60 + dense_rank)

        bm25_weight / (60 + bm25_rank)

    This avoids combining incompatible raw score scales.
    """

    fused: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    for result in dense_results:
        result_id = result.get("id")

        if not result_id:
            continue

        if result_id not in fused:
            fused[result_id] = {
                "id": result_id,
                "text": result.get("text"),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "source": result.get("source"),
                "filename": result.get("filename"),
                "page": result.get("page"),
                "category": result.get("category"),
                "chunk_number": result.get(
                    "chunk_number"
                ),
                "dense_rank": None,
                "bm25_rank": None,
            }

        rank = result.get("rank", 999)

        fused[result_id]["dense_rank"] = rank

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for result in bm25_results:
        result_id = result.get("id")

        if not result_id:
            continue

        if result_id not in fused:
            fused[result_id] = {
                "id": result_id,
                "text": result.get("text"),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "source": result.get("source"),
                "filename": result.get("filename"),
                "page": result.get("page"),
                "category": result.get("category"),
                "chunk_number": result.get(
                    "chunk_number"
                ),
                "dense_rank": None,
                "bm25_rank": None,
            }

        rank = result.get("rank", 999)

        fused[result_id]["bm25_rank"] = rank

    # --------------------------------------------------------
    # Calculate RRF score
    # --------------------------------------------------------

    for result in fused.values():

        dense_rank = result["dense_rank"]
        bm25_rank = result["bm25_rank"]

        dense_score = 0.0
        bm25_score = 0.0

        if dense_rank is not None:
            dense_score = (
                dense_weight
                / (60 + dense_rank)
            )

        if bm25_rank is not None:
            bm25_score = (
                bm25_weight
                / (60 + bm25_rank)
            )

        result["hybrid_score"] = (
            dense_score + bm25_score
        )

    ranked = sorted(
        fused.values(),
        key=lambda result: result[
            "hybrid_score"
        ],
        reverse=True,
    )

    for rank, result in enumerate(
        ranked[:final_k],
        start=1,
    ):
        result["rank"] = rank

    return ranked[:final_k]


# ============================================================
# EVALUATE WEIGHT
# ============================================================

def evaluate_weight(
    questions: list[dict[str, Any]],
    dense_results_by_question: dict[str, list],
    bm25_results_by_question: dict[str, list],
    dense_weight: float,
    bm25_weight: float,
) -> dict[str, float]:

    total = len(questions)

    hit1 = 0
    hit3 = 0

    precision3 = 0.0
    recall3 = 0.0
    mrr = 0.0
    coverage3 = 0.0

    all_expected3 = 0

    for item in questions:

        question_id = item["id"]
        expected_sources = item[
            "expected_sources"
        ]

        dense_results = (
            dense_results_by_question[
                question_id
            ]
        )

        bm25_results = (
            bm25_results_by_question[
                question_id
            ]
        )

        hybrid_results = hybrid_search(
            dense_results=dense_results,
            bm25_results=bm25_results,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            final_k=FINAL_K,
        )

        top1 = hybrid_results[:1]
        top3 = hybrid_results[:3]

        if hit_at_k(
            top1,
            expected_sources,
        ):
            hit1 += 1

        if hit_at_k(
            top3,
            expected_sources,
        ):
            hit3 += 1

        precision3 += precision_at_k(
            top3,
            expected_sources,
        )

        recall3 += recall_at_k(
            top3,
            expected_sources,
        )

        mrr += reciprocal_rank(
            top3,
            expected_sources,
        )

        coverage3 += source_coverage(
            top3,
            expected_sources,
        )

        if all_expected_retrieved(
            top3,
            expected_sources,
        ):
            all_expected3 += 1

    return {
        "dense_weight": dense_weight,
        "bm25_weight": bm25_weight,
        "hit_at_1": hit1 / total,
        "hit_at_3": hit3 / total,
        "precision_at_3": precision3 / total,
        "recall_at_3": recall3 / total,
        "mrr": mrr / total,
        "coverage_at_3": coverage3 / total,
        "all_expected_at_3": (
            all_expected3 / total
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("HYBRID RETRIEVAL WEIGHT EXPERIMENT")
    print("=" * 70)

    print(
        f"Candidate K       : {CANDIDATE_K}"
    )

    print(
        f"Final K           : {FINAL_K}"
    )

    print(
        f"Weight combinations: {len(WEIGHTS)}"
    )

    # --------------------------------------------------------
    # Initialize retriever
    # --------------------------------------------------------

    retriever = Retriever(
        chroma_dir=CHROMA_DIR,
        collection_name="docmind_documents",
    )

    print(
        f"Dense indexed chunks: {retriever.count()}"
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    bm25_index = BM25Index(
        retriever
    )

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = get_evaluation_questions()

    print(
        f"Evaluation questions: {len(questions)}"
    )

    # --------------------------------------------------------
    # Precompute retrieval results
    # --------------------------------------------------------

    dense_results_by_question = {}
    bm25_results_by_question = {}

    print("\nCollecting retrieval candidates...")

    start_time = time.time()

    for item in questions:

        question_id = item["id"]
        question = item["question"]

        dense_results_by_question[
            question_id
        ] = retriever.retrieve(
            query=question,
            top_k=CANDIDATE_K,
        )

        bm25_results_by_question[
            question_id
        ] = bm25_index.search(
            query=question,
            top_k=CANDIDATE_K,
        )

    retrieval_time = (
        time.time() - start_time
    )

    print(
        f"Candidate retrieval time: "
        f"{retrieval_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Evaluate weights
    # --------------------------------------------------------

    all_results = []

    for dense_weight, bm25_weight in WEIGHTS:

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Testing Dense={dense_weight:.1f} "
            f"/ BM25={bm25_weight:.1f}"
        )

        metrics = evaluate_weight(
            questions=questions,
            dense_results_by_question=(
                dense_results_by_question
            ),
            bm25_results_by_question=(
                bm25_results_by_question
            ),
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        all_results.append(metrics)

        print(
            f"Hit@1             : "
            f"{metrics['hit_at_1']:.3f}"
        )

        print(
            f"Hit@3             : "
            f"{metrics['hit_at_3']:.3f}"
        )

        print(
            f"Precision@3       : "
            f"{metrics['precision_at_3']:.3f}"
        )

        print(
            f"Recall@3          : "
            f"{metrics['recall_at_3']:.3f}"
        )

        print(
            f"MRR               : "
            f"{metrics['mrr']:.3f}"
        )

        print(
            f"Source Coverage@3 : "
            f"{metrics['coverage_at_3']:.3f}"
        )

        print(
            f"All Expected@3    : "
            f"{metrics['all_expected_at_3']:.3f}"
        )

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    print("\n")
    print("=" * 95)
    print("HYBRID WEIGHT COMPARISON")
    print("=" * 95)

    print(
        f"{'Dense/BM25':<14}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Prec@3':>10}"
        f"{'Recall@3':>10}"
        f"{'MRR':>10}"
        f"{'Coverage':>12}"
        f"{'All Exp.':>12}"
    )

    print("-" * 95)

    for result in all_results:

        weight_label = (
            f"{result['dense_weight']:.1f}/"
            f"{result['bm25_weight']:.1f}"
        )

        print(
            f"{weight_label:<14}"
            f"{result['hit_at_1']:>10.3f}"
            f"{result['hit_at_3']:>10.3f}"
            f"{result['precision_at_3']:>10.3f}"
            f"{result['recall_at_3']:>10.3f}"
            f"{result['mrr']:>10.3f}"
            f"{result['coverage_at_3']:>12.3f}"
            f"{result['all_expected_at_3']:>12.3f}"
        )

    print("=" * 95)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dense_weight",
        "bm25_weight",
        "hit_at_1",
        "hit_at_3",
        "precision_at_3",
        "recall_at_3",
        "mrr",
        "coverage_at_3",
        "all_expected_at_3",
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

        for result in all_results:
            writer.writerow(result)

    # --------------------------------------------------------
    # Best metrics by individual metric
    # --------------------------------------------------------

    print("\nBest weight by metric:")

    metric_labels = [
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("precision_at_3", "Precision@3"),
        ("recall_at_3", "Recall@3"),
        ("mrr", "MRR"),
        ("coverage_at_3", "Source Coverage@3"),
        (
            "all_expected_at_3",
            "All Expected@3",
        ),
    ]

    for key, label in metric_labels:

        best = max(
            all_results,
            key=lambda result: result[key],
        )

        print(
            f"{label:<22}: "
            f"Dense={best['dense_weight']:.1f}, "
            f"BM25={best['bm25_weight']:.1f} "
            f"({best[key]:.3f})"
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nExperiment complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()