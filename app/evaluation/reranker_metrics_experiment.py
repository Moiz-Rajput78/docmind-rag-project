from pathlib import Path
import csv
import math

from sentence_transformers import CrossEncoder

from app.evaluation.dataset import get_evaluation_questions
from app.retrieval.retriever import Retriever


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_DIR / "docs" / "reranker_metrics_experiment.csv"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 10
FINAL_K = 3


def normalize_filename(value):
    if not value:
        return ""

    return Path(str(value)).name.lower().strip()


def expected_sources(question):
    sources = question.get("expected_sources", [])

    if isinstance(sources, str):
        sources = [sources]

    return [
        normalize_filename(source)
        for source in sources
        if source
    ]


def retrieved_sources(results):
    return [
        normalize_filename(result.get("filename"))
        for result in results
        if result.get("filename")
    ]


def hit_at_k(results, expected, k):
    retrieved = set(retrieved_sources(results[:k]))
    return any(source in retrieved for source in expected)


def source_coverage_at_k(results, expected, k):
    if not expected:
        return 0.0

    retrieved = set(retrieved_sources(results[:k]))

    matched = sum(
        1
        for source in expected
        if source in retrieved
    )

    return matched / len(set(expected))


def reciprocal_rank(results, expected):
    expected_set = set(expected)

    for rank, result in enumerate(results, start=1):
        filename = normalize_filename(
            result.get("filename")
        )

        if filename in expected_set:
            return 1.0 / rank

    return 0.0


def best_expected_rank(results, expected):
    expected_set = set(expected)

    ranks = []

    for rank, result in enumerate(results, start=1):
        filename = normalize_filename(
            result.get("filename")
        )

        if filename in expected_set:
            ranks.append(rank)

    return min(ranks) if ranks else None


def all_expected_retrieved(results, expected, k):
    if not expected:
        return False

    retrieved = set(retrieved_sources(results[:k]))

    return all(
        source in retrieved
        for source in set(expected)
    )


def rerank_results(question, results, reranker):
    pairs = [
        (question, result["text"])
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for result, score in zip(results, scores):
        item = dict(result)
        item["reranker_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    for rank, result in enumerate(
        reranked,
        start=1,
    ):
        result["rank"] = rank

    return reranked


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_metric_comparison(
    name,
    baseline_values,
    reranker_values,
):
    baseline = mean(baseline_values)
    reranker = mean(reranker_values)
    change = reranker - baseline

    print(
        f"{name:<25}"
        f"Baseline: {baseline:.3f}   "
        f"Reranker: {reranker:.3f}   "
        f"Change: {change:+.3f}"
    )


def main():
    print("=" * 75)
    print("CROSS-ENCODER RERANKER METRICS EXPERIMENT")
    print("=" * 75)

    print()
    print(f"Model          : {RERANKER_MODEL}")
    print(f"Candidate K    : {CANDIDATE_K}")
    print(f"Final K        : {FINAL_K}")
    print()

    questions = get_evaluation_questions()

    print(f"Evaluation questions: {len(questions)}")
    print()

    print("Loading retriever...")
    retriever = Retriever()

    print("Loading cross-encoder...")
    reranker = CrossEncoder(RERANKER_MODEL)

    rows = []

    baseline_hit1 = []
    baseline_hit3 = []
    baseline_coverage3 = []
    baseline_mrr = []
    baseline_all3 = []

    reranker_hit1 = []
    reranker_hit3 = []
    reranker_coverage3 = []
    reranker_mrr = []
    reranker_all3 = []

    baseline_ranks = []
    reranker_ranks = []

    print()
    print("=" * 75)
    print("QUESTION-BY-QUESTION METRICS")
    print("=" * 75)

    for question_data in questions:
        question_id = question_data["id"]
        question = question_data["question"]
        difficulty = question_data["difficulty"]
        category = question_data["category"]

        expected = expected_sources(question_data)

        # ------------------------------------------------------------
        # Dense baseline
        # ------------------------------------------------------------
        baseline = retriever.retrieve(
            query=question,
            top_k=FINAL_K,
        )

        # ------------------------------------------------------------
        # Dense retrieval candidate pool
        # ------------------------------------------------------------
        candidates = retriever.retrieve(
            query=question,
            top_k=CANDIDATE_K,
        )

        # ------------------------------------------------------------
        # Cross-encoder reranking
        # ------------------------------------------------------------
        reranked_all = rerank_results(
            question,
            candidates,
            reranker,
        )

        reranked = reranked_all[:FINAL_K]

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
        b_hit1 = hit_at_k(baseline, expected, 1)
        b_hit3 = hit_at_k(baseline, expected, 3)
        b_cov3 = source_coverage_at_k(
            baseline,
            expected,
            3,
        )
        b_mrr = reciprocal_rank(
            baseline,
            expected,
        )
        b_all3 = all_expected_retrieved(
            baseline,
            expected,
            3,
        )
        b_rank = best_expected_rank(
            baseline,
            expected,
        )

        r_hit1 = hit_at_k(reranked, expected, 1)
        r_hit3 = hit_at_k(reranked, expected, 3)
        r_cov3 = source_coverage_at_k(
            reranked,
            expected,
            3,
        )
        r_mrr = reciprocal_rank(
            reranked,
            expected,
        )
        r_all3 = all_expected_retrieved(
            reranked,
            expected,
            3,
        )
        r_rank = best_expected_rank(
            reranked,
            expected,
        )

        baseline_hit1.append(int(b_hit1))
        baseline_hit3.append(int(b_hit3))
        baseline_coverage3.append(b_cov3)
        baseline_mrr.append(b_mrr)
        baseline_all3.append(int(b_all3))

        reranker_hit1.append(int(r_hit1))
        reranker_hit3.append(int(r_hit3))
        reranker_coverage3.append(r_cov3)
        reranker_mrr.append(r_mrr)
        reranker_all3.append(int(r_all3))

        if b_rank is not None:
            baseline_ranks.append(b_rank)

        if r_rank is not None:
            reranker_ranks.append(r_rank)

        if b_rank is None:
            rank_change = "not_retrieved"
        elif r_rank is None:
            rank_change = "lost"
        elif r_rank < b_rank:
            rank_change = "moved_up"
        elif r_rank > b_rank:
            rank_change = "moved_down"
        else:
            rank_change = "same_rank"

        row = {
            "id": question_id,
            "question": question,
            "difficulty": difficulty,
            "category": category,
            "expected_sources": " | ".join(expected),

            "baseline_hit_at_1": b_hit1,
            "baseline_hit_at_3": b_hit3,
            "baseline_source_coverage_at_3": round(
                b_cov3,
                4,
            ),
            "baseline_mrr": round(
                b_mrr,
                4,
            ),
            "baseline_all_expected_at_3": b_all3,
            "baseline_best_rank": b_rank,

            "reranker_hit_at_1": r_hit1,
            "reranker_hit_at_3": r_hit3,
            "reranker_source_coverage_at_3": round(
                r_cov3,
                4,
            ),
            "reranker_mrr": round(
                r_mrr,
                4,
            ),
            "reranker_all_expected_at_3": r_all3,
            "reranker_best_rank": r_rank,

            "rank_change": rank_change,

            "baseline_sources": " | ".join(
                retrieved_sources(baseline)
            ),

            "reranker_sources": " | ".join(
                retrieved_sources(reranked)
            ),
        }

        rows.append(row)

        print(
            f"[{question_id:02d}] "
            f"Rank {str(b_rank):>4} -> "
            f"{str(r_rank):<4} "
            f"| Coverage "
            f"{b_cov3:.2f} -> {r_cov3:.2f} "
            f"| {rank_change:<13} "
            f"| {question}"
        )

    # ------------------------------------------------------------
    # Save detailed CSV
    # ------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

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

    # ------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------
    print()
    print("=" * 75)
    print("AGGREGATE RESULTS")
    print("=" * 75)

    print_metric_comparison(
        "Hit@1",
        baseline_hit1,
        reranker_hit1,
    )

    print_metric_comparison(
        "Hit@3",
        baseline_hit3,
        reranker_hit3,
    )

    print_metric_comparison(
        "Source Coverage@3",
        baseline_coverage3,
        reranker_coverage3,
    )

    print_metric_comparison(
        "MRR",
        baseline_mrr,
        reranker_mrr,
    )

    print_metric_comparison(
        "All Expected@3",
        baseline_all3,
        reranker_all3,
    )

    print()
    print(
        f"Average best source rank - "
        f"Baseline : {mean(baseline_ranks):.3f}"
    )

    print(
        f"Average best source rank - "
        f"Reranker : {mean(reranker_ranks):.3f}"
    )

    # ------------------------------------------------------------
    # Count rank movements
    # ------------------------------------------------------------
    moved_up = sum(
        1
        for row in rows
        if row["rank_change"] == "moved_up"
    )

    moved_down = sum(
        1
        for row in rows
        if row["rank_change"] == "moved_down"
    )

    same_rank = sum(
        1
        for row in rows
        if row["rank_change"] == "same_rank"
    )

    lost = sum(
        1
        for row in rows
        if row["rank_change"] == "lost"
    )

    not_retrieved = sum(
        1
        for row in rows
        if row["rank_change"] == "not_retrieved"
    )

    print()
    print("=" * 75)
    print("RANK MOVEMENT")
    print("=" * 75)

    print(f"Moved up       : {moved_up}")
    print(f"Moved down     : {moved_down}")
    print(f"Same rank      : {same_rank}")
    print(f"Lost           : {lost}")
    print(f"Not retrieved  : {not_retrieved}")

    print()
    print("=" * 75)
    print("DIFFICULTY BREAKDOWN")
    print("=" * 75)

    difficulties = ["easy", "medium", "difficult"]

    for difficulty in difficulties:
        difficulty_rows = [
            row
            for row in rows
            if row["difficulty"] == difficulty
        ]

        if not difficulty_rows:
            continue

        b_mrr_values = [
            row["baseline_mrr"]
            for row in difficulty_rows
        ]

        r_mrr_values = [
            row["reranker_mrr"]
            for row in difficulty_rows
        ]

        b_cov_values = [
            row["baseline_source_coverage_at_3"]
            for row in difficulty_rows
        ]

        r_cov_values = [
            row["reranker_source_coverage_at_3"]
            for row in difficulty_rows
        ]

        print()
        print(difficulty.upper())

        print(
            f"  MRR       : "
            f"{mean(b_mrr_values):.3f} -> "
            f"{mean(r_mrr_values):.3f}"
        )

        print(
            f"  Coverage  : "
            f"{mean(b_cov_values):.3f} -> "
            f"{mean(r_cov_values):.3f}"
        )

    print()
    print("=" * 75)
    print("RESULT FILE")
    print("=" * 75)

    print(OUTPUT_FILE)

    print()
    print("The production Retriever was not modified.")


if __name__ == "__main__":
    main()