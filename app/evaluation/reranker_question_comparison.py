from pathlib import Path
import csv

from sentence_transformers import CrossEncoder

from app.evaluation.dataset import get_evaluation_questions
from app.retrieval.retriever import Retriever


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_DIR / "docs" / "reranker_question_comparison.csv"

TOP_K_BASELINE = 3
TOP_K_CANDIDATES = 10
TOP_K_RERANKED = 3

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def normalize_filename(value):
    """Normalize a filename for reliable comparison."""
    if not value:
        return ""

    return Path(str(value)).name.lower().strip()


def get_source_names(results):
    """Return normalized source filenames from retrieval results."""
    sources = []

    for result in results:
        filename = result.get("filename")

        if filename:
            normalized = normalize_filename(filename)

            if normalized and normalized not in sources:
                sources.append(normalized)

    return sources


def get_expected_sources(question):
    """Return normalized expected source filenames."""
    expected = question.get("expected_sources", [])

    if isinstance(expected, str):
        expected = [expected]

    return [
        normalize_filename(source)
        for source in expected
        if source
    ]


def calculate_hit(results, expected_sources):
    """Check whether any expected source appears in the results."""
    retrieved_sources = get_source_names(results)

    return any(
        expected in retrieved_sources
        for expected in expected_sources
    )


def get_matching_ranks(results, expected_sources):
    """Return ranks at which expected sources were retrieved."""
    ranks = []

    expected_set = set(expected_sources)

    for result in results:
        filename = normalize_filename(result.get("filename"))

        if filename in expected_set:
            ranks.append(result.get("rank"))

    return ranks


def format_sources(results):
    """Format retrieved sources for CSV output."""
    sources = []

    for result in results:
        filename = result.get("filename")

        if filename:
            rank = result.get("rank")
            sources.append(f"{rank}:{filename}")

    return " | ".join(sources)


def format_scores(results):
    """Format reranker scores for CSV output."""
    scores = []

    for result in results:
        score = result.get("reranker_score")

        if score is not None:
            scores.append(
                f"{result.get('rank')}:{score:.4f}"
            )

    return " | ".join(scores)


def classify_change(baseline_hit, reranker_hit):
    """Classify how reranking changed the result."""
    if baseline_hit and reranker_hit:
        return "same_success"

    if not baseline_hit and reranker_hit:
        return "improved"

    if baseline_hit and not reranker_hit:
        return "worsened"

    return "same_failure"


def main():
    print("=" * 70)
    print("QUESTION-BY-QUESTION RERANKER COMPARISON")
    print("=" * 70)

    print()
    print(f"Reranker model: {RERANKER_MODEL}")
    print(f"Baseline Top-K: {TOP_K_BASELINE}")
    print(f"Candidate Top-K: {TOP_K_CANDIDATES}")
    print(f"Reranked Top-K: {TOP_K_RERANKED}")
    print()

    questions = get_evaluation_questions()

    print(f"Evaluation questions: {len(questions)}")
    print()

    print("Loading retriever...")
    retriever = Retriever()

    print("Loading cross-encoder...")
    reranker = CrossEncoder(RERANKER_MODEL)

    rows = []

    improved = 0
    worsened = 0
    same_success = 0
    same_failure = 0

    baseline_hits = 0
    reranker_hits = 0

    print()
    print("=" * 70)
    print("QUESTION RESULTS")
    print("=" * 70)

    for question_data in questions:
        question_id = question_data["id"]
        question = question_data["question"]
        difficulty = question_data["difficulty"]
        category = question_data["category"]

        expected_sources = get_expected_sources(question_data)

        # ------------------------------------------------------------
        # Baseline dense retrieval
        # ------------------------------------------------------------
        baseline_results = retriever.retrieve(
            query=question,
            top_k=TOP_K_BASELINE,
        )

        # ------------------------------------------------------------
        # Retrieve larger candidate pool for reranking
        # ------------------------------------------------------------
        candidate_results = retriever.retrieve(
            query=question,
            top_k=TOP_K_CANDIDATES,
        )

        # ------------------------------------------------------------
        # Cross-encoder reranking
        # ------------------------------------------------------------
        pairs = [
            (question, result["text"])
            for result in candidate_results
        ]

        reranker_scores = reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        scored_results = []

        for result, score in zip(
            candidate_results,
            reranker_scores,
        ):
            updated_result = dict(result)
            updated_result["reranker_score"] = float(score)
            scored_results.append(updated_result)

        scored_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        # Reassign ranks after reranking
        for rank, result in enumerate(
            scored_results,
            start=1,
        ):
            result["rank"] = rank

        reranked_results = scored_results[:TOP_K_RERANKED]

        # ------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------
        baseline_hit = calculate_hit(
            baseline_results,
            expected_sources,
        )

        reranker_hit = calculate_hit(
            reranked_results,
            expected_sources,
        )

        baseline_ranks = get_matching_ranks(
            baseline_results,
            expected_sources,
        )

        reranker_ranks = get_matching_ranks(
            reranked_results,
            expected_sources,
        )

        change = classify_change(
            baseline_hit,
            reranker_hit,
        )

        baseline_hits += int(baseline_hit)
        reranker_hits += int(reranker_hit)

        if change == "improved":
            improved += 1
        elif change == "worsened":
            worsened += 1
        elif change == "same_success":
            same_success += 1
        else:
            same_failure += 1

        row = {
            "id": question_id,
            "question": question,
            "difficulty": difficulty,
            "category": category,
            "expected_sources": " | ".join(expected_sources),

            "baseline_hit": baseline_hit,
            "baseline_ranks": " | ".join(
                str(rank)
                for rank in baseline_ranks
            ),
            "baseline_sources": format_sources(
                baseline_results
            ),

            "reranker_hit": reranker_hit,
            "reranker_ranks": " | ".join(
                str(rank)
                for rank in reranker_ranks
            ),
            "reranker_sources": format_sources(
                reranked_results
            ),
            "reranker_scores": format_scores(
                reranked_results
            ),

            "change": change,
        }

        rows.append(row)

        status = {
            "improved": "IMPROVED",
            "worsened": "WORSENED",
            "same_success": "SAME",
            "same_failure": "FAILED",
        }[change]

        print(
            f"[{question_id:02d}] "
            f"{status:<9} "
            f"Baseline={'HIT' if baseline_hit else 'MISS':4} "
            f"Reranker={'HIT' if reranker_hit else 'MISS':4} "
            f"| {question}"
        )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "id",
        "question",
        "difficulty",
        "category",
        "expected_sources",
        "baseline_hit",
        "baseline_ranks",
        "baseline_sources",
        "reranker_hit",
        "reranker_ranks",
        "reranker_sources",
        "reranker_scores",
        "change",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    total = len(rows)

    baseline_hit_rate = (
        baseline_hits / total
        if total
        else 0
    )

    reranker_hit_rate = (
        reranker_hits / total
        if total
        else 0
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total questions       : {total}")
    print(
        f"Baseline hit rate     : "
        f"{baseline_hit_rate:.3f}"
    )
    print(
        f"Reranker hit rate     : "
        f"{reranker_hit_rate:.3f}"
    )

    print()
    print(f"Improved questions    : {improved}")
    print(f"Worsened questions    : {worsened}")
    print(f"Same successful       : {same_success}")
    print(f"Same failed           : {same_failure}")

    print()
    print("Results saved to:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if improved > 0:
        print(
            f"- Reranking improved {improved} question(s)."
        )

    if worsened > 0:
        print(
            f"- Reranking worsened {worsened} question(s)."
        )

    if improved == 0 and worsened == 0:
        print(
            "- Reranking produced the same hit/miss result "
            "for every question."
        )

    print(
        "- This experiment compares retrieval behavior only."
    )
    print(
        "- The production Retriever was not modified."
    )


if __name__ == "__main__":
    main()