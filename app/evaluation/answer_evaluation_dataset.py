"""
DocMind — Answer Evaluation Dataset

Builds the answer-quality evaluation dataset directly from the
existing 30-question retrieval evaluation dataset.

The existing dataset.py is the source of truth.

This module deliberately does NOT invent reference answers.
It reuses:
    - question
    - difficulty
    - category
    - expected source information
    - any optional evaluation metadata already present

This allows answer evaluation to measure groundedness, citation
correctness, context relevance, abstention, and retrieval support
without fabricating ground-truth answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.evaluation.dataset import EVALUATION_QUESTIONS


@dataclass(frozen=True)
class AnswerEvaluationCase:
    """
    One answer-quality evaluation case.

    Attributes
    ----------
    question:
        User question.

    difficulty:
        easy / medium / difficult.

    category:
        Knowledge-base category.

    expected_sources:
        Source filenames expected to contain supporting information.

    reference_answer:
        Optional human-validated reference answer.

    required_facts:
        Optional facts that should appear in a correct answer.

    forbidden_facts:
        Optional facts that should not appear in the answer.

    abstention_expected:
        Whether the question is expected to produce an abstention.
    """

    question: str
    difficulty: str
    category: str
    expected_sources: tuple[str, ...]
    reference_answer: str | None = None
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    abstention_expected: bool = False


def _normalise_sources(item: dict[str, Any]) -> tuple[str, ...]:
    """
    Extract expected source filenames from an existing evaluation item.

    The function supports several possible field names so that this
    module remains compatible with the existing dataset structure.
    """

    source_fields = (
        "expected_sources",
        "expected_source",
        "relevant_sources",
        "source",
        "sources",
    )

    for field in source_fields:
        if field not in item:
            continue

        value = item[field]

        if value is None:
            continue

        if isinstance(value, str):
            return (value,)

        if isinstance(value, (list, tuple, set)):
            return tuple(
                str(source)
                for source in value
                if source is not None and str(source).strip()
            )

    return ()


def _get_question(item: dict[str, Any]) -> str:
    """Extract and validate the question."""

    question = item.get("question")

    if question is None:
        raise ValueError(
            f"Evaluation item is missing the 'question' field: {item}"
        )

    question = str(question).strip()

    if not question:
        raise ValueError(
            f"Evaluation item contains an empty question: {item}"
        )

    return question


def _get_difficulty(item: dict[str, Any]) -> str:
    """Extract the difficulty level."""

    return str(
        item.get("difficulty", "unknown")
    ).strip().lower()


def _get_category(item: dict[str, Any]) -> str:
    """Extract the evaluation category."""

    return str(
        item.get("category", "general")
    ).strip().lower()


def _normalise_string_tuple(
    value: Any,
) -> tuple[str, ...]:
    """
    Convert optional metadata into a tuple of non-empty strings.
    """

    if value is None:
        return ()

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        return ()

    return tuple(
        str(item).strip()
        for item in value
        if item is not None and str(item).strip()
    )


def _extract_optional_metadata(
    item: dict[str, Any],
) -> tuple[
    str | None,
    tuple[str, ...],
    tuple[str, ...],
]:
    """
    Extract optional answer-quality metadata.

    These fields are optional because the original retrieval dataset
    may not contain human-written reference answers.
    """

    reference_answer = item.get("reference_answer")

    if reference_answer is not None:
        reference_answer = str(reference_answer).strip()

        if not reference_answer:
            reference_answer = None

    required_facts = _normalise_string_tuple(
        item.get("required_facts")
    )

    forbidden_facts = _normalise_string_tuple(
        item.get("forbidden_facts")
    )

    return (
        reference_answer,
        required_facts,
        forbidden_facts,
    )


def _is_abstention_case(
    item: dict[str, Any],
    category: str,
) -> bool:
    """
    Determine whether a case expects abstention.

    Explicit dataset metadata takes priority.
    """

    explicit_fields = (
        "abstention_expected",
        "expected_abstention",
        "should_abstain",
    )

    for field in explicit_fields:
        if field in item:
            return bool(item[field])

    return category == "abstention"


def build_answer_evaluation_dataset() -> list[AnswerEvaluationCase]:
    """
    Convert the existing retrieval dataset into answer-evaluation cases.

    No new questions are created here.
    No answers are fabricated here.
    """

    cases: list[AnswerEvaluationCase] = []

    for item in EVALUATION_QUESTIONS:
        if not isinstance(item, dict):
            raise TypeError(
                "Every item in EVALUATION_QUESTIONS must be a dictionary."
            )

        question = _get_question(item)
        difficulty = _get_difficulty(item)
        category = _get_category(item)

        expected_sources = _normalise_sources(item)

        (
            reference_answer,
            required_facts,
            forbidden_facts,
        ) = _extract_optional_metadata(item)

        abstention_expected = _is_abstention_case(
            item,
            category,
        )

        cases.append(
            AnswerEvaluationCase(
                question=question,
                difficulty=difficulty,
                category=category,
                expected_sources=expected_sources,
                reference_answer=reference_answer,
                required_facts=required_facts,
                forbidden_facts=forbidden_facts,
                abstention_expected=abstention_expected,
            )
        )

    return cases


# ------------------------------------------------------------------
# Public dataset
# ------------------------------------------------------------------

ANSWER_EVALUATION_DATASET = build_answer_evaluation_dataset()


def get_answer_evaluation_dataset() -> list[AnswerEvaluationCase]:
    """
    Return the answer evaluation dataset.
    """

    return list(ANSWER_EVALUATION_DATASET)


def dataset_summary() -> dict[str, Any]:
    """
    Return summary statistics for the answer evaluation dataset.
    """

    cases = ANSWER_EVALUATION_DATASET

    difficulty_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}

    for case in cases:
        difficulty_counts[case.difficulty] = (
            difficulty_counts.get(case.difficulty, 0) + 1
        )

        category_counts[case.category] = (
            category_counts.get(case.category, 0) + 1
        )

    return {
        "total_cases": len(cases),
        "difficulty_counts": difficulty_counts,
        "category_counts": category_counts,
        "abstention_cases": sum(
            1
            for case in cases
            if case.abstention_expected
        ),
        "reference_answer_cases": sum(
            1
            for case in cases
            if case.reference_answer
        ),
        "required_fact_cases": sum(
            1
            for case in cases
            if case.required_facts
        ),
    }


def dataset_as_dicts() -> list[dict[str, Any]]:
    """
    Convert cases to JSON/CSV-friendly dictionaries.
    """

    output: list[dict[str, Any]] = []

    for case in ANSWER_EVALUATION_DATASET:
        data = asdict(case)

        data["expected_sources"] = list(
            case.expected_sources
        )

        data["required_facts"] = list(
            case.required_facts
        )

        data["forbidden_facts"] = list(
            case.forbidden_facts
        )

        output.append(data)

    return output


def validate_answer_evaluation_dataset() -> dict[str, Any]:
    """
    Validate the dataset against the internship requirement.

    Expected:
        - 30 total questions
        - 10 easy
        - 10 medium
        - 10 difficult
    """

    summary = dataset_summary()

    errors: list[str] = []

    if summary["total_cases"] != 30:
        errors.append(
            "Expected 30 evaluation questions, "
            f"found {summary['total_cases']}."
        )

    expected_difficulty_counts = {
        "easy": 10,
        "medium": 10,
        "difficult": 10,
    }

    for difficulty, expected_count in (
        expected_difficulty_counts.items()
    ):
        actual_count = summary["difficulty_counts"].get(
            difficulty,
            0,
        )

        if actual_count != expected_count:
            errors.append(
                f"Expected {expected_count} {difficulty} questions, "
                f"found {actual_count}."
            )

    questions = [
        case.question
        for case in ANSWER_EVALUATION_DATASET
    ]

    duplicate_questions = (
        len(questions) != len(set(questions))
    )

    if duplicate_questions:
        errors.append(
            "Duplicate evaluation questions were detected."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "summary": summary,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("DOCMIND — ANSWER EVALUATION DATASET")
    print("=" * 70)

    summary = dataset_summary()

    print("\nDataset summary:")
    print("-" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nValidation:")
    print("-" * 70)

    validation = validate_answer_evaluation_dataset()

    if validation["valid"]:
        print("PASS — dataset satisfies the 30-question structure.")
    else:
        print("WARNING — dataset validation found issues:")

        for error in validation["errors"]:
            print(f"  - {error}")

    print("\nQuestions:")
    print("-" * 70)

    for index, case in enumerate(
        ANSWER_EVALUATION_DATASET,
        start=1,
    ):
        sources = ", ".join(case.expected_sources)

        if not sources:
            sources = "No expected source metadata"

        print(
            f"{index:02d}. "
            f"[{case.difficulty.upper():8}] "
            f"[{case.category:15}] "
            f"{case.question}"
        )

        print(
            f"    Expected source(s): {sources}"
        )

    print("\n" + "=" * 70)