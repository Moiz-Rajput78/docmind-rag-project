"""
DocMind RAG - Generation Tests

Tests the complete question -> retrieval -> context -> LLM pipeline.

These tests verify:
- Known questions produce grounded answers.
- Unknown questions trigger abstention.
- Source metadata is returned correctly.
- Empty questions are rejected.

Note:
These tests call the real Ollama Cloud API, so they require:
- OLLAMA_API_KEY in .env
- OLLAMA_MODEL in .env
- ChromaDB populated with the knowledge base
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ============================================================
# PROJECT ROOT
# ============================================================

# Add the project root to Python's import path so that:
#
#     from app.generation.answer_service import ...
#
# works when pytest runs this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from app.generation.answer_service import (  # noqa: E402
    ABSTENTION_MESSAGE,
    AnswerService,
    answer_question,
)


# ============================================================
# KNOWN QUESTION TESTS
# ============================================================


def test_known_hr_question():
    """A known HR question should return the correct grounded answer."""

    result = answer_question(
        "How many annual leave days do full-time employees receive?"
    )

    answer = result["answer"].lower()

    assert "20" in answer
    assert "annual leave" in answer

    assert result["retrieved_count"] > 0
    assert len(result["sources"]) > 0


def test_known_technical_question():
    """A known technical question should return relevant troubleshooting information."""

    result = answer_question(
        "What should I do if NovaDesk fails to start?"
    )

    answer = result["answer"].lower()

    assert result["retrieved_count"] > 0
    assert len(result["sources"]) > 0

    expected_terms = [
        "configuration",
        "services",
        "logs",
        "database",
    ]

    matched_terms = [
        term
        for term in expected_terms
        if term in answer
    ]

    assert len(matched_terms) >= 2


# ============================================================
# ABSTENTION TEST
# ============================================================


def test_unknown_question_abstains():
    """A question outside the knowledge base should trigger abstention."""

    result = answer_question(
        "What is the capital city of France?"
    )

    assert result["answer"].strip() == ABSTENTION_MESSAGE


# ============================================================
# SOURCE TESTS
# ============================================================


def test_sources_have_required_metadata():
    """Retrieved sources should contain authoritative metadata."""

    result = answer_question(
        "How many annual leave days do full-time employees receive?"
    )

    assert len(result["sources"]) > 0

    for source in result["sources"]:
        assert "source_number" in source
        assert "filename" in source
        assert "category" in source
        assert "distance" in source

        assert isinstance(
            source["source_number"],
            int,
        )

        assert isinstance(
            source["filename"],
            str,
        )

        assert source["filename"]


def test_source_numbers_are_application_generated():
    """
    Source numbering should come from AnswerService rather than the LLM.

    The numbering should start at 1 and increment sequentially.
    """

    result = answer_question(
        "How many annual leave days do full-time employees receive?"
    )

    source_numbers = [
        source["source_number"]
        for source in result["sources"]
    ]

    assert source_numbers == list(
        range(1, len(source_numbers) + 1)
    )


# ============================================================
# INPUT VALIDATION TESTS
# ============================================================


def test_empty_question_is_rejected():
    """An empty question should raise ValueError."""

    service = AnswerService()

    with pytest.raises(ValueError):
        service.ask("")


def test_whitespace_question_is_rejected():
    """A whitespace-only question should raise ValueError."""

    service = AnswerService()

    with pytest.raises(ValueError):
        service.ask("   ")


# ============================================================
# TOP-K TEST
# ============================================================


def test_top_k_is_respected():
    """The answer service should respect the requested top_k value."""

    result = answer_question(
        "How many annual leave days do full-time employees receive?",
        top_k=1,
    )

    assert result["top_k"] == 1
    assert result["retrieved_count"] <= 1
    assert len(result["sources"]) <= 1