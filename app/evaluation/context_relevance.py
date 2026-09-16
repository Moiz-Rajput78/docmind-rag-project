"""
DocMind RAG - Context Relevance Evaluation

Evaluates whether retrieved context is relevant to the user's question.

This is intentionally separate from answer correctness.

A system can retrieve relevant documents but still generate a bad answer,
or generate a good-looking answer from poor evidence.

This module provides lightweight, reproducible retrieval-context metrics
without requiring another LLM call.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _normalize_text(text: str) -> str:
    """Normalize text for lexical comparison."""

    text = str(text or "").lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> set[str]:
    """
    Convert text into normalized content words.
    """

    normalized = _normalize_text(text)

    return {
        token
        for token in normalized.split()
        if token
        and token not in STOPWORDS
        and len(token) > 1
    }


def keyword_overlap(
    question: str,
    context: str,
) -> float:
    """
    Calculate lexical overlap between question and context.

    Returns:
        Value between 0 and 1.

    This is not semantic similarity. It is a lightweight diagnostic
    indicating whether important query terms are represented in the
    retrieved context.
    """

    question_tokens = tokenize(question)
    context_tokens = tokenize(context)

    if not question_tokens:
        return 0.0

    overlap = question_tokens.intersection(
        context_tokens
    )

    return len(overlap) / len(question_tokens)


def source_relevance(
    sources: Iterable[dict[str, Any]],
    expected_source: str | None,
) -> bool | None:
    """
    Check whether an expected source was retrieved.

    Returns:
        True  -> expected source retrieved
        False -> expected source not retrieved
        None  -> no source expectation
    """

    if not expected_source:
        return None

    expected = expected_source.strip().lower()

    for source in sources:
        filename = str(
            source.get("filename") or ""
        ).strip().lower()

        if filename == expected:
            return True

    return False


def evidence_keyword_coverage(
    context: str,
    expected_keywords: Iterable[str],
) -> float:
    """
    Measure how many expected evidence keywords occur in context.

    Returns:
        Value between 0 and 1.
    """

    keywords = [
        _normalize_text(keyword)
        for keyword in expected_keywords
        if str(keyword).strip()
    ]

    keywords = [
        keyword
        for keyword in keywords
        if keyword
    ]

    if not keywords:
        return 1.0

    normalized_context = _normalize_text(context)

    matched = sum(
        1
        for keyword in keywords
        if keyword in normalized_context
    )

    return matched / len(keywords)


def calculate_context_relevance(
    question: str,
    context: str,
    sources: list[dict[str, Any]],
    expected_source: str | None = None,
    expected_keywords: Iterable[str] = (),
) -> dict[str, Any]:
    """
    Calculate context-relevance diagnostics.

    Metrics:

    - query_keyword_overlap
    - expected_source_retrieved
    - evidence_keyword_coverage
    - retrieved_source_count

    A combined score is also returned when enough reference information
    exists.
    """

    query_overlap = keyword_overlap(
        question=question,
        context=context,
    )

    source_match = source_relevance(
        sources=sources,
        expected_source=expected_source,
    )

    evidence_coverage = evidence_keyword_coverage(
        context=context,
        expected_keywords=expected_keywords,
    )

    components: list[float] = []

    components.append(query_overlap)
    components.append(evidence_coverage)

    if source_match is not None:
        components.append(
            1.0 if source_match else 0.0
        )

    relevance_score = (
        sum(components) / len(components)
        if components
        else 0.0
    )

    return {
        "query_keyword_overlap": round(
            query_overlap,
            4,
        ),
        "expected_source_retrieved": source_match,
        "evidence_keyword_coverage": round(
            evidence_coverage,
            4,
        ),
        "retrieved_source_count": len(sources),
        "context_relevance_score": round(
            relevance_score,
            4,
        ),
    }


def evaluate_context_relevance(
    question: str,
    results: list[dict[str, Any]],
    expected_source: str | None = None,
    expected_keywords: Iterable[str] = (),
) -> dict[str, Any]:
    """
    Evaluate context relevance directly from AnswerService-style
    retrieval results.

    Each result can contain:

        {
            "text": "...",
            "content": "...",
            "metadata": {...}
        }
    """

    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []

    for result in results:
        content = (
            result.get("text")
            or result.get("content")
            or ""
        )

        if content:
            context_parts.append(
                str(content)
            )

        metadata = result.get(
            "metadata",
            {},
        )

        sources.append(
            {
                "filename": (
                    metadata.get("filename")
                    or result.get("filename")
                ),
                "page": (
                    metadata.get("page")
                    if metadata.get("page") is not None
                    else result.get("page")
                ),
                "category": (
                    metadata.get("category")
                    or result.get("category")
                ),
            }
        )

    context = "\n\n".join(
        context_parts
    )

    return calculate_context_relevance(
        question=question,
        context=context,
        sources=sources,
        expected_source=expected_source,
        expected_keywords=expected_keywords,
    )