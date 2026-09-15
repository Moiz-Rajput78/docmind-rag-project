# ============================================================
# DOCMIND — RETRIEVAL TEST
# ============================================================

import sys
from pathlib import Path


# Add project root to Python path
PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from app.retrieval.retriever import Retriever


print("=" * 70)
print("DOCMIND — RETRIEVAL TEST")
print("=" * 70)


# ------------------------------------------------------------
# Initialize retriever
# ------------------------------------------------------------

print("\nInitializing retriever...")

retriever = Retriever()

print(f"  ChromaDB chunks: {retriever.count()}")


if retriever.count() == 0:
    raise AssertionError(
        "ChromaDB is empty. Run tests/test_chroma.py first."
    )


# ------------------------------------------------------------
# Test questions
# ------------------------------------------------------------

questions = [
    "How many annual leave days do full-time employees receive?",
    "What does NovaDesk do?",
    "How do I install NovaFlow?",
    "What should I do if NovaDesk fails to start?",
]


# ------------------------------------------------------------
# Test Top-K values
# ------------------------------------------------------------

top_k_values = [1, 3, 5, 10]


for question in questions:

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    for top_k in top_k_values:

        print(f"\n--- TOP-{top_k} ---")

        results = retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        print(f"Retrieved chunks: {len(results)}")

        for result in results:

            metadata = result["metadata"]

            print(
                f"\n  Rank: {result['rank']}"
                f"\n  File: {result['filename']}"
                f"\n  Category: {result['category']}"
                f"\n  Page: {result['page']}"
                f"\n  Chunk: {result['chunk_number']}"
                f"\n  Distance: {result['distance']}"
                f"\n  Text: {result['text'][:250]}"
            )


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)


test_query = "How many annual leave days do full-time employees receive?"

results = retriever.retrieve(
    query=test_query,
    top_k=3,
)


assert len(results) == 3, (
    f"Expected 3 results, got {len(results)}"
)


first_result = results[0]


assert first_result["filename"] is not None
assert first_result["text"]
assert first_result["distance"] is not None
assert first_result["metadata"]


combined_text = " ".join(
    result["text"].lower()
    for result in results
)


assert "annual leave" in combined_text
assert "20" in combined_text


print("  Top-3 retrieval returned exactly 3 chunks")
print("  Metadata is present")
print("  Distance scores are present")
print("  Relevant annual-leave information was retrieved")


print("\n" + "=" * 70)
print("RETRIEVAL TEST PASSED")
print("=" * 70)