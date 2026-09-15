from pathlib import Path
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from app.embeddings.embedding_service import EmbeddingService


def cosine_similarity(
    vector_a,
    vector_b,
):
    """
    Calculate cosine similarity between two vectors.
    """

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def main():

    print("=" * 70)
    print("DOCMIND — EMBEDDING TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    embedding_service = EmbeddingService()

    dimension = embedding_service.dimension()

    print("\nModel:")
    print(f"  {embedding_service.model_name}")

    print("\nEmbedding dimension:")
    print(f"  {dimension}")

    if dimension != 384:
        raise AssertionError(
            f"Expected 384 dimensions, got {dimension}"
        )

    # ---------------------------------------------------------
    # Test individual embedding
    # ---------------------------------------------------------

    text = (
        "Full-time employees receive 20 annual "
        "leave days per calendar year."
    )

    embedding = embedding_service.embed_text(text)

    print("\nSingle text embedding:")
    print(f"  Text: {text}")
    print(f"  Vector length: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")

    if len(embedding) != 384:
        raise AssertionError(
            "Embedding does not contain 384 values."
        )

    # ---------------------------------------------------------
    # Semantic similarity experiment
    # ---------------------------------------------------------

    texts = [
        "Full-time employees receive 20 annual leave days.",
        "Employees get 20 days of annual vacation each year.",
        "NovaDesk is a customer-support management platform.",
        "Bananas are a type of fruit.",
    ]

    embeddings = embedding_service.embed_texts(texts)

    print("\nSemantic similarity experiment:")
    print("-" * 70)

    similarity_01 = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    similarity_02 = cosine_similarity(
        embeddings[0],
        embeddings[2],
    )

    similarity_03 = cosine_similarity(
        embeddings[0],
        embeddings[3],
    )

    print(
        "\nAnnual leave ↔ Annual vacation:"
    )
    print(
        f"  Similarity: {similarity_01:.4f}"
    )

    print(
        "\nAnnual leave ↔ NovaDesk:"
    )
    print(
        f"  Similarity: {similarity_02:.4f}"
    )

    print(
        "\nAnnual leave ↔ Bananas:"
    )
    print(
        f"  Similarity: {similarity_03:.4f}"
    )

    # ---------------------------------------------------------
    # Expected semantic behavior
    # ---------------------------------------------------------

    if similarity_01 <= similarity_02:
        raise AssertionError(
            "Related annual-leave sentences should have "
            "higher similarity than unrelated text."
        )

    if similarity_01 <= similarity_03:
        raise AssertionError(
            "Related annual-leave sentences should have "
            "higher similarity than unrelated text."
        )

    # ---------------------------------------------------------
    # Normalization test
    # ---------------------------------------------------------

    norm = np.linalg.norm(
        np.array(embedding)
    )

    print("\nEmbedding normalization:")
    print(f"  Vector norm: {norm:.6f}")

    if not np.isclose(norm, 1.0, atol=0.01):
        raise AssertionError(
            "Embedding is not normalized."
        )

    print("\n" + "=" * 70)
    print("EMBEDDING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()