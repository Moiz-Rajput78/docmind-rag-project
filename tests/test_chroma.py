from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from app.ingestion.loader import load_all_documents
from app.ingestion.chunker import create_chunks
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


KNOWLEDGE_BASE_DIR = (
    PROJECT_DIR / "knowledge_base"
)

CHROMA_DIR = (
    PROJECT_DIR / "data" / "chroma"
)


def main():

    print("=" * 70)
    print("DOCMIND — CHROMADB TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load documents
    # ---------------------------------------------------------

    print("\nLoading documents...")

    records = load_all_documents(
        KNOWLEDGE_BASE_DIR
    )

    print(
        f"  Loaded records: {len(records)}"
    )

    # ---------------------------------------------------------
    # Create chunks
    # ---------------------------------------------------------

    print("\nCreating chunks...")

    chunks = create_chunks(
        records,
        chunk_size=500,
        chunk_overlap=100,
    )

    print(
        f"  Created chunks: {len(chunks)}"
    )

    # ---------------------------------------------------------
    # Create embeddings
    # ---------------------------------------------------------

    print("\nCreating embeddings...")

    embedding_service = EmbeddingService()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = (
        embedding_service.embed_texts(
            texts
        )
    )

    print(
        f"  Embeddings created: {len(embeddings)}"
    )

    print(
        f"  Embedding dimension: "
        f"{len(embeddings[0])}"
    )

    # ---------------------------------------------------------
    # Create Chroma store
    # ---------------------------------------------------------

    print("\nInitializing ChromaDB...")

    store = ChromaStore(
        persist_directory=CHROMA_DIR
    )

    # Start clean for this test
    store.clear()

    print(
        f"  Collection: "
        f"{store.collection_name}"
    )

    # ---------------------------------------------------------
    # Store chunks
    # ---------------------------------------------------------

    print("\nStoring chunks...")

    stored = store.add_chunks(
        chunks,
        embeddings,
    )

    print(
        f"  Chunks stored: {stored}"
    )

    print(
        f"  ChromaDB count: {store.count()}"
    )

    if store.count() != len(chunks):
        raise AssertionError(
            "ChromaDB count does not match "
            "the number of chunks."
        )

    # ---------------------------------------------------------
    # Test semantic search
    # ---------------------------------------------------------

    query = (
        "How many annual leave days "
        "do full-time employees receive?"
    )

    print("\nSemantic search:")
    print(f"  Query: {query}")

    query_embedding = (
        embedding_service.embed_text(
            query
        )
    )

    results = store.query(
        query_embedding=query_embedding,
        top_k=3,
    )

    print("\nTop 3 results:")
    print("-" * 70)

    result_documents = (
        results.get("documents", [[]])[0]
    )

    result_metadatas = (
        results.get("metadatas", [[]])[0]
    )

    result_distances = (
        results.get("distances", [[]])[0]
    )

    for index, document in enumerate(
        result_documents,
        start=1,
    ):

        metadata = (
            result_metadatas[index - 1]
        )

        distance = (
            result_distances[index - 1]
            if result_distances
            else None
        )

        print(f"\n[{index}]")
        print(
            f"  File: "
            f"{metadata['filename']}"
        )
        print(
            f"  Category: "
            f"{metadata['category']}"
        )
        print(
            f"  Page: "
            f"{metadata['page']}"
        )
        print(
            f"  Chunk: "
            f"{metadata['chunk_number']}"
        )
        print(
            f"  Distance: "
            f"{distance}"
        )
        print(
            f"  Text: "
            f"{document[:300]}"
        )

    # ---------------------------------------------------------
    # Validate retrieval
    # ---------------------------------------------------------

    if not result_documents:
        raise AssertionError(
            "ChromaDB returned no search results."
        )

    top_result = (
        result_documents[0].lower()
    )

    expected_terms = [
        "annual leave",
        "20",
    ]

    if not all(
        term in top_result
        for term in expected_terms
    ):
        raise AssertionError(
            "The top result did not contain "
            "the expected annual-leave information."
        )

    print("\n" + "=" * 70)
    print("CHROMADB TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()