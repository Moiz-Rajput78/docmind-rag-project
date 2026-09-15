from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from app.ingestion.loader import load_all_documents
from app.ingestion.chunker import (
    clean_text,
    chunk_text,
    create_chunks,
)


KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"


def main():
    print("=" * 70)
    print("DOCMIND — CHUNKING TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load documents
    # ---------------------------------------------------------

    records = load_all_documents(KNOWLEDGE_BASE_DIR)

    print(f"\nLoaded documents/records: {len(records)}")

    # ---------------------------------------------------------
    # Test cleaning
    # ---------------------------------------------------------

    sample = "  Hello   world  \n\n   This is a test.  "

    cleaned = clean_text(sample)

    print("\nCleaning test:")
    print(f"  Original: {repr(sample)}")
    print(f"  Cleaned:  {repr(cleaned)}")

    if cleaned != "Hello world This is a test.":
        raise AssertionError(
            "Text cleaning produced an unexpected result."
        )

    # ---------------------------------------------------------
    # Test basic chunking
    # ---------------------------------------------------------

    test_text = "A" * 1200

    chunks = chunk_text(
        test_text,
        chunk_size=500,
        chunk_overlap=100,
    )

    print("\nBasic chunking test:")
    print(f"  Input characters:  {len(test_text)}")
    print(f"  Chunk size:        500")
    print(f"  Chunk overlap:     100")
    print(f"  Chunks created:    {len(chunks)}")

    if not chunks:
        raise AssertionError("No chunks were created.")

    for chunk in chunks:
        if len(chunk) > 500:
            raise AssertionError(
                f"Chunk exceeds configured size: {len(chunk)}"
            )

    # ---------------------------------------------------------
    # Create real document chunks
    # ---------------------------------------------------------

    chunked_records = create_chunks(
        records,
        chunk_size=500,
        chunk_overlap=100,
    )

    print("\nReal knowledge-base chunking:")
    print(f"  Documents/records: {len(records)}")
    print(f"  Chunks created:    {len(chunked_records)}")

    if not chunked_records:
        raise AssertionError(
            "No chunks were created from the knowledge base."
        )

    # ---------------------------------------------------------
    # Show chunk samples
    # ---------------------------------------------------------

    print("\nChunk samples:")
    print("-" * 70)

    for index, record in enumerate(chunked_records[:5], start=1):
        metadata = record["metadata"]
        text = record["text"]

        print(f"\n[{index}] {metadata['filename']}")
        print(f"    Page: {metadata['page']}")
        print(f"    Chunk: {metadata['chunk_number']}")
        print(f"    Characters: {len(text)}")
        print(f"    Preview: {text[:200]}")

    # ---------------------------------------------------------
    # Metadata validation
    # ---------------------------------------------------------

    print("\nMetadata validation:")

    required_fields = [
        "source",
        "filename",
        "file_type",
        "document_type",
        "page",
        "chunk_number",
        "chunk_size",
        "chunk_overlap",
    ]

    for field in required_fields:
        for record in chunked_records:
            if field not in record["metadata"]:
                raise AssertionError(
                    f"Missing metadata field: {field}"
                )

    print("  ✓ All required metadata fields are present")

    print("\n" + "=" * 70)
    print("CHUNKING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()