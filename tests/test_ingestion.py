from pathlib import Path
import sys


# Add project root to Python import path
PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from app.ingestion.loader import (
    discover_documents,
    load_all_documents,
)


KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"


def main():
    print("=" * 70)
    print("DOCMIND — DOCUMENT INGESTION TEST")
    print("=" * 70)

    print("\nKnowledge base:")
    print(f"  {KNOWLEDGE_BASE_DIR}")

    documents = discover_documents(KNOWLEDGE_BASE_DIR)

    print(f"\nDocuments discovered: {len(documents)}")

    for document in documents:
        print(f"  ✓ {document.relative_to(KNOWLEDGE_BASE_DIR)}")

    records = load_all_documents(KNOWLEDGE_BASE_DIR)

    print(f"\nLoaded records: {len(records)}")

    print("\nLoaded content summary:")
    print("-" * 70)

    for index, record in enumerate(records, start=1):
        text = record["text"]
        metadata = record["metadata"]

        preview = text[:150].replace("\n", " ")

        print(f"\n[{index}] {metadata['filename']}")
        print(f"    Type: {metadata['file_type']}")
        print(f"    Page: {metadata['page']}")
        print(f"    Characters: {len(text)}")
        print(f"    Preview: {preview}")

    print("\n" + "=" * 70)

    if len(documents) != 12:
        raise AssertionError(
            f"Expected 12 documents, found {len(documents)}"
        )

    if not records:
        raise AssertionError("No document content was loaded.")

    for record in records:
        if not record["text"].strip():
            raise AssertionError(
                f"Empty document content: "
                f"{record['metadata']['filename']}"
            )

    print("INGESTION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()