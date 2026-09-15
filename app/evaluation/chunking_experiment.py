from pathlib import Path
import sys
import csv


# Add project root
PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from app.ingestion.loader import load_all_documents
from app.ingestion.chunker import create_chunks


KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
DOCS_DIR = PROJECT_DIR / "docs"

CHUNK_SIZES = [300, 500, 800, 1200]
CHUNK_OVERLAPS = [0, 50, 100, 200]


def run_experiment():
    print("=" * 80)
    print("DOCMIND — CHUNKING EXPERIMENT")
    print("=" * 80)

    records = load_all_documents(KNOWLEDGE_BASE_DIR)

    print(f"\nLoaded records: {len(records)}")

    results = []

    for chunk_size in CHUNK_SIZES:

        for chunk_overlap in CHUNK_OVERLAPS:

            # Invalid configurations are skipped
            if chunk_overlap >= chunk_size:
                continue

            chunks = create_chunks(
                records,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            lengths = [
                len(chunk["text"])
                for chunk in chunks
            ]

            if lengths:
                average_length = sum(lengths) / len(lengths)
                minimum_length = min(lengths)
                maximum_length = max(lengths)
                total_characters = sum(lengths)
            else:
                average_length = 0
                minimum_length = 0
                maximum_length = 0
                total_characters = 0

            result = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_count": len(chunks),
                "average_length": round(average_length, 2),
                "minimum_length": minimum_length,
                "maximum_length": maximum_length,
                "total_characters": total_characters,
            }

            results.append(result)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\nExperiment results:")
    print("-" * 80)

    header = (
        f"{'Size':>6} "
        f"{'Overlap':>8} "
        f"{'Chunks':>8} "
        f"{'Avg':>10} "
        f"{'Min':>8} "
        f"{'Max':>8} "
        f"{'Total':>10}"
    )

    print(header)
    print("-" * 80)

    for result in results:

        print(
            f"{result['chunk_size']:>6} "
            f"{result['chunk_overlap']:>8} "
            f"{result['chunk_count']:>8} "
            f"{result['average_length']:>10.2f} "
            f"{result['minimum_length']:>8} "
            f"{result['maximum_length']:>8} "
            f"{result['total_characters']:>10}"
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    output_file = DOCS_DIR / "chunking_experiment_results.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        fieldnames = [
            "chunk_size",
            "chunk_overlap",
            "chunk_count",
            "average_length",
            "minimum_length",
            "maximum_length",
            "total_characters",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:")
    print(f"  {output_file}")

    print("\n" + "=" * 80)
    print("CHUNKING EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()