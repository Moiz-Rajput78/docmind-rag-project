from pathlib import Path
import sys
import traceback


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow direct execution:
# python app/utils/rebuild_knowledge_base.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.ingestion.loader import load_document
from app.ingestion.chunker import create_chunks
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


# ============================================================
# CONFIGURATION
# ============================================================

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "docmind_documents"

# Established project baseline
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


# ============================================================
# FIND KNOWLEDGE BASE DOCUMENTS
# ============================================================

def find_knowledge_base_documents():
    """
    Find all supported documents inside knowledge_base/.

    knowledge_base/ is the authoritative bundled/demo
    knowledge base used for this rebuild.
    """

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base directory does not exist:\n"
            f"{KNOWLEDGE_BASE_DIR}"
        )

    documents = []

    for path in sorted(KNOWLEDGE_BASE_DIR.rglob("*")):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append(path)

    return documents


# ============================================================
# PRINT DOCUMENT LIST
# ============================================================

def print_document_list(documents):
    """Print all documents that will be indexed."""

    print(
        f"\nFound {len(documents)} supported documents:\n"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):

        relative_path = document.relative_to(
            PROJECT_ROOT
        )

        print(
            f"  {index:02d}. {relative_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DOCMIND — KNOWLEDGE BASE CHROMA REBUILD")
    print("=" * 70)

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nAuthoritative knowledge base:\n"
        f"{KNOWLEDGE_BASE_DIR}"
    )

    print(
        f"\nChroma directory:\n"
        f"{CHROMA_DIR}"
    )

    print(
        "\nChunk configuration:"
        f"\n  Chunk size   : {CHUNK_SIZE}"
        f"\n  Chunk overlap: {CHUNK_OVERLAP}"
    )

    # ========================================================
    # FIND DOCUMENTS
    # ========================================================

    documents = find_knowledge_base_documents()

    if not documents:
        raise RuntimeError(
            "No supported documents were found inside "
            f"{KNOWLEDGE_BASE_DIR}"
        )

    print_document_list(documents)

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    print("\n" + "-" * 70)
    print("Loading embedding model...")
    print("-" * 70)

    embedding_service = EmbeddingService()

    # ========================================================
    # CONNECT TO CHROMA
    # ========================================================

    print("\n" + "-" * 70)
    print("Connecting to Chroma...")
    print("-" * 70)

    store = ChromaStore(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    old_count = store.count()

    print(
        f"Existing Chroma chunks: {old_count}"
    )

    # ========================================================
    # CLEAR EXISTING CHROMA DATA
    # ========================================================

    print("\n" + "-" * 70)
    print("Clearing existing Chroma collection...")
    print("-" * 70)

    print(
        "\nIMPORTANT:"
        "\nOnly the Chroma collection will be cleared."
        "\n"
        "\nThe following will NOT be deleted:"
        "\n  - knowledge_base/"
        "\n  - data/documents/"
        "\n  - uploaded documents"
        "\n  - source files"
    )

    store.clear()

    print(
        "\nChroma collection cleared successfully."
    )

    # ========================================================
    # INDEX DOCUMENTS
    # ========================================================

    print("\n" + "-" * 70)
    print("INDEXING AUTHORITATIVE KNOWLEDGE BASE")
    print("-" * 70)

    total_chunks = 0
    successful_documents = 0
    failed_documents = []

    for index, document_path in enumerate(
        documents,
        start=1,
    ):

        relative_path = document_path.relative_to(
            PROJECT_ROOT
        )

        print("\n" + "=" * 70)
        print(
            f"[{index}/{len(documents)}] "
            f"{relative_path}"
        )
        print("=" * 70)

        try:

            # ------------------------------------------------
            # LOAD DOCUMENT
            # ------------------------------------------------

            loaded = load_document(
                document_path
            )

            print(
                f"Loaded pages/sections: "
                f"{len(loaded)}"
            )

            # ------------------------------------------------
            # CREATE CHUNKS
            # ------------------------------------------------

            chunks = create_chunks(
                loaded,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            # ------------------------------------------------
            # REMOVE EMPTY CHUNKS
            # ------------------------------------------------

            chunks = [
                chunk
                for chunk in chunks
                if chunk.get("text", "").strip()
            ]

            print(
                f"Usable chunks: {len(chunks)}"
            )

            if not chunks:

                raise RuntimeError(
                    "No usable chunks were created."
                )

            # ------------------------------------------------
            # GENERATE EMBEDDINGS
            #
            # IMPORTANT:
            # EmbeddingService uses embed_texts().
            # ------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            print(
                f"Generating {len(texts)} embeddings..."
            )

            embeddings = (
                embedding_service.embed_texts(
                    texts
                )
            )

            print(
                f"Generated embeddings: "
                f"{len(embeddings)}"
            )

            # ------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------

            if len(embeddings) != len(chunks):

                raise RuntimeError(
                    "Embedding count does not match "
                    "chunk count. "
                    f"Chunks={len(chunks)}, "
                    f"Embeddings={len(embeddings)}"
                )

            # ------------------------------------------------
            # ADD TO CHROMA
            # ------------------------------------------------

            store.add_chunks(
                chunks,
                embeddings,
            )

            total_chunks += len(chunks)
            successful_documents += 1

            print(
                "Indexed successfully."
            )

            print(
                f"Running chunk total: "
                f"{total_chunks}"
            )

        except Exception as error:

            print(
                "\nERROR while indexing:"
            )

            print(
                f"  Document: {relative_path}"
            )

            print(
                f"  Error: {error}"
            )

            failed_documents.append(
                (
                    str(relative_path),
                    str(error),
                )
            )

            print(
                "\nTraceback:"
            )

            traceback.print_exc()

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print("\n" + "=" * 70)
    print("REBUILD VERIFICATION")
    print("=" * 70)

    final_count = store.count()

    print(
        f"\nDocuments discovered : "
        f"{len(documents)}"
    )

    print(
        f"Documents indexed    : "
        f"{successful_documents}"
    )

    print(
        f"Documents failed     : "
        f"{len(failed_documents)}"
    )

    print(
        f"Chunks indexed       : "
        f"{total_chunks}"
    )

    print(
        f"Chroma total         : "
        f"{final_count}"
    )

    # ========================================================
    # FAILED DOCUMENT REPORT
    # ========================================================

    if failed_documents:

        print(
            "\n" + "-" * 70
        )

        print(
            "DOCUMENTS WITH ERRORS"
        )

        print(
            "-" * 70
        )

        for document, error in failed_documents:

            print(
                f"\nDocument: {document}"
            )

            print(
                f"Error: {error}"
            )

    # ========================================================
    # FINAL SUCCESS / FAILURE LOGIC
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    if (
        successful_documents == len(documents)
        and final_count == total_chunks
        and total_chunks > 0
    ):

        print(
            "SUCCESS: All knowledge-base documents "
            "were indexed successfully."
        )

        print(
            "SUCCESS: Chroma count matches "
            "the indexed chunk count."
        )

    elif total_chunks == 0:

        print(
            "REBUILD FAILED: No chunks were indexed."
        )

        print(
            "Chroma currently contains 0 chunks."
        )

    else:

        print(
            "REBUILD INCOMPLETE: Some documents "
            "could not be indexed."
        )

        print(
            f"Indexed chunks: {total_chunks}"
        )

        print(
            f"Chroma chunks: {final_count}"
        )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "KNOWLEDGE BASE REBUILD FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        "\nAuthoritative source:"
        "\n  knowledge_base/"
    )

    print(
        "\nUntouched:"
        "\n  data/documents/"
    )

    print(
        "\nChroma collection:"
        f"\n  {COLLECTION_NAME}"
    )

    print(
        "\nFinal searchable chunks:"
        f"\n  {final_count}"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()