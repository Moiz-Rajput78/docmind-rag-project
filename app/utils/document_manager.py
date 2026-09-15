from pathlib import Path
from typing import List, Dict

from app.ingestion.loader import load_document
from app.ingestion.chunker import create_chunks
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ============================================================
# DOCUMENT MANAGER
# ============================================================

class DocumentManager:
    """
    Handles document upload, indexing, listing, re-indexing,
    deletion, and statistics.

    Pipeline:

        File
          ↓
        Loader
          ↓
        Cleaning + Chunking
          ↓
        Embeddings
          ↓
        ChromaDB
    """

    def __init__(self):
        DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_service = EmbeddingService()

        self.vector_store = ChromaStore(
            str(CHROMA_DIR)
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_file(
        self,
        filename: str,
    ) -> None:

        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:

            supported = ", ".join(
                sorted(SUPPORTED_EXTENSIONS)
            )

            raise ValueError(
                f"Unsupported file type: "
                f"{extension or 'unknown'}. "
                f"Supported types: {supported}"
            )

    # ========================================================
    # SAVE UPLOADED FILE
    # ========================================================

    def save_uploaded_file(
        self,
        uploaded_file,
    ) -> Path:

        self.validate_file(
            uploaded_file.name
        )

        filename = Path(
            uploaded_file.name
        ).name

        destination = DOCUMENTS_DIR / filename

        if destination.exists():

            raise FileExistsError(
                f"A document named '{filename}' "
                f"already exists. "
                f"Delete it or use re-index instead."
            )

        with open(
            destination,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        return destination

    # ========================================================
    # INDEX DOCUMENT
    # ========================================================

    def index_document(
        self,
        file_path: Path,
    ) -> Dict:

        file_path = Path(file_path)

        if not file_path.exists():

            raise FileNotFoundError(
                f"Document does not exist: "
                f"{file_path}"
            )

        self.validate_file(
            file_path.name
        )

        # ----------------------------------------------------
        # 1. LOAD
        # ----------------------------------------------------

        documents = load_document(
            file_path
        )

        if not documents:

            raise ValueError(
                f"No readable text was found "
                f"in {file_path.name}"
            )

        # ----------------------------------------------------
        # 2. CHUNK
        # ----------------------------------------------------

        chunks = create_chunks(
            documents,
            chunk_size=500,
            chunk_overlap=100,
        )

        if not chunks:

            raise ValueError(
                f"No chunks were created for "
                f"{file_path.name}"
            )

        # ----------------------------------------------------
        # 3. REMOVE EMPTY CHUNKS
        # ----------------------------------------------------

        usable_chunks = [
            chunk
            for chunk in chunks
            if chunk.get(
                "text",
                ""
            ).strip()
        ]

        if not usable_chunks:

            raise ValueError(
                f"No usable text chunks were "
                f"created for {file_path.name}"
            )

        # ----------------------------------------------------
        # 4. TEXT
        # ----------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in usable_chunks
        ]

        # ----------------------------------------------------
        # 5. EMBEDDINGS
        # ----------------------------------------------------

        embeddings = (
            self.embedding_service.embed_texts(
                texts
            )
        )

        # ----------------------------------------------------
        # 6. CHROMADB
        # ----------------------------------------------------

        added_count = (
            self.vector_store.add_chunks(
                usable_chunks,
                embeddings,
            )
        )

        return {
            "filename": file_path.name,
            "chunks": len(usable_chunks),
            "indexed": added_count,
            "status": "success",
        }

    # ========================================================
    # UPLOAD + INDEX
    # ========================================================

    def upload_and_index(
        self,
        uploaded_file,
    ) -> Dict:

        file_path = self.save_uploaded_file(
            uploaded_file
        )

        try:

            result = self.index_document(
                file_path
            )

            return result

        except Exception:

            # If indexing fails, remove the newly
            # uploaded physical file.

            if file_path.exists():
                file_path.unlink()

            raise

    # ========================================================
    # LIST DOCUMENTS
    # ========================================================

    def list_documents(self) -> List[Dict]:

        documents = []

        if not DOCUMENTS_DIR.exists():
            return documents

        for path in sorted(
            DOCUMENTS_DIR.iterdir()
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in SUPPORTED_EXTENSIONS
            ):
                continue

            documents.append(
                {
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "path": str(path),
                }
            )

        return documents

    # ========================================================
    # GET DOCUMENT
    # ========================================================

    def get_document(
        self,
        filename: str,
    ) -> Dict:

        filename = Path(
            filename
        ).name

        file_path = (
            DOCUMENTS_DIR / filename
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Document not found: "
                f"{filename}"
            )

        self.validate_file(
            filename
        )

        return {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size,
            "path": str(file_path),
        }

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        filename: str,
    ) -> Dict:

        filename = Path(
            filename
        ).name

        file_path = (
            DOCUMENTS_DIR / filename
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Document not found: "
                f"{filename}"
            )

        # IMPORTANT:
        # Delete only vectors belonging to the exact
        # managed document path.
        deleted_chunks = (
            self.vector_store.delete_by_source_path(
                str(file_path)
            )
        )

        file_path.unlink()

        return {
            "filename": filename,
            "deleted_chunks": deleted_chunks,
            "status": "deleted",
        }

    # ========================================================
    # RE-INDEX ONE DOCUMENT
    # ========================================================

    def reindex_document(
        self,
        filename: str,
    ) -> Dict:

        filename = Path(
            filename
        ).name

        file_path = (
            DOCUMENTS_DIR / filename
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Document not found: "
                f"{filename}"
            )

        self.validate_file(
            filename
        )

        # IMPORTANT:
        # Remove only vectors belonging to this exact
        # managed document path.

        deleted_chunks = (
            self.vector_store.delete_by_source_path(
                str(file_path)
            )
        )

        try:

            result = self.index_document(
                file_path
            )

            result["reindexed"] = True

            result[
                "previous_chunks_removed"
            ] = deleted_chunks

            return result

        except Exception as error:

            raise RuntimeError(
                f"Re-indexing failed for "
                f"'{filename}'. "
                f"The original file was preserved, "
                f"but it could not be indexed again. "
                f"Error: {error}"
            ) from error

    # ========================================================
    # RE-INDEX ALL DOCUMENTS
    # ========================================================

    def reindex_all(self) -> Dict:
        """
        Re-index every managed document.

        Returns a summary of successful
        and failed documents.
        """

        documents = self.list_documents()

        successful = []
        failed = []

        for document in documents:

            filename = document[
                "filename"
            ]

            try:

                result = (
                    self.reindex_document(
                        filename
                    )
                )

                successful.append(
                    result
                )

            except Exception as error:

                failed.append(
                    {
                        "filename": filename,
                        "error": str(error),
                    }
                )

        return {
            "total_documents": len(
                documents
            ),
            "successful": successful,
            "failed": failed,
            "success_count": len(
                successful
            ),
            "failure_count": len(
                failed
            ),
            "status": (
                "success"
                if not failed
                else "partial"
            ),
        }

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(
        self,
    ) -> Dict:

        documents = self.list_documents()

        total_size = sum(
            document["size_bytes"]
            for document in documents
        )

        # Count only chunks belonging to managed
        # documents, rather than all chunks in ChromaDB.

        managed_sources = {
            str(
                DOCUMENTS_DIR
                / document["filename"]
            )
            for document in documents
        }

        indexed_chunks = 0

        if managed_sources:

            try:

                results = (
                    self.vector_store.collection.get(
                        include=["metadatas"]
                    )
                )

                metadatas = results.get(
                    "metadatas",
                    [],
                )

                for metadata in metadatas:

                    if not metadata:
                        continue

                    source = str(
                        metadata.get(
                            "source",
                            ""
                        )
                    )

                    if source in managed_sources:
                        indexed_chunks += 1

            except Exception:
                indexed_chunks = 0

        return {
            "document_count": len(
                documents
            ),
            "indexed_chunks": indexed_chunks,
            "total_size_bytes": total_size,
            "supported_extensions": sorted(
                SUPPORTED_EXTENSIONS
            ),
        }