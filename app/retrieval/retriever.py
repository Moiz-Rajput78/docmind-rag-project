# ============================================================
# DOCMIND — RETRIEVER
# ============================================================

from pathlib import Path
from typing import Any

from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"


class Retriever:
    """
    Retrieval layer for DocMind.

    Converts a user question into an embedding and retrieves
    the most semantically relevant document chunks from ChromaDB.

    An existing EmbeddingService can be supplied so multiple
    retrieval components can share the same embedding model.
    """

    def __init__(
        self,
        chroma_dir: str | Path = CHROMA_DIR,
        collection_name: str = "docmind_documents",
        embedding_service: EmbeddingService | None = None,
    ):
        # Reuse an existing embedding service when provided.
        # This prevents loading the same Sentence Transformer model
        # multiple times inside AnswerService.
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )

        self.store = ChromaStore(
            persist_directory=chroma_dir,
            collection_name=collection_name,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-K most relevant chunks for a query.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = query.strip()

        # Convert question into embedding.
        query_embedding = self.embedding_service.embed_text(
            query
        )

        # Search ChromaDB using the existing ChromaStore interface.
        results = self.store.query(
            query_embedding,
            top_k,
        )

        retrieved: list[dict[str, Any]] = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            chunk_id = (
                ids[index]
                if index < len(ids)
                else None
            )

            retrieved.append(
                {
                    "rank": index + 1,
                    "id": chunk_id,
                    "text": document,
                    "distance": distance,
                    "metadata": metadata,
                    "source": metadata.get("source"),
                    "filename": metadata.get("filename"),
                    "file_type": metadata.get("file_type"),
                    "document_type": metadata.get("document_type"),
                    "page": metadata.get("page"),
                    "category": metadata.get("category"),
                    "chunk_number": metadata.get("chunk_number"),
                }
            )

        return retrieved

    def retrieve_multiple(
        self,
        queries: list[str],
        top_k: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retrieve results for multiple questions.
        """

        results: dict[str, list[dict[str, Any]]] = {}

        for query in queries:
            results[query] = self.retrieve(
                query=query,
                top_k=top_k,
            )

        return results

    def count(self) -> int:
        """Return the number of chunks currently stored."""

        return self.store.count()