from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb


class ChromaStore:
    """
    Local ChromaDB vector store for DocMind.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "docmind_documents",
    ):
        self.persist_directory = Path(persist_directory)

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "DocMind RAG document chunks"
            },
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Add chunk records and their embeddings to ChromaDB.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):

            metadata = dict(
                chunk.get("metadata", {})
            )

            source = metadata.get(
                "source",
                "unknown"
            )

            chunk_number = metadata.get(
                "chunk_number",
                index + 1
            )

            page = metadata.get("page")

            # Chroma metadata values cannot be None.
            # Convert missing page numbers to -1.
            if page is None:
                page = -1

            # Create a deterministic unique ID.
            chunk_id = (
                f"{source}"
                f"::page_{page}"
                f"::chunk_{chunk_number}"
            )

            ids.append(chunk_id)

            documents.append(
                chunk.get("text", "")
            )

            chroma_metadata = {
                "source": str(source),
                "filename": str(
                    metadata.get(
                        "filename",
                        "unknown"
                    )
                ),
                "file_type": str(
                    metadata.get(
                        "file_type",
                        "unknown"
                    )
                ),
                "document_type": str(
                    metadata.get(
                        "document_type",
                        "unknown"
                    )
                ),
                "page": int(page),
                "chunk_number": int(
                    chunk_number
                ),
                "chunk_size": int(
                    metadata.get(
                        "chunk_size",
                        0
                    )
                ),
                "chunk_overlap": int(
                    metadata.get(
                        "chunk_overlap",
                        0
                    )
                ),
            }

            # Add category based on knowledge-base directory.
            source_path = Path(str(source))

            category = (
                source_path.parent.name
            )

            chroma_metadata["category"] = (
                category
            )

            metadatas.append(
                chroma_metadata
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def count(self) -> int:
        """
        Return number of stored chunks.
        """

        return self.collection.count()

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search the vector store using an embedding.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": top_k,
        }

        if where is not None:
            kwargs["where"] = where

        return self.collection.query(
            **kwargs
        )

    def clear(self) -> None:
        """
        Delete the current collection and recreate it.
        """

        self.client.delete_collection(
            name=self.collection_name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": (
                        "DocMind RAG document chunks"
                    )
                },
            )
        )