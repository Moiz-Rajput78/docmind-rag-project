from typing import List, Union

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Local embedding service using Sentence Transformers.

    Default model:
        sentence-transformers/all-MiniLM-L6-v2

    The model produces 384-dimensional embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.")

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Convert one piece of text into an embedding vector.
        """

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Convert multiple pieces of text into embedding vectors.
        """

        if not texts:
            return []

        for text in texts:
            if not text or not text.strip():
                raise ValueError(
                    "Cannot embed an empty text."
                )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def dimension(self) -> int:
        """
        Return the embedding vector dimension.
        """

        return self.model.get_sentence_embedding_dimension()