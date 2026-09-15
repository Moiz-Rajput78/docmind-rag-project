"""
DocMind RAG - Hybrid Retrieval

Combines:

    Dense semantic retrieval
            +
    BM25 lexical retrieval
            ↓
       Score fusion
            ↓
    Optional cross-encoder reranking

The existing Retriever class remains unchanged.

This module is intentionally separate so the hybrid pipeline can be
evaluated before becoming the production retrieval strategy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"


class HybridRetriever:
    """
    Hybrid retrieval using Dense + BM25, with optional cross-encoder
    reranking.

    Default experimental configuration:

        Dense weight = 0.6
        BM25 weight  = 0.4

    These weights were selected from the project's 30-question
    hybrid weight experiment. They should be treated as experiment
    results for this dataset, not as universal optimal weights.

    An existing EmbeddingService can be supplied so the Dense
    retrieval component can share the embedding model with other
    retrievers.
    """

    def __init__(
        self,
        chroma_dir: str | Path = CHROMA_DIR,
        collection_name: str = "docmind_documents",
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_service: EmbeddingService | None = None,
    ) -> None:

        if dense_weight < 0:
            raise ValueError(
                "dense_weight cannot be negative."
            )

        if bm25_weight < 0:
            raise ValueError(
                "bm25_weight cannot be negative."
            )

        total_weight = dense_weight + bm25_weight

        if total_weight <= 0:
            raise ValueError(
                "dense_weight + bm25_weight must be greater than zero."
            )

        self.dense_weight = dense_weight / total_weight
        self.bm25_weight = bm25_weight / total_weight

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

        self.reranker_model_name = reranker_model
        self._reranker: CrossEncoder | None = None

        # BM25 is built lazily because the collection may change after
        # document uploads/deletions.
        self._bm25: BM25Okapi | None = None
        self._bm25_documents: list[dict[str, Any]] = []

    # ============================================================
    # BM25 INITIALIZATION
    # ============================================================

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple tokenization for BM25.

        Lowercasing keeps lexical matching case-insensitive.
        """

        return text.lower().split()

    def _load_all_documents(self) -> list[dict[str, Any]]:
        """
        Load all indexed Chroma chunks.

        Returns normalized records containing the same fields used
        by the normal Retriever.
        """

        results = self.store.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        ids = results.get("ids") or []

        records: list[dict[str, Any]] = []

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            chunk_id = (
                ids[index]
                if index < len(ids)
                else None
            )

            records.append(
                self._make_result(
                    document=document or "",
                    metadata=metadata,
                    chunk_id=chunk_id,
                    distance=None,
                    rank=index + 1,
                )
            )

        return records

    def _build_bm25_index(self) -> None:
        """
        Build or rebuild the BM25 index from current Chroma data.
        """

        self._bm25_documents = self._load_all_documents()

        tokenized_documents = [
            self._tokenize(record["text"])
            for record in self._bm25_documents
        ]

        if tokenized_documents:
            self._bm25 = BM25Okapi(
                tokenized_documents
            )
        else:
            self._bm25 = None

    # ============================================================
    # RESULT NORMALIZATION
    # ============================================================

    @staticmethod
    def _make_result(
        document: str,
        metadata: dict[str, Any],
        chunk_id: str | None,
        distance: float | None,
        rank: int,
    ) -> dict[str, Any]:
        """
        Create the same result structure used by Retriever.
        """

        return {
            "rank": rank,
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

    # ============================================================
    # DENSE RETRIEVAL
    # ============================================================

    def _dense_retrieve(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidate chunks using ChromaDB semantic search.
        """

        query_embedding = self.embedding_service.embed_text(
            query
        )

        results = self.store.query(
            query_embedding,
            candidate_k,
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        ids = results.get(
            "ids",
            [[]],
        )[0]

        retrieved: list[dict[str, Any]] = []

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
                self._make_result(
                    document=document,
                    metadata=metadata,
                    chunk_id=chunk_id,
                    distance=distance,
                    rank=index + 1,
                )
            )

        return retrieved

    # ============================================================
    # BM25 RETRIEVAL
    # ============================================================

    def _bm25_retrieve(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidate chunks using BM25 lexical matching.
        """

        self._build_bm25_index()

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results: list[dict[str, Any]] = []

        for index in ranked_indices[:candidate_k]:

            index = int(index)

            if index >= len(
                self._bm25_documents
            ):
                continue

            original = self._bm25_documents[
                index
            ]

            result = dict(original)

            # BM25 does not have a Chroma distance.
            result["distance"] = None

            # Keep the lexical score available for debugging.
            result["bm25_score"] = float(
                scores[index]
            )

            result["rank"] = len(results) + 1

            results.append(result)

        return results

    # ============================================================
    # SCORE NORMALIZATION
    # ============================================================

    @staticmethod
    def _min_max_normalize(
        scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Normalize scores to [0, 1].

        This prevents raw Dense/BM25 scores from dominating simply
        because they use different numerical scales.
        """

        if not scores:
            return {}

        values = list(
            scores.values()
        )

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:

            return {
                key: 1.0
                for key in scores
            }

        return {
            key: (
                (value - minimum)
                / (maximum - minimum)
            )
            for key, value in scores.items()
        }

    # ============================================================
    # HYBRID FUSION
    # ============================================================

    def _fuse_results(
        self,
        dense_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fuse Dense and BM25 rankings using normalized scores.

        A weighted score fusion is used so both retrieval methods
        contribute even though their raw score scales differ.
        """

        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        records: dict[str, dict[str, Any]] = {}

        # --------------------------------------------------------
        # Dense scores
        # --------------------------------------------------------

        for result in dense_results:

            chunk_id = result.get("id")

            if not chunk_id:
                continue

            distance = result.get("distance")

            if distance is None:
                continue

            # Chroma distance is lower-is-better.
            # Convert it to a similarity-like score.
            dense_scores[chunk_id] = 1.0 / (
                1.0 + float(distance)
            )

            records[chunk_id] = result

        # --------------------------------------------------------
        # BM25 scores
        # --------------------------------------------------------

        for result in bm25_results:

            chunk_id = result.get("id")

            if not chunk_id:
                continue

            bm25_scores[chunk_id] = float(
                result.get(
                    "bm25_score",
                    0.0,
                )
            )

            if chunk_id not in records:
                records[chunk_id] = result

        normalized_dense = self._min_max_normalize(
            dense_scores
        )

        normalized_bm25 = self._min_max_normalize(
            bm25_scores
        )

        # --------------------------------------------------------
        # Weighted score fusion
        # --------------------------------------------------------

        fused_scores: dict[str, float] = {}

        all_ids = set(records)

        for chunk_id in all_ids:

            dense_score = normalized_dense.get(
                chunk_id,
                0.0,
            )

            bm25_score = normalized_bm25.get(
                chunk_id,
                0.0,
            )

            fused_scores[chunk_id] = (
                self.dense_weight
                * dense_score
                + self.bm25_weight
                * bm25_score
            )

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        fused_results: list[dict[str, Any]] = []

        for rank, chunk_id in enumerate(
            ranked_ids,
            start=1,
        ):

            result = dict(
                records[chunk_id]
            )

            result["rank"] = rank

            result["dense_score"] = normalized_dense.get(
                chunk_id,
                0.0,
            )

            result["bm25_score"] = normalized_bm25.get(
                chunk_id,
                0.0,
            )

            result["hybrid_score"] = fused_scores[
                chunk_id
            ]

            fused_results.append(result)

        return fused_results

    # ============================================================
    # CROSS-ENCODER
    # ============================================================

    def _get_reranker(self) -> CrossEncoder:
        """
        Lazily load the cross-encoder.

        Loading is delayed until reranking is explicitly requested.
        """

        if self._reranker is None:

            self._reranker = CrossEncoder(
                self.reranker_model_name
            )

        return self._reranker

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        final_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Rerank hybrid candidates with a cross-encoder.
        """

        if not results:
            return []

        if final_k < 1:
            raise ValueError(
                "final_k must be at least 1."
            )

        reranker = self._get_reranker()

        pairs = [
            (
                query,
                result.get("text", ""),
            )
            for result in results
        ]

        scores = reranker.predict(
            pairs
        )

        reranked: list[dict[str, Any]] = []

        for result, score in zip(
            results,
            scores,
        ):

            updated = dict(result)

            updated["reranker_score"] = float(
                score
            )

            reranked.append(
                updated
            )

        reranked.sort(
            key=lambda item: item.get(
                "reranker_score",
                float("-inf"),
            ),
            reverse=True,
        )

        reranked = reranked[
            :final_k
        ]

        for rank, result in enumerate(
            reranked,
            start=1,
        ):
            result["rank"] = rank

        return reranked

    # ============================================================
    # PUBLIC RETRIEVAL API
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run the complete hybrid retrieval pipeline.

        Pipeline:

            Dense
              +
            BM25
              ↓
            Fusion
              ↓
            Optional Cross-Encoder
              ↓
            Top-K
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than "
                "or equal to top_k."
            )

        query = query.strip()

        dense_results = self._dense_retrieve(
            query=query,
            candidate_k=candidate_k,
        )

        bm25_results = self._bm25_retrieve(
            query=query,
            candidate_k=candidate_k,
        )

        fused_results = self._fuse_results(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        candidates = fused_results[
            :candidate_k
        ]

        if use_reranker:

            final_results = self.rerank(
                query=query,
                results=candidates,
                final_k=top_k,
            )

        else:

            final_results = candidates[
                :top_k
            ]

            for rank, result in enumerate(
                final_results,
                start=1,
            ):
                result["rank"] = rank

        return final_results

    # ============================================================
    # MULTIPLE QUESTIONS
    # ============================================================

    def retrieve_multiple(
        self,
        queries: list[str],
        top_k: int = 3,
        candidate_k: int = 10,
        use_reranker: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retrieve results for multiple questions.
        """

        results: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for query in queries:

            results[query] = self.retrieve(
                query=query,
                top_k=top_k,
                candidate_k=candidate_k,
                use_reranker=use_reranker,
            )

        return results

    # ============================================================
    # COUNT
    # ============================================================

    def count(self) -> int:
        """
        Return the number of indexed chunks.
        """

        return self.store.count()