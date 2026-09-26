"""
CABI knowledge-base retriever.

Loads a FAISS index + chunk metadata once and provides
semantic-search retrieval over the CABI corpus.
"""

import logging

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class CabiRetriever:
    """Semantic retriever backed by FAISS + BGE-M3."""

    def __init__(self):
        logger.info("Loading CABI chunks from %s …", settings.chunks_csv_path)
        self.chunks = pd.read_csv(
            settings.chunks_csv_path,
            encoding="utf-8-sig",
        )

        logger.info("Loading FAISS index from %s …", settings.faiss_index_path)
        self.index = faiss.read_index(settings.faiss_index_path)

        logger.info("Loading embedding model %s …", settings.embedding_model)
        self.embedder = SentenceTransformer(settings.embedding_model)
        logger.info("Retriever ready — %d chunks indexed.", len(self.chunks))

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Embed the *query* at runtime, search FAISS, and return
        the top-k matching CABI chunks with metadata.
        """
        if top_k is None:
            top_k = settings.rag_top_k

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            row = self.chunks.iloc[int(idx)]
            results.append({
                "score": round(float(score), 4),
                "chunk_id": int(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "title": str(row["title"]),
                "section": str(row["section"]),
                "text": str(row["text"]),
            })

        return results
