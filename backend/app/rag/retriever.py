from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

CHUNKS_PATH = DATA_DIR / "cabi_chunks.csv"
FAISS_PATH = DATA_DIR / "cabi_faiss.index"

EMBEDDING_MODEL = "BAAI/bge-m3"


class CabiRetriever:

    def __init__(self):

        self.chunks = pd.read_csv(
            CHUNKS_PATH,
            encoding="utf-8-sig"
        )

        self.index = faiss.read_index(
            str(FAISS_PATH)
        )

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def retrieve(self, query: str, top_k: int = 5):

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            row = self.chunks.iloc[int(idx)]

            results.append({
                "score": float(score),
                "chunk_id": int(row["chunk_id"]),
                "document_id": row["document_id"],
                "title": row["title"],
                "section": row["section"],
                "text": row["text"]
            })

        return results