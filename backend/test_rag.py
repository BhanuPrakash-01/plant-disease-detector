from app.rag.retriever import CabiRetriever
from app.rag.generator import GeminiGenerator


retriever = CabiRetriever()
generator = GeminiGenerator()

question = "What are the symptoms of tomato leaf miner?"

import time
import numpy as np

question = "What are the symptoms of tomato leaf miner?"

# Warm up embedding
query_embedding = retriever.embedder.encode(
    [question],
    normalize_embeddings=True,
    convert_to_numpy=True
)

# FAISS benchmark
times = []

for _ in range(20):
    start = time.perf_counter()

    retriever.index.search(
        query_embedding,
        5
    )

    times.append(time.perf_counter() - start)

print("FAISS mean:", np.mean(times))
print("FAISS median:", np.median(times))
print("FAISS min:", np.min(times))
print("FAISS max:", np.max(times))


times = []

for _ in range(10):
    start = time.perf_counter()

    retriever.embedder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    times.append(time.perf_counter() - start)

print("Embedding mean:", np.mean(times))
print("Embedding median:", np.median(times))
print("Embedding min:", np.min(times))
print("Embedding max:", np.max(times))