"""
Retrieve relevant chunks for a user query using FAISS + SBERT.
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, EMBEDDING_MODEL, TOP_K


class Retriever:
    """Semantic retrieval over the indexed chunk corpus."""

    def __init__(self):
        print("Loading retriever...")

        # Load SBERT model for query encoding
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Load FAISS index
        index_path = PROCESSED_DIR / "index.faiss"
        self.index = faiss.read_index(str(index_path))

        # Load chunk metadata and texts
        with open(PROCESSED_DIR / "chunk_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(PROCESSED_DIR / "chunk_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)

        print(f"Retriever ready: {self.index.ntotal} chunks indexed")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Returns:
            List of dicts with keys: text, doc_name, doc_id, chunk_id, score, chunk_index
        """
        # Encode query
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Search
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots
            results.append({
                "text": self.texts[idx],
                "doc_name": self.metadata[idx]["doc_name"],
                "doc_id": self.metadata[idx]["doc_id"],
                "chunk_id": self.metadata[idx]["chunk_id"],
                "chunk_index": self.metadata[idx]["chunk_index"],
                "score": float(score),
            })

        return results


if __name__ == "__main__":
    # Quick test
    retriever = Retriever()
    query = "What is a transformer architecture?"
    print(f"\nQuery: {query}")
    results = retriever.retrieve(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Source: {r['doc_name']}")
        print(f"Text: {r['text'][:200]}...")
