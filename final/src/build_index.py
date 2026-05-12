"""
Build FAISS index from precomputed embeddings.
Usage: python -m src.build_index
"""
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR


def build_index():
    """Build and save a FAISS FlatIP index (cosine similarity for normalized vectors)."""
    emb_path = PROCESSED_DIR / "embeddings.npy"
    if not emb_path.exists():
        print(f"No embeddings found at {emb_path}")
        print("Run 'python -m src.embed_chunks' first.")
        return

    embeddings = np.load(emb_path).astype("float32")
    print(f"Loaded embeddings: {embeddings.shape}")

    # Ensure normalization (for cosine similarity via inner product)
    faiss.normalize_L2(embeddings)

    # Build index — Inner Product on normalized vectors = cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    index_path = PROCESSED_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))

    # Sanity check: query first vector, should return itself as top-1
    D, I = index.search(embeddings[:1], k=3)
    print(f"Sanity check — top 3 for chunk 0: indices={I[0].tolist()}, scores={D[0].tolist()}")
    assert I[0][0] == 0, "Sanity check failed: first result should be the query itself!"

    print(f"\nIndex built: {index.ntotal} vectors, dim={dim}")
    print(f"Index size: {index_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Saved to: {index_path}")


if __name__ == "__main__":
    build_index()
