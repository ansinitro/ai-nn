"""
Compute SBERT embeddings for all text chunks.
Usage: python -m src.embed_chunks
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


def embed_chunks():
    """Encode all chunks with SBERT and save embeddings."""
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        print(f"No chunks found at {chunks_path}")
        print("Run 'python -m src.chunk_docs' first.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    print(f"Loaded {len(texts)} chunks")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("-" * 50)

    # Load model
    print("Loading SBERT model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Encode
    print(f"Encoding {len(texts)} chunks (batch_size={EMBEDDING_BATCH_SIZE})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity via inner product
    )
    elapsed = time.time() - start

    # Save embeddings
    emb_path = PROCESSED_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)

    # Save metadata (without text to save space)
    meta_path = PROCESSED_DIR / "chunk_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save texts separately (needed for retrieval)
    texts_path = PROCESSED_DIR / "chunk_texts.json"
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    # Report
    print("-" * 50)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Time: {elapsed:.1f}s ({len(texts) / elapsed:.1f} chunks/s)")
    print(f"Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    print(f"Saved: {emb_path}, {meta_path}, {texts_path}")


if __name__ == "__main__":
    embed_chunks()
