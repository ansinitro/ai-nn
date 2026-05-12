"""
Chunk documents into smaller passages for embedding and retrieval.
Uses LangChain's RecursiveCharacterTextSplitter for robust chunking.
Usage: python -m src.chunk_docs
"""
import json
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents():
    """Split all documents into overlapping chunks."""
    docs_path = PROCESSED_DIR / "documents.json"
    if not docs_path.exists():
        print(f"No documents found at {docs_path}")
        print("Run 'python -m src.parse_pdfs' first.")
        return

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents")
    print(f"Chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars")
    print("-" * 50)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk_{i:04d}",
                "doc_id": doc["doc_id"],
                "doc_name": doc["filename"],
                "chunk_index": i,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })

        print(f"  {doc['filename']}: {len(chunks)} chunks")

    # Save chunks
    output_path = PROCESSED_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Statistics
    avg_len = sum(c["char_count"] for c in all_chunks) / max(len(all_chunks), 1)
    print("-" * 50)
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Avg chunk length: {avg_len:.0f} chars (~{avg_len / 4:.0f} tokens)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    chunk_documents()
