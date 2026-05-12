"""
Extract and clean text from PDF files.
Usage: python -m src.parse_pdfs
"""
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PROCESSED_DIR, MIN_TEXT_LENGTH


def extract_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Normalize unicode dashes
    text = re.sub(r'[\u2013\u2014]', '-', text)
    # Remove arXiv headers
    text = re.sub(r'arXiv:\d+\.\d+v\d+\s*\[.*?\]\s*\d+\s+\w+\s+\d+', '', text)
    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    # Collapse excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove hyphenated line breaks (e.g., "con-\ntinue" → "continue")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()


def parse_all_pdfs():
    """Parse all PDFs in data/papers/ and save cleaned text."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        print("Run 'python download_papers.py' first, or place PDFs there manually.")
        return

    print(f"Found {len(pdf_files)} PDF files in {DATA_DIR}")
    print("-" * 50)

    documents = []
    skipped = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            raw_text = extract_text(pdf_path)
            cleaned = clean_text(raw_text)

            if len(cleaned) < MIN_TEXT_LENGTH:
                print(f"  [{i}/{len(pdf_files)}] SKIP {pdf_path.name} ({len(cleaned)} chars - too short)")
                skipped += 1
                continue

            documents.append({
                "doc_id": pdf_path.stem,
                "filename": pdf_path.name,
                "text": cleaned,
                "char_count": len(cleaned),
            })
            print(f"  [{i}/{len(pdf_files)}] OK   {pdf_path.name} ({len(cleaned):,} chars)")

        except Exception as e:
            print(f"  [{i}/{len(pdf_files)}] ERR  {pdf_path.name}: {e}")
            skipped += 1

    # Save
    output_path = PROCESSED_DIR / "documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    total_chars = sum(d["char_count"] for d in documents)
    print("-" * 50)
    print(f"Parsed: {len(documents)} documents ({skipped} skipped)")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parse_all_pdfs()
