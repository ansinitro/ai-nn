"""
Download ML research papers from arXiv for the RAG corpus.
Downloads 50-100 papers across various ML topics.
"""
import os
import time
import arxiv
from pathlib import Path

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR

# Search queries covering diverse ML topics
QUERIES = [
    "deep learning survey",
    "transformer architecture attention",
    "convolutional neural network image classification",
    "recurrent neural network sequence modeling",
    "generative adversarial network",
    "reinforcement learning policy gradient",
    "natural language processing BERT GPT",
    "graph neural network",
    "diffusion model generative",
    "meta-learning few-shot learning",
    "self-supervised learning representation",
    "neural architecture search",
    "federated learning privacy",
    "knowledge distillation model compression",
    "object detection YOLO",
]

MAX_PAPERS = 75
PAPERS_PER_QUERY = 7


def download_papers():
    """Download ML papers from arXiv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing papers
    existing = set(f.stem for f in DATA_DIR.glob("*.pdf"))
    print(f"Found {len(existing)} existing papers in {DATA_DIR}")

    downloaded = len(existing)
    new_downloads = 0

    for i, query in enumerate(QUERIES):
        if downloaded >= MAX_PAPERS:
            break

        print(f"\n[{i+1}/{len(QUERIES)}] Searching: '{query}'")

        try:
            search = arxiv.Search(
                query=query,
                max_results=PAPERS_PER_QUERY,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            client = arxiv.Client()
            for result in client.results(search):
                if downloaded >= MAX_PAPERS:
                    break

                paper_id = result.get_short_id().replace("/", "_")
                filename = f"{paper_id}.pdf"
                filepath = DATA_DIR / filename

                if filepath.exists() or paper_id in existing:
                    continue

                try:
                    result.download_pdf(dirpath=str(DATA_DIR), filename=filename)
                    downloaded += 1
                    new_downloads += 1
                    title = result.title[:70]
                    print(f"  [{downloaded}/{MAX_PAPERS}] {title}")
                    time.sleep(1)  # Be nice to arXiv servers
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)[:60]}")

        except Exception as e:
            print(f"  ✗ Search failed: {str(e)[:60]}")

    print(f"\n{'='*50}")
    print(f"Total papers: {downloaded} ({new_downloads} new)")
    print(f"Location: {DATA_DIR}")


if __name__ == "__main__":
    download_papers()
