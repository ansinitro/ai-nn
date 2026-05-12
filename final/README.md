# RAG QA System — Variant 5

> Retrieval-Augmented Generation system for question-answering on ML research papers.

## Overview

This system answers questions about machine learning by retrieving relevant passages from a corpus of 50–100 research papers and generating answers using an LLM.

**Pipeline:** PDF Papers → Text Extraction → Chunking → SBERT Embeddings → FAISS Index → Retrieval → LLM Generation → Gradio UI

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | Sentence-BERT (`all-mpnet-base-v2`, 768-dim) |
| **Vector DB** | FAISS (FlatIP, cosine similarity) |
| **LLM** | Llama 2 7B Chat (GGUF Q4, 4-bit quantized) |
| **LLM Fallback** | OpenAI API (GPT-3.5-turbo) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |
| **Web UI** | Gradio |
| **Evaluation** | ROUGE-1/L, BERTScore |

## Requirements

- Python 3.10+
- NVIDIA GPU with 6+ GB VRAM (for local LLM) or OpenAI API key
- ~15 GB disk space (models + papers)

## Quick Start

```bash
# 1. Setup environment
bash setup.sh

# 2. Activate
source rag_env/bin/activate

# 3. Download papers (or place your own PDFs in data/papers/)
python download_papers.py

# 4. Download LLM model (~4 GB)
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
    llama-2-7b-chat.Q4_K_M.gguf --local-dir models/

# 5. Run the pipeline
python -m src.parse_pdfs       # Extract text from PDFs
python -m src.chunk_docs       # Chunk into passages
python -m src.embed_chunks     # Compute SBERT embeddings
python -m src.build_index      # Build FAISS index

# 6. Launch the web demo
python app.py
# Open http://localhost:7860

# 7. Run evaluation
python -m src.evaluate
```

## Project Structure

```
final/
├── app.py                  # Gradio web interface
├── config.py               # Central configuration
├── download_papers.py      # ArXiv paper downloader
├── setup.sh                # One-command setup
├── requirements.txt        # Dependencies
├── src/
│   ├── parse_pdfs.py       # PDF text extraction
│   ├── chunk_docs.py       # Text chunking
│   ├── embed_chunks.py     # SBERT embedding
│   ├── build_index.py      # FAISS index building
│   ├── retriever.py        # Semantic retrieval
│   ├── llm.py              # LLM inference (GGUF + API)
│   ├── rag_pipeline.py     # Full RAG orchestration
│   └── evaluate.py         # ROUGE + BERTScore evaluation
├── data/papers/            # PDF files go here
├── processed/              # Generated artifacts
├── models/                 # LLM model files
└── eval/                   # Evaluation results
```

## Evaluation

The system is evaluated on 15 manually created test questions covering:
- Transformers & attention mechanisms
- CNNs & computer vision
- GANs & generative models
- Reinforcement learning
- NLP & transfer learning
- Optimization techniques
- And more...

Metrics computed:
- **ROUGE-1 F1** — unigram overlap with reference answers
- **ROUGE-L F1** — longest common subsequence overlap
- **BERTScore F1** — semantic similarity using BERT embeddings

Results are saved to `eval/eval_results.csv` and `eval/eval_report.json`.

## API Fallback

If the local GGUF model is unavailable, set an OpenAI API key:

```bash
export OPENAI_API_KEY="sk-your-key-here"
python app.py
```

The system will automatically use GPT-3.5-turbo as a fallback.

## Hardware Tested

- GPU: NVIDIA RTX 2060 (6 GB VRAM)
- CPU: Intel i5-9400 @ 2.90GHz
- RAM: 16 GB
- OS: Linux (Ubuntu)
