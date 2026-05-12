"""
Central configuration for the RAG QA System.
All paths, model names, and hyperparameters in one place.
"""
import os
from pathlib import Path

# ─── Directories ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "papers"
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "eval"

# ─── PDF Parsing ───────────────────────────────────────────────
MIN_TEXT_LENGTH = 500  # Skip PDFs shorter than this (chars)

# ─── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE = 3000        # characters (~750 tokens)
CHUNK_OVERLAP = 400      # characters (~100 tokens)

# ─── Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL = "all-mpnet-base-v2"   # 768-dim, best quality
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIM = 768

# ─── Retrieval ─────────────────────────────────────────────────
TOP_K = 5

# ─── LLM ───────────────────────────────────────────────────────
GGUF_MODEL_PATH = MODELS_DIR / "llama-2-7b-chat.Q4_K_M.gguf"
LLM_CONTEXT_LENGTH = 2048     # Reduced from 4096 to fit 6GB VRAM with display
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.1
LLM_N_GPU_LAYERS = 25        # 25 of 33 layers on GPU (leaves room for display ~1.4GB)

# ─── API Fallback ──────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# ─── Gradio ────────────────────────────────────────────────────
GRADIO_PORT = 7860
