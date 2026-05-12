#!/bin/bash
# RAG QA System - One-command Setup
set -e

echo "═══════════════════════════════════════════════"
echo "  RAG QA System - Environment Setup"
echo "═══════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Create virtual environment
if [ ! -d "rag_env" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv rag_env
else
    echo "[1/5] Virtual environment already exists."
fi

source rag_env/bin/activate
echo "  Python: $(python --version)"

# 2. Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip -q

# 3. Install PyTorch with CUDA support
echo "[3/5] Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# 4. Install remaining requirements
echo "[4/5] Installing project dependencies..."
pip install -r requirements.txt -q

# 5. Create directories
echo "[5/5] Creating project directories..."
mkdir -p data/papers processed models eval

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup Complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Activate environment:  source rag_env/bin/activate"
echo "  2. Download papers:       python download_papers.py"
echo "  3. Or place PDFs in:      data/papers/"
echo ""
echo "To download the LLM model (~4 GB):"
echo "  pip install huggingface-hub"
echo "  huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir models/"
echo ""
